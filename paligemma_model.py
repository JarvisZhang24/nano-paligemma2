"""PaliGemma For Conditional Generation.

This module implements the PaliGemma model for conditional generation, which combines
a vision encoder (SigLIP) with a language model (Gemma) to generate text from images.

The model processes images through a vision tower, projects the visual features into
the language model's embedding space, and then generates text conditioned on both
the image and input text tokens.

Example:
    >>> config = PaliGemmaConfig()
    >>> model = PaliGemmaForConditionalGeneration(config)
    >>> outputs = model(pixel_values=pixel_values, input_ids=input_ids)

The code is annotated with Google-style docstrings and type hints.
"""

import torch
from torch import nn
from typing import Optional, Tuple

from vision_encoder import SiglipVisionModel
from config import PaliGemmaConfig, SiglipVisionConfig, GemmaConfig
from gemma_model import GemmaForCausalLM
from kv_cache import KVCache


class PaliGemmaMultiModalProjector(nn.Module):
    '''
    This class implements the multi-modal projector for the PaliGemma model.
    It projects the image features to the text embedding space.
    '''

    def __init__(self, config: PaliGemmaConfig) -> None:
        '''
        Initialize the multi-modal projector.

        Args:
            config: PaliGemmaConfig containing model parameters.
        '''
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Projection_Dim]
        hidden_states = self.linear(image_features)
        return hidden_states

class PaliGemmaForConditionalGeneration(nn.Module):
    """PaliGemma model for conditional generation from images and text.

    This model combines a vision encoder (SigLIP) with a language model (Gemma)
    to generate text conditioned on input images. The model processes images
    through the vision tower, projects the visual features into the language
    model's embedding space, and generates text autoregressively.

    The forward pass involves:
    1. Extracting visual features from input images
    2. Projecting visual features to match language model embeddings
    3. Merging visual and text features in the input sequence
    4. Generating output through the language model

    Attributes:
        config: Model configuration.
        vision_tower: Vision encoder (SigLIP model).
        multi_modal_projector: Projects vision features to text embedding space.
        language_model: Gemma language model for text generation.
        vocab_size: Size of the vocabulary.
        pad_token_id: ID of the padding token.

    Example:
        >>> config = PaliGemmaConfig()
        >>> model = PaliGemmaForConditionalGeneration(config)
        >>> outputs = model(pixel_values=pixel_values, input_ids=input_ids)
        
    """

    def __init__(self, config: PaliGemmaConfig) -> None:
        """Initialize PaliGemmaForConditionalGeneration.

        Args:
            config: PaliGemmaConfig containing model parameters.
        """
        super().__init__()
        self.config = config

        # Initialize vision tower: [batch_size, 3, 224, 224] -> [batch_size, 196, 768]
        self.vision_tower = SiglipVisionModel(config.vision_config)

        # Initialize multimodal projector: [batch_size, 196, 768] -> [batch_size, 196, 2048]
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)

        # Set vocabulary size
        self.vocab_size = config.vocab_size

        # Initialize language model
        self.language_model = GemmaForCausalLM(config.text_config)

        # Set pad token ID
        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else -1
	
    def tie_weights(self):
        return self.language_model.tie_weights()


    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        input_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Merge image features with text embeddings.
        
        Args:
            image_features: Image features.
            input_embeds: Text embeddings.
            input_ids: Input IDs.
            attention_mask: Attention mask.
            kv_cache: Key-value cache.

        Returns:
            final_embedding: Merged embeddings.
            causal_mask: Causal mask.
            position_ids: Position IDs.
        """
        # Get the shape of the image features and the input embeddings
        # [batch_size, num_patches, embed_dim]
        _, _, embed_dim = image_features.shape

        # Get the shape of the input embeddings
        # [batch_size, seq_len]
        batch_size, sequence_length = input_ids.shape
        
        # Get the dtype and device of the input embeddings
        dtype, device = input_embeds.dtype, input_embeds.device
        
        # Scale the image features by the square root of the hidden size
        # Shape: [batch_size, num_patches, embed_dim]
        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        # Shape: [batch_size, seq_len, embed_dim]
        final_embedding = torch.zeros(
            batch_size, sequence_length, embed_dim, dtype=input_embeds.dtype, device=input_embeds.device
        )

        '''
        Suppose we have a sequence of tokens ids: [567, 567, 567, 567, 567, 1, 56, 67, 89, 11, 2]
        where 567 is the image token, 1 is the padding token and 56, 67, 89, 11 are text tokens, 2 is the \n token.

        We want to merge the image features with the text features and mask out all the padding tokens.

        We can do this by creating a mask that is True for the text tokens and False for the image tokens and padding tokens.

        We can then use this mask to merge the image features with the text features.

        '''

        # Text tokens are the ones that are not image tokens and not padding tokens
        # Shape: [Batch_Size, Seq_Len]. True for text tokens
        # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)

        # Image tokens are the ones that are image tokens
        # Shape: [Batch_Size, Seq_Len]. True for image tokens
        # [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        image_mask = input_ids == self.config.image_token_index

        # Padding tokens are the ones that are padding tokens ,but we don't use them in the final embedding.
        # Shape: [Batch_Size, Seq_Len]. True for padding tokens
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pad_mask = input_ids == self.pad_token_id

        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        # Shape: [Batch_Size, Seq_Len, Embed_Dim]. True for text tokens
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        # Shape: [Batch_Size, Seq_Len, Embed_Dim]. True for padding tokens
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        # Shape: [Batch_Size, Seq_Len, Embed_Dim]. True for image tokens
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        # Shape: [batch_size, seq_len, embed_dim]
        final_embedding = torch.where(text_mask_expanded, input_embeds, final_embedding)

        # Mask out the image tokens
        # Shape: [batch_size, seq_len, embed_dim]
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)

        # Mask out the padding tokens
        # Shape: [batch_size, seq_len, embed_dim]
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        #### CREATE THE ATTENTION MASK ####

        # Create the attention mask
        dtype, device = input_embeds.dtype, input_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = input_embeds.shape[1]

        # If we are in the prefill phase, we do not mask any token, because we're in the prefill phase
        # This only works when we have no padding
        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't need to mask anything, since each query should be able to attend all previous tokens. 
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)
        
        # If we are generating tokens, we need to create the position ids
        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            # [0, 1,2,3,4,... 255 , 256 ,257] 0~255 : image tokens, 256 : text tokens, 257 : \n token
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            # [Batch_Size, Seq_Len] -> [Batch_Size, Seq_Len]
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass for the PaliGemmaForConditionalGeneration model.

        This method processes input images and text tokens through the vision-language
        model to generate text conditioned on the input image. The process involves
        three main steps: extracting visual features, merging them with text features,
        and generating output through the language model.

        Args:
            pixel_values: Input image tensor.
                        Shape: [batch_size, 3, height, width]
            input_ids: Text token IDs.
                     Shape: [batch_size, seq_len]
            attention_mask: Attention mask for text tokens.
                         Shape: [batch_size, seq_len]
            kv_cache: Optional key-value cache for faster inference.

        Returns:
            Model outputs from the language model, typically containing:
            - logits: Prediction logits. Shape: [batch_size, seq_len, vocab_size]

        Raises:
            AssertionError: If attention_mask is provided and contains non-1 values.
        """
        # Validate attention mask if provided
        if attention_mask is not None:
            assert torch.all(attention_mask == 1), "attention_mask must contain only 1s"

        # Step 1: Extract text embeddings from input tokens
        # Convert token IDs to embeddings: 
        # [batch_size, seq_len] -> [batch_size, seq_len, 2048]
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Step 2: Extract and project image features
        # Process image through vision tower: [batch_size, 3, H, W] -> [batch_size, num_patches, 768]
        image_embeds = self.vision_tower(pixel_values.to(input_embeds.dtype))

        # Project visual features to text embedding space: 
        # [batch_size, num_patches, 768] -> [batch_size, num_patches, 2048]
        image_features = self.multi_modal_projector(image_embeds)

        # Merge image features with text embeddings
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features, input_embeds, input_ids, attention_mask, kv_cache
        )

        # Step 3: Generate output through language model
        # Pass merged features through language model: 
        # [batch_size, 392, 2048] -> [batch_size, 392, vocab_size]
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs
        