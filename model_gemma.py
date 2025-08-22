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
from typing import Optional, Tuple, List, Dict, Any
from torch.nn import CrossEntropyLoss
import math
from model_siglip import SiglipVisionConfig, SiglipVisionModel


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
        # Convert token IDs to embeddings: [batch_size, seq_len, hidden_size]
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Step 2: Extract and project image features
        # Process image through vision tower: [batch_size, 3, H, W] -> [batch_size, 196, 768]
        image_embeds = self.vision_tower(pixel_values.to(input_embeds.dtype))

        # Project visual features to text embedding space: [batch_size, 196, 768] -> [batch_size, 196, 2048]
        image_features = self.multi_modal_projector(image_embeds)

        # Merge image features with text embeddings
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features, input_embeds, input_ids, attention_mask, kv_cache
        )

        # Step 3: Generate output through language model
        # Pass merged features through language model: [batch_size, 392, 768] -> [batch_size, 392, vocab_size]
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs
        