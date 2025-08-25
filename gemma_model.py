"""
This module implements the Gemma model, which is a causal language model that uses the Gemma model.
It is used to generate text from a given input text.
"""
import torch
from torch import nn
from typing import Optional, Tuple


from config import GemmaConfig
from kv_cache import KVCache


class GemmaModel(nn.Module):
    '''
    This class implements the GemmaModel
    Given the input embeddings, it passes them through the decoder layers and returns the hidden states.
    
    input_embeds: [Batch_Size, Seq_Len, Hidden_Size]
    -> GemmaDecoderLayer * num_layers # [Batch_Size, Seq_Len, Hidden_Size]
    -> GemmaRMSNorm # [Batch_Size, Seq_Len, Hidden_Size]
    Outputs: 
    -> hidden_states  #[Batch_Size, Seq_Len, Hidden_Size]

    Args:
        config (GemmaConfig): The configuration of the Gemma model.

    Attributes:
        config (GemmaConfig): The configuration of the Gemma model.
        padding_idx (int): The padding index of the model.
        vocab_size (int): The vocabulary size of the model.
        embed_tokens (nn.Embedding): The embedding layer to project the input ids to the hidden size.
        layers (nn.ModuleList): The list of decoder layers of the model.
        norm (GemmaRMSNorm): The normalization layer of the model.

    Methods:
        get_input_embeds(self) -> torch.Tensor: Get the input embeddings for the input ids.
        forward(self, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, inputs_embeds: Optional[torch.FloatTensor] = None, kv_cache: Optional[KVCache] = None) -> torch.FloatTensor: Forward pass of the GemmaModel.

    Example:
        >>> config = GemmaConfig()
        >>> model = GemmaModel(config)
        >>> inputs_embeds = model.get_input_embeds() # [Batch_Size, Seq_Len, Hidden_Size]
        >>> outputs = model(inputs_embeds=inputs_embeds) # [Batch_Size, Seq_Len, Hidden_Size]
        >>> print(outputs.shape) # [Batch_Size, Seq_Len, Hidden_Size]
    
    '''

    def __init__(self, config: GemmaConfig):
        """
        Initialize the GemmaModel.
        """
        super().__init__()
        # Initialize the configuration
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeds(self) -> torch.Tensor:
        """
        Get the input embeddings for the input ids.
        """
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        """
        Forward pass of the GemmaModel.
        """
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = inputs_embeds
        # [Batch_Size, Seq_Len, Hidden_Size]
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        # Normalize the hidden states
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            # Pass the hidden states through the decoder layer
            # [Batch_Size, Seq_Len, Hidden_Size]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.norm(hidden_states)

        # Return the hidden states
        # [Batch_Size, Seq_Len, Hidden_Size]
        return hidden_states



class GemmaForCausalLM(nn.Module):
    """
    This class implements the GemmaForCausalLM model, which is a causal language model that uses the Gemma model.
    It is used to generate text from a given input text.
    
    Args:
        config (GemmaConfig): The configuration of the Gemma model.
    
    Attributes:
        config (GemmaConfig): The configuration of the Gemma model.
        model (GemmaModel): The Gemma model.
        vocab_size (int): The vocabulary size of the model.
        lm_head (nn.Linear): The linear layer to project the hidden states to the vocabulary size.

    Example:
        >>> config = GemmaConfig()
        >>> model = GemmaForCausalLM(config)
        >>> inputs_embeds = model.get_input_embeds() # [Batch_Size, Seq_Len, Hidden_Size]
        >>> outputs = model(inputs_embeds=inputs_embeds) # [Batch_Size, Seq_Len, Hidden_Size]
        >>> print(outputs.shape) # [Batch_Size, Seq_Len, Hidden_Size]
        >>> print(outputs["logits"].shape) # [Batch_Size, Seq_Len, Vocab_Size]
        >>> print(outputs["kv_cache"].shape) # [Batch_Size, Seq_Len, Hidden_Size]

        input_embeds: [Batch_Size, Seq_Len, Hidden_Size]
        -> GemmaModel # [Batch_Size, Seq_Len, Hidden_Size]
        -> lm_head # [Batch_Size, Seq_Len, Vocab_Size]
        Outputs:
        -> logits: [Batch_Size, Seq_Len, Vocab_Size]
        -> kv_cache: [Batch_Size, Seq_Len, Hidden_Size] (if kv_cache is not None)
    """
    def __init__(self, config: GemmaConfig):
        super().__init__()
        # Initialize the configuration
        self.config = config
        # Initialize the Gemma model
        self.model = GemmaModel(config)
        # Initialize the vocabulary size
        self.vocab_size = config.vocab_size
        # Initialize the linear layer to project the hidden states to the vocabulary size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def tie_weights(self):
        """
        Tie the weights of the linear layer to the weights of the model.
        """
        self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeds(self):
        """
        Get the input embeddings for the input ids.
        """
        return self.model.embed_tokens
    
    
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        """
        Forward pass of the GemmaForCausalLM model.
        """
        # input_embeds: [Batch_Size, Seq_Len, Hidden_Size]
        # outputs: [Batch_Size, Seq_Len, Hidden_Size]
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )
        # outputs: [Batch_Size, Seq_Len, Hidden_Size]
        # Get the hidden states from the outputs
        hidden_states = outputs
        # Get the logits from the linear layer
        logits = self.lm_head(hidden_states)
        # Convert the logits to float
        logits = logits.float()
        
        # Return the logits and the kv_cache if it is not None
        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            # Return the updated cache
            return_data["kv_cache"] = kv_cache

        return return_data