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
    """
    This class implements the Gemma model, which is a causal language model that uses the Gemma model.
    It is used to generate text from a given input text.
    
    Args:
        config (GemmaConfig): The configuration of the Gemma model.

    Attributes:
        config (GemmaConfig): The configuration of the Gemma model.
        padding_idx (int): The padding index of the model.
        vocab_size (int): The vocabulary size of the model.
        embed_tokens (nn.Embedding): The embedding layer to project the input ids to the hidden size.
        layers (nn.ModuleList): The list of decoder layers of the model.
        norm (GemmaRMSNorm): The normalization layer of the model.
    """
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