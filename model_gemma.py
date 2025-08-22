"""PaliGemma For Conditional Generation

This model is a conditional generation model that uses a vision encoder and a language model to generate text.
It is used to generate text from an image.

The code is annotated with Google-style docstrings and type hints.
"""

import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from model_siglip import SiglipVisionConfig, SiglipVisionModel

# PaliGemma For Conditional Generation
class PaliGemmaForConditionalGeneration(nn.Module):
    """PaliGemma For Conditional Generation

    This model is a conditional generation model that uses a vision encoder and a language model to generate text.
    It is used to generate text from an image.

    Args:
        config: PaliGemmaConfig

    Returns:
        PaliGemmaForConditionalGeneration
    """
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config

        # [batch_size, 3, 224, 224] -> [batch_size, 196, 768]
        self.vision_tower = SiglipVisionModel(config.vision_config)        # vision encoder

        # [batch_size, 196, 768] -> [batch_size, 196, 2048]
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)   # Multimodel encoder
        
        # vocab_size
        self.vocab_size = config.vocab_size

        # Language Model
        language_model = GemmaForCausalLM(config.text_config)               
        self.language_model = language_model

        # pad_token_id
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
	
    def tie_weights(self):
        """Tie the weights of the language model and the vision model."""
        return self.language_model.tie_weights()
    
    def forward(
        self, 
        pixel_values: torch.Tensor = None, 
        input_ids: torch.Tensor = None, 
        attention_mask: torch.Tensor = None, 
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        """Forward pass for the PaliGemmaForConditionalGeneration model.

        Args:
            pixel_values: Tensor of shape [batch_size, 3, 224, 224]
            input_ids: Tensor of shape [batch_size, sequence_length]
            attention_mask: Tensor of shape [batch_size, sequence_length]
            labels: Tensor of shape [batch_size, sequence_length]

        Returns:
            Tensor of shape [batch_size, sequence_length, vocab_size]
        """
        