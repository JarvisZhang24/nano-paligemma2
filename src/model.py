import torch
import math
from torch import nn
from typing import Optional, Tuple
from src.configs import PaliGemma2Config
from src.vision.siglip import SiglipVisionModel
from src.text.gemma2_wrapper import Gemma2
from src.kv_cache import KVCache

from src.projector import MultiModalProjector

from src.merge import merge_input_ids_with_image_features


################################### PaliGemma 2 Model ###################################


class PaliGemma2(nn.Module):
    def __init__(self, config: PaliGemma2Config):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = MultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = Gemma2(config.text_config)
        self.language_model = language_model

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

    # We reuse the embeddings of the language model in the last linear layer
    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ):
        """
        Call the merge function with the correct parameters.
        """
        return merge_input_ids_with_image_features(
            image_features=image_features,
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_token_index=self.config.image_token_index,
            pad_token_id=self.pad_token_id,
            hidden_size=self.config.hidden_size,
            kv_cache=kv_cache,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

        # Make sure the input is right-padded
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # Extract the input embeddings
        # (batch_size, seq_len, hidden_size)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # We merge the image features with the input embeddings
        # (batch_size, channels, height, width) -> (batch_size, num_patches, embed_dim)
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        # (batch_size, num_patches, embed_dim) -> (batch_size, num_patches, Hidden_Size)
        image_features = self.multi_modal_projector(selected_image_feature)

        # Merge the embeddings of the text tokens and the image tokens
        inputs_embeds, attention_mask, position_ids = (
            self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask, kv_cache
            )
        )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs
