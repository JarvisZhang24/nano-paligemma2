# src_jarvis/merge.py
# -*- coding: utf-8 -*-
"""
Utilities to merge image features into the text sequence and build attention artifacts
(causal mask and position_ids) for PaliGemma2-style VLMs.

Assumptions:
- Inputs are RIGHT-PAD FREE for now (i.e., attention_mask must be all ones).
- Image tokens in `input_ids` are placeholders to be replaced by `image_features`.
- Padding is not supported in this minimal implementation.
"""

import torch
from torch import nn
from typing import Optional, Tuple
from src.kv_cache import KVCache


def merge_input_ids_with_image_features(
    image_features: torch.Tensor,
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_token_index: int,
    pad_token_id: int,
    hidden_size: int,
    kv_cache: Optional[KVCache] = None,
):
    """
    Merge image features with input embeddings, following the original PaliGemma2 implementation.
    
    This function is an exact copy of the _merge_input_ids_with_image_features method 
    from the original src/paligemma2.py implementation.
    """
    _, _, embed_dim = image_features.shape
    batch_size, sequence_length = input_ids.shape
    dtype, device = inputs_embeds.dtype, inputs_embeds.device
    # Shape: [Batch_Size, Seq_Len, Hidden_Size]
    scaled_image_features = image_features / (hidden_size**0.5)

    # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
    final_embedding = torch.zeros(
        batch_size,
        sequence_length,
        embed_dim,
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device,
    )
    # (batch_size, seq_len)
    text_mask = (input_ids != image_token_index) & (
        input_ids != pad_token_id
    )
    # (batch_size, seq_len)
    image_mask = input_ids == image_token_index
    # (batch_size, seq_len)
    pad_mask = input_ids == pad_token_id

    # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
    text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
    pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
    image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

    # Add the text embeddings
    final_embedding = torch.where(
        text_mask_expanded, inputs_embeds, final_embedding
    )
    # Insert image embeddings. We can't use torch.where because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
    final_embedding = final_embedding.masked_scatter(
        image_mask_expanded, scaled_image_features
    )
    # Zero out padding tokens
    final_embedding = torch.where(
        pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding
    )

    # Attention Mask
    dtype, device = inputs_embeds.dtype, inputs_embeds.device
    q_len = inputs_embeds.shape[1]

    if kv_cache is None or kv_cache.num_items() == 0:
        # Do not mask any token, because we're in the prefill phase
        # This implementation does not support padding.
        causal_mask = torch.full(
            (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
        )
    else:
        # Since we are generating tokens, the query must be one single token
        assert q_len == 1
        kv_len = kv_cache.num_items() + q_len
        # Also in this case we don't need to mask anything, since each query should be able to attend all previous tokens.
        # This implementation does not support padding.
        causal_mask = torch.full(
            (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
        )

    # Add the head dimension
    # (batch_size, q_len, kv_len) -> (batch_size, 1, q_len, kv_len)
    causal_mask = causal_mask.unsqueeze(1)

    if kv_cache is not None and kv_cache.num_items() > 0:
        # The position of the query is just the last position
        position_ids = attention_mask.cumsum(-1)[:, -1]
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
    else:
        # Create a position_ids based on the size of the attention_mask
        # For masked tokens, use the number 1 as position.
        position_ids = (
            (attention_mask.cumsum(-1))
            .masked_fill_((attention_mask == 0), 1)
            .to(device)
        )

    return final_embedding, causal_mask, position_ids
