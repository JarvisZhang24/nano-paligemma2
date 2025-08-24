"""
This module implements the KVCache class, which is used to cache the key and value tensors for the attention mechanism.
"""
from typing import List, Tuple

import torch


class KVCache():
    """
    This class implements the KVCache, which is used to cache the key and value tensors for the attention mechanism.

    Args:
        config (GemmaConfig): The configuration of the Gemma model.

    Attributes:
        key_cache (List[torch.Tensor]): The list of key tensors.
        value_cache (List[torch.Tensor]): The list of value tensors. 
        num_items (int): The number of items in the KVCache.
    """

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def num_items(self) -> int:
        """
        Get the number of items in the KVCache.
        """
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the KVCache with the new key and value states.
        """
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, let's create it.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # ... otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # ... and then we return all the existing keys + the new ones.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]