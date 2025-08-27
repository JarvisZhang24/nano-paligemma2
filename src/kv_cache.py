

import torch
from typing import List, Tuple


class KVCache:

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is (batch_size, num_key_value_heads, sequence_length, head_dim)
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, we create it.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # We concatenate the new keys with the existing ones.
            # (batch_size, num_key_value_heads, sequence_length, head_dim)
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        # We return all the existing keys and the new ones.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
