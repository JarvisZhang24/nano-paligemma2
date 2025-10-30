import torch
from typing import List, Tuple


# class KVCache:

#     def __init__(self) -> None:
#         self.key_cache: List[torch.Tensor] = []
#         self.value_cache: List[torch.Tensor] = []

#     def num_items(self) -> int:
#         if len(self.key_cache) == 0:
#             return 0
#         else:
#             # The shape of the key_cache is (batch_size, num_key_value_heads, sequence_length, head_dim)
#             return self.key_cache[0].shape[-2]

#     def update(
#         self,
#         key_states: torch.Tensor,
#         value_states: torch.Tensor,
#         layer_idx: int,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         if len(self.key_cache) <= layer_idx:
#             # If we never added anything to the KV-Cache of this layer, we create it.
#             self.key_cache.append(key_states)
#             self.value_cache.append(value_states)
#         else:
#             # We concatenate the new keys with the existing ones.
#             # (batch_size, num_key_value_heads, sequence_length, head_dim)
#             self.key_cache[layer_idx] = torch.cat(
#                 [self.key_cache[layer_idx], key_states], dim=-2
#             )
#             self.value_cache[layer_idx] = torch.cat(
#                 [self.value_cache[layer_idx], value_states], dim=-2
#             )

#         # We return all the existing keys and the new ones.
#         return self.key_cache[layer_idx], self.value_cache[layer_idx]


import torch
from typing import List, Tuple, Optional
import warnings


class KVCache:
    def __init__(
        self, 
        max_batch_size: int = 1,
        max_seq_length: int = 2048,
        num_layers: int = None,
        num_heads: int = None,
        head_dim: int = None,
        dtype: torch.dtype = torch.float16,
        device: torch.device = None,
        enable_sliding_window: bool = False,
        window_size: int = 1024
    ) -> None:
        """
        Optimized KV cache implementation
        
        Args:
            max_batch_size: Maximum batch size
            max_seq_length: Maximum sequence length
            num_layers: Number of Transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            dtype: Data type
            device: Device
            enable_sliding_window: Whether to enable sliding window
            window_size: Window size
        """
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.device = device or torch.device('cpu')
        self.enable_sliding_window = enable_sliding_window
        self.window_size = window_size
        
        # Preallocate cache if model parameters are known
        if all(x is not None for x in [num_layers, num_heads, head_dim]):
            self._preallocate_cache(num_layers, num_heads, head_dim)
        else:
            # Fall back to dynamic allocation
            self.key_cache: List[torch.Tensor] = []
            self.value_cache: List[torch.Tensor] = []
            self._preallocated = False
        
        self._current_length = 0
        self._initialized_layers = set()

    def _preallocate_cache(self, num_layers: int, num_heads: int, head_dim: int):
        """Preallocate cache memory"""
        cache_shape = (self.max_batch_size, num_heads, self.max_seq_length, head_dim)
        
        self.key_cache = []
        self.value_cache = []
        
        for _ in range(num_layers):
            key_tensor = torch.zeros(cache_shape, dtype=self.dtype, device=self.device)
            value_tensor = torch.zeros(cache_shape, dtype=self.dtype, device=self.device)
            self.key_cache.append(key_tensor)
            self.value_cache.append(value_tensor)
        
        self._preallocated = True

    def num_items(self) -> int:
        """Return the current cache length"""
        if self._preallocated:
            return self._current_length
        elif len(self.key_cache) == 0:
            return 0
        else:
            return self.key_cache[0].shape[-2]

    def clear(self):
        """Clear the cache"""
        if self._preallocated:
            self._current_length = 0
            self._initialized_layers.clear()
        else:
            self.key_cache.clear()
            self.value_cache.clear()

    def _apply_sliding_window(self, layer_idx: int):
        """Apply sliding window mechanism"""
        if not self.enable_sliding_window or self._current_length <= self.window_size:
            return
        
        if self._preallocated:
            # Shift data to the left
            shift_size = self._current_length - self.window_size
            remaining_size = self.window_size
            
            self.key_cache[layer_idx][:, :, :remaining_size] = \
                self.key_cache[layer_idx][:, :, shift_size:self._current_length]
            self.value_cache[layer_idx][:, :, :remaining_size] = \
                self.value_cache[layer_idx][:, :, shift_size:self._current_length]
                
            self._current_length = self.window_size
        else:
            # Truncate to window size
            self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, -self.window_size:]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, -self.window_size:]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the KV cache
        
        Args:
            key_states: New key states [batch_size, num_heads, seq_len, head_dim]
            value_states: New value states
            layer_idx: Layer index
            
        Returns:
            Complete key and value cache
        """
        # Input validation
        if key_states.shape != value_states.shape:
            raise ValueError(f"Key and value shapes must match: {key_states.shape} vs {value_states.shape}")
        
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        if self._preallocated:
            return self._update_preallocated(key_states, value_states, layer_idx, seq_len)
        else:
            return self._update_dynamic(key_states, value_states, layer_idx)

    def _update_preallocated(
        self, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        layer_idx: int,
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update in preallocated mode"""
        if layer_idx not in self._initialized_layers:
            self._initialized_layers.add(layer_idx)
        
        # Check if sliding window needs to be applied
        if self._current_length + seq_len > self.max_seq_length:
            if self.enable_sliding_window:
                self._apply_sliding_window(layer_idx)
            else:
                warnings.warn(f"Sequence length {self._current_length + seq_len} exceeds max_seq_length {self.max_seq_length}")
                # Truncate to max length
                available_space = self.max_seq_length - self._current_length
                if available_space > 0:
                    seq_len = min(seq_len, available_space)
                    key_states = key_states[:, :, :seq_len]
                    value_states = value_states[:, :, :seq_len]
                else:
                    return self.key_cache[layer_idx][:, :, :self._current_length], \
                           self.value_cache[layer_idx][:, :, :self._current_length]
        
        # Copy new data to preallocated cache
        end_pos = self._current_length + seq_len
        self.key_cache[layer_idx][:, :, self._current_length:end_pos] = key_states
        self.value_cache[layer_idx][:, :, self._current_length:end_pos] = value_states
        
        # Only update current length in the last layer to avoid duplicate updates
        if layer_idx == len(self.key_cache) - 1 or layer_idx not in self._initialized_layers:
            self._current_length = end_pos
        
        return self.key_cache[layer_idx][:, :, :end_pos], \
               self.value_cache[layer_idx][:, :, :end_pos]

    def _update_dynamic(
        self, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update in dynamic allocation mode (compatible with original implementation)"""
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Apply sliding window
            if self.enable_sliding_window:
                current_len = self.key_cache[layer_idx].shape[-2]
                if current_len >= self.window_size:
                    self._apply_sliding_window(layer_idx)
            
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_memory_usage(self) -> dict:
        """Get memory usage"""
        if not self.key_cache:
            return {"total_mb": 0, "per_layer_mb": 0, "num_layers": 0}
        
        # Calculate memory usage per tensor
        sample_tensor = self.key_cache[0]
        bytes_per_element = sample_tensor.element_size()
        elements_per_tensor = sample_tensor.numel()
        
        mb_per_tensor = (elements_per_tensor * bytes_per_element) / (1024 * 1024)
        total_mb = mb_per_tensor * len(self.key_cache) * 2  # key + value
        
        return {
            "total_mb": total_mb,
            "per_layer_mb": mb_per_tensor * 2,
            "num_layers": len(self.key_cache),
            "current_length": self.num_items(),
            "max_length": self.max_seq_length if self._preallocated else "dynamic"
        }

    def to(self, device: torch.device):
        """Move cache to specified device"""
        self.device = device
        for i in range(len(self.key_cache)):
            self.key_cache[i] = self.key_cache[i].to(device)
            self.value_cache[i] = self.value_cache[i].to(device)
        return self

