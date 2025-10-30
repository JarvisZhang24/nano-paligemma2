import torch
from torch import nn
from typing import Optional, Tuple
import math
from src.text.gemma2_config import Gemma2Config
from src.kv_cache import KVCache
from src.attention.rotary import RotaryEmbedding
import torch.nn.functional as F

################################### Useful functions ###################################


def rotate_half(x):
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1 = x[..., : x.shape[-1] // 2]  # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :]  # Takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)  # Add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim)  # Add the head dimension
    # Formula from the RoPE paper
    q_embed = (query * cos) + (rotate_half(query) * sin)
    k_embed = (key * cos) + (rotate_half(key) * sin)
    return q_embed, k_embed

class Gemma2GQA(nn.Module):

    def __init__(self, config: Gemma2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        # Specific to Gemma2
        self.sliding_window_size = config.sliding_window_size
        self.attn_logit_softcapping = config.attn_logit_softcapping
        self.attn_types = config.attn_types

        assert (
            self.hidden_size % self.num_heads == 0
        ), "Hidden size must be divisible by the number of heads."

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.size()
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch_size, num_key_value_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(
            batch_size, num_key_value_heads * n_rep, seq_len, head_dim
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.size()

        query = (
            self.q_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.k_proj(hidden_states)
            .view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        cos, sin = self.rotary_emb(value, position_ids, seq_len=None)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Update key and value caches if provided
        if kv_cache is not None:
            key, value = kv_cache.update(key, value, self.layer_idx)

        # Repeat key and value to match the number of query heads
        key = self.repeat_kv(key, self.num_key_value_groups)
        value = self.repeat_kv(value, self.num_key_value_groups)

        # Prepare attention mask for scaled_dot_product_attention
        if attention_mask is not None:
            # Convert attention_mask to boolean mask (True = attend, False = mask out)
            # Original mask uses large negative values for masking
            bool_mask = attention_mask > -1e4
            
            # Handle sliding window attention
            if (
                self.attn_types[self.layer_idx] == "local_sliding"
                and self.sliding_window_size is not None
            ):
                batch_size, seq_len = bool_mask.shape[-2:]
                # Create sliding window mask
                sliding_mask = torch.ones((seq_len, seq_len), device=bool_mask.device, dtype=torch.bool)
                sliding_mask = torch.triu(sliding_mask, -self.sliding_window_size + 1) & \
                              torch.tril(sliding_mask, self.sliding_window_size - 1)
                # Apply sliding window to existing mask
                bool_mask = bool_mask & sliding_mask.unsqueeze(0).unsqueeze(0)
        else:
            bool_mask = None

        # Handle Gemma2's attention logit softcapping
        if self.attn_logit_softcapping is not None:
            # For softcapping, we need to do it manually since SDPA doesn't support it
            attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Apply softcapping
            attn_weights = attn_weights / self.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.attn_logit_softcapping
            
            # Apply mask manually
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value)
        else:
            # Use optimized scaled_dot_product_attention when no softcapping
            attn_output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=bool_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=self.is_causal if bool_mask is None else False,  # Don't use is_causal with custom mask
                scale=1.0 / math.sqrt(self.head_dim)
            )
            # Note: We can't easily extract attention weights from SDPA, so set to None
            attn_weights = None

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights