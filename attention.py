from typing import Optional, Tuple
import torch
import torch.nn as nn

from model_siglip import SiglipVisionConfig

'''
This class implements the multi-head self-attention mechanism for the SIGLIP Vision Transformer.
'''

class SiglipAttention(nn.Module):
    """Multi-head self-attention mechanism for SIGLIP Vision Transformer.

    This implementation follows the standard multi-head attention architecture
    from "Attention Is All You Need" with the following steps:

    1. Project input embeddings into Query, Key, and Value spaces
    2. Split each projection into multiple attention heads
    3. Compute scaled dot-product attention for each head
    4. Concatenate and project attention outputs back to embedding space

    The attention mechanism allows the model to attend to different parts of
    the input sequence simultaneously, capturing various types of relationships
    between patches in the image.

    Attributes:
        config: Model configuration containing attention parameters.
        embed_dim: Dimensionality of input embeddings.
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        scale: Scaling factor for attention scores (1/sqrt(head_dim)).
        dropout: Dropout probability for attention weights.
        q_proj: Linear projection for queries.
        k_proj: Linear projection for keys.
        v_proj: Linear projection for values.
        out_proj: Output projection layer.
    """

    def __init__(self, config: SiglipVisionConfig) -> None:
        """Initialize SiglipAttention.

        Args:
            config: Model configuration containing attention parameters.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        # Scaling factor for attention scores: 1/sqrt(head_dim)
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Output projection
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply multi-head self-attention to input hidden states.

        This method implements the complete multi-head attention mechanism:
        1. Project inputs to Query, Key, Value spaces
        2. Reshape for multi-head processing
        3. Compute attention scores and weights
        4. Apply attention to values
        5. Concatenate and project outputs

        Args:
            hidden_states: Input tensor of shape [batch_size, num_patches, embed_dim].

        Returns:
            A tuple containing:
            - attn_output: Attention output of shape [batch_size, num_patches, embed_dim].
            - attn_weights: Attention weights of shape [batch_size, num_heads, num_patches, num_patches].

        Raises:
            ValueError: If intermediate attention tensor shapes do not match expectations.

        Example:
            >>> config = SiglipVisionConfig()
            >>> attention = SiglipAttention(config)
            >>> hidden_states = torch.randn(2, 196, 768)
            >>> output, weights = attention(hidden_states)
            >>> output.shape  # [2, 196, 768]
        """
        batch_size, num_patches, embed_dim = hidden_states.shape

        # Step 1: Project inputs to Query, Key, and Value spaces
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Step 2: Reshape for multi-head attention
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, num_heads, head_dim]
        query_states = query_states.view(batch_size, num_patches, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, num_patches, self.num_heads, self.head_dim)
        value_states = value_states.view(batch_size, num_patches, self.num_heads, self.head_dim)

        # Step 3: Transpose to bring heads to the second dimension
        # [batch_size, num_patches, num_heads, head_dim] -> [batch_size, num_heads, num_patches, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Step 4: Compute attention scores
        # [batch_size, num_heads, num_patches, head_dim] @ [batch_size, num_heads, head_dim, num_patches]
        # -> [batch_size, num_heads, num_patches, num_patches]
        attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale

        # Validate attention scores shape
        if attn_scores.size() != (batch_size, self.num_heads, num_patches, num_patches):
            raise ValueError(
                "Attention scores should be of size "
                f"(batch_size, num_heads, num_patches, num_patches), but got {attn_scores.size()}"
            )

        # Step 5: Apply softmax to get attention weights
        # [batch_size, num_heads, num_patches, num_patches]
        attn_weights = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Step 6: Apply dropout to attention weights
        # [batch_size, num_heads, num_patches, num_patches]
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Step 7: Apply attention weights to values
        # [batch_size, num_heads, num_patches, num_patches] @ [batch_size, num_heads, num_patches, head_dim]
        # -> [batch_size, num_heads, num_patches, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        # Validate attention output shape
        if attn_output.size() != (batch_size, self.num_heads, num_patches, self.head_dim):
            raise ValueError(
                "Attention output should be of size "
                f"(batch_size, num_heads, num_patches, head_dim), but got {attn_output.size()}"
            )

        # Step 8: Reshape back to original format
        # [batch_size, num_heads, num_patches, head_dim] -> [batch_size, num_patches, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_patches, self.embed_dim)

        # Step 9: Final linear projection
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

