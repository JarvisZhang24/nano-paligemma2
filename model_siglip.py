"""SIGLIP Vision Transformer components.

This module implements a lightweight SIGLIP-style Vision Transformer stack,
including configuration, patch embeddings, multi-head attention, MLP blocks,
encoder layers, and a top-level vision model wrapper.

The code is annotated with Google-style docstrings and type hints.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


# Siglip Vision Config
class SiglipVisionConfig:
    """Configuration for the SIGLIP Vision Transformer blocks.

    Args:
        hidden_size: Dimension of the hidden states of the encoder.
        intermediate_size: Dimension of the MLP inside each encoder layer.
        num_hidden_layers: Number of transformer layers in the encoder.
        num_attention_heads: Number of attention heads.
        num_channels: Number of input image channels.
        image_size: Input image spatial size (assumes square input).
        patch_size: Patch size used by the patch embedding convolution.
        layer_norm_eps: Epsilon used in layer normalization.
        attention_dropout: Dropout probability applied to attention weights.
        num_image_tokens: Optional number of image tokens if precomputed.
        **kwargs: Additional unused keyword arguments for forward compatibility.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 16,
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        num_image_tokens: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    """Patch embedding with learnable position embeddings.

    Converts an input image tensor into a sequence of patch embeddings and
    adds a learnable position embedding for each patch.

    Args:
        config: Model configuration.
    """

    def __init__(self , config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size # 768
        self.image_size = config.image_size # 224
        self.patch_size = config.patch_size # 16

        # Patch Embedding
        # [batch_size , channel , Height , Width] -> [Batch size , Embed_dim , Num_patch_H , Num_patch_W]
        self.patch_embeddings = nn.Conv2d(
            config.num_channels , # 3
            self.embed_dim , # 768
            kernel_size = self.patch_size , # 16
            stride = self.patch_size , # 16
            padding = 'valid' # 0
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2 # 14 * 14 = 196
        self.num_positions = self.num_patches  # 196 
        self.position_embeddings = nn.Embedding(
            self.num_positions , # 196
            self.embed_dim # 768
        )

        self.register_buffer(
            "position_ids" ,
            torch.arange(self.num_positions).expand((1 , -1)) ,
            persistent = False
        )

    def forward(self , pixel_values: torch.Tensor) -> torch.Tensor:
        """Compute patch + position embeddings.

        Args:
            pixel_values: Image tensor of shape [batch_size, channels, height, width].

        Returns:
            Tensor of shape [batch_size, num_patches, embed_dim].
        """
        _batch_size , _num_channels , _height , _width = pixel_values.shape

        # [batch_size , channel , Height , Width] -> [Batch size , Embed_dim , Num_patch_H , Num_patch_W]
        embeddings = self.patch_embeddings(pixel_values)

        # [Batch size , Embed_dim , Num_patch_H , Num_patch_W] -> [Batch size , Embed_dim , Num_patches]
        embeddings = embeddings.flatten(2)

        # [Batch size , Embed_dim , Num_Patches] -> [Batch size , Num_Patches , Embed_dim]
        embeddings = embeddings.transpose(1 , 2)

        # [Batch size , Num_Patches , Embed_dim] -> [Batch size , Num_Patches , Embed_dim]
        embeddings = embeddings + self.position_embeddings(self.position_ids)

        return embeddings



class SiglipAttention(nn.Module):
    """Multi-head self-attention.

    Based on "Attention Is All You Need". Projects inputs to Q, K, V, computes
    scaled dot-product attention, and returns mixed-head outputs.
    """
    
    def __init__(self , config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        # scale for the dot product, equal to 1 / sqrt(head_dim)
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.embed_dim , self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim , self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim , self.embed_dim)

        self.out_proj = nn.Linear(self.embed_dim , self.embed_dim)

        
    # [Batch size , Num_Patches , Embed_dim] -> [Batch size , Num_Patches , Embed_dim]
    def forward(self , hidden_states: torch.Tensor) -> Tuple[torch.Tensor , Optional[torch.Tensor]]:
        """Apply multi-head self-attention.

        Args:
            hidden_states: Input tensor of shape [batch_size, num_patches, embed_dim].

        Returns:
            A tuple of:
            - attn_output: Tensor of shape [batch_size, num_patches, embed_dim].
            - attn_weights: Tensor of shape [batch_size, num_heads, num_patches, num_patches].

        Raises:
            ValueError: If intermediate attention tensor shapes do not match expectations.
        """
        batch_size , num_patches , embed_dim = hidden_states.shape

        # [Batch size , Num_Patches , Embed_dim] -> [Batch size , Num_Patches , Embed_dim]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # [Batch size , Num_Patches , Embed_dim] -> [Batch size , Num_Patches , Num_heads , Head_dim]
        query_states = query_states.view(batch_size , num_patches , self.num_heads , self.head_dim)
        key_states = key_states.view(batch_size , num_patches , self.num_heads , self.head_dim)
        value_states = value_states.view(batch_size , num_patches , self.num_heads , self.head_dim)


        # [Batch size , Num_Patches , Num_heads , Head_dim] -> [Batch size , Num_heads , Num_Patches , Head_dim]
        query_states = query_states.transpose(1 , 2)
        key_states = key_states.transpose(1 , 2)
        value_states = value_states.transpose(1 , 2)

        # attention weights
        # [Batch size , Num_heads , Num_Patches , Head_dim] -> [Batch size , Num_heads , Num_Patches , Num_Patches]
        attn_weights = torch.matmul(query_states , key_states.transpose(-2 , -1)) * self.scale

        if attn_weights.size() != (batch_size , self.num_heads , num_patches , num_patches):
            raise ValueError(f"Attention weights should be of size (batch_size , num_heads , num_patches , num_patches), but got {attn_weights.size()}")


        # [Batch size , Num_heads , Num_Patches , Num_Patches] -> [Batch size , Num_heads , Num_Patches , Num_Patches]
        attn_weights = nn.functional.softmax(attn_weights , dim = -1 , dtype = torch.float32).to(query_states.dtype)

        # [Batch size , Num_heads , Num_Patches , Num_Patches] -> [Batch size , Num_heads , Num_Patches , Num_Patches]
        attn_weights = nn.functional.dropout(attn_weights , p = self.dropout , training = self.training)

        # [Batch size , Num_heads , Num_Patches , Num_Patches] -> [Batch size , Num_heads , Num_Patches , Head_dim]
        attn_output = torch.matmul(attn_weights , value_states)

        if attn_output.size() != (batch_size , self.num_heads , num_patches , self.head_dim):
            raise ValueError(f"Attention output should be of size (batch_size , num_heads , num_patches , head_dim), but got {attn_output.size()}")

        # [Batch size , Num_heads , Num_Patches , Head_dim] -> [Batch size , Num_Patches , Embed_dim]
        attn_output = attn_output.transpose(1 , 2).contiguous()
        attn_output = attn_output.view(batch_size , num_patches , self.embed_dim)

        # Mix up the heads 
        # [Batch size , Num_Patches , Embed_dim] -> [Batch size , Num_Patches , Embed_dim]
        attn_output = self.out_proj(attn_output)

        return attn_output , attn_weights


class SiglipMLP(nn.Module):
    """Feed-forward MLP block used inside transformer layers."""

    def __init__(self , config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config

        self.fc1 = nn.Linear(config.hidden_size , config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size , config.hidden_size)

    def forward(self , hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the MLP on hidden states.

        Args:
            hidden_states: Tensor of shape [batch_size, num_patches, embed_dim].

        Returns:
            Tensor of shape [batch_size, num_patches, embed_dim].
        """

        # [Batch size , Num_Patches , Embed_dim] -> [Batch size , Num_Patches , Intermediate_size]
        hidden_states = self.fc1(hidden_states)

        # [Batch size , Num_Patches , Intermediate_size] -> [Batch size , Num_Patches , Intermediate_size]
        hidden_states = nn.functional.gelu(hidden_states , approximate = 'tanh')

        # [Batch size , Num_Patches , Intermediate_size] -> [Batch size , Num_Patches , Embed_dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states

class SiglipEncoderLayer(nn.Module):
    """Single transformer encoder layer (pre-norm).

    Applies LayerNorm -> Self-Attention -> Residual, then LayerNorm -> MLP ->
    Residual.
    """

    def __init__(self , config: SiglipVisionConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attention = SiglipAttention(config)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim , eps = config.layer_norm_eps)

        self.mlp = SiglipMLP(config)

        self.layer_norm2 = nn.LayerNorm(self.embed_dim , eps = config.layer_norm_eps)

    # [Batch size , Num_Patches , Embed_dim] -> [Batch size , Num_Patches , Embed_dim]
    def forward(self , hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single encoder layer.

        Args:
            hidden_states: Tensor of shape [batch_size, num_patches, embed_dim].

        Returns:
            Tensor of shape [batch_size, num_patches, embed_dim].
        """
        # save residual , [Batch size , Num_Patches , Embed_dim]
        residual = hidden_states

        # layer norm , [Batch size , Num_Patches , Embed_dim]
        hidden_states = self.layer_norm1(hidden_states)

        # self attention , [Batch size , Num_Patches , Embed_dim]
        attn_output, _ = self.self_attention(hidden_states)
        hidden_states = attn_output

        # residual connection , [Batch size , Num_Patches , Embed_dim]
        hidden_states = residual + hidden_states

        # save residual , [Batch size , Num_Patches , Embed_dim]
        residual = hidden_states

        # layer norm , [Batch size , Num_Patches , Embed_dim]
        hidden_states = self.layer_norm2(hidden_states)

        # mlp , [Batch size , Num_Patches , Embed_dim]
        hidden_states = self.mlp(hidden_states)

        # residual connection , [Batch size , Num_Patches , Embed_dim]
        hidden_states = residual + hidden_states

        return hidden_states


class SiglipEncoder(nn.Module):
    """Stack of transformer encoder layers."""

    def __init__(self , config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self , input_embeds: torch.Tensor) -> torch.Tensor:
        """Run the encoder over input embeddings.

        Args:
            input_embeds: Tensor of shape [batch_size, num_patches, embed_dim].

        Returns:
            Tensor of shape [batch_size, num_patches, embed_dim].
        """
        # [Batch size , Num_Patches , Embed_dim] -> [Batch size , Num_Patches , Embed_dim]
        # [batch_size , 196 , 768] -> [batch_size , 196 , 768]
        hidden_states = input_embeds

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states

class SiglipVisionTransformer(nn.Module):
    """Full Vision Transformer model (embeddings + encoder + final norm)."""

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config 
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim , eps = config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Compute visual features from input images.

        Args:
            pixel_values: Image tensor of shape [batch_size, channels, height, width].

        Returns:
            Tensor of shape [batch_size, num_patches, embed_dim].
        """
        # [batch_size , channel , Height , Width] -> [Batch size , Num_Patches , Each_Dim]
        # [batch_size , 3 , 224 , 224] -> [batch_size , 196 , 768]
        embedding_output = self.embeddings(pixel_values)
        
        encoder_output = self.encoder(embedding_output)

        pooled_output = self.post_layernorm(encoder_output)
        return pooled_output

class SiglipVisionModel(nn.Module):
    """Convenience wrapper around the vision transformer."""

    def __init__(self , config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the wrapped vision transformer.

        Args:
            pixel_values: Image tensor of shape [batch_size, channels, height, width].

        Returns:
            Tensor of shape [batch_size, num_patches, embed_dim].
        """
        # [batch_size , channel , Height , Width] -> [Batch size , Num_Patches , Each_Dim]
        # [batch_size , 3 , 224 , 224] -> [batch_size , 196 , 768]
        return self.vision_model(pixel_values = pixel_values)
        
        