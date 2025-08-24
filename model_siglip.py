"""SIGLIP Vision Transformer implementation.

This module provides a complete implementation of the SIGLIP (Sigmoid Loss for
Language-Image Pre-training) vision transformer architecture. The implementation
includes all necessary components for processing images through a transformer-based
vision model:

- Configuration classes for model hyperparameters
- Patch embedding layers to convert images into sequences
- Multi-head self-attention mechanisms
- MLP blocks with GELU activation
- Transformer encoder layers with pre-norm architecture
- Complete vision transformer model wrapper

The architecture follows the standard Vision Transformer design with modifications
specific to SIGLIP, including optimized attention patterns and layer normalization
placement for improved training stability.

Example:
    >>> config = SiglipVisionConfig()
    >>> model = SiglipVisionModel(config)
    >>> pixel_values = torch.randn(1, 3, 224, 224)
    >>> features = model(pixel_values)  # Shape: [1, 196, 768]

The code is annotated with comprehensive Google-style docstrings and type hints.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from paligemma_config import SiglipVisionConfig
from attention import SiglipAttention


class SiglipVisionEmbeddings(nn.Module):
    """Patch embedding layer with learnable position embeddings.

    This module converts input images into sequences of patch embeddings and adds
    learnable positional information to preserve spatial relationships between
    patches. The process involves:

    1. Dividing the input image into fixed-size patches using convolution
    2. Linearly projecting each patch into an embedding vector
    3. Adding learnable position embeddings to encode patch positions

    The convolutional approach to patch embedding is more efficient than direct
    reshaping and provides better inductive bias for vision tasks.

    Attributes:
        config: Model configuration containing embedding parameters.
        embed_dim: Dimensionality of patch embeddings (typically 768).
        image_size: Input image height and width.
        patch_size: Size of patches for embedding.
        num_patches: Total number of patches after dividing the image.
        patch_embeddings: Convolutional layer for patch embedding.
        position_embeddings: Learnable position embedding layer.
    """

    def __init__(self, config: SiglipVisionConfig) -> None:
        """Initialize SiglipVisionEmbeddings.

        Args:
            config: Model configuration containing embedding parameters.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # 768
        self.image_size = config.image_size  # 224
        self.patch_size = config.patch_size  # 16

        # Calculate number of patches: (224 // 16) ** 2 = 196
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches

        # Convolutional patch embedding layer
        # Input: [batch_size, 3, 224, 224] -> Output: [batch_size, 768, 14, 14]
        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels,  # 3 for RGB
            out_channels=self.embed_dim,      # 768
            kernel_size=self.patch_size,      # 16
            stride=self.patch_size,           # 16
            padding='valid'                   # No padding
        )

        # Learnable position embeddings to encode spatial positions
        # Input: [batch_size, 196] -> Output: [batch_size, 196, 768]
        self.position_embeddings = nn.Embedding(
            num_embeddings=self.num_positions,  # 196
            embedding_dim=self.embed_dim        # 768
        )

        # Register position IDs as buffer (not a parameter)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Compute patch embeddings with positional information.

        This method processes input images through the patch embedding pipeline:
        1. Extract patch embeddings using convolution
        2. Reshape to sequence format
        3. Add positional embeddings

        Args:
            pixel_values: Input image tensor.
                        Shape: [batch_size, channels, height, width]

        Returns:
            Patch embeddings with positional information.
            Shape: [batch_size, num_patches, embed_dim]

        Example:
            >>> config = SiglipVisionConfig()
            >>> embeddings = SiglipVisionEmbeddings(config)
            >>> images = torch.randn(2, 3, 224, 224)
            >>> output = embeddings(images)  # Shape: [2, 196, 768]
        """
        batch_size, num_channels, height, width = pixel_values.shape

        # Step 1: Extract patch embeddings using convolution
        # Input: [batch_size, 3, 224, 224] -> Output: [batch_size, 768, 14, 14]
        patch_embeddings = self.patch_embeddings(pixel_values)

        # Step 2: Reshape from spatial to sequence format
        # [batch_size, 768, 14, 14] -> [batch_size, 768, 196]
        patch_embeddings = patch_embeddings.flatten(start_dim=2)

        # Step 3: Transpose to sequence-first format
        # [batch_size, 768, 196] -> [batch_size, 196, 768]
        patch_embeddings = patch_embeddings.transpose(1, 2)

        # Step 4: Add learnable positional embeddings
        # [batch_size, 196, 768] + [1, 196, 768] -> [batch_size, 196, 768]
        embeddings_with_position = patch_embeddings + self.position_embeddings(self.position_ids)

        return embeddings_with_position



class SiglipMLP(nn.Module):
    """Feed-forward MLP block used inside transformer encoder layers.

    This module implements the two-layer MLP with GELU activation that is
    standard in transformer architectures. The MLP expands the hidden
    dimension (typically 4x) and then projects back to the original size,
    allowing the model to capture complex non-linear relationships.

    Attributes:
        config: Model configuration containing MLP parameters.
        fc1: First linear layer that expands to intermediate size.
        fc2: Second linear layer that projects back to hidden size.
    """

    def __init__(self, config: SiglipVisionConfig) -> None:
        """Initialize SiglipMLP.

        Args:
            config: Model configuration containing MLP parameters.
        """
        super().__init__()
        self.config = config

        # First linear layer: expand to intermediate size (4x hidden_size)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)

        # Second linear layer: project back to hidden size
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the MLP transformation to hidden states.

        The MLP applies the following transformation:
        hidden_states -> fc1 -> GELU -> fc2 -> output

        Args:
            hidden_states: Input tensor of shape [batch_size, num_patches, hidden_size].

        Returns:
            Transformed tensor of shape [batch_size, num_patches, hidden_size].

        Example:
            >>> config = SiglipVisionConfig()
            >>> mlp = SiglipMLP(config)
            >>> hidden_states = torch.randn(2, 196, 768)
            >>> output = mlp(hidden_states)  # Shape: [2, 196, 768]
        """
        # Step 1: Expand to intermediate size
        # [batch_size, num_patches, hidden_size] -> [batch_size, num_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)

        # Step 2: Apply GELU activation (with tanh approximation for efficiency)
        # [batch_size, num_patches, intermediate_size] -> [batch_size, num_patches, intermediate_size]
        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh')

        # Step 3: Project back to hidden size
        # [batch_size, num_patches, intermediate_size] -> [batch_size, num_patches, hidden_size]
        hidden_states = self.fc2(hidden_states)

        return hidden_states

class SiglipEncoderLayer(nn.Module):
    """Single transformer encoder layer with pre-normalization.

    This layer implements the standard transformer encoder block with pre-norm
    architecture. The layer applies the following sequence:

    1. LayerNorm -> Self-Attention -> Residual connection
    2. LayerNorm -> MLP -> Residual connection

    The pre-norm architecture places layer normalization before attention and
    MLP blocks, which typically leads to more stable training compared to
    post-norm architectures.

    Attributes:
        embed_dim: Dimensionality of embeddings.
        self_attention: Multi-head self-attention module.
        layer_norm1: First layer normalization (before attention).
        mlp: Feed-forward MLP module.
        layer_norm2: Second layer normalization (before MLP).
    """

    def __init__(self, config: SiglipVisionConfig) -> None:
        """Initialize SiglipEncoderLayer.

        Args:
            config: Model configuration containing layer parameters.
        """
        super().__init__()
        self.embed_dim = config.hidden_size

        # Multi-head self-attention module
        self.self_attention = SiglipAttention(config)

        # First layer normalization (applied before attention)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        # Feed-forward MLP module
        self.mlp = SiglipMLP(config)

        # Second layer normalization (applied before MLP)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single transformer encoder layer.

        This method implements the pre-norm transformer encoder layer:
        1. Apply layer norm before attention
        2. Apply self-attention and add residual connection
        3. Apply layer norm before MLP
        4. Apply MLP and add residual connection

        Args:
            hidden_states: Input tensor of shape [batch_size, num_patches, embed_dim].

        Returns:
            Output tensor of shape [batch_size, num_patches, embed_dim].

        Example:
            >>> config = SiglipVisionConfig()
            >>> layer = SiglipEncoderLayer(config)
            >>> hidden_states = torch.randn(2, 196, 768)
            >>> output = layer(hidden_states)  # Shape: [2, 196, 768]
        """
        # Step 1: Self-attention block with residual connection
        # Save residual connection
        residual = hidden_states

        # Apply layer normalization before attention
        hidden_states = self.layer_norm1(hidden_states)

        # Apply multi-head self-attention
        attn_output, _ = self.self_attention(hidden_states)
        hidden_states = attn_output

        # Add residual connection (first residual)
        hidden_states = residual + hidden_states

        # Step 2: MLP block with residual connection
        # Save residual connection
        residual = hidden_states

        # Apply layer normalization before MLP
        hidden_states = self.layer_norm2(hidden_states)

        # Apply feed-forward MLP
        hidden_states = self.mlp(hidden_states)

        # Add residual connection (second residual)
        hidden_states = residual + hidden_states

        return hidden_states


class SiglipEncoder(nn.Module):
    """Stack of transformer encoder layers for SIGLIP vision model.

    This module creates a sequential stack of transformer encoder layers.
    Input embeddings are processed through each layer in sequence, with
    each layer applying self-attention and feed-forward transformations.

    Attributes:
        config: Model configuration containing encoder parameters.
        layers: ModuleList of SiglipEncoderLayer instances.
    """

    def __init__(self, config: SiglipVisionConfig) -> None:
        """Initialize SiglipEncoder.

        Args:
            config: Model configuration containing encoder parameters.
        """
        super().__init__()
        self.config = config
        # Create stack of encoder layers
        self.layers = nn.ModuleList([
            SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """Process input embeddings through the encoder stack.

        Args:
            input_embeds: Input embeddings of shape [batch_size, num_patches, embed_dim].

        Returns:
            Processed embeddings of shape [batch_size, num_patches, embed_dim].

        Example:
            >>> config = SiglipVisionConfig()
            >>> encoder = SiglipEncoder(config)
            >>> embeddings = torch.randn(2, 196, 768)
            >>> output = encoder(embeddings)  # Shape: [2, 196, 768]
        """
        # Pass embeddings through each encoder layer sequentially
        hidden_states = input_embeds

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)

        return hidden_states

class SiglipVisionTransformer(nn.Module):
    """Complete Vision Transformer model for SIGLIP.

    This module combines patch embeddings, transformer encoder layers, and
    final layer normalization to create a full vision transformer. The model
    processes input images through the following pipeline:

    1. Patch embedding: Convert images to sequence of patch embeddings
    2. Transformer encoder: Process embeddings through multiple layers
    3. Final normalization: Apply layer normalization to final output

    Attributes:
        config: Model configuration containing all parameters.
        embeddings: Patch embedding layer with positional information.
        encoder: Stack of transformer encoder layers.
        post_layernorm: Final layer normalization.
    """

    def __init__(self, config: SiglipVisionConfig) -> None:
        """Initialize SiglipVisionTransformer.

        Args:
            config: Model configuration containing all parameters.
        """
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        # Patch embedding layer with positional embeddings
        self.embeddings = SiglipVisionEmbeddings(config)

        # Stack of transformer encoder layers
        self.encoder = SiglipEncoder(config)

        # Final layer normalization
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract visual features from input images.

        This method processes images through the complete vision transformer
        pipeline to extract rich visual representations.

        Args:
            pixel_values: Input images of shape [batch_size, channels, height, width].

        Returns:
            Visual features of shape [batch_size, num_patches, embed_dim].

        Example:
            >>> config = SiglipVisionConfig()
            >>> model = SiglipVisionTransformer(config)
            >>> images = torch.randn(2, 3, 224, 224)
            >>> features = model(images)  # Shape: [2, 196, 768]
        """
        # Step 1: Extract patch embeddings with positional information
        # [batch_size, 3, 224, 224] -> [batch_size, 196, 768]
        embedding_output = self.embeddings(pixel_values)

        # Step 2: Process through transformer encoder layers
        # [batch_size, 196, 768] -> [batch_size, 196, 768]
        encoder_output = self.encoder(embedding_output)

        # Step 3: Apply final layer normalization
        # [batch_size, 196, 768] -> [batch_size, 196, 768]
        final_output = self.post_layernorm(encoder_output)

        return final_output

class SiglipVisionModel(nn.Module):
    """Convenience wrapper for the SIGLIP Vision Transformer model.

    This class provides a simple interface around the full vision transformer,
    making it easy to use the vision model as a component in larger architectures
    like vision-language models. It handles the configuration and provides
    a clean forward method for feature extraction.

    Attributes:
        config: Model configuration containing all parameters.
        vision_model: The underlying SiglipVisionTransformer instance.
    """

    def __init__(self, config: SiglipVisionConfig) -> None:
        """Initialize SiglipVisionModel.

        Args:
            config: Model configuration containing all parameters.
        """
        super().__init__()
        self.config = config
        # Initialize the complete vision transformer
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract visual features from input images.

        This method serves as a simple interface to the underlying vision
        transformer, making it easy to extract features for downstream tasks.

        Args:
            pixel_values: Input images of shape [batch_size, channels, height, width].

        Returns:
            Visual features of shape [batch_size, num_patches, embed_dim].

        Example:
            >>> config = SiglipVisionConfig()
            >>> model = SiglipVisionModel(config)
            >>> images = torch.randn(2, 3, 224, 224)
            >>> features = model(images)  # Shape: [2, 196, 768]
        """
        # Forward pass through the vision transformer
        # [batch_size, 3, 224, 224] -> [batch_size, 196, 768]
        return self.vision_model(pixel_values)
        
        