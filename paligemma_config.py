from model_siglip import SiglipVisionConfig
from typing import Optional, Dict, Any

class SiglipVisionConfig:
    """Configuration class for SIGLIP Vision Transformer hyperparameters.

    This class encapsulates all the configuration parameters needed to initialize
    and train a SIGLIP vision transformer model. The parameters control various
    aspects of the model architecture including embedding dimensions, attention
    mechanisms, and regularization.

    Attributes:
        hidden_size: Dimensionality of hidden states throughout the encoder.
        intermediate_size: Size of the intermediate layer in MLP blocks.
        num_hidden_layers: Number of transformer encoder layers.
        num_attention_heads: Number of attention heads in multi-head attention.
        num_channels: Number of input image channels (typically 3 for RGB).
        image_size: Input image height and width (assumed square).
        patch_size: Size of patches for patch embedding (height and width).
        layer_norm_eps: Epsilon value for layer normalization stability.
        attention_dropout: Dropout probability for attention weights.
        num_image_tokens: Optional precomputed number of image tokens.

    Example:
        >>> config = SiglipVisionConfig(
        ...     hidden_size=768,
        ...     num_hidden_layers=12,
        ...     num_attention_heads=12
        ... )
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
        """Initialize SiglipVisionConfig.

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

class GemmaConfig():
    '''
    This class encapsulates all the configuration parameters needed to initialize
    and train a Gemma model. The parameters control various aspects of the model architecture including embedding dimensions, attention
    mechanisms, and regularization.

    Attributes:
        vocab_size: Size of the vocabulary.
        max_position_embeddings: Maximum length of the input sequence.
        hidden_size: Dimensionality of hidden states throughout the encoder.
        intermediate_size: Size of the intermediate layer in MLP blocks.
        num_hidden_layers: Number of transformer encoder layers.
        num_attention_heads: Number of attention heads in multi-head attention.
        num_key_value_heads: Number of key value heads in multi-head attention.
        head_dim: Dimensionality of each head in multi-head attention.
        rms_norm_eps: Epsilon value for layer normalization stability.
        rope_theta: The base of the exponential function used for RoPE.
        attention_bias: Whether to use attention bias.
        attention_dropout: Dropout probability for weights.
        pad_token_id: Token ID for padding.
        **kwargs: Additional unused keyword arguments for forward compatibility.

    Example:
        >>> config = GemmaConfig(
            vocab_size=257152,
            hidden_size=2048,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=12,
            head_dim=256,
            max_position_embeddings=8192,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            attention_bias=False,
            attention_dropout=0.0,
            pad_token_id=None
        )
    '''



    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig():

    '''
    This class encapsulates all the configuration parameters needed to initialize
    and train a PaliGemma model. The parameters control various aspects of the model architecture including embedding dimensions, attention
    mechanisms, and regularization.

    Attributes:
        vocab_size: Size of the vocabulary.
        max_position_embeddings: Maximum length of the input sequence.
        hidden_size: Dimensionality of hidden states throughout the encoder.
        intermediate_size: Size of the intermediate layer in MLP blocks.
        num_hidden_layers: Number of transformer encoder layers.
        num_attention_heads: Number of attention heads in multi-head attention.
        num_key_value_heads: Number of key value heads in multi-head attention.
        head_dim: Dimensionality of each head in multi-head attention.
        rms_norm_eps: Epsilon value for layer normalization stability.
        rope_theta: The base of the exponential function used for RoPE.
        attention_bias: Whether to use attention bias.
        attention_dropout: Dropout probability for attention weights.
        pad_token_id: Token ID for padding.
        vision_config: Configuration for the vision transformer.
        text_config: Configuration for the text transformer.
        ignore_index: Index for ignored tokens.
        image_token_index: Index for the image token.
        vocab_size: Size of the vocabulary.
        projection_dim: Dimensionality of the projection layer.
        hidden_size: Dimensionality of hidden states throughout the encoder.
        pad_token_id: Token ID for padding.
        **kwargs: Additional unused keyword arguments for forward compatibility.

    Example:
        >>> config = PaliGemmaConfig(
            vocab_size=257152,
            hidden_size=2048,
            projection_dim=2048,
            vision_config=SiglipVisionConfig(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12
            ),
            text_config=GemmaConfig(
                vocab_size=257152,
                hidden_size=2048,
                num_hidden_layers=12,
                num_attention_heads=12
            )
        )
    '''

    def __init__(
        self,
        # Vision config
        vision_config=None,
        # Text config   
        text_config=None,
        # Other configs
        ignore_index=-100,
        image_token_index=256000,
        # Vocab size
        vocab_size=257152,
        # Projection dim (hidden size of the vision model)
        projection_dim=2048,
        # Hidden size (hidden size of the text model)
        hidden_size=2048,
        # Pad token id
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id )

        # Set the vocab size to the vocab size of the text model
        self.vocab_size = self.text_config.vocab_size

        # Set the number of image tokens to the number of patches in the image
        self.vision_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2

        # Set the projection dim to the projection dim of the vision model
        self.vision_config.projection_dim = projection_dim