"""PaliGemma processor for vision-language model inputs.

This module provides comprehensive preprocessing utilities for PaliGemma
vision-language models. It handles the complex task of preparing multimodal
inputs by combining text and image data into a unified format that the model
can process.

Key Features:
- Image preprocessing: Resize, rescale, normalize, and convert to model format
- Text preprocessing: Add special image tokens and tokenize prompts
- Multimodal integration: Combine image and text features seamlessly
- Flexible configuration: Support for various image sizes and processing options

The processor follows the specific requirements of PaliGemma models, including:
- Adding configurable numbers of image tokens to prompts
- Proper image preprocessing with ImageNet-style normalization
- Handling special tokens for object detection and segmentation
- Maintaining correct tensor shapes and data types

Example:
    >>> processor = PaligemmaProcessor(tokenizer, num_image_tokens=256, image_size=224)
    >>> inputs = processor(text=["What is in this image?"], images=[image])
    >>> # Returns: {"pixel_values": ..., "input_ids": ..., "attention_mask": ...}

The code is annotated with comprehensive Google-style docstrings and type hints.
"""

from typing import Dict, List, Optional, Union, Tuple, Iterable, Any
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

class PaligemmaProcessor:
    """Comprehensive processor for PaliGemma vision-language model inputs.

    This processor handles the complete preprocessing pipeline for PaliGemma models,
    transforming raw text and images into model-ready inputs. The processing includes:

    - Text Processing:
        * Adding configurable number of image tokens to prompts
        * Tokenization with special handling for multimodal sequences
        * Integration of location and segmentation tokens for downstream tasks

    - Image Processing:
        * Resizing to target dimensions (square images assumed)
        * Channel-first transformation (HWC -> CHW)
        * ImageNet-style normalization with configurable mean/std
        * Optional rescaling and advanced interpolation

    - Multimodal Integration:
        * Seamless combination of image and text features
        * Proper attention mask generation
        * Correct tensor shape management

    The processor is specifically designed for PaliGemma's requirements and includes
    special tokens for object detection (location tokens) and segmentation tasks.

    Attributes:
        IMAGE_TOKEN: Special token representing image content in text.
        image_seq_length: Number of image tokens to add to each prompt.
        image_size: Target size for image preprocessing.
        tokenizer: Configured tokenizer with special tokens.
        image_token_id: ID of the image token in tokenizer vocabulary.
    """

    # Special token used to represent image content in text sequences
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer: Any, num_image_tokens: int, image_size: int) -> None:
        """Initialize the PaliGemmaProcessor.

        Sets up the tokenizer with all necessary special tokens and configures
        image processing parameters.

        Args:
            tokenizer: Tokenizer instance that supports adding special tokens.
                     Must be compatible with PaliGemma's token requirements.
            num_image_tokens: Number of image tokens to prepend to each prompt.
                            Typically 256 for base models.
            image_size: Target image size (height and width, square assumed).
                      Typically 224 for standard models.
        """
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        # Configure tokenizer with special tokens for multimodal processing
        # Reference: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        # Add location tokens for object detection (bounding boxes)
        # Format: <loc0000> to <loc1023> for 1024 possible locations
        location_tokens = [f"<loc{i:04d}>" for i in range(1024)]

        # Add segmentation tokens for object segmentation
        # Format: <seg000> to <seg127> for 128 possible segments
        segmentation_tokens = [f"<seg{i:03d}>" for i in range(128)]

        # Add all extra tokens to tokenizer vocabulary
        extra_tokens = location_tokens + segmentation_tokens
        tokenizer.add_tokens(extra_tokens)

        # Store image token ID for efficient processing
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        # Configure tokenizer to not add BOS/EOS tokens automatically
        # We handle these manually for precise control over multimodal sequences
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Process text and images into model-ready multimodal inputs.

        This method handles the complete preprocessing pipeline for PaliGemma:
        1. Validate input lengths (single image-text pair expected)
        2. Process images through resizing, normalization, and format conversion
        3. Prepare text by adding image tokens and tokenization
        4. Combine processed inputs into model-ready format

        Args:
            text: List containing exactly one prompt string. Must correspond
                 to the single image being processed.
            images: List containing exactly one PIL Image. Must correspond
                   to the single text prompt being processed.
            padding: Tokenization padding strategy. Options: "longest", "max_length",
                    or False. Default: "longest".
            truncation: Whether to truncate sequences exceeding max_length.
                      Default: True.

        Returns:
            Dictionary containing model-ready inputs:
            - "pixel_values": Processed image tensor of shape [1, 3, H, W]
            - "input_ids": Tokenized input sequence with image tokens
            - "attention_mask": Attention mask for the tokenized sequence

        Raises:
            AssertionError: If input lengths don't match (must be 1:1 correspondence).

        Example:
            >>> processor = PaligemmaProcessor(tokenizer, 256, 224)
            >>> inputs = processor(text=["What is in this image?"], images=[image])
            >>> # Returns: {"pixel_values": ..., "input_ids": ..., "attention_mask": ...}
        """
        # Validate input correspondence - expect single image-text pair
        if len(images) != 1 or len(text) != 1:
            raise AssertionError(
                f"Expected exactly 1 image and 1 text prompt, "
                f"but received {len(images)} images and {len(text)} prompts"
            )

        # Step 1: Process images through complete preprocessing pipeline
        pixel_values = process_images(
            images=images,
            target_size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,  # High-quality resampling
            rescale_factor=1 / 255.0,          # Convert to [0,1] range
            image_mean=IMAGENET_STANDARD_MEAN,  # Standard ImageNet normalization
            image_std=IMAGENET_STANDARD_STD,
        )

        # Convert list of processed images to batched tensor
        # [num_images, height, width, channels] -> [num_images, channels, height, width]
        pixel_values = np.stack(pixel_values, axis=0)
        pixel_values = torch.tensor(pixel_values, dtype=torch.float32)

        # Step 2: Prepare text inputs with image tokens
        # Add image tokens to each prompt for multimodal processing
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Step 3: Tokenize the prepared input strings
        # Convert text to token IDs and create attention masks
        tokenized_inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",  # Return PyTorch tensors
            padding=padding,
            truncation=truncation,
        )

        # Step 4: Combine all inputs into model-ready format
        processed_inputs = {
            "pixel_values": pixel_values,
            **tokenized_inputs
        }

        return processed_inputs

# output:
'''
{
    'pixel_values': tensor([[[[-0.3333, -0.3333, ...],    # cat image normalized to [-1,1]
                            [-0.3333, -0.3333, ...],    # R channel
                            ...],
                           [[-0.3333, -0.3333, ...],    # G channel
                            [-0.3333, -0.3333, ...],
                            ...],
                           [[-0.3333, -0.3333, ...],    # B channel
                            [-0.3333, -0.3333, ...],
                            ...]]]),                    # shape: [1, 3, 224, 224]
    
    'input_ids': tensor([[32000, 32000, ..., 32000,      # 256 <image> tokens
                          2,                             # <BOS> token
                          235290, 318, 263, 1339, 1234,  # "What is in this image?" tokens
                          1]]),                          # possible <EOS> or newline
    
    'attention_mask': tensor([[1, 1, 1, ..., 1, 1, 1, 1]]) # all 1 attention mask
}
'''

def add_image_tokens_to_prompt(
    prefix_prompt: str,
    bos_token: str,
    image_seq_len: int,
    image_token: str
) -> str:
    """Create a multimodal prompt by adding image tokens and BOS token.

    This function prepares a text prompt for multimodal processing by prepending
    the appropriate number of image tokens followed by the beginning-of-sequence
    token. This allows the model to understand where image information should be
    integrated into the text sequence.

    Args:
        prefix_prompt: The original text prompt describing the image content.
                     Should be a clear, descriptive question or instruction.
        bos_token: The tokenizer's beginning-of-sequence token string.
                  Used to properly start the multimodal sequence.
        image_seq_len: Number of image tokens to prepend. This determines
                      how many visual features will be integrated into the prompt.
        image_token: The special token representing image content.
                    Typically "<image>" for PaliGemma models.

    Returns:
        A formatted prompt string ready for tokenization:
        "{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

    Example:
        >>> prompt = "What is in this image?"
        >>> formatted = add_image_tokens_to_prompt(prompt, "<s>", 256, "<image>")
        >>> # Result: "<image><image>...256 times...<s>What is in this image?\n"
    """
    # Construct multimodal prompt with image tokens, BOS token, and original prompt
    multimodal_prompt = f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"
    return multimodal_prompt

#  output: "<image><image>...256 times ...<BOS>What is in this image?\n"

def rescale(
    image: np.ndarray,
    scale: float,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """Rescale image pixel values by a multiplicative factor.

    This function applies a simple linear scaling to image pixel values,
    typically used to normalize pixel ranges (e.g., from [0,255] to [0,1]).

    Args:
        image: Input image as numpy array with pixel values in any range.
        scale: Multiplicative scaling factor. Use 1/255.0 to convert
               from [0,255] to [0,1] range.
        dtype: Target data type for the output array. Default: float32.

    Returns:
        Rescaled image array with the specified dtype.

    Example:
        >>> image = np.array([[255, 128], [64, 0]])  # [0,255] range
        >>> rescaled = rescale(image, 1/255.0)       # Convert to [0,1]
        >>> # Result: [[1.0, 0.5], [0.25, 0.0]]
    """
    # Apply linear scaling transformation
    rescaled_image = image * scale

    # Convert to target data type
    rescaled_image = rescaled_image.astype(dtype)

    return rescaled_image


def resize(
    image: Image.Image,
    size: Tuple[int, int],
    resample: Optional[Image.Resampling] = None,
    reducing_gap: Optional[int] = None,
) -> Image.Image:
    """Resize a PIL image to the specified dimensions.

    This function handles image resizing with configurable resampling methods
    and optimization parameters. The resizing is typically used to standardize
    input images to the model's expected dimensions.

    Args:
        image: Input PIL Image object to be resized.
        size: Target size as (height, width) tuple. Both dimensions should
              be positive integers.
        resample: PIL resampling filter to use for resizing. Options include:
                 - Image.Resampling.BICUBIC (high quality, recommended)
                 - Image.Resampling.BILINEAR (balanced quality/speed)
                 - Image.Resampling.NEAREST (fastest, lowest quality)
                 If None, PIL uses its default resampling method.
        reducing_gap: Optimization parameter for PIL's resize algorithm.
                     Controls the threshold for using optimized resizing.
                     If None, PIL uses its default value.

    Returns:
        A new PIL Image object resized to the specified dimensions.
        The original image is not modified.

    Example:
        >>> from PIL import Image
        >>> img = Image.open("image.jpg")  # 1000x800 image
        >>> resized = resize(img, (224, 224), Image.Resampling.BICUBIC)
        >>> resized.size  # (224, 224)
    """
    # Extract target dimensions
    height, width = size

    # Perform resizing with specified parameters
    resized_image = image.resize(
        size=(width, height),      # Note: PIL expects (width, height)
        resample=resample,
        reducing_gap=reducing_gap
    )

    return resized_image


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    """Normalize image pixel values using mean and standard deviation.

    This function applies standard normalization (zero mean, unit variance)
    to image pixel values. Supports both scalar and per-channel normalization
    for RGB images. The normalization is crucial for model convergence and
    typically uses ImageNet statistics.

    Args:
        image: Input image array to normalize. Can be grayscale (H,W) or
               RGB (H,W,C) format.
        mean: Normalization mean value(s). Can be:
             - Scalar: Single mean applied to all channels
             - List/Array: Per-channel means for RGB images
        std: Normalization standard deviation value(s). Can be:
            - Scalar: Single std applied to all channels
            - List/Array: Per-channel stds for RGB images

    Returns:
        Normalized image array with zero mean and unit variance.
        Output has the same shape as input.

    Example:
        >>> # Normalize RGB image with ImageNet statistics
        >>> image = np.random.rand(224, 224, 3)  # [0,1] range
        >>> mean = [0.485, 0.456, 0.406]  # ImageNet RGB means
        >>> std = [0.229, 0.224, 0.225]   # ImageNet RGB stds
        >>> normalized = normalize(image, mean, std)
        >>> # Result: ~zero mean, unit variance per channel
    """
    # Convert mean and std to numpy arrays for broadcasting
    # This handles both scalar and per-channel cases
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)

    # Apply normalization: (pixel - mean) / std
    # Broadcasting handles scalar vs per-channel automatically
    normalized_image = (image - mean) / std

    return normalized_image



def process_images(
    images: List[Image.Image],
    target_size: Tuple[int, int],
    resample: Optional[Image.Resampling] = None,
    rescale_factor: Optional[float] = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    """Complete image preprocessing pipeline for vision models.

    This function applies a standard computer vision preprocessing pipeline
    to prepare PIL images for input to vision-language models. The pipeline
    includes resizing, format conversion, optional rescaling, optional
    normalization, and channel reordering.

    Processing Steps:
    1. Resize images to target dimensions
    2. Convert PIL Images to numpy arrays
    3. Optional: Rescale pixel values (e.g., [0,255] -> [0,1])
    4. Optional: Normalize using mean and standard deviation
    5. Reorder dimensions from HWC to CHW format

    Args:
        images: List of input PIL Image objects to process.
        target_size: Target dimensions as (height, width) tuple.
                    Images will be resized to these dimensions.
        resample: PIL resampling method for resizing. Recommended:
                 Image.Resampling.BICUBIC for high quality.
        rescale_factor: Optional scaling factor for pixel values.
                       Use 1/255.0 to convert from [0,255] to [0,1].
        image_mean: Optional mean values for normalization. Can be:
                   - Scalar: Single mean for all channels
                   - List: Per-channel means [R_mean, G_mean, B_mean]
                   Typically uses ImageNet means: [0.485, 0.456, 0.406]
        image_std: Optional standard deviation values for normalization. Can be:
                  - Scalar: Single std for all channels
                  - List: Per-channel stds [R_std, G_std, B_std]
                  Typically uses ImageNet stds: [0.229, 0.224, 0.225]

    Returns:
        List of processed numpy arrays with shape [channels, height, width].
        Each array is ready for input to vision models.

    Example:
        >>> from PIL import Image
        >>> images = [Image.open("image1.jpg"), Image.open("image2.jpg")]
        >>> processed = process_images(
        ...     images=images,
        ...     target_size=(224, 224),
        ...     resample=Image.Resampling.BICUBIC,
        ...     rescale_factor=1/255.0,
        ...     image_mean=[0.485, 0.456, 0.406],
        ...     image_std=[0.229, 0.224, 0.225]
        ... )
        >>> # Result: List of [3, 224, 224] arrays, normalized and ready for model
    """
    # Extract target dimensions
    height, width = target_size

    # Step 1: Resize all images to target dimensions
    resized_images = [
        resize(image=image, size=(height, width), resample=resample)
        for image in images
    ]

    # Step 2: Convert PIL Images to numpy arrays (HWC format)
    # [height, width, channels]
    numpy_images = [np.array(image) for image in resized_images]

    # Step 3: Optional rescaling of pixel values
    if rescale_factor is not None:
        numpy_images = [
            rescale(image, scale=rescale_factor) for image in numpy_images
        ]

    # Step 4: Optional normalization using mean and std
    if image_mean is not None and image_std is not None:
        numpy_images = [
            normalize(image, mean=image_mean, std=image_std)
            for image in numpy_images
        ]

    # Step 5: Convert from HWC to CHW format (required by PyTorch models)
    # [height, width, channels] -> [channels, height, width]
    chw_images = [image.transpose(2, 0, 1) for image in numpy_images]

    return chw_images


# Example usage and testing of PaligemmaProcessor
"""
This section demonstrates how to use the PaligemmaProcessor with a sample setup.
In practice, you would replace the test image with real images and use an
appropriate tokenizer for your specific PaliGemma model variant.
"""

# Example setup (requires transformers and PIL to be installed)
try:
    from transformers import AutoTokenizer
    from PIL import Image

    # Initialize tokenizer (using BERT as example - replace with actual PaliGemma tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", padding_side="right")

    # Configure special tokens if not present
    special_tokens = {}
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = "<s>"
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = "</s>"
    # Use EOS token as PAD token if PAD token is not defined
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = special_tokens.get("eos_token", tokenizer.eos_token or "</s>")

    # Add special tokens to tokenizer
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)

    # Initialize processor with typical PaliGemma parameters
    processor = PaligemmaProcessor(
        tokenizer=tokenizer,
        num_image_tokens=256,  # Standard for base models
        image_size=224         # Standard input size
    )

    # Prepare sample inputs
    text_prompts = ["What is in this image?"]
    # Create a test image (in practice, load real images)
    test_image = Image.new('RGB', (224, 224), color='red')

    # Process inputs - note that images must be passed as a list
    processed_inputs = processor(text=text_prompts, images=[test_image])

    # Display output information
    print("Processed input keys:", list(processed_inputs.keys()))
    print("Pixel values shape:", processed_inputs["pixel_values"].shape)  # Expected: [1, 3, 224, 224]
    print("Input IDs shape:", processed_inputs["input_ids"].shape)
    print("Attention mask shape:", processed_inputs["attention_mask"].shape)

    # Verify the processing worked correctly
    assert processed_inputs["pixel_values"].shape == (1, 3, 224, 224), "Incorrect pixel values shape"
    assert processed_inputs["input_ids"].shape[0] == 1, "Expected batch size of 1"
    assert processed_inputs["attention_mask"].shape[0] == 1, "Expected batch size of 1"

    print("\n✅ PaligemmaProcessor test completed successfully!")

except ImportError as e:
    print(f"❌ Test skipped: Required dependencies not available ({e})")
    print("To run this test, install: pip install transformers pillow")

except Exception as e:
    print(f"❌ Test failed: {e}")
    print("This may indicate an issue with the processor implementation or dependencies")