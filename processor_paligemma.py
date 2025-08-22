"""Processing utilities for PaliGemma vision-language inputs.

Provides a `PaligemmaProcessor` that prepares text and image inputs for
vision-language models, including adding image tokens to prompts and
preprocessing images. Functions are documented with Google-style docstrings
and type hints.
"""

from typing import Dict, List, Optional, Union, Tuple, Iterable, Any
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

class PaligemmaProcessor:
    """Processor for preparing text and image inputs.

    This processor:
    - Adds a configurable number of image tokens to each prompt
    - Resizes, rescales, normalizes, and channels-first transforms images
    - Tokenizes prompts with the provided tokenizer

    Args:
        tokenizer: Tokenizer that supports adding special tokens and processing text.
        num_image_tokens: Number of image tokens to prepend to each prompt.
        image_size: Target image height and width (square images assumed).
    """

    # add image token to the prompt
    IMAGE_TOKEN = "<image>"

    def __init__(self ,tokenizer: Any , num_image_tokens: int , image_size: int ) -> None:
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}

        tokenizer.add_special_tokens(tokens_to_add)

        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # These tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # We will add the BOS and EOS tokens ourselves
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
        """Prepare batched inputs for the model.

        Args:
            text: A list of exactly one prompt string.
            images: A list containing exactly one PIL Image corresponding to the prompt.
            padding: Padding strategy for the tokenizer (e.g., "longest").
            truncation: Whether to enable truncation in the tokenizer.

        Returns:
            A dictionary containing:
            - "pixel_values": Tensor [batch, channels, height, width]
            - "input_ids": Token IDs tensor from the tokenizer
            - "attention_mask": Attention mask tensor from the tokenizer
        """
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

        pixel_values = process_images(
            images,
            target_size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )
        # Convert the list of numpy arrays to a single numpy array with shape 
        # [Batch_Size, Channel, Height, Width]
        pixel_values = np.stack(pixel_values, axis=0)

        # Convert the numpy array to a PyTorch tensor
        pixel_values = torch.tensor(pixel_values)

        # Prepend a `self.image_seq_length` number of image tokens to the prompt
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Returns the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data

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

# Add image tokens to the prompt
def add_image_tokens_to_prompt(prefix_prompt: str, bos_token: str, image_seq_len: int, image_token: str) -> str:
    """Prepend repeated image tokens and a BOS token to a prompt.

    Args:
        prefix_prompt: The human-readable prompt to send to the model.
        bos_token: The tokenizer's BOS token string.
        image_seq_len: Number of image tokens to repeat.
        image_token: The image token string to repeat.

    Returns:
        A single prompt string with image tokens and BOS prepended, ending with a newline.
    """
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

#  output: "<image><image>...256 times ...<BOS>What is in this image?\n"

# Rescale the pixel values to be in the range [0, 1]
def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32
) -> np.ndarray:
    """Rescale pixel values by a multiplicative factor and cast dtype.

    Args:
        image: Numpy array representing the image.
        scale: Multiplicative rescale factor.
        dtype: Output dtype (default: float32).

    Returns:
        Rescaled image array.
    """
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


# Resize the images to the desired height and width
def resize(
    image: Image.Image,
    size: Tuple[int, int],
    resample: Optional[Image.Resampling] = None,
    reducing_gap: Optional[int] = None,
) -> Image.Image:
    """Resize an image to the requested size.

    Args:
        image: PIL image to resize.
        size: Tuple of (height, width).
        resample: Optional PIL resampling filter.
        reducing_gap: Optional optimization parameter for PIL.

    Returns:
        A resized PIL image.
    """
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image


# Normalize the images to have mean 0 and standard deviation 1
def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    """Normalize image by mean and std (channel-wise supported).

    Args:
        image: Input image array.
        mean: Scalar or per-channel means.
        std: Scalar or per-channel standard deviations.

    Returns:
        Normalized image array.
    """
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image



def process_images(
    images: List[Image.Image],
    target_size: Tuple[int, int],
    resample: Optional[Image.Resampling] = None,
    rescale_factor: Optional[float] = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    """Preprocess a list of PIL images for the vision model.

    Steps: resize -> to numpy -> optional rescale -> optional normalize -> HWC->CHW.

    Args:
        images: List of PIL images.
        target_size: Tuple of (height, width) to resize images to.
        resample: Optional PIL resampling filter to use when resizing.
        rescale_factor: If provided, multiply pixel values by this factor.
        image_mean: If provided, mean(s) for normalization.
        image_std: If provided, std(s) for normalization.

    Returns:
        A list of numpy arrays shaped [channels, height, width].
    """
    # Get the height and width of the images
    height, width = target_size

    # Resize the images to the desired height and width
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    # Convert each image to a numpy array
    images = [np.array(image) for image in images]
    # Rescale the pixel values to be in the range [0, 1]
    if rescale_factor is not None:
        images = [rescale(image, scale=rescale_factor) for image in images]
    # Normalize the images to have mean 0 and standard deviation 1
    if image_mean is not None and image_std is not None:
        images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # Move the channel dimension to the first dimension. The model expects images in the format [Channel, Height, Width]
    images = [image.transpose(2, 0, 1) for image in images]
    return images


# Test PaligemmaProcessor
from transformers import AutoTokenizer
from PIL import Image

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", padding_side="right")

# BOS and EOS 
specials = {}
if tokenizer.bos_token is None: specials["bos_token"] = "<s>"
if tokenizer.eos_token is None: specials["eos_token"] = "</s>"
# if pad_token is None, use eos_token
if tokenizer.pad_token is None: specials["pad_token"] = specials.get("eos_token", tokenizer.eos_token or "</s>")
if specials: tokenizer.add_special_tokens(specials)

processor = PaligemmaProcessor(tokenizer, num_image_tokens=256, image_size=224)

text = ["What is in this image?"]
test_image = Image.new('RGB', (224, 224), color='red')

# Note: images need to be a list
inputs = processor(text, [test_image])

print("Keys:", list(inputs.keys()))
print("pixel_values:", inputs["pixel_values"].shape)   # Target: torch.Size([1, 3, 224, 224])
print("input_ids:", inputs["input_ids"].shape)
print("attention_mask:", inputs["attention_mask"].shape)