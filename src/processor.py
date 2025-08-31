import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Iterable, Union
import torchvision.transforms as T


################################### Constants ###################################

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]  # From HF code
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]  # From HF code

################################### Utility functions ###################################


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    # The input text is tokenized normally.
    # A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    # This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    # The tokenized text is also prefixed with a fixed number of <image> tokens.
    # Unlike in the PaliGemma paper, the Hugging Face code doesn't tokenize \n separately.
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def preprocess_single_image(image, size=(224, 224), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    # Support PIL.Image or str(path)
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
    elif not isinstance(image, Image.Image):
        raise ValueError("Unsupported image type")
    
    # Resize
    transform = T.Compose([
        T.Resize(size, antialias=True),
        T.ToTensor(),  # [C, H, W] Tensor
        T.Normalize(mean=mean, std=std)
    ])
    image_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]
    
    return image_tensor


################################### PaliGemma Processor ###################################


class PaliGemmaProcessor:

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()
        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        # Tokens for object segmentation
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # The tokenizer will not automatically prepend a BOS token or append an EOS token when encoding text.
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        image: Image.Image,
        padding: str = "longest",
        truncation: bool = False,
    ) -> dict:

        pixel_values = preprocess_single_image(
            image,
            size=(self.image_size, self.image_size),
            mean=IMAGENET_STANDARD_MEAN,
            std=IMAGENET_STANDARD_STD,
        )

        # The image tokens act as placeholders and will be later replaced by the image embeddings.
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
        # The attention mask is only 1s as we don't use padding
        # The model has been trained with a maximum sequence length of 128
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return_data = {"pixel_values": pixel_values, **inputs}

        # we return the preprocessed image tensor and the tokenized input with the <image> placeholders, BOS token, prefix prompt and the separator.
        return return_data
