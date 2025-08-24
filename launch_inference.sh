#!/bin/bash
'''
This script is used to launch the inference of the model.
It is used to generate the caption for the image.

The model is a PaliGemma model.
The model is trained on the PALI dataset.
The model is a 3B model.
The model is a 224x224 model.

Args:
    --model_path: The path to the model.
    --prompt: The prompt to generate the caption.
    --image_file_path: The path to the image file.
    --max_tokens_to_generate: The maximum tokens to generate.
    --temperature: The temperature for the generation.
    --top_p: The top p for the generation.
    --do_sample: Whether to sample the generation.
    --only_cpu: Whether to run the inference on the CPU.

Example:
    ./launch_inference.sh
    ./launch_inference.sh --model_path $MODEL_PATH --prompt $PROMPT --image_file_path $IMAGE_FILE_PATH --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE --temperature $TEMPERATURE --top_p $TOP_P --do_sample $DO_SAMPLE --only_cpu $ONLY_CPU



'''

# Set the model path
MODEL_PATH="$HOME/projects/paligemma-weights/paligemma-3b-pt-224"

# Set the prompt
PROMPT="this building is "

# Set the image file path
IMAGE_FILE_PATH="test_images/pic1.jpeg"

# Set the maximum tokens to generate
MAX_TOKENS_TO_GENERATE=100

# Set the temperature
TEMPERATURE=0.8

# Set the top p
TOP_P=0.9

# Set the do sample
DO_SAMPLE="False"

# Set the only cpu
ONLY_CPU="False"

# Run the inference
python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \