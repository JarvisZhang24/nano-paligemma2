from huggingface_hub import snapshot_download, login
from dotenv import load_dotenv
import os
import json
import glob
from safetensors import safe_open
from transformers import AutoTokenizer

from src.model import PaliGemma2
from src.configs import PaliGemma2Config
from pathlib import Path
import sys

"""
Step 1. Create a .env file in the root directory with HF_TOKEN = <your token>
Step 2. Run this script to download the model to ./paligemma2-3b-pt-224
"""

PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# # Load environment variables from .env file
# load_dotenv()
# HF_token = os.getenv("HF_TOKEN")

# login(token=HF_token)
# download the model
# snapshot_download(
#     repo_id="google/paligemma2-3b-mix-224",
#     local_dir=str(PROJECT_ROOT / "paligemma2-3b-mix-224"),
#     local_dir_use_symlinks=False 
# )


################################### Load Hugging Face weights ###################################


def load_weights(model_path: str, model_type: str, device: str):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config

    if model_type == "paligemma2":
        with open(os.path.join(model_path, "config.json"), "r") as f:
            model_config_file = json.load(f)
            config = PaliGemma2Config(**model_config_file)
        model = PaliGemma2(config).to(device)
    else:
        raise ValueError(f"Model type {model_type} not supported.")

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)
