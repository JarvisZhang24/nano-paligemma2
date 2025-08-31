from huggingface_hub import snapshot_download, login
from dotenv import load_dotenv
import os
import json
import glob
from safetensors import safe_open
from transformers import AutoTokenizer

from pathlib import Path
import sys

"""
Step 1. Create a .env file in the root directory with HF_TOKEN = <your token>
Step 2. Run this script to download the model to ./paligemma2-3b-pt-224
"""

PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT.parent / "paligemma2-3b-mix-224"
sys.path.append(str(PROJECT_ROOT))

# # Load environment variables from .env file
load_dotenv()
HF_token = os.getenv("HF_TOKEN")

login(token=HF_token)
# download the model
snapshot_download(
    repo_id="google/paligemma2-3b-mix-224",
    local_dir=str(MODEL_DIR),
    local_dir_use_symlinks=False 
)



