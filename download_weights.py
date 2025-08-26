from huggingface_hub import snapshot_download

import os
from huggingface_hub import login
from dotenv import load_dotenv

"""
Step 1. Create a .env file in the root directory with HF_TOKEN = <your token>
Step 2. Run this script to download the model to ./paligemma2-3b-pt-224
"""

# Load environment variables from .env file
load_dotenv()
HF_token = os.getenv("HF_TOKEN")

login(token=HF_token)
# download the model
snapshot_download(
    repo_id="google/paligemma2-3b-pt-224",
    local_dir="./paligemma2-3b-pt-224",
    local_dir_use_symlinks=False 
)