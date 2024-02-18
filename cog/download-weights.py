#!/usr/bin/env python

import os
import sys
import json
import torch
import time
import subprocess
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionXLPipeline
from diffusers.models import ControlNetModel

# append project directory to path so predict.py can be imported
sys.path.append('.')

from predict import (
    CHECKPOINTS_CACHE,
    SAFETY_MODEL_CACHE,
    POSE_CHKPT_CACHE,
    CANNY_CHKPT_CACHE,
    DEPTH_CHKPT_CACHE,
)

# for `models/antelopev2`
MODELS_CACHE = "./models"
MODELS_URL = "https://weights.replicate.delivery/default/InstantID/models.tar"

# for safety checker
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"

# Download the ip-adapter and ControlNetModel checkpoints
def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

if not os.path.exists(MODELS_CACHE):
    download_weights(MODELS_URL, MODELS_CACHE)

if not os.path.exists(SAFETY_MODEL_CACHE):
    download_weights(SAFETY_URL, SAFETY_MODEL_CACHE)

model_list = [
    { 
        "repo_id": "InstantX/InstantID",
        "filename": "ip-adapter.bin",
        "use_symlinks": False,
        "local_dir": CHECKPOINTS_CACHE,
    },
    {
        "repo_id": "InstantX/InstantID",
        "filename": "ControlNetModel/config.json",
        "use_symlinks": False,
        "local_dir": CHECKPOINTS_CACHE,
    },
    {
        "repo_id": "InstantX/InstantID",
        "filename": "ControlNetModel/diffusion_pytorch_model.safetensors",
        "use_symlinks": False,
        "local_dir": CHECKPOINTS_CACHE,
    },
    {
        "repo_id": "latent-consistency/lcm-lora-sdxl",
        "filename": "pytorch_lora_weights.safetensors",
        "use_symlinks": False,
        "local_dir": CHECKPOINTS_CACHE,
    },
]

if not os.path.exists(CHECKPOINTS_CACHE):
    for model in model_list:
        hf_hub_download(
            repo_id=model["repo_id"],
            filename=model["filename"],
            local_dir_use_symlinks=model["use_symlinks"],
            local_dir=model["local_dir"]
        )

# Download and save the controlnet model weights
pipe = ControlNetModel.from_pretrained(
    "thibaud/controlnet-openpose-sdxl-1.0",
    torch_dtype=torch.float16,
)
# Save to cache folder. Will be created if doesn't exist.
pipe.save_pretrained(POSE_CHKPT_CACHE)

# Download and save the controlnet model weights
pipe = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
)
# Save to cache folder. Will be created if doesn't exist.
pipe.save_pretrained(CANNY_CHKPT_CACHE)

# Download and save the controlnet model weights
pipe = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0-small",
    torch_dtype=torch.float16,
)
# Save to cache folder. Will be created if doesn't exist.
pipe.save_pretrained(DEPTH_CHKPT_CACHE)

# Download and save SDXL image models
with open("./cog/img_models.json", "r") as f:
    data = json.load(f)
    for model in data["model"]:
        file_path = os.path.join(model["cacheFolder"], model["filename"])
        subprocess.check_call(["mkdir", "-p", model["cacheFolder"]], close_fds=False)
        download_weights(model["url"], file_path)
        pipe = StableDiffusionXLPipeline.from_single_file(
            file_path,
            torch_dtype=torch.float16,
        )
        # Save to cache folder. Will be created if doesn't exist.
        pipe.save_pretrained(model["cacheFolder"])
