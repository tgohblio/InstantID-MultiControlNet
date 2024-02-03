#!/usr/bin/env python

import os
import sys
import torch
import time
import subprocess
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionXLPipeline

# append project directory to path so predict.py can be imported
sys.path.append('.')

from predict import SD_MODEL_NAME, SD_MODEL_CACHE, SAFETY_MODEL_CACHE

# for ip-adapter and ControlNetModel
CHECKPOINTS_CACHE = "./checkpoints"

# for `models/antelopev2`
MODELS_CACHE = "./models"
MODELS_URL = "https://weights.replicate.delivery/default/InstantID/models.tar"

# for safety checker
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"

# Download and save the SD model weights
pipe = StableDiffusionXLPipeline.from_pretrained(
    SD_MODEL_NAME,
    torch_dtype=torch.float16,
)
# Save to cache folder. Will be created if doesn't exist.
pipe.save_pretrained(SD_MODEL_CACHE)

# Download the ip-adapter and ControlNetModel checkpoints
def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

if not os.path.exists(MODELS_CACHE):
    download_weights(MODELS_URL, MODELS_CACHE)

if not os.path.exists(SAFETY_MODEL_CACHE):
    download_weights(SAFETY_URL, SAFETY_MODEL_CACHE)

if not os.path.exists(CHECKPOINTS_CACHE):
    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ip-adapter.bin",
        local_dir_use_symlinks=False,
        local_dir=CHECKPOINTS_CACHE
    )
    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ControlNetModel/config.json",
        local_dir=CHECKPOINTS_CACHE
    )
    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ControlNetModel/diffusion_pytorch_model.safetensors",
        local_dir_use_symlinks=False,
        local_dir=CHECKPOINTS_CACHE
    )
    hf_hub_download(
        repo_id="latent-consistency/lcm-lora-sdxl",
        filename="pytorch_lora_weights.safetensors",
        local_dir_use_symlinks=False,
        local_dir=CHECKPOINTS_CACHE
    )
