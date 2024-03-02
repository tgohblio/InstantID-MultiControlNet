#!/usr/bin/env python

import os
import sys
import torch

from huggingface_hub import hf_hub_download
from diffusers.models import ControlNetModel
from controlnet_aux import OpenposeDetector

# append project directory to path so predict.py can be imported
sys.path.append('.')

from depth_anything.dpt import DepthAnything
from predict import (
    download_weights,
    setup_sdxl_pipeline,
    DEFAULT_SDXL_MODEL,
    CHECKPOINTS_CACHE,
    SAFETY_MODEL_CACHE,
    POSE_CHKPT_CACHE,
    CANNY_CHKPT_CACHE,
    DEPTH_CHKPT_CACHE,
    LORA_CHECKPOINTS_CACHE
)

# for `models/antelopev2`
MODELS_CACHE = "./models"
MODELS_URL = "https://weights.replicate.delivery/default/InstantID/models.tar"

# for safety checker
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"

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
        "repo_id": "ByteDance/SDXL-Lightning",
        "filename": "sdxl_lightning_2step_lora.safetensors",
        "use_symlinks": False,
        "local_dir": LORA_CHECKPOINTS_CACHE,
    },
    {
        "repo_id": "ByteDance/SDXL-Lightning",
        "filename": "sdxl_lightning_4step_lora.safetensors",
        "use_symlinks": False,
        "local_dir": LORA_CHECKPOINTS_CACHE,
    },
    {
        "repo_id": "ByteDance/SDXL-Lightning",
        "filename": "sdxl_lightning_8step_lora.safetensors",
        "use_symlinks": False,
        "local_dir": LORA_CHECKPOINTS_CACHE,
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

# Download to cache
OpenposeDetector.from_pretrained(
    "lllyasviel/ControlNet",
    cache_dir=CHECKPOINTS_CACHE
)
DepthAnything.from_pretrained(
    'LiheYoung/depth_anything_vitl14',
    cache_dir=CHECKPOINTS_CACHE
)

# Download and save default SDXL model only
setup_sdxl_pipeline(DEFAULT_SDXL_MODEL)
