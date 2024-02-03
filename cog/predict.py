# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import sys

from cog import BasePredictor, Input, Path

import cv2
import torch
import numpy as np
from PIL import Image

from diffusers import LCMScheduler
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
from insightface.app import FaceAnalysis

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pipeline_stable_diffusion_xl_instantid import (
    StableDiffusionXLInstantIDPipeline,
    draw_kps,
)

# for ip-adapter, ControlNetModel
CHECKPOINTS_CACHE = "./checkpoints"

# for SDXL model
SD_MODEL_CACHE = "./sd_model"
SD_MODEL_NAME = "GraydientPlatformAPI/albedobase2-xl"

# safety checker model
SAFETY_MODEL_CACHE = "./safety_cache"
FEATURE_EXTRACT_CACHE = "feature_extractor"

def resize_img(
    input_image,
    max_side=1280,
    min_side=1024,
    size=None,
    pad_to_max_side=False,
    mode=Image.BILINEAR,
    base_pixel_number=64,
):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[
            offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new
        ] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load safety checker"""
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_MODEL_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACT_CACHE)

        """Load the model into memory to make running multiple predictions efficient"""
        self.width, self.height = 640, 640
        self.app = FaceAnalysis(
            name="antelopev2",
            root="./",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(self.width, self.height))

        # Path to InstantID models
        face_adapter = f"{CHECKPOINTS_CACHE}/ip-adapter.bin"
        controlnet_path = f"{CHECKPOINTS_CACHE}/ControlNetModel"

        # Load pipeline
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch.float16,
            cache_dir=CHECKPOINTS_CACHE,
            use_safetensors=True,
            local_files_only=True,
        )

        self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            SD_MODEL_NAME,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            cache_dir=SD_MODEL_CACHE,
            use_safetensors=True,
        )

        # load LCM LoRA
        self.pipe.load_lora_weights(f"{CHECKPOINTS_CACHE}/pytorch_lora_weights.safetensors")
        self.pipe.fuse_lora()
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        self.pipe.cuda()
        self.pipe.load_ip_adapter_instantid(face_adapter)

    def run_safety_checker(self, image) -> (list, list):
        """Detect nsfw content"""
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to("cuda")
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(
            description="Input prompt",
            default="analog film photo of a man. faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, masterpiece, best quality",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        width: int = Input(
            description="Width of output image",
            default=640,
            ge=512,
            le=2048,
        ),
        height: int = Input(
            description="Height of output image",
            default=640,
            ge=512,
            le=2048,
        ),
        ip_adapter_scale: float = Input(
            description="Scale for IP adapter",
            default=0.8,
            ge=0,
            le=1,
        ),
        controlnet_conditioning_scale: float = Input(
            description="Scale for ControlNet conditioning",
            default=0.8,
            ge=0,
            le=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps. With LCM-LoRA, optimum is 6-8.",
            default=6,
            ge=1,
            le=30,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance. With LCM-LoRA, optimum is 0-5.",
            default=0,
            ge=0,
            le=10,
        ),
        safety_checker: bool = Input(
            description="Safety checker is enabled by default. Un-tick to expose unfiltered results.",
            default=True,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if self.width != width or self.height != height:
            print(f"[!] Resizing output to {width}x{height}")
            self.width = width
            self.height = height
            self.app.prepare(ctx_id=0, det_size=(self.width, self.height))

        face_image = load_image(str(image))
        face_image = resize_img(face_image)

        face_info = self.app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        face_info = sorted(
            face_info,
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
            reverse=True,
        )[0]  # only use the maximum face
        face_emb = face_info["embedding"]
        face_kps = draw_kps(face_image, face_info["kps"])

        self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        output_path = "result.jpg"

        output = [image]
        if safety_checker:
            image_list, has_nsfw_content = self.run_safety_checker(output)
            if has_nsfw_content[0]:
                print("NSFW content detected. Try running it again, rephrase different prompt or add 'nsfw' in the negative prompt.")
                black = Image.fromarray(np.uint8(image_list[0])).convert('RGB')    # black box image
                black.save(output_path)
            else:
                image.save(output_path)
        else:
            image.save(output_path)       

        return Path(output_path)
