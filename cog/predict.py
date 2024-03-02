# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import sys
import json
import subprocess
import time
from cog import BasePredictor, Input, Path

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import random
import diffusers

from PIL import Image
from torchvision.transforms import Compose
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
from insightface.app import FaceAnalysis
from controlnet_aux import OpenposeDetector

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pipeline_stable_diffusion_xl_instantid_full import (
    StableDiffusionXLInstantIDPipeline,
    draw_kps,
)
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# for ip-adapter, ControlNetModel
CHECKPOINTS_CACHE = "./checkpoints"
POSE_CHKPT_CACHE = f"{CHECKPOINTS_CACHE}/pose"
CANNY_CHKPT_CACHE = f"{CHECKPOINTS_CACHE}/canny"
DEPTH_CHKPT_CACHE = f"{CHECKPOINTS_CACHE}/depth"

# safety checker model
SAFETY_MODEL_CACHE = "./safety_cache"
FEATURE_EXTRACT_CACHE = "feature_extractor"

# for SDXL lightning LoRA
LORA_CHECKPOINTS_CACHE = f"{CHECKPOINTS_CACHE}/lora"

# default SDXL model
DEFAULT_SDXL_MODEL = "AlbedoBase XL V2"

# global variable
MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32


def resize_img(
    input_image,
    max_side=1280,
    min_side=1024,
    size=None,
    pad_to_max_side=False,
    mode=Image.BILINEAR,
    base_pixel_number=64,
):
    """Resize input image"""
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

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def list_models(path:str) -> list:
    """Return all model names in the json file given by path
    
    Arguments:
        path:str path to json file
    """
    model_list = []
    with open(path, "r") as f:
        data = json.load(f)
        model_list = [model.get("name") for model in data["model"]]
        return model_list

def download_weights(url, dest, extract=True) -> None:
    """Helper function to download model weights"""
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    if extract:
        subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    else:
        subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

def setup_sdxl_pipeline(model_name: str) -> str:
    """Helper function to download and load weights into SDXL pipeline"""
    cache_dir = ""
    with open("./cog/img_models.json", "r") as f:
        data = json.load(f)
        for model in data["model"]:
            if model["name"] == model_name:
                cache_dir = model["cacheFolder"]
                if not os.path.exists(cache_dir):
                    file_path = os.path.join(cache_dir, model["filename"])
                    subprocess.check_call(["mkdir", "-p", cache_dir], close_fds=False)
                    print(f"Downloading new SDXL weights: {file_path}")
                    download_weights(model["url"], file_path, False)
                    pipe = StableDiffusionXLPipeline.from_single_file(
                        file_path,
                        torch_dtype=torch.float16
                    )
                    # Save to cache folder. Will be created if doesn't exist.
                    pipe.save_pretrained(cache_dir)
                break
    return cache_dir

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load safety checker"""
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_MODEL_CACHE, torch_dtype=dtype
        ).to(device)
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACT_CACHE)

        """Load the model into memory to make running multiple predictions efficient"""
        self.width, self.height = 640, 640
        self.app = FaceAnalysis(
            name="antelopev2",
            root="./",
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(self.width, self.height))

        # Load openpose and depth-anything controlnet pipelines
        self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        self.depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14').to(device).eval()

        self.transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        # Path to InstantID models
        self.face_adapter = f"{CHECKPOINTS_CACHE}/ip-adapter.bin"
        controlnet_path = f"{CHECKPOINTS_CACHE}/ControlNetModel"

        # Load pipeline face ControlNetModel
        self.controlnet_identitynet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=dtype,
            cache_dir=CHECKPOINTS_CACHE,
            use_safetensors=True,
            local_files_only=True,
        )

        # Load controlnet-pose/canny/depth from local cache
        self.controlnet_pose = ControlNetModel.from_pretrained(
            POSE_CHKPT_CACHE,
            torch_dtype=dtype,
            use_safetensors=True,
            cache_dir=POSE_CHKPT_CACHE,
            local_files_only=True,
        ).to(device)
        self.controlnet_canny = ControlNetModel.from_pretrained(
            CANNY_CHKPT_CACHE,
            torch_dtype=dtype,
            use_safetensors=True,
            cache_dir=CANNY_CHKPT_CACHE,
            local_files_only=True,
        ).to(device)
        self.controlnet_depth = ControlNetModel.from_pretrained(
            DEPTH_CHKPT_CACHE,
            torch_dtype=dtype,
            use_safetensors=True,
            cache_dir=DEPTH_CHKPT_CACHE,
            local_files_only=True,    
        ).to(device)

        # setup the InstantID pipeline
        self.model = DEFAULT_SDXL_MODEL
        self.setup_instantID_pipeline(self.model)

    def setup_instantID_pipeline(self, model: str) -> None:
        cache_folder = setup_sdxl_pipeline(model)

        self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            cache_folder,
            controlnet=[self.controlnet_identitynet],
            torch_dtype=dtype,
            cache_dir=cache_folder,
            use_safetensors=True,
            local_files_only=True,
        )
        # pre-load lightning LoRA weights, just in case is selected
        self.lightning_steps = "4step"
        self.pipe.load_lora_weights(
            LORA_CHECKPOINTS_CACHE,
            cache_dir=LORA_CHECKPOINTS_CACHE,
            weight_name=f"sdxl_lightning_{self.lightning_steps}_lora.safetensors",
            local_files_only=True,
        )
        # Ensure sampler uses "trailing" timesteps for lightning LoRA
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config,
            timestep_spacing="trailing"
        )

        self.pipe.disable_lora()
        self.pipe.cuda()
        self.pipe.load_ip_adapter_instantid(self.face_adapter)
        self.pipe.image_proj_model.to("cuda")
        self.pipe.unet.to("cuda")

    def load_lightning_lora(self, steps: str):
        self.pipe.unload_lora_weights()
        self.pipe.load_lora_weights(
            LORA_CHECKPOINTS_CACHE,
            cache_dir=LORA_CHECKPOINTS_CACHE,
            weight_name=f"sdxl_lightning_{steps}_lora.safetensors",
            local_files_only=True,
        )
        self.pipe.enable_lora()

    def get_depth_map(self, image):
        """Get the depth map from input image"""
        image = np.array(image) / 255.0
        h, w = image.shape[:2]

        image = self.transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to("cuda")

        with torch.no_grad():
            depth = self.depth_anything(image)

        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)
        depth_image = Image.fromarray(depth)
        return depth_image

    def get_canny_image(self, image, t1=100, t2=200):
        """Get the canny edges from input image"""
        image = convert_from_image_to_cv2(image)
        edges = cv2.Canny(image, t1, t2)
        return Image.fromarray(edges, "L")

    def run_safety_checker(self, image) -> (list, list):
        """Detect nsfw content"""
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(device)
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(dtype),
        )
        return image, has_nsfw_concept

    @torch.inference_mode()
    def predict(
        self,
        face_image_path: Path = Input(description="Image of your face"),
        pose_image_path: Path = Input(
            description="Reference pose image",
            default=None,
        ),
        prompt: str = Input(
            description="Input prompt",
            default="a person",
        ),
        negative_prompt: str = Input(
            description="Input negative prompt",
            default="(lowres, low quality, worst quality:1.2), (text:1.2), watermark, glitch, deformed, mutated, cross-eyed, ugly, disfigured, blurry, grainy",
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
        model: str = Input(
            description="Select SDXL model",
            default="AlbedoBase XL V2",
            choices=list_models("./cog/img_models.json"),
        ),
        enable_fast_mode: bool = Input(
            description="Enable SDXL-lightning fast inference. If pose, canny or depth map is used, disable it for better quality images.",
            default=True,
        ),
        lightning_steps: str = Input(
            description="if enable fast mode, choose number of denoising steps",
            choices=[
                "2step",
                "4step",
                "8step",
            ],
            default="4step",
        ),
        scheduler: str = Input(
            description="Scheduler options. If enable fast mode, this is not used.",
            choices=[
                "DEISMultistepScheduler",
                "HeunDiscreteScheduler",
                "EulerDiscreteScheduler",
                "DPMSolverMultistepScheduler",
                "DPMSolverMultistepScheduler-Karras",
                "DPMSolverMultistepScheduler-Karras-SDE",
            ],
            default="DPMSolverMultistepScheduler",
        ),
        adapter_strength_ratio: float = Input(
            description="Image adapter strength (for detail)",
            default=0.8,
            ge=0,
            le=1,
        ),
        identitynet_strength_ratio: float = Input(
            description="IdentityNet strength (for fidelity)",
            default=0.8,
            ge=0,
            le=1,
        ),
        pose: bool = Input(
            description="Use pose for skeleton inference",
            default=False,
        ),
        pose_strength: float = Input(
            default=1.0,
            ge=0,
            le=1.5,
        ),
        canny: bool = Input(
            description="Use canny for edge detection",
            default=False,
        ),
        canny_strength: float = Input(
            default=0.5,
            ge=0,
            le=1.5,
        ),
        depth_map: bool = Input(
            description="Use depth for depth map estimation",
            default=False,
        ),
        depth_strength: float = Input(
            default=0.5,
            ge=0,
            le=1.5,
        ),
        num_steps: int = Input(
            description="Number of denoising steps. If enable fast mode, this is not used.",
            default=25,
            ge=1,
            le=50,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance. Optimum is 4-8. If enable fast mode, this is not used.",
            default=7,
            ge=0,
            le=10,
        ),
        seed: int = Input(
            description="Seed number. Set to non-zero to make the image reproducible.",
            default=0,
            ge=0,
            le=MAX_SEED,
        ),
        enhance_non_face_region: bool = Input(
            description="Enhance non-face region",
            default=True
        ),
        safety_checker: bool = Input(
            description="Safety checker is enabled by default. Un-tick to expose unfiltered results.",
            default=True,
        ),
    ) -> Path:
        """Run a single prediction on the model"""    
        # Load the weights if they are different from the base weights
        if model != self.model:
            setup_sdxl_pipeline(model)
            self.model = model

        # Resize the output if the provided dimensions are different from the current ones
        if self.width != width or self.height != height:
            print(f"[!] Resizing output to {width}x{height}")
            self.width = width
            self.height = height
            self.app.prepare(ctx_id=0, det_size=(self.width, self.height))

        # Load and resize the face image
        face_image = load_image(str(face_image_path))
        face_image = resize_img(face_image, max_side=1024)
        face_image_cv2 = convert_from_image_to_cv2(face_image)
        height, width, _ = face_image_cv2.shape

        # Extract face features
        face_info = self.app.get(face_image_cv2)
        if len(face_info) == 0:
            raise ValueError(
                "Unable to detect your face in the photo. Please upload a different photo with a clear face."
            )
        # only use the maximum face
        face_info = sorted(
            face_info,
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
        )[-1]
        face_emb = face_info["embedding"]
        face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info["kps"])
        img_controlnet = face_image

        # If pose image is provided, use it to extra the pose
        if pose_image_path is not None:
            pose_image = load_image(str(pose_image_path))
            pose_image = resize_img(pose_image, max_side=1024)
            img_controlnet = pose_image
            pose_image_cv2 = convert_from_image_to_cv2(pose_image)

            # Extract face features from the reference pose image
            face_info = self.app.get(pose_image_cv2)
            if len(face_info) == 0:
                raise ValueError(
                    "Unable to detect a face in the reference image. Please upload another person's image."
                )
            # only use the maximum face
            face_info = sorted(
                face_info,
                key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
            )[-1]
            face_kps = draw_kps(pose_image, face_info["kps"])

        if enhance_non_face_region:
            control_mask = np.zeros([height, width, 3])
            x1, y1, x2, y2 = face_info["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            control_mask[y1:y2, x1:x2] = 255
            control_mask = Image.fromarray(control_mask.astype(np.uint8))
        else:
            control_mask = None

        controlnet_map = {
            "pose": self.controlnet_pose,
            "canny": self.controlnet_canny,
            "depth": self.controlnet_depth,
        }
        controlnet_map_fn = {
            "pose": self.openpose,
            "canny": self.get_canny_image,
            "depth": self.get_depth_map,
        }

        controlnet_selection = []
        if pose:
            controlnet_selection.append("pose")
        if canny:
            controlnet_selection.append("canny")
        if depth_map:
            controlnet_selection.append("depth")

        if len(controlnet_selection) > 0:
            controlnet_scales = {
                "pose": pose_strength,
                "canny": canny_strength,
                "depth": depth_strength,
            }
            self.pipe.controlnet = MultiControlNetModel([self.controlnet_identitynet] + [controlnet_map[s] for s in controlnet_selection])
            control_scales = [float(identitynet_strength_ratio)] + [controlnet_scales[s] for s in controlnet_selection]
            control_images = [face_kps] + [
                controlnet_map_fn[s](img_controlnet).resize((width, height))
                for s in controlnet_selection
            ]
        else:
            self.pipe.controlnet = self.controlnet_identitynet
            control_scales = float(identitynet_strength_ratio)
            control_images = face_kps

        # load LCM LoRA if enabled, else use other schedulers
        if enable_fast_mode:
            if self.lightning_steps != lightning_steps:
                self.load_lightning_lora(lightning_steps)
                self.lightning_steps = lightning_steps
            else:
                self.pipe.enable_lora()
            guidance_scale = 0
            num_steps = int(self.lightning_steps.strip("step"))
        else:
            self.pipe.disable_lora()
            scheduler_class_name = scheduler.split("-")[0]

            add_kwargs = {}
            if len(scheduler.split("-")) > 1:
                add_kwargs["use_karras_sigmas"] = True
            if len(scheduler.split("-")) > 2:
                add_kwargs["algorithm_type"] = "sde-dpmsolver++"
            scheduler = getattr(diffusers, scheduler_class_name)
            self.pipe.scheduler = scheduler.from_config(
                self.pipe.scheduler.config,
                **add_kwargs,
            )

        if seed == 0:
            seed = random.randint(1, MAX_SEED)
        generator = torch.Generator(device=device).manual_seed(seed)

        self.pipe.set_ip_adapter_scale(adapter_strength_ratio)
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=control_images,
            control_mask=control_mask,
            controlnet_conditioning_scale=control_scales,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=height,
            width=width,
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
