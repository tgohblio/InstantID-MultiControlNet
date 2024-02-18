# InstantID Cog Model

## Overview
This repository contains the implementation of [InstantID](https://github.com/InstantID/InstantID) as a [Cog](https://github.com/replicate/cog) model. 

Using [Cog](https://github.com/replicate/cog) allows any users with a GPU to run the model locally easily, without the hassle of downloading weights, installing libraries, or managing CUDA versions. Everything just works.

## Development
To push your own fork of InstantID to [Replicate](https://replicate.com), follow the [Model Pushing Guide](https://replicate.com/docs/guides/push-a-model).

## Basic Usage
To make predictions using the model, execute the following command from the root of this project:

```bash
cog predict \
-i face_image_path=@examples/halle-berry.jpeg \
-i prompt="woman as elven princess, with blue sheen dress" \
-i negative_prompt="nsfw" \
-i width=1024 \
-i height=1024 \
-i adapter_strength_ratio=0.8 \
-i identitynet_strength_ratio=0.8 \
-i num_steps=6 \
-i guidance_scale=0 \
-i safety_checker=True
```

<table>
  <tr>
    <td>
      <p align="center">Input</p>
      <img src="examples/halle-berry.jpeg" alt="Sample Input Image" width="100%"/>
    </td>
    <td>
      <p align="center">Output</p>
      <img src="examples/result.jpg" alt="Sample Output Image" width="100%"/>
    </td>
  </tr>
</table>

```bash
cog predict \
-i face_image_path=@examples/halle-berry.jpeg \
-i pose_image_path=@examples/poses/ballet-pose.jpg \
-i prompt="photo of a ballerina on stage" \
-i model="Juggernaut XL V8" \
-i width=1024 \
-i height=1024 \
-i adapter_strength_ratio=0.8 \
-i identitynet_strength_ratio=0.8 \
-i scheduler="DPMSolverMultistepScheduler-Karras" \
-i pose=True \
-i pose_strength=0.4 \
-i enable_LCM=False \
-i num_steps=30 \
-i guidance_scale=4 \
-i safety_checker=True
```

## Input Parameters

The following table provides details about each input parameter for the `predict` function:

| Parameter                       | Description                        | Default Value                                     | Range       |
| ------------------------------- | ---------------------------------- | --------------------------------------------------| ----------- |
| `face_image_path`               | Input image                        | A path to the input image file                    | Path string |
| `pose_image_path`               | Input image                        | A path to the reference pose image file           | Path string |
| `prompt`                        | Input prompt                       | "a person"                                        | String      |
| `negative_prompt`               | Input Negative Prompt              | "ugly, low quality, deformed face"                | String      |
| `model`                         | SDXL image model choices           | "AlbedoBase XL V2"                                | String      |
| `enable_LCM`                    | enable LCM LoRA                    | False                                             | Boolean     |
| `scheduler`                     | scheduler algorithm choices        | "DPMSolverMultistepScheduler"                     | String      |
| `width`                         | Width of output image              | 640                                               | 512 - 2048  |
| `height`                        | Height of output image             | 640                                               | 512 - 2048  |
| `adapter_strength_ratio`        | Scale for IP adapter               | 0.8                                               | 0.0 - 1.0   |
| `identitynet_strength_ratio`    | Scale for ControlNet conditioning  | 0.8                                               | 0.0 - 1.0   |
| `pose`                          | select ControlNet pose model       | False                                             | Boolean     |
| `canny`                         | select ControlNet canny edge model | False                                             | Boolean     |
| `depth_map`                     | select ControlNet depth model      | False                                             | Boolean     |
| `pose_strength`                 | Scale for pose conditioning        | 0.5                                               | 0.0 - 1.5   |
| `canny_strength`                | Scale for canny edge conditioning  | 0.5                                               | 0.0 - 1.5   |
| `depth_strength`                | Scale for depth map conditioning   | 0.5                                               | 0.0 - 1.5   |
| `num_steps`                     | Number of denoising steps          | 25                                                | 1 - 50      |
| `guidance_scale`                | Scale for classifier-free guidance | 7                                                 | 1 - 10      |
| `seed`                          | RNG seed number                    | 0 (= random seed)                                 | 0 - int MAX |
| `safety_checker`                | Enable or disable NSFW filter      | True                                              | Boolean     |

This table provides a quick reference to understand and modify the inputs for generating predictions using the model.
