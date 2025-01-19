# Image Generation Resources

## Image Generation Models (Open Source vs Closed Source)

| **Model Name**           | **Year Released** | **Open Source/Commercial**   | **Popularity**   | **Usable in ComfyUI** | **Outputs Usable Commercially**  | **Model Size (Parameters)** | **Link**                                                                 |
|--------------------------|-------------------|------------------------------|------------------|-----------------------|----------------------------------|------------------------------------|--------------------------------------------------------------------------|
| **HunYuan DiT**          | 2024             | Open Source                  | Growing          | Yes                   | Yes, with restrictions           | 1.5 Billion                     | [HunYuan DiT v1.1](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1) |
| **PixArt-α**             | 2023             | Open Source                  | High             | Yes                   | Yes                              | 0.6 Billion                     | [PixArt-XL-2-1024-MS](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS) |
| **Playground 2.5**       | 2024             | Open Source                  | High             | Yes                   | Yes, with restrictions           | -                                | [Playground 2.5](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic) |
| **Stable Diffusion 1.5** | 2022             | Open Source                  | Very High        | Yes                   | Yes                              | 860 Million                     | [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| **Stable Diffusion 2.1** | 2022             | Open Source                  | Very High        | Yes                   | Yes                              | -                                | [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) |
| **Stable Diffusion XL**  | 2023             | Open Source                  | Very High        | Yes                   | Yes                              | -                                | [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| **Stable Diffusion XL Turbo** | 2024       | Commercial                   | High             | Yes                   | Yes, with paid license           | -                                | [SDXL Turbo](https://huggingface.co/stabilityai/sdxl-turbo) |
| **Stable Cascade**       | 2024             | Commercial                   | Growing          | Yes                   | Yes, with paid license           | -                                | -                                                                        |
| **Flux Schnell**         | 2024             | Open Source                  | Growing          | Yes                   | Yes                              | -                                | [Flux Schnell GitHub](https://github.com/black-forest-labs/flux)          |
| **Flux Dev**             | 2024             | Partially Open Source (Restricted Use) | Growing          | Yes                   | Yes, with restrictions           | -                                | [FLUX.1 Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)         |
| **Flux Pro**             | 2024             | Commercial                   | Growing          | Yes                   | Yes, with paid license           | -                                | -                                                                        |
| **Midjourney v1**        | 2022             | Commercial                   | Very High        | With custom nodes     | Yes, with paid license           | -                                | -                                                                        |
| **Midjourney v2**        | 2022             | Commercial                   | Very High        | With custom nodes     | Yes, with paid license           | -                                | -                                                                        |
| **Midjourney v3**        | 2022             | Commercial                   | Very High        | With custom nodes     | Yes, with paid license           | -                                | -                                                                        |
| **Midjourney v4**        | 2022             | Commercial                   | Very High        | With custom nodes     | Yes, with paid license           | -                                | -                                                                        |
| **Midjourney v5**        | 2023             | Commercial                   | Very High        | With custom nodes     | Yes, with paid license           | -                                | -                                                                        |
| **Midjourney v6**        | 2024             | Commercial                   | Very High        | With custom nodes     | Yes, with paid license           | -                                | -                                                                        |
| **DALL-E 1**             | 2021             | Commercial                   | Moderate         | No                    | Yes, with paid credits           | -                                | -                                                                        |
| **DALL-E 2**             | 2022             | Commercial                   | High             | With API calls        | Yes, with paid credits           | -                                | -                                                                        |
| **DALL-E 3**             | 2023             | Commercial                   | High             | With API calls        | Yes, with paid credits           | -                                | -                                                                        |

**Model evaluation:** https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0#evaluation

| 🌟 **Checkpoint**                  | 🖼️ **Purpose/Notes**                                | 🔗 **Link**                                                                                     |
|------------------------------------|-----------------------------------------------------|-----------------------------------------------------------------------------------------------|
| **FLUX.1-dev**                     | Experimental, versatile model for creative tasks   | [FLUX.1-dev on Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-dev)             |
| **Stable Diffusion v1.5**          | Classic, widely-used model for general use         | [Stable Diffusion v1.5 on Hugging Face](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| **Juggernaut-XL-v9**               | Advanced, high-quality XL model                   | [Juggernaut-XL-v9 on Hugging Face](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9)      |
| **Animagine XL 3.1**               | Focused on animations and creative illustrations  | [Animagine XL 3.1 on Hugging Face](https://huggingface.co/cagliostrolab/animagine-xl-3.1)     |
| **COSXL (Change Image Style)**     | Flexible style adjustments, workflows available    | [COSXL on Civiti](https://civitai.com/models/397741?modelVersionId=443550) <br> [Workflows on Reddit](https://www.reddit.com/r/StableDiffusion/comments/1e2l6px/quickly_and_easily_change_image_style_with_cosxl/) |


## Generative Image Tools

| Feature             | ComfyUI                     | Automatic1111            | Fooocus/RuinedFooocus     | SD Next                   | SwarmUI                     |
|---------------------|-----------------------------|--------------------------|---------------------------|---------------------------|-----------------------------|
| Interface           | 🟢 Advanced                | 🟠 Friendly              | 🟠 Simplified             | 🟠 Simplified            | 🟠 Friendly                   |
| Models              | 🌟 SD + Flux     | 🎯 Stable Diffusion only | 🎯 SD XL only             | 🌀 Various txt2img/img2img| 🌀 SD + Flux               |
| Performance         | 🟢 Efficient               | 🟠 Optimizable           | 🟠 Solid                  | 🟢 Optimized              | 🟢 High                    |
| Features            | 🧬 Graph workflows, memory | 🧩 Plugins, upscaling    | 🎨 Styles, Img2Img        | 🌐 Multi-platform, models | 🎥 Modular, AI video       |
| License             | 🟢 Open-source (GPL)       | 🟢 Open-source (Free)    | 🟢 Open-source (Free)     | 🟢 Open-source (Free)     | 🟢 Open-source (Free)       |

## Running ComfyUI

| 🛠️ **Method**            | 🌐 **Ease of Use**             | 💸 **Cost**                | 🕒 **Setup Time**        | 🔗 **Details/Tutorial**                                                                                         |
|--------------------------|--------------------------------|---------------------------|--------------------------|---------------------------------------------------------------------------------------------------------------|
| **RunComfy**            | 🟢 One-click deploy           | 💰 More expensive          | 🟢 Quick                 | [RunComfy](https://www.runcomfy.com/)                                                                         |
| **Runpod**              | 🟠 Requires setup (tutorial)  | 💰 Expensive               | 🟠 Moderate               | [Runpod Tutorial](https://blog.runpod.io/how-to-run-flux-image-generator-with-comfyui-2/)                      |
| **ComfyAI.run**         | 🟢 Very easy  (restrictive)               | 💰 Most expensive          | 🟢 Quick | [Visit ComfyAI.run](https://comfyai.run)                                 |
| **GCP (VM)**            | 🟠 Manual (needs tutorial)    | 💵 ~$25-30/month           | 🟠 Moderate               | [GCP Tutorial](https://www.youtube.com/watch?v=PZwnbBaJH3I)                                                   |
| **Vast.ai**             | 🟠 Needs setup (tutorial)     | 💵 Affordable              | 🟠 Moderate               | [Vast.ai Tutorial](https://www.youtube.com/watch?v=PvW0VbBK7CY)                                               |


## Popular Technologies

| **Technology**          | **What It Does**                       | **How It Works**                                              | **Example Use Case**                                                 |
|--------------------------|----------------------------------------|--------------------------------------------------------------|-----------------------------------------------------------------------|
| **ControlNets**           | Guides generation with edges, depth, etc. | Structured input + compatible model = precise alignment      | Design a room using a depth map or generate characters using pose maps. |
| **iAdapters**            | Adds fine-tuned behavior or style      | Lightweight adaptation of base models                        | Apply a custom art style or adapt for branding.                       |
| **Inpainting**           | Edits specific parts of an image       | Uses masks to target areas                                   | Replace an object in a photo or fix damaged parts.                    |
| **Outpainting**          | Expands image boundaries               | Fills in new areas beyond the original frame                 | Extend a portrait into a full-body scene.                             |
| **Upscaling**            | Improves resolution and sharpness      | Enhances image quality with upscaling models                | Create high-res assets for print or displays.                         |
| **Masking/Segmentation** | Isolates or labels image regions       | Masks or segmentation maps define areas to edit             | Change colors or replace objects without affecting the rest of the image. |
| **LoRAs**                | Adds specific styles or fine-tuned capabilities | Fine-tunes parts of a base model to add features or aesthetics | Add a custom anime style, enhance realism, or apply niche aesthetics. |

### Controlnet Models

| **ControlNets**          | **Links**                                                                                                           |
|--------------------------|---------------------------------------------------------------------------------------------------------------------|
| Flux Dev (Unofficial)    | [Depth](https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Depth) • [Surface Normals](https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Surface-Normals) • [Upscaler](https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler) • [Canny](https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny) • [Union](https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union) |
| Flux Dev (Official)      | [Depth](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev-lora) • [Canny](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev) |
| SDXL                     | [Union SDXL 1.0](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0)                                           |
| SD 1.5                   | [T2I Adapters](https://civitai.com/models/17220/controlnet-t2i-adapter-models) • [ControlNetXL](https://civitai.com/models/136070/controlnetxl-cnxl) |
| Stable Diffusion         | [Control Collection (Hugging Face)](https://huggingface.co/lllyasviel/sd_control_collection/tree/main) • [ControlNet Models (Civitai)](https://civitai.com/models/38784/controlnet-11-models) |
| HunYan                   | [HYDiT ControlNet](https://huggingface.co/Tencent-Hunyuan/HYDiT-ControlNet-v1.2)                                    |
| PixArt                   | [PixArt ControlNet](https://huggingface.co/PixArt-alpha/PixArt-ControlNet/tree/main)                                |

### IP Adapters, Upscalers & Inpainting Models

| **iAdapters**            | **Links**                                                                                                           |
|--------------------------|---------------------------------------------------------------------------------------------------------------------|
| Flux Dev (Unofficial)    | [SD3.5 Large IP Adapter](https://huggingface.co/InstantX/SD3.5-Large-IP-Adapter)                                    |
| Flux Dev (Official)      | [Redux Dev](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev)                                              |
| SD 1.5                   | [IP Adapter](https://huggingface.co/h94/IP-Adapter) • [ComfyUI IP Adapter Plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) • [Face Adapter](https://huggingface.co/h94/IP-Adapter/blob/main/models/ip-adapter-plus-face_sd15.safetensors) • [Face ID](https://huggingface.co/h94/IP-Adapter-FaceID/blob/main/ip-adapter-faceid_sd15.bin) |
| SDXL                     | [IP Adapter SDXL](https://huggingface.co/h94/IP-Adapter/blob/main/sdxl_models/ip-adapter_sdxl.safetensors) • [ViT-H](https://huggingface.co/h94/IP-Adapter/blob/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors) • [InstantID](https://huggingface.co/InstantX/InstantID) |

| **Upscalers**            | **Links**                                                                                                           |
|--------------------------|---------------------------------------------------------------------------------------------------------------------|
| General Upscalers        | [4x Nomos8kDAT](https://openmodeldb.info/models/4x-Nomos8kDAT) • [StableSR GitHub](https://github.com/IceClear/StableSR?tab=readme-ov-file) • [UltraSharp](https://civitai.com/models/116225/4x-ultrasharp) |

| **Inpainting**           | **Links**                                                                                                           |
|--------------------------|---------------------------------------------------------------------------------------------------------------------|
| SD Models                | [Stable Diffusion 2 Inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)                   |
| Flux Dev (Unofficial)    | [Inpainting Beta](https://huggingface.co/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta)                    |
| Flux Dev (Official)      | [Fill Dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)                                                |

### Example Use Case (w/ Said Tech)

| **Example Use Case**                                      | **Description**                                                                                                           | **Technologies**                                                                                  |
|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| Turn faces into cartoons                                 | Transform realistic faces into cartoon-style images.                                                                    | LoRAs (cartoon styles), Face ID IP-Adapter (for facial feature transfer).                        |
| Create realistic images from 3D model sketches           | Take a sketch of a 3D model and generate a realistic image in a specific style (e.g., Japanese).                        | ControlNet (Depth), LoRAs (Japanese art style), or IP-Adapter with a reference image.            |
| Generate variations of an existing image                | Create multiple versions of an image with slight differences.                                                           | IP-Adapter models, LoRAs.                                                                        |
| Expand an image beyond its original borders             | Add new content to extend the scene while maintaining consistency with the original.                                    | Outpainting (using inpainting models and/or ControlNet's inpainting module).                     |
| Scale up the quality of an image                        | Increase the resolution and sharpness of an image.                                                                      | Upscaling models (e.g., ESRGAN, SwinIR).                                                         |
| Create different styles of a bedroom image              | Apply different styles to a bedroom while preserving its structure.                                                     | ControlNet (Canny Edge for structure or Depth), LoRAs (specific styles), or IP-Adapter with a reference image. |
| Remove or add a feature in an existing image            | Edit specific parts of an image, like removing objects or adding new ones.                                              | Inpainting.                                                                                      |
| Product placements in AI-generated images              | Add branded products into AI images by isolating areas and replacing them.                                              | Segment Anything (to create a mask), Inpainting.                                                |
| Change texture or color on walls                       | Identify the background and modify its color or texture.                                                                | Segmentation (to isolate walls), Inpainting.                                                    |
| Transfer logo                                          | Transfer Logo to different images, play around with the borders                                                          | Depth or canny (outline) OR Redux/IP Adapter for style transfer                                 |



## Interesting Repositories

| **Repository**                                                                                                  | **Description**                                      | **Stars**                                                                                                                              |
|-----------------------------------------------------------------------------------------------------------------|------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| [style_aligned_comfy](https://github.com/brianfitzgerald/style_aligned_comfy?tab=readme-ov-file)                 | Transfer styles between images                      | [![GitHub Repo stars](https://img.shields.io/github/stars/brianfitzgerald/style_aligned_comfy?style=flat-square&color=purple)](https://github.com/brianfitzgerald/style_aligned_comfy)                 |
| [Auto-Photoshop-StableDiffusion-Plugin](https://github.com/AbdullahAlfaraj/Auto-Photoshop-StableDiffusion-Plugin?tab=readme-ov-file) | Use ComfyUI in Photoshop for designers             | [![GitHub Repo stars](https://img.shields.io/github/stars/AbdullahAlfaraj/Auto-Photoshop-StableDiffusion-Plugin?style=flat-square&color=purple)](https://github.com/AbdullahAlfaraj/Auto-Photoshop-StableDiffusion-Plugin) |
| [ComfyUI-Workflows-ZHO](https://github.com/ZHO-ZHO-ZHO/ComfyUI-Workflows-ZHO?tab=readme-ov-file)                 | Workflow collection                                 | [![GitHub Repo stars](https://img.shields.io/github/stars/ZHO-ZHO-ZHO/ComfyUI-Workflows-ZHO?style=flat-square&color=purple)](https://github.com/ZHO-ZHO-ZHO/ComfyUI-Workflows-ZHO)                 |
| [ComfyUI-3D-Pack](https://github.com/MrForExample/ComfyUI-3D-Pack)                                              | Create 3D images                                    | [![GitHub Repo stars](https://img.shields.io/github/stars/MrForExample/ComfyUI-3D-Pack?style=flat-square&color=purple)](https://github.com/MrForExample/ComfyUI-3D-Pack)                              |
| [ComfyUI-AdvancedLivePortrait](https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait)                    | Create live portraits                               | [![GitHub Repo stars](https://img.shields.io/github/stars/PowerHouseMan/ComfyUI-AdvancedLivePortrait?style=flat-square&color=purple)](https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait)  |
| [comfyui_LLM_party](https://github.com/heshengtao/comfyui_LLM_party)                                            | Create agents with ComfyUI and LLMs                | [![GitHub Repo stars](https://img.shields.io/github/stars/heshengtao/comfyui_LLM_party?style=flat-square&color=purple)](https://github.com/heshengtao/comfyui_LLM_party)                              |
| [ComfyUI-Griptape](https://github.com/griptape-ai/ComfyUI-Griptape)                                             | Use ComfyUI for creating LLM agents                | [![GitHub Repo stars](https://img.shields.io/github/stars/griptape-ai/ComfyUI-Griptape?style=flat-square&color=purple)](https://github.com/griptape-ai/ComfyUI-Griptape)                             |
| [ComfyUI_Custom_Nodes_AlekPet](https://github.com/AlekPet/ComfyUI_Custom_Nodes_AlekPet)                          | Custom nodes for ComfyUI                           | [![GitHub Repo stars](https://img.shields.io/github/stars/AlekPet/ComfyUI_Custom_Nodes_AlekPet?style=flat-square&color=purple)](https://github.com/AlekPet/ComfyUI_Custom_Nodes_AlekPet)               |
| [comfyui-portrait-master](https://github.com/florestefano1975/comfyui-portrait-master)                          | Create human-like portraits                        | [![GitHub Repo stars](https://img.shields.io/github/stars/florestefano1975/comfyui-portrait-master?style=flat-square&color=purple)](https://github.com/florestefano1975/comfyui-portrait-master)       |
| [comfyui_LLM_party](https://github.com/heshengtao/comfyui_LLM_party)                                            | Auto-prompting with LLMs in ComfyUI                | [![GitHub Repo stars](https://img.shields.io/github/stars/heshengtao/comfyui_LLM_party?style=flat-square&color=purple)](https://github.com/heshengtao/comfyui_LLM_party)                              |

## Use Cases (Tutorials)

| **Example Use Case**                                      | **Description**                                                                                   | **Link**                                                                                   |
|----------------------------------------------------------|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| Animate videos                                           | Create animations from static images                                                             | [Watch Video](https://www.youtube.com/watch?v=AugFKDGyVuw)                                |
| Consistent characters                                    | Maintain consistent character design across images                                               | [Watch Video](https://www.youtube.com/watch?v=MbQv8zoNEfY)                                |
| Video workflow                                           | Generate videos using ComfyUI                                                                   | [Watch Video](https://www.youtube.com/watch?v=gHI6PjTkBF4)                                |
| Inpainting                                               | Nodes for inpainting tasks                                                                       | [Inpainting Nodes](https://github.com/Acly/comfyui-inpaint-nodes)                         |
| Upscale                                                 | Enhance image resolution                                                                         | [GitHub](https://github.com/ssitu/ComfyUI_UltimateSDUpscale?tab=readme-ov-file) • [Reddit](https://www.reddit.com/r/comfyui/comments/1e9d0ex/comfyui_image_upscale_simple_guide_and_workflow/) |
| Face swap                                               | Swap faces in images                                                                            | [Workflow](https://comfyworkflows.com/workflows/cd26cfd4-abf6-449c-b058-c26c44b12c93)     |
| Face to cartoon                                         | Turn realistic faces into cartoon-style images                                                  | [Watch Video](https://www.youtube.com/watch?v=XIRIwGOd1H8)                                |
| Train a LoRA directly in ComfyUI                        | Use ComfyUI to fine-tune a LoRA                                                                 | [GitHub](https://github.com/LarryJane491/Lora-Training-in-Comfy)                          |

## Learning Resources

| **Topic**           | **Description**                              | **Link**                                                                                      |
|----------------------|----------------------------------------------|----------------------------------------------------------------------------------------------|
| **ControlNets**      | Comprehensive guide to using ControlNets.    | [Civitai Guide to ControlNet](https://education.civitai.com/civitai-guide-to-controlnet)     |
| **ComfyUI Basics**   | Introduction to using ComfyUI for beginners. | [YouTube: ComfyUI Basics](https://www.youtube.com/)                                          |
| **Style Prompts**    | Which styles suit you and then use them in your promps | [SDXL Prompt Artist Study](https://rikkar69.github.io/SDXL-artist-study/)         | 

