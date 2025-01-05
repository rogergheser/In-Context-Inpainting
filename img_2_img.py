from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from diffusers.utils import load_image, make_image_grid

import torch
from PIL import Image

# use from_pipe to avoid consuming additional memory when loading a checkpoint
pipeline = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base",
                                                torch_dtype = torch.float16,
                                                variant = "fp16",
                                                use_safetensor=True).to("mps")

img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
mask_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-inpaint-mask.png"

# init_image = load_image(img_url)
# mask_image = load_image(mask_url)

init_image = Image.open("inputs/img.png")
mask_image = Image.open("inputs/mask.png")

prompt = "A dog"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, strength=0.85, guidance_scale=12.5).images[0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3).save("outputs/img.png")