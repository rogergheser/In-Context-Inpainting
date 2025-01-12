from diffusers.pipelines import StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image, make_image_grid

import torch
from PIL import Image

# use from_pipe to avoid consuming additional memory when loading a checkpoint
pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base",
                                                use_safetensor=True).to("cuda")

img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
mask_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-inpaint-mask.png"

# init_image = load_image(img_url)
# mask_image = load_image(mask_url)

init_image = Image.open("blended-latent-diffusion/inputs/img.png")
mask_image = Image.open("blended-latent-diffusion/inputs/mask.png")

prompt = "In a world where the sky is green and the grass is blue."
images = pipeline(prompt=prompt, image=init_image, strength=0.85, guidance_scale=12.5, num_images_per_prompt=3).images
make_image_grid(images, rows=1, cols=3).save("outputs/img.png")