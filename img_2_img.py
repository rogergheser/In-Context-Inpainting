# Description: Sample code to generate images from text using the Stable Diffusion default pipeline

from diffusers.pipelines import StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image, make_image_grid
from Image2TextEmbedder import Image2TextEmbedder
from PIL import Image

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
# use from_pipe to avoid consuming additional memory when loading a checkpoint
pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4"
                                                          ).to(device)
tokenizer = pipeline.tokenizer
text_encoder = pipeline.text_encoder

init_image_path = "blended-latent-diffusion/inputs/img.png"
init_image = Image.open(init_image_path)
mask_image = Image.open("blended-latent-diffusion/inputs/mask.png")
img2text = Image2TextEmbedder(
    clip_path="openai/clip-vit-large-patch14",
    device=device,
    alpha=0.6,
    onlyprompt=False,
    edit=True
)
prompt = "A dog in a grass field"

inputs = tokenizer(prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
text_input = inputs.input_ids.to(device)
text_masks = inputs.attention_mask.to(device)
text_embeddings = text_encoder(text_input)[0]
text_embeddings = img2text(init_image_path, text_embeddings, text_masks)

images = pipeline(prompt=None, image=init_image, strength=0.85,
                   guidance_scale=12.5, num_images_per_prompt=3,
                   prompt_embeds=text_embeddings).images
make_image_grid(images, rows=1, cols=3).save("outputs/img.png")