from diffusers.pipelines import StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image, make_image_grid
from Image2TextEmbedder import Image2TextEmbedder
from PIL import Image

import torch

device = 'mps'
# use from_pipe to avoid consuming additional memory when loading a checkpoint
pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4"
                                                          ).to(device)
tokenizer = pipeline.tokenizer
text_encoder = pipeline.text_encoder
# img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
# mask_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-inpaint-mask.png"
# init_image = load_image(img_url)
# mask_image = load_image(mask_url)

init_image_path = "blended-latent-diffusion/inputs/img.png"
init_image = Image.open("blended-latent-diffusion/inputs/img.png")
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