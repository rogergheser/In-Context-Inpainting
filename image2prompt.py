import torch
import os
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from diffusers import DDIMScheduler, StableDiffusionPipeline

class Image2TextEmbedder():
    def __init__(self,
                 clip_path: str = "openai/clip-vit-large-patch14",
                 device: str = "cuda",
                 alpha: float = 0.6,
                 onlyprompt: bool = False,
                 edit: bool = False):
        self.clip_model = CLIPModel.from_pretrained(clip_path).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_path)
        self.inv_text = torch.linalg.pinv(self.clip_model.text_projection.weight, atol=0.3)
        self.visual_projection = self.clip_model.visual_projection.weight
        self.alpha = alpha
        self.onlyprompt = onlyprompt
        self.edit = edit

    def __call__(self, image, text_embeddings, text_masks):
        image = Image.open(image).convert("RGB")
        clip_image = self.processor(None, image, return_tensors='pt').pixel_values.to(device)
        image_emb = self.clip_model.vision_model(pixel_values=clip_image)
        image_emb = image_emb.pooler_output

        image_emb_proj = image_emb @ self.visual_projection.T
        image_emb_proj = image_emb_proj @ self.inv_text.T
        image_emb_proj = image_emb_proj / image_emb_proj.norm(dim=1, keepdim=True)
        image_emb_proj = 27.5 * image_emb_proj # Empirically determined text embedding norm

        convert_text_embeddings = torch.zeros_like(text_embeddings)
        convert_text_embeddings[:, 0] = text_embeddings[:, 0]
        convert_text_embeddings[:, 1:] = image_emb_proj.unsqueeze(1)

        convert_edit_embeddings  = text_embeddings.clone()
        convert_edit_embeddings[:, text_masks.sum(1)[0]-1:] = image_emb_proj.unsqueeze(1) + self.alpha * text_embeddings[:, text_masks.sum(1)[0]-1:]

        if self.onlyprompt:
            prompt_embeds = text_embeddings
        elif self.edit:
            prompt_embeds = convert_edit_embeddings
        else:
            prompt_embeds = convert_text_embeddings

        return prompt_embeds.squeeze(0)

class BlendedLatentDiffusion():
    def __init__(self, device):
        self.init_args(device=device)
        self.load_models()

    def __call__(self, x):
        image_path, mask_path, guidance_image, prompt, blending_percentage = x['image_path'], x['mask_path'], x['guidance_image'], x['prompt'], x['blending_percentage']


        # inputs = tokenizer(
        # args.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        # )
        # text_input = inputs.input_ids.to(device)
        # text_masks = inputs.attention_mask.to(device)
        # text_embeddings = pipe.text_encoder(text_input)[0]
        
        inputs = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        text_input = inputs.input_ids.to(self.device)
        text_masks = inputs.attention_mask.to(self.device)
        text_embeddings = self.text_encoder(text_input)[0]

        text_embedding = self.image2text_embedder(guidance_image, text_embeddings, text_masks)        
        results = self.edit_image(image_path, mask_path, text_embedding, batch_size=self.batch_size, blending_percentage=blending_percentage)
    
        return results

    # def get_embeddings(self, guidance_image):
    #     guidance_image = Image.open(guidance_image)
    #     W_t_plus, f_c_img, f_cnvrt_txt = compute_f_cnvrt_txt(guidance_image, self.clip_visual_model, self.clip_text_model, self.processor, device)
    #     pseudo_prompt = generate_pseudo_prompt(f_cnvrt_txt, self.clip_text_model, self.tokenizer)

    #     return pseudo_prompt        

    def init_args(self,
                  model_path = "CompVis/stable-diffusion-v1-4",
                  clip_path = "openai/clip-vit-large-patch14",
                  batch_size = 4,
                  blending_start_percentage = 0.25,
                  device = "cuda",
                  alpha = 0.6,
                  output_path = "outputs/res.jpg",
                  onlyprompt = False,
                  edit = True):
        
        self.model_path = model_path # the path to the HuggingFace model
        self.clip_path = clip_path # the path to the CLIP model
        self.batch_size = batch_size # Number of images to be generated
        self.blending_start_percentage = 0.25 # The diffusion steps percentage to jump to
        self.device = device
        self.output_path = output_path
        self.onlyprompt = onlyprompt
        self.edit = edit

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        return (model_path, clip_path, batch_size, blending_start_percentage, device, output_path)
        

    def load_models(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path, torch_dtype=torch.float16
        )
        self.vae = pipe.vae.to(self.device)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to(self.device)
        self.unet = pipe.unet.to(self.device)
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.image2text_embedder = Image2TextEmbedder(self.clip_path, self.device, onlyprompt=self.onlyprompt, edit=self.edit)

    @torch.no_grad()
    def edit_image(
        self,
        image_path,
        mask_path,
        text_embedding,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=torch.manual_seed(42),
        blending_percentage=0.25,
    ):
        image = Image.open(image_path)
        image = image.resize((height, width), Image.BILINEAR)
        image = np.array(image)[:, :, :3]
        source_latents = self._image2latent(image)
        latent_mask, org_mask = self._read_mask(mask_path)

        text_embeddings = text_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
        max_length = text_embeddings.shape[-2]

        uncond_input = self.tokenizer(
            [""],
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        ).input_ids.to(self.device).repeat(batch_size, 1)
        
        uncond_embeddings = self.text_encoder(uncond_input)[0]

        text_embeddings = torch.vstack([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(self.device).half()

        self.scheduler.set_timesteps(num_inference_steps)

        loop = tqdm(self.scheduler.timesteps[
            int(len(self.scheduler.timesteps) * blending_percentage) :
        ], desc="Blending", leave=False)
        for t in loop:
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, timestep=t
            )
            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # Blending
            noise_source_latents = self.scheduler.add_noise(
                source_latents, torch.randn_like(latents), t
            )
            latents = latents * latent_mask + noise_source_latents * (1 - latent_mask)

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")

        return images

    @torch.no_grad()
    def _image2latent(self, image):
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
        image = image.half()
        latents = self.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.18215

        return latents

    def _read_mask(self, mask_path: str, dest_size=(64, 64)):
        org_mask = Image.open(mask_path).convert("L")
        mask = org_mask.resize(dest_size, Image.NEAREST)
        mask = np.array(mask) / 255
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = mask[np.newaxis, np.newaxis, ...]
        mask = torch.from_numpy(mask).half().to(self.device)

        return mask, org_mask
    
class UnCLIPBLD():
    def __init__(self, device):
        self.init_args(device=device)
        self.load_models()

    def __call__(self, x):
        image_path, mask_path, guidance_image, blending_percentage = x['image_path'], x['mask_path'], x['guidance_image'], x['blending_percentage']
        
        text_embedding = self.get_embeddings(guidance_image)
        results = self.edit_image(image_path, mask_path, text_embedding, blending_percentage=blending_percentage)
        
        return results

    def get_embeddings(self, guidance_image):
        guidance_image = Image.open(guidance_image)
        W_t_plus, f_c_img, f_cnvrt_txt = compute_f_cnvrt_txt(guidance_image, self.image_encoder, self.text_model, self.processor, device)
        pseudo_prompt = generate_pseudo_prompt(f_cnvrt_txt, self.text_encoder, self.tokenizer)

        return pseudo_prompt

    def init_args(self,
                  model_path = "stabilityai/stable-diffusion-2-1-unclip-small",
                  batch_size = 4,
                  blending_start_percentage = 0.25,
                  device = "cuda",
                  output_path = "outputs/res.jpg"):
         
        self.model_path = model_path # the path to the HuggingFace model
        
        self.batch_size = 4 # Number of images to be generated
        self.blending_start_percentage = 0.25 # The diffusion steps percentage to jump to
        self.device = device
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        return (model_path, batch_size, blending_start_percentage, device, output_path)
        

    def load_models(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path, torch_dtype=torch.float16
        )
        self.vae = pipe.vae.to(self.device)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to(self.device)
        self.unet = pipe.unet.to(self.device)
        self.image_encoder = pipe.image_encoder.to(device)
        self.text_model = CLIPTextModelWithProjection("openai/clip-vit-large-patch14").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

@torch.no_grad()
def compute_f_cnvrt_txt(image: Image,
                        vision_model: CLIPVisionModelWithProjection,
                        text_model: CLIPTextModelWithProjection,
                        processor: CLIPProcessor,
                        device: str):
    # Move W_t and compute W_t^+
    W_t =text_model.text_projection.weight.to(device)
    W_t_plus = torch.linalg.pinv(W_t)  # Compute pseudo-inverse

    f_c_txt_norm = torch.tensor(27.0, device=device) # Empirically determined text embedding norm  

    inputs = processor(images=image, return_tensors="pt").to(device)
    f_c_img = vision_model(**inputs).image_embeds  # Shape: (batch_size, projection_dim)
    f_c_img_norm = f_c_img.norm(dim=-1, keepdim=True)  # Shape: (batch_size, 1)

    # Compute f_cnvrt_txt
    f_cnvrt_txt = (f_c_txt_norm / f_c_img_norm) * (W_t_plus @ f_c_img.T)  # Shape: (d_text, batch_size)

    f_cnvrt_txt = f_cnvrt_txt.T  # Final shape: (batch_size, d_text)

    return W_t_plus, f_c_img, f_cnvrt_txt

def generate_pseudo_prompt(f_cnvrt_txt: torch.Tensor,
                            text_model: CLIPTextModelWithProjection,
                            tokenizer: CLIPProcessor):
    sos_token = tokenizer.bos_token
    
    # todo encode text properly
    sos_embedding = text_model(sos_token).text_embeds  # Shape: (batch_size, d_text)

    # cat sos_embedding and f_cnvrt_txt repeated 76 times
    pseudo_prompt = torch.vstack((
        sos_embedding,
        # 1 <sos> + 76 tokens = 77 tokens
        f_cnvrt_txt.repeat(76, 1)
        )
    )
    return pseudo_prompt

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    bld = BlendedLatentDiffusion(device=device)

    src_path = 'inputs/img.png'
    mask_path = 'inputs/mask.png'
    guidance_path = 'inputs/guidance.png'

    sample = {
        'image_path': src_path,
        'mask_path': mask_path,
        'prompt': 'a dog',
        'guidance_image': guidance_path,
        'blending_percentage': 0.25
    }

    results = bld(sample)
    
    results_flat = np.concatenate(results, axis=1)
    Image.fromarray(results_flat).save(bld.output_path)
