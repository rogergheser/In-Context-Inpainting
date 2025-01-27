import os
import torch
import argparse
import numpy as np
import sys
import cv2

from Image2TextEmbedder import Image2TextEmbedder
from diffusers import DDIMScheduler, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, AutoPipelineForInpainting, StableDiffusionInpaintPipeline
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm
    
class BlendedLatentDiffusion:
    def __init__(self):
        self.parse_args()
        self.load_models()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--prompt", type=str, default="", help="The target text prompt"
        )
        parser.add_argument(
            "--init_image", type=str, required=True, help="The path to the input image"
        )
        parser.add_argument(
            "--mask", type=str, required=True, help="The path to the input mask"
        )
        parser.add_argument(
            "--guiding_image",
            type=str,
            help="The path to the guiding image",
        )
        parser.add_argument(
            "--model_path",
            type=str,
            # default="stabilityai/stable-diffusion-2-1-base",
            default="CompVis/stable-diffusion-v1-4",
            help="The path to the HuggingFace model",
        )
        parser.add_argument(
            "--batch_size", type=int, default=4, help="The number of images to generate"
        )
        parser.add_argument(
            "--strength", type=float, default=0.5, help="The strength of the guiding image"
        )
        parser.add_argument(
            "--blending_start_percentage",
            type=float,
            default=0.25,
            help="The diffusion steps percentage to jump",
        )
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--alpha", type=float, default=0.6)
        parser.add_argument(
            "--output_path",
            type=str,
            default="outputs/",
            help="The destination output path",
        )
        parser.add_argument(
            "--image_guided_prompt_gen",
            action="store_true",
            default=False,
            help="Whether to use image guided prompt generation",
        )

        parser.add_argument(
            "--clip_path",
            type=str,
            default="openai/clip-vit-large-patch14",
            help="The path to the CLIP model",
        )

        parser.add_argument(
            "--save_frequency",
            type=float,
            default=0,
            help="The frequency of intermediate steps to save",
        )

            
        self.args = parser.parse_args()
        
    def load_models(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            self.args.model_path, torch_dtype=torch.float16,
        )
        self.vae = pipe.vae.to(self.args.device)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to(self.args.device)
        self.unet = pipe.unet.to(self.args.device)
        self.mask_processor = AutoPipelineForInpainting.from_pipe(pipe).mask_processor
        if self.args.image_guided_prompt_gen:
            self.image2text_embedder = Image2TextEmbedder(
                self.args.clip_path,
                self.args.device, 
                alpha=self.args.alpha, 
                onlyprompt=False, 
                edit=True
            )

        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

    def save_intermediate_step(self, latents, step):
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")

        images_flat = np.concatenate(images, axis=1)
        Image.fromarray(images_flat).save(f"{self.args.output_path}steps/{step}.png")

    @torch.no_grad()
    def edit_image(
        self,
        image_path,
        mask_path,
        prompts=[],
        guiding_image=None,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=100,
        guidance_scale=12.5,
        strength=0.5,
        generator=torch.manual_seed(42),
        blending_percentage=0.25
    ):
        # Background image processing
        image = Image.open(image_path)
        image = image.resize((height, width), Image.BILINEAR)
        image = np.array(image)[:, :, :3]
        source_latents = self._image2latent(image)
        latent_mask, org_mask = self._read_mask(mask_path)

        if self.args.image_guided_prompt_gen:
            text_embeddings = self._fuse_text_img_embeds(prompts, guiding_image)
        else:
            print("Using text embeddings")
            # Conditioning text processing
            text_input = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.args.device))[0]

        max_length = text_embeddings.shape[-2]
        uncond_input = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.args.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Foreground image processing
        guiding_image_name = guiding_image.split("/")[-1].split(".")[0]
        if guiding_image is not None and strength < 1: # If guiding image is provided
            guiding_image = Image.open(guiding_image)
            guiding_image = guiding_image.resize((height, width), Image.BILINEAR)
            guiding_image = np.array(guiding_image)[:, :, :3]
            init_latents = self._image2latent(guiding_image).clone()
            init_latents = torch.cat([init_latents] * batch_size)
            init_latents = init_latents.to(self.args.device).half()

            # Add noise to the guiding image
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

            timesteps, num_inference_steps = timesteps, num_inference_steps - t_start
            latent_timestep = timesteps[:1].repeat(batch_size)

            noise = torch.randn(init_latents.shape, device=init_latents.device, dtype=init_latents.dtype)
            init_latents = self.scheduler.add_noise(init_latents, noise, latent_timestep)
            latents = init_latents
            loop = tqdm(timesteps, desc="Blending ")
        else: # If guiding image is not provided I will use random latents
            latents = torch.randn(
                (batch_size, self.unet.config.in_channels, height // 8, width // 8),
                generator=generator,
            )
            latents = latents.to(self.args.device).half()

            self.scheduler.set_timesteps(num_inference_steps)
            loop = tqdm(self.scheduler.timesteps[int(len(self.scheduler.timesteps) * blending_percentage) :],
                        desc="Blending", 
                        total=len(self.scheduler.timesteps) - int(len(self.scheduler.timesteps) * blending_percentage)
                    )
        
        if self.args.save_frequency > 0:
            os.makedirs(self.args.output_path + "steps", exist_ok=True)
            save_every = int(len(loop) * self.args.save_frequency)
        else:
            save_every = num_inference_steps * 2 


        loop.set_description(f"Blending {guiding_image_name}")
        for idx, t in enumerate(loop):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, 
                    t, 
                    encoder_hidden_states=text_embeddings
                ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_image - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # Blending
            noise_source_latents = self.scheduler.add_noise(
                source_latents, torch.randn_like(latents), t
            )
            latents = latents * latent_mask + noise_source_latents * (1.0 - latent_mask)
            if (1 + idx) % save_every == 0:
                self.save_intermediate_step(latents, idx + 1)

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")

        return images

    @torch.no_grad()
    def _fuse_text_img_embeds(self, prompt, guidance_image):
        inputs = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        text_input = inputs.input_ids.to(self.args.device)
        text_masks = inputs.attention_mask.to(self.args.device)
        text_embeddings = self.text_encoder(text_input)[0]

        text_embedding = self.image2text_embedder(guidance_image, text_embeddings, text_masks)
        
        return text_embedding
        
    @torch.no_grad()
    def _encode_image(self, image):
        image = torch.from_numpy(image)
        input = self.processor(images=image, return_tensors="pt", padding=True).to(self.args.device)
        image_features = self.image_encoder(**input).last_hidden_state

        return image_features

    @torch.no_grad()
    def _image2latent(self, image):
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(self.args.device)
        image = image.half()
        latents = self.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.18215

        return latents

    def mask_postprocess(self, mask, blur_factor):
        # 1. Blur the borders of the mask
        pass

    def blur(self, image: Image, blur_factor):
        image = image.filter(ImageFilter.GaussianBlur(blur_factor))

        return image

    def dilate_mask(self, mask: Image, kernel_size: int, iterations: int=1):
        mask = np.array(mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
        dilated_mask = Image.fromarray(dilated_mask)
        
        return dilated_mask

    def _read_mask(self, mask_path: str, dest_size=(64, 64), dilate_mask: bool=False):
        org_mask = Image.open(mask_path).convert("L")
        dilated_mask = self.dilate_mask(org_mask, 25, 7)
        mask = self.blur(dilated_mask, 33)
        mask.save(self.args.output_path + 'mask.png')

        mask = mask.resize(dest_size, Image.NEAREST)
        mask = np.array(mask) / 255
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = mask[np.newaxis, np.newaxis, ...]
        mask = torch.from_numpy(mask).half().to(self.args.device)

        return mask, org_mask

if __name__ == "__main__":
    bld = BlendedLatentDiffusion()
    results = bld.edit_image(
        bld.args.init_image,
        bld.args.mask,
        prompts=[bld.args.prompt] * bld.args.batch_size,
        guiding_image=bld.args.guiding_image,
        batch_size=bld.args.batch_size,
        blending_percentage=bld.args.blending_start_percentage,
        strength=bld.args.strength,
    )
    results_flat = np.concatenate(results, axis=1)
    Image.fromarray(results_flat).save(bld.args.output_path + "res.jpg")
