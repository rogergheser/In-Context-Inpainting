import torch

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class Image2TextEmbedder():
    def __init__(self,
                 clip_path: str = "openai/clip-vit-large-patch14",
                 device: str = "cuda",
                 alpha: float = 0.6,
                 onlyprompt: bool = False,
                 edit: bool = True):
        self.clip_model = CLIPModel.from_pretrained(clip_path).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_path)
        self.inv_text = torch.linalg.pinv(self.clip_model.text_projection.weight, atol=0.3)
        self.visual_projection = self.clip_model.visual_projection.weight
        self.alpha = alpha
        self.onlyprompt = onlyprompt
        self.edit = edit
        self.device = device

    def __call__(self, image, text_embeddings, text_masks):
        image = Image.open(image).convert("RGB")
        clip_image = self.processor(None, image, return_tensors='pt').pixel_values.to(self.device)
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