import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from .layers import MLP


class CLIPVLModel(nn.Module):

    def __init__(self, emb_dim, mlp_dims, dropout):
        super().__init__()
        local_path = "/home/edis/biswajit/models/clip-vit-base-patch32/"
        self.clip = CLIPModel.from_pretrained(
            local_path,
            local_files_only=True
        )

        self.processor = CLIPProcessor.from_pretrained(
            local_path,
            local_files_only=True
        )

        # freeze backbone
        for p in self.clip.parameters():
            p.requires_grad = False

        hidden_dim = self.clip.config.projection_dim

        # image + text features
        fusion_dim = hidden_dim * 2

        self.mlp = MLP(fusion_dim, mlp_dims, dropout)

    def forward(self, **kwargs):

        images = kwargs["image"]
        texts = kwargs["raw_text"]

        pil_images = []

        for img in images:
            img = img.detach().cpu().permute(1,2,0).numpy()
            img = (img * 255).clip(0,255).astype("uint8")
            pil_images.append(Image.fromarray(img))

        '''inputs = self.processor(
            text=texts,
            images=pil_images,
            padding=True,
            return_tensors="pt"
        )'''
        inputs = self.processor(
            text=texts,
            images=pil_images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        device = images.device
        inputs = {k:v.to(device) for k,v in inputs.items()}

        with torch.no_grad():

            image_features = self.clip.get_image_features(
                pixel_values=inputs["pixel_values"]
            )

            text_features = self.clip.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        image_features = image_features.float()
        text_features = text_features.float()

        fused = torch.cat([image_features, text_features], dim=1)

        output = self.mlp(fused)

        return torch.sigmoid(output.squeeze(1))