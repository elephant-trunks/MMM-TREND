import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from .layers import MLP


class CLIPImageModel(nn.Module):

    def __init__(self, emb_dim, mlp_dims, dropout):
        super().__init__()
        local_path = "/home/edis/biswajit/models/clip-vit-base-patch32/"
        # Load pretrained CLIP
        self.clip = CLIPModel.from_pretrained(
            local_path,
            local_files_only=True
        )

        self.processor = CLIPProcessor.from_pretrained(
            local_path,
            local_files_only=True
        )

        # Freeze CLIP backbone
        for p in self.clip.parameters():
            p.requires_grad = False

        # CLIP image embedding dimension
        hidden_dim = self.clip.config.projection_dim   # usually 512

        # Classifier head
        self.mlp = MLP(hidden_dim, mlp_dims, dropout)

    def forward(self, **kwargs):

        images = kwargs["image"]

        pil_images = []

        # Convert tensor → PIL (same logic as Qwen)
        for img in images:
            img = img.detach().cpu().permute(1, 2, 0).numpy()
            img = (img * 255).clip(0,255).astype("uint8")
            pil_images.append(Image.fromarray(img))

        # CLIP preprocessing
        inputs = self.processor(
            images=pil_images,
            return_tensors="pt"
        )

        device = images.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass through CLIP
        with torch.no_grad():
            image_features = self.clip.get_image_features(**inputs)

        image_features = image_features.float()

        # Classification head
        output = self.mlp(image_features)

        return torch.sigmoid(output.squeeze(1))