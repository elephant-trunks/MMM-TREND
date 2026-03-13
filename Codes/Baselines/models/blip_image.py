import torch
import torch.nn as nn
from transformers import BlipModel, BlipProcessor
from PIL import Image
from .layers import MLP

from transformers.utils import logging
logging.set_verbosity_error()

class BLIPImageModel(nn.Module):

    def __init__(self, emb_dim, mlp_dims, dropout):
        super().__init__()
        local_path = "/home/edis/biswajit/models/blip-image-captioning-base/"
        # Load pretrained BLIP
        self.blip = BlipModel.from_pretrained(
            local_path,
            local_files_only=True
        )

        self.processor = BlipProcessor.from_pretrained(
            local_path,
            local_files_only=True
        )

        # Freeze backbone
        for p in self.blip.parameters():
            p.requires_grad = False

        hidden_dim = self.blip.config.vision_config.hidden_size

        self.mlp = MLP(hidden_dim, mlp_dims, dropout)

    def forward(self, **kwargs):

        images = kwargs["image"]

        pil_images = []

        for img in images:
            img = img.detach().cpu().permute(1, 2, 0).numpy()
            img = (img * 255).clip(0,255).astype("uint8")
            pil_images.append(Image.fromarray(img))

        inputs = self.processor(
            images=pil_images,
            return_tensors="pt"
        )

        device = images.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            vision_outputs = self.blip.vision_model(
                pixel_values=inputs["pixel_values"]
            )

        hidden = vision_outputs.last_hidden_state
        image_features = hidden.mean(dim=1)

        image_features = image_features.float()

        output = self.mlp(image_features)

        return torch.sigmoid(output.squeeze(1))