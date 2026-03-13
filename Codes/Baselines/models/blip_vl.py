import torch
import torch.nn as nn
from transformers import BlipModel, BlipProcessor
from PIL import Image
from .layers import MLP

from transformers.utils import logging
logging.set_verbosity_error()

class BLIPVLModel(nn.Module):

    def __init__(self, emb_dim, mlp_dims, dropout):
        super().__init__()
        local_path = "/home/edis/biswajit/models/blip-image-captioning-base/"
        self.blip = BlipModel.from_pretrained(
            local_path,
            local_files_only=True
        )

        self.processor = BlipProcessor.from_pretrained(
            local_path,
            local_files_only=True
        )

        # freeze backbone
        for p in self.blip.parameters():
            p.requires_grad = False

        hidden_dim = self.blip.config.vision_config.hidden_size

        # image + text fusion
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

        inputs = self.processor(
            text=texts,
            images=pil_images,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        device = images.device
        inputs = {k:v.to(device) for k,v in inputs.items()}

        with torch.no_grad():

            # image features
            vision_outputs = self.blip.vision_model(
                pixel_values=inputs["pixel_values"]
            )
            image_hidden = vision_outputs.last_hidden_state
            image_features = image_hidden.mean(dim=1)

            # text features
            text_outputs = self.blip.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            text_hidden = text_outputs.last_hidden_state
            text_features = text_hidden.mean(dim=1)

        fused = torch.cat([image_features, text_features], dim=1)

        output = self.mlp(fused)

        return torch.sigmoid(output.squeeze(1))