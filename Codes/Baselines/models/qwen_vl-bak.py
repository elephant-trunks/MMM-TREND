import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
from .layers import MLP


class QwenVLModel(nn.Module):

    def __init__(self, emb_dim, mlp_dims, dropout):
        super().__init__()
        local_path = "/home/edis/biswajit/models/Qwen2-VL-2B/"
        # Load Qwen2-VL model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            local_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            local_files_only=True
        )

        # Processor
        self.processor = AutoProcessor.from_pretrained(
            local_path,
            local_files_only=True
        )

        # Freeze backbone
        for p in self.model.parameters():
            p.requires_grad = False

        # Hidden size from model config
        hidden_dim = self.model.config.hidden_size

        # Classification head
        self.mlp = MLP(hidden_dim, mlp_dims, dropout)

    def forward(self, **kwargs):

        images = kwargs["image"]
        texts = kwargs["raw_text"]

        # Ensure python string list
        texts = [str(t) for t in texts]

        # Add image placeholder required by Qwen2-VL
        texts = [f"<|image_pad|> {t}" for t in texts]

        # Convert tensor images → PIL
        pil_images = []

        for img in images:
            img = img.detach().cpu().permute(1, 2, 0).numpy()
            img = (img * 255).clip(0, 255).astype("uint8")
            pil_images.append(Image.fromarray(img))

        # Processor
        inputs = self.processor(
            text=texts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
            do_rescale=False
        )

        # Move inputs to device
        device = images.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass through Qwen
        outputs = self.model(
            **inputs,
            output_hidden_states=True
        )

        # Last hidden layer
        hidden = outputs.hidden_states[-1]

        # Global Average Pooling
        pooled = hidden.mean(dim=1)

        # Fix dtype mismatch
        pooled = pooled.float()

        # Classification head
        logits = self.mlp(pooled)

        return torch.sigmoid(logits.squeeze(1))