import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
from .layers import MLP

class QwenImageModel(nn.Module):

    def __init__(self, emb_dim, mlp_dims, dropout):
        super().__init__()
        local_path = "/home/edis/biswajit/models/Qwen2-VL-2B/"
        # Load Qwen model in low-memory mode
        self.qwen = Qwen2VLForConditionalGeneration.from_pretrained(
            local_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            local_files_only=True
        )

        self.processor = AutoProcessor.from_pretrained(
            local_path,
            local_files_only=True
        )

        # freeze backbone
        for p in self.qwen.parameters():
            p.requires_grad = False

        hidden_dim = self.qwen.config.hidden_size
        self.mlp = MLP(hidden_dim, mlp_dims, dropout)

    def forward(self, **kwargs):

        images = kwargs["image"]

        pil_images = []

        for img in images:
            img = img.detach().cpu().permute(1,2,0).numpy()
            img = (img * 255).clip(0,255).astype("uint8")
            pil_images.append(Image.fromarray(img))


        inputs = self.processor(
            text=["<|image_pad|>"] * len(pil_images),
            images=pil_images,
            return_tensors="pt"
        )


        device = images.device
        inputs = {k:v.to(device) for k,v in inputs.items()}

        outputs = self.qwen(
            **inputs,
            output_hidden_states=True
        )

        hidden = outputs.hidden_states[-1]

        pooled = hidden.mean(dim=1)

        pooled = pooled.float()   

        output = self.mlp(pooled)

        return torch.sigmoid(output.squeeze(1))
