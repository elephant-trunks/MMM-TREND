import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
from .layers import MLP
from torch.cuda.amp import autocast


class QwenVLModel(nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout):
        super().__init__()

        local_path = "/home/edis/biswajit/models/Qwen2-VL-2B/"

        # Load Qwen2-VL in FP16 from local files
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            local_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        # Put model on GPU 0 if available, freeze backbone
        if torch.cuda.is_available():
            self.model.to(torch.device("cuda:0"))
        self.model.requires_grad_(False)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False  # reduce mem

        self.processor = AutoProcessor.from_pretrained(
            local_path,
            local_files_only=True,
        )

        hidden_dim = self.model.config.hidden_size
        self.mlp = MLP(hidden_dim, mlp_dims, dropout)

    @staticmethod
    def _to_pil_rgb(img_tensor, size=(224, 224)):
        """
        img_tensor: (C,H,W) in [0,1]
        Convert to PIL RGB and resize to control memory/latency.
        """
        arr = (
            img_tensor.detach().cpu().permute(1, 2, 0)
            .mul(255).clamp(0, 255).byte().numpy()
        )
        pil = Image.fromarray(arr)
        if pil.mode in ("P", "RGBA", "LA"):
            pil = pil.convert("RGBA").convert("RGB")
        elif pil.mode != "RGB":
            pil = pil.convert("RGB")
        if size is not None:
            pil = pil.resize(size, Image.BICUBIC)
        return pil

    def _move_to_device(self, obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._move_to_device(v, device) for v in obj)
        return obj

    def forward(self, **kwargs):
        images = kwargs["image"]                           # (B, C, H, W), [0,1]
        raw_texts = kwargs.get("raw_text", [""] * images.size(0))
        B = images.size(0)

        # Convert tensors -> PIL RGB (224x224)
        pil_images = [self._to_pil_rgb(img, size=(224, 224)) for img in images]

        # ---- CRITICAL: the processor in your build looks for the literal "<|image_pad|>" ----
        # Prepend it so it can expand to the correct number of image tokens.
        texts = [f"<|image_pad|> {str(raw_texts[i])[:256]}" for i in range(B)]

        # Single processor call with BOTH text and images ensures alignment
        inputs = self.processor(
            text=texts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )

        # Dtypes
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].long()
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].long()
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)

        # Move everything to the model's device
        model_device = next(self.model.parameters()).device
        try:
            inputs = inputs.to(model_device)  # BatchEncoding path
        except AttributeError:
            inputs = self._move_to_device(inputs, model_device)

        # Extra belt & suspenders
        for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
            if k in inputs and isinstance(inputs[k], torch.Tensor) and inputs[k].device != model_device:
                inputs[k] = inputs[k].to(model_device, non_blocking=True)
        # ... keep everything above unchanged ...

        # Frozen backbone: do NOT track gradients through Qwen (saves memory)
        with torch.no_grad():
            with autocast(dtype=torch.float16):
                outputs = self.model(
                    **inputs,
                    # ✅ we need hidden states from the last layer to pool
                    output_hidden_states=True,   # <-- was False
                )       
                # ✅ for causal LMs, use hidden_states[-1] (not last_hidden_state)
                hidden = outputs.hidden_states[-1]   # (B, T, D)
                pooled = hidden.mean(dim=1).float()  # (B, D)

        # Trainable head
        logits = self.mlp(pooled)
        return torch.sigmoid(logits.squeeze(1))
