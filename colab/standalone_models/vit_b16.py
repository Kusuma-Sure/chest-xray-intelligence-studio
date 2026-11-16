"""Vision Transformer (ViT-B/16) for Multi-Label Chest Radiograph Classification."""
from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models as tv_models

CHEXPERT_FINDINGS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

class CheXpertViT(nn.Module):
    """Vision Transformer (ViT-B/16) initialized with ImageNet weights and 13-Finding Head."""

    def __init__(self, num_classes: int = len(CHEXPERT_FINDINGS), pretrained: bool = False, hf_model_id: str = "google/vit-base-patch16-224"):
        super().__init__()
        self.num_classes = num_classes
        self._backend_type = "torchvision"
        try:
            from transformers import ViTForImageClassification, ViTConfig
            if pretrained:
                self.vit_model = ViTForImageClassification.from_pretrained(
                    hf_model_id,
                    num_labels=num_classes,
                    problem_type="multi_label_classification",
                    ignore_mismatched_sizes=True,
                    attn_implementation="eager",
                )
            else:
                config = ViTConfig.from_pretrained(hf_model_id)
                config.num_labels = num_classes
                config.problem_type = "multi_label_classification"
                config._attn_implementation = "eager"
                self.vit_model = ViTForImageClassification(config)
            self._backend_type = "huggingface"
        except Exception:
            weights = tv_models.ViT_B_16_Weights.DEFAULT if pretrained else None
            tv_vit = tv_models.vit_b_16(weights=weights)
            tv_vit.heads.head = nn.Linear(tv_vit.heads.head.in_features, num_classes)
            self.vit_model = tv_vit
            self._backend_type = "torchvision"

    def forward(self, x: torch.Tensor, output_attentions: bool = False):
        if self._backend_type == "huggingface":
            outputs = self.vit_model(pixel_values=x, output_attentions=output_attentions)
            logits = outputs.logits
            if output_attentions:
                return logits, outputs.attentions
            return logits
        else:
            logits = self.vit_model(x)
            if output_attentions:
                return logits, None
            return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if self._backend_type == "huggingface":
            outputs = self.vit_model.vit(pixel_values=x)
            return outputs.last_hidden_state[:, 0]
        else:
            x_proc = self.vit_model._process_input(x)
            batch_class_token = self.vit_model.class_token.expand(x.shape[0], -1, -1)
            x_tokens = torch.cat([batch_class_token, x_proc], dim=1)
            x_tokens = self.vit_model.encoder(x_tokens)
            return x_tokens[:, 0]

if __name__ == "__main__":
    model = CheXpertViT(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print(f"[+] ViT-B/16 output shape: {logits.shape} (Expected: [2, 13])")
