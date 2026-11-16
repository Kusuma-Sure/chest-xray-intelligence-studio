"""DenseNet121 Architecture for 13-Finding Multi-Label Chest Radiograph Classification."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models

CHEXPERT_FINDINGS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

class CheXpertDenseNet121(nn.Module):
    """DenseNet121 with feature reuse across dense blocks and 13-finding classification head."""

    def __init__(self, num_classes: int = len(CHEXPERT_FINDINGS), pretrained: bool = False):
        super().__init__()
        weights = tv_models.DenseNet121_Weights.DEFAULT if pretrained else None
        densenet = tv_models.densenet121(weights=weights)
        self.features = densenet.features
        in_features = densenet.classifier.in_features
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)

if __name__ == "__main__":
    model = CheXpertDenseNet121(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print(f"[+] DenseNet121 output shape: {logits.shape} (Expected: [2, 13])")
