"""Capsule Network (CapsNet) with Dynamic Routing by Agreement (Sabour et al., 2017)."""
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

def squash(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Non-linear vector squashing function for capsule vectors."""
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * x / torch.sqrt(squared_norm + 1e-8)

class CapsuleLayer(nn.Module):
    """Dynamic routing capsule layer."""

    def __init__(self, num_input_capsules: int, input_dim: int, num_output_capsules: int, output_dim: int, routing_iterations: int = 3):
        super().__init__()
        self.num_input_capsules = num_input_capsules
        self.input_dim = input_dim
        self.num_output_capsules = num_output_capsules
        self.output_dim = output_dim
        self.routing_iterations = routing_iterations
        std = 1.0 / (input_dim ** 0.5)
        self.W = nn.Parameter(std * torch.randn(1, num_input_capsules, num_output_capsules, output_dim, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_size = x.size(0)
        x_exp = x.unsqueeze(2).unsqueeze(-1)
        W_exp = self.W.expand(b_size, -1, -1, -1, -1)
        u_hat = torch.matmul(W_exp, x_exp).squeeze(-1)

        b = torch.zeros(b_size, self.num_input_capsules, self.num_output_capsules, device=x.device, dtype=x.dtype)
        for it in range(self.routing_iterations):
            c = F.softmax(b, dim=2)
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)
            v = squash(s)
            if it < self.routing_iterations - 1:
                b = b + (u_hat * v.unsqueeze(1)).sum(dim=-1)
        return v

class CapsNet(nn.Module):
    """Full Capsule Network for 13 CheXpert multi-label findings."""

    def __init__(self, num_classes: int = len(CHEXPERT_FINDINGS), routing_iterations: int = 3, pretrained: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.routing_iterations = routing_iterations
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        base = tv_models.resnet18(weights=weights)
        self.conv1 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.conv2 = base.layer1
        self.conv3 = base.layer2

        self.primary_conv = nn.Conv2d(128, 16 * 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.num_primary_capsules = 7 * 7 * 16  # 784

        self.class_capsules = CapsuleLayer(
            num_input_capsules=self.num_primary_capsules,
            input_dim=8,
            num_output_capsules=num_classes,
            output_dim=16,
            routing_iterations=routing_iterations,
        )
        self.class_weights = nn.Parameter(torch.randn(num_classes, 16) * 0.1)
        self.class_bias = nn.Parameter(torch.full((num_classes,), -1.5))

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)

        pc = self.pool(self.primary_conv(c3))
        b_size = pc.size(0)
        pc = pc.view(b_size, 16, 8, 7, 7).permute(0, 1, 3, 4, 2).contiguous().view(b_size, -1, 8)
        pc = squash(pc)

        class_capsules = self.class_capsules(pc)
        logits = (class_capsules * self.class_weights.unsqueeze(0)).sum(dim=-1) + self.class_bias.unsqueeze(0)
        probabilities = torch.sigmoid(logits)

        if return_dict:
            return {"logits": logits, "probabilities": probabilities, "capsules": class_capsules}
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        return F.adaptive_avg_pool2d(c3, (1, 1)).flatten(1)

if __name__ == "__main__":
    model = CapsNet(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print(f"[+] CapsNet output shape: {logits.shape} (Expected: [2, 13])")
