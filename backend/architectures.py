"""Multi-Model Deep Learning Architectures for CheXpert Multi-Label Classification.

Implements all four distinct model families:
1. Vision Transformer (ViT-B/16) - Hugging Face Transformers baseline
2. DenseNet121 - Densely Connected Convolutional Network
3. Capsule Network (CapsNet) - Part-whole modeling with dynamic routing
4. Graph Attention Network (GAT) - Spatial region graph reasoning
"""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models

from .config import FINDINGS
from .gat import ImageGraphGAT


# ==============================================================================
# 1. CAPSULE NETWORK COMPONENTS
# ==============================================================================

def squash(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Non-linear vector squashing function for capsule vectors (Sabour et al., 2017).

    Ensures short vectors get shrunk to almost zero length and long vectors
    get shrunk to a length slightly below 1 while preserving orientation.
    """
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * x / torch.sqrt(squared_norm + 1e-8)


class CapsuleLayer(nn.Module):
    """Dynamic routing capsule layer."""

    def __init__(
        self,
        num_input_capsules: int,
        input_dim: int,
        num_output_capsules: int,
        output_dim: int,
        routing_iterations: int = 3,
    ) -> None:
        super().__init__()
        self.num_input_capsules = num_input_capsules
        self.input_dim = input_dim
        self.num_output_capsules = num_output_capsules
        self.output_dim = output_dim
        self.routing_iterations = routing_iterations

        # Transformation weight tensor: [1, input_caps, output_caps, out_dim, in_dim]
        self.W = nn.Parameter(
            0.01 * torch.randn(1, num_input_capsules, num_output_capsules, output_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, num_input_capsules, input_dim]
        batch_size = x.size(0)

        # Expand dims for matrix multiplication with W
        x_expanded = x.unsqueeze(2).unsqueeze(-1)  # [B, in_caps, 1, in_dim, 1]
        W_expanded = self.W.expand(batch_size, -1, -1, -1, -1)  # [B, in_caps, out_caps, out_dim, in_dim]

        # Predicted output capsule vectors u_hat: [B, in_caps, out_caps, out_dim]
        u_hat = torch.matmul(W_expanded, x_expanded).squeeze(-1)

        # Routing logits b initialized to 0
        b = torch.zeros(
            batch_size,
            self.num_input_capsules,
            self.num_output_capsules,
            device=x.device,
            dtype=x.dtype,
        )

        # Dynamic routing agreement loop
        for iteration in range(self.routing_iterations):
            # Coupling coefficients c_ij = softmax_j(b_ij)
            c = F.softmax(b, dim=2)  # [B, in_caps, out_caps]

            # Weighted sum: s_j = sum_i c_ij * u_hat_j|i
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)  # [B, out_caps, out_dim]

            # Squash into output capsules v_j
            v = squash(s)  # [B, out_caps, out_dim]

            if iteration < self.routing_iterations - 1:
                # Agreement update: b_ij += u_hat_j|i . v_j
                agreement = (u_hat * v.unsqueeze(1)).sum(dim=-1)
                b = b + agreement

        return v


class CapsNet(nn.Module):
    """Capsule Network with Conv1-3, Primary Capsules, and Disease Capsules.

    Matches the exact architecture from implementation_capsnet.ipynb:
    - Input: Grayscale [B, 1, 224, 224]
    - Conv1-3: Convolutional feature extraction down to [B, 128, 28, 28]
    - Primary Capsules: 16 types x 8 dims x 7 x 7 = 784 primary capsules
    - Disease Capsules: 13 multi-label findings x 16 dims with 3 routing iterations
    - Multi-label output: Capsule vector lengths serve as class probabilities
    """

    def __init__(self, num_classes: int = 13, routing_iterations: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes

        # 1. Convolutional feature extractor
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 2. Primary capsules conversion
        self.primary_conv = nn.Conv2d(128, 16 * 8, kernel_size=3, stride=1, padding=1)
        self.num_primary_types = 16
        self.primary_dim = 8
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.num_primary_capsules = 7 * 7 * 16  # 784

        # 3. Disease capsules layer
        self.class_capsules = CapsuleLayer(
            num_input_capsules=self.num_primary_capsules,
            input_dim=8,
            num_output_capsules=num_classes,
            output_dim=16,
            routing_iterations=routing_iterations,
        )

    def forward(self, x: torch.Tensor, return_dict: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        # Handle RGB input if passed by converting to 1-channel luminance
        if x.size(1) == 3:
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        # Conv feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Primary capsules
        x = self.primary_conv(x)
        x = self.pool(x)

        batch_size = x.size(0)
        x = x.view(batch_size, self.num_primary_types, self.primary_dim, 7, 7)
        x = x.permute(0, 1, 3, 4, 2).contiguous().view(batch_size, -1, self.primary_dim)
        x = squash(x)

        # Dynamic routing
        class_capsules = self.class_capsules(x)  # [B, 13, 16]

        # Capsule lengths represent multi-label probabilities
        probabilities = torch.norm(class_capsules, dim=-1)
        probabilities = torch.clamp(probabilities, 1e-7, 1.0 - 1e-7)

        # Logit conversion for BCEWithLogitsLoss compatibility
        logits = torch.log(probabilities / (1.0 - probabilities))

        if return_dict:
            return {
                "logits": logits,
                "probabilities": probabilities,
                "capsules": class_capsules,
            }
        return logits


# ==============================================================================
# 2. DENSENET121 ARCHITECTURE
# ==============================================================================

class CheXpertDenseNet121(nn.Module):
    """DenseNet121 specialized for 13-Finding Multi-Label Chest X-ray Classification."""

    def __init__(self, num_classes: int = 13, pretrained: bool = False) -> None:
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
        logits = self.classifier(out)
        return logits


# ==============================================================================
# 3. VISION TRANSFORMER (ViT-B/16) - PRIMARY ARCHITECTURE
# ==============================================================================

class CheXpertViT(nn.Module):
    """Vision Transformer (ViT-B/16) using Hugging Face Transformers.

    Initializes from google/vit-base-patch16-224 with a 13-label multi-label head.
    Exposes all 12 multi-head self-attention layers for Attention Rollout
    and embeds hooks on patch embeddings for class-specific Gradient × Activation.
    """

    def __init__(
        self,
        num_classes: int = 13,
        pretrained: bool = False,
        hf_model_id: str = "google/vit-base-patch16-224",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.hf_model_id = hf_model_id

        try:
            from transformers import ViTForImageClassification, ViTConfig
            if pretrained:
                self.vit_model = ViTForImageClassification.from_pretrained(
                    hf_model_id,
                    num_labels=num_classes,
                    problem_type="multi_label_classification",
                    ignore_mismatched_sizes=True,
                )
            else:
                config = ViTConfig.from_pretrained(hf_model_id)
                config.num_labels = num_classes
                config.problem_type = "multi_label_classification"
                self.vit_model = ViTForImageClassification(config)
            self._backend_type = "huggingface"
        except Exception:
            # Fallback to TorchVision ViT if HF is temporarily unavailable in environment
            weights = tv_models.ViT_B_16_Weights.DEFAULT if pretrained else None
            tv_vit = tv_models.vit_b_16(weights=weights)
            tv_vit.heads.head = nn.Linear(tv_vit.heads.head.in_features, num_classes)
            self.vit_model = tv_vit
            self._backend_type = "torchvision"

    def forward(
        self,
        x: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
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

    @property
    def embeddings(self):
        """Exposes patch embeddings for Gradient × Activation attribution hook."""
        if self._backend_type == "huggingface":
            return self.vit_model.vit.embeddings
        return self.vit_model.conv_proj


# ==============================================================================
# MODEL BUILDER FACTORY
# ==============================================================================

def build_model(model_name: str, pretrained: bool = False) -> nn.Module:
    """Instantiates any of the four supported architectures.

    Args:
        model_name: One of 'vit', 'densenet', 'capsnet', 'gat'.
        pretrained: Whether to load ImageNet pretrained initialization.
    """
    key = model_name.strip().lower()
    if key == "vit":
        return CheXpertViT(num_classes=len(FINDINGS), pretrained=pretrained)
    elif key == "densenet":
        return CheXpertDenseNet121(num_classes=len(FINDINGS), pretrained=pretrained)
    elif key == "capsnet":
        return CapsNet(num_classes=len(FINDINGS), routing_iterations=3)
    elif key == "gat":
        return ImageGraphGAT(num_classes=len(FINDINGS), pretrained=pretrained)
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from: vit, densenet, capsnet, gat."
        )


def forward_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Executes model forward pass and guarantees [Batch, 13] logits tensor output."""
    output = model(images)
    if isinstance(output, dict):
        output = output.get("logits", output)
    elif isinstance(output, tuple):
        output = output[0]

    if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[1] != len(FINDINGS):
        raise RuntimeError(
            f"Model output must be a 2D tensor [batch, {len(FINDINGS)}], got {type(output)} "
            f"with shape {getattr(output, 'shape', None)}"
        )
    return output
