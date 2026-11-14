"""Four 13-label CheXpert architectures used by the shared training and API code."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models

from .config import FINDINGS


def squash(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * x / torch.sqrt(squared_norm + 1e-8)


class CapsuleLayer(nn.Module):
    def __init__(self, num_input_capsules: int, input_dim: int, num_output_capsules: int,
                 output_dim: int, routing_iterations: int = 3) -> None:
        super().__init__()
        self.num_input_capsules = num_input_capsules
        self.num_output_capsules = num_output_capsules
        self.output_dim = output_dim
        self.routing_iterations = routing_iterations
        self.W = nn.Parameter(0.01 * torch.randn(1, num_input_capsules, num_output_capsules, output_dim, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.unsqueeze(2).unsqueeze(-1)
        predictions = torch.matmul(self.W.expand(batch_size, -1, -1, -1, -1), x).squeeze(-1)
        logits = torch.zeros(batch_size, self.num_input_capsules, self.num_output_capsules,
                             device=x.device, dtype=x.dtype)
        for iteration in range(self.routing_iterations):
            coupling = F.softmax(logits, dim=2)
            combined = (coupling.unsqueeze(-1) * predictions).sum(dim=1)
            output = squash(combined)
            if iteration < self.routing_iterations - 1:
                logits = logits + (predictions * output.unsqueeze(1)).sum(dim=-1)
        return output


class CapsNet(nn.Module):
    """Architecture matching the uploaded implementation_capsnet notebook."""
    def __init__(self, num_classes: int = 13, routing_iterations: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, 5, 2, 2), nn.BatchNorm2d(64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 5, 2, 2), nn.BatchNorm2d(128), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.primary_conv = nn.Conv2d(128, 16 * 8, 3, 1, 1)
        self.num_primary_types = 16
        self.primary_dim = 8
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.class_capsules = CapsuleLayer(7 * 7 * 16, 8, num_classes, 16, routing_iterations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.primary_conv(x)
        x = self.pool(x)
        b = x.size(0)
        x = x.view(b, self.num_primary_types, self.primary_dim, 7, 7)
        x = x.permute(0, 1, 3, 4, 2).contiguous().view(b, -1, self.primary_dim)
        x = squash(x)
        capsules = self.class_capsules(x)
        probabilities = torch.norm(capsules, dim=-1).clamp(1e-7, 1 - 1e-7)
        return torch.log(probabilities / (1 - probabilities))


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, heads: int = 4,
                 concat: bool = True, dropout: float = 0.1, alpha: float = 0.2) -> None:
        super().__init__()
        self.heads = heads
        self.out_features = out_features
        self.concat = concat
        self.dropout = dropout
        self.weight = nn.Parameter(torch.empty(heads, in_features, out_features))
        self.attn_src = nn.Parameter(torch.empty(heads, out_features))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_features))
        self.leaky_relu = nn.LeakyReLU(alpha)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(-1))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(-1))

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor):
        # nodes [B,N,F], adjacency [N,N] bool
        h = torch.einsum("bnf,hfo->bhno", nodes, self.weight)
        src = (h * self.attn_src[None, :, None, :]).sum(-1)
        dst = (h * self.attn_dst[None, :, None, :]).sum(-1)
        scores = self.leaky_relu(src.unsqueeze(-1) + dst.unsqueeze(-2))
        scores = scores.masked_fill(~adjacency[None, None, :, :], torch.finfo(scores.dtype).min)
        attention = F.softmax(scores, dim=-1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        out = torch.matmul(attention, h)
        if self.concat:
            out = out.permute(0, 2, 1, 3).reshape(nodes.size(0), nodes.size(1), -1)
        else:
            out = out.mean(dim=1)
        return F.elu(out), attention


class ImageGraphGAT(nn.Module):
    """CNN features are converted into a 7x7 grid graph and processed by GAT."""
    def __init__(self, num_classes: int = 13, pretrained: bool = False) -> None:
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = tv_models.resnet18(weights=weights)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        self.projection = nn.Linear(512, 128)
        self.gat1 = GraphAttentionLayer(128, 32, heads=4, concat=True)
        self.gat2 = GraphAttentionLayer(128, 128, heads=1, concat=False)
        self.norm = nn.LayerNorm(128)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(128, num_classes)
        self.register_buffer("adjacency", self._make_adjacency(7), persistent=False)
        self.last_attention_map = None

    @staticmethod
    def _make_adjacency(side: int) -> torch.Tensor:
        n = side * side
        adj = torch.zeros(n, n, dtype=torch.bool)
        for y in range(side):
            for x in range(side):
                i = y * side + x
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < side and 0 <= xx < side:
                            adj[i, yy * side + xx] = True
        return adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)  # B,512,7,7 for 224x224 input
        nodes = features.flatten(2).transpose(1, 2)
        nodes = self.projection(nodes)
        nodes = F.dropout(nodes, 0.1, self.training)
        nodes, attn1 = self.gat1(nodes, self.adjacency)
        nodes = self.norm(nodes)
        nodes, attn2 = self.gat2(nodes, self.adjacency)
        self.last_attention_map = attn2.mean(dim=1).mean(dim=1).view(-1, 7, 7)
        pooled = nodes.mean(dim=1)
        return self.classifier(self.dropout(pooled))


def build_model(model_name: str, pretrained: bool = False) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "vit":
        weights = tv_models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = tv_models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, len(FINDINGS))
        return model
    if model_name == "densenet":
        weights = tv_models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = tv_models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, len(FINDINGS))
        return model
    if model_name == "capsnet":
        return CapsNet(num_classes=len(FINDINGS), routing_iterations=3)
    if model_name == "gat":
        return ImageGraphGAT(num_classes=len(FINDINGS), pretrained=False)
    raise ValueError(f"Unknown model '{model_name}'. Choose from vit, densenet, capsnet, gat.")


def forward_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = model(images)
    if isinstance(output, dict):
        output = output.get("logits", output)
    if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[1] != len(FINDINGS):
        raise RuntimeError(f"Model must return [batch, {len(FINDINGS)}] logits; received {type(output)} {getattr(output, 'shape', None)}")
    return output
