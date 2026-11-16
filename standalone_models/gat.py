"""Graph Attention Network (GAT) with Spatial Region Graph (Veličković et al., 2018)."""
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

class GraphAttentionLayer(nn.Module):
    """Multi-Head Graph Attention Layer with 8-neighbor spatial attention."""

    def __init__(self, in_features: int, out_features: int, heads: int = 4, concat: bool = True, dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
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
        h = torch.einsum("bnf,hfo->bhno", nodes, self.weight)
        src_scores = (h * self.attn_src[None, :, None, :]).sum(dim=-1)
        dst_scores = (h * self.attn_dst[None, :, None, :]).sum(dim=-1)
        scores = self.leaky_relu(src_scores.unsqueeze(-1) + dst_scores.unsqueeze(-2))
        mask = adjacency[None, None, :, :]
        scores = scores.masked_fill(~mask, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        att_dropped = F.dropout(attention_weights, p=self.dropout, training=self.training)
        out = torch.matmul(att_dropped, h)
        if self.concat:
            out = out.permute(0, 2, 1, 3).contiguous().reshape(nodes.size(0), nodes.size(1), -1)
        else:
            out = out.mean(dim=1)
        return F.elu(out), attention_weights

class ImageGraphGAT(nn.Module):
    """Image-Region Graph Attention Network (49 Spatial Nodes, 8-Neighbor Mesh)."""

    def __init__(self, num_classes: int = len(CHEXPERT_FINDINGS), pretrained: bool = False, grid_size: int = 7):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = tv_models.resnet18(weights=weights)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        self.projection = nn.Sequential(nn.Linear(512, 128), nn.LayerNorm(128), nn.ReLU(inplace=True))
        self.gat1 = GraphAttentionLayer(128, 32, heads=4, concat=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(128)
        self.gat2 = GraphAttentionLayer(128, 128, heads=1, concat=False, dropout=0.1)
        self.norm2 = nn.LayerNorm(128)
        self.classifier = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.register_buffer("adjacency", self._build_spatial_adjacency(grid_size), persistent=False)
        self.last_node_importance = None

    @staticmethod
    def _build_spatial_adjacency(side: int) -> torch.Tensor:
        total = side * side
        adj = torch.zeros(total, total, dtype=torch.bool)
        for y in range(side):
            for x in range(side):
                s_idx = y * side + x
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < side and 0 <= nx < side:
                            adj[s_idx, ny * side + nx] = True
        return adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)
        b = feat.size(0)
        nodes = self.projection(feat.flatten(2).transpose(1, 2))
        nodes1, _ = self.gat1(nodes, self.adjacency)
        nodes = self.norm1(nodes + nodes1)
        nodes2, attn2 = self.gat2(nodes, self.adjacency)
        nodes = self.norm2(nodes + nodes2)
        incoming_attention = attn2.squeeze(1).mean(dim=1)
        self.last_node_importance = incoming_attention.view(b, self.grid_size, self.grid_size)
        return self.classifier(self.dropout(nodes.mean(dim=1)))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)
        nodes = self.projection(feat.flatten(2).transpose(1, 2))
        nodes1, _ = self.gat1(nodes, self.adjacency)
        nodes = self.norm1(nodes + nodes1)
        nodes2, _ = self.gat2(nodes, self.adjacency)
        nodes = self.norm2(nodes + nodes2)
        return nodes.mean(dim=1)

if __name__ == "__main__":
    model = ImageGraphGAT(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print(f"[+] GAT output shape: {logits.shape} (Expected: [2, 13])")
