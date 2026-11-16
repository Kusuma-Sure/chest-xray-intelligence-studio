"""Graph Attention Network (GAT) for Chest X-ray Region-Level Representation.

Translates 2D radiograph feature maps into an image-region graph (49 spatial nodes)
and applies multi-head graph self-attention across 8-neighbor spatial adjacencies.
Exposes regional attention weights and influential neighbor links for XAI.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models


class GraphAttentionLayer(nn.Module):
    """Multi-Head Graph Attention Layer (Veličković et al., 2018)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.1,
        alpha: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        # Weight matrices: [heads, in_features, out_features]
        self.weight = nn.Parameter(torch.empty(heads, in_features, out_features))
        self.attn_src = nn.Parameter(torch.empty(heads, out_features))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_features))
        self.leaky_relu = nn.LeakyReLU(alpha)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(-1))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(-1))

    def forward(
        self, nodes: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            nodes: [Batch, N_nodes, in_features]
            adjacency: [N_nodes, N_nodes] boolean adjacency matrix
        Returns:
            out: [Batch, N_nodes, heads * out_features] (if concat) or [Batch, N_nodes, out_features]
            attention_weights: [Batch, heads, N_nodes, N_nodes]
        """
        # Linear projection: h = [B, heads, N_nodes, out_features]
        h = torch.einsum("bnf,hfo->bhno", nodes, self.weight)

        # Source and destination score components
        src_scores = (h * self.attn_src[None, :, None, :]).sum(dim=-1)  # [B, heads, N]
        dst_scores = (h * self.attn_dst[None, :, None, :]).sum(dim=-1)  # [B, heads, N]

        # Combine pairwise scores: e_ij = LeakyReLU(src_i + dst_j)
        scores = self.leaky_relu(src_scores.unsqueeze(-1) + dst_scores.unsqueeze(-2))

        # Mask disconnected edges with -inf
        mask = adjacency[None, None, :, :]
        scores = scores.masked_fill(~mask, -1e9)

        # Attention coefficients: alpha_ij = softmax_j(scores_ij)
        attention_weights = F.softmax(scores, dim=-1)
        attention_dropped = F.dropout(attention_weights, p=self.dropout, training=self.training)

        # Aggregation: sum_j alpha_ij * h_j -> [B, heads, N, out_features]
        out = torch.matmul(attention_dropped, h)

        if self.concat:
            out = out.permute(0, 2, 1, 3).contiguous().reshape(nodes.size(0), nodes.size(1), -1)
        else:
            out = out.mean(dim=1)

        return F.elu(out), attention_weights


class ImageGraphGAT(nn.Module):
    """Image-Region Graph Attention Network for Multi-Label Chest Radiograph Analysis.

    Architecture:
    1. Convolutional Backbone (ResNet18 conv layers) -> [B, 512, 7, 7]
    2. Spatial Node Tokenizer -> 49 nodes of dim 512, projected to 128
    3. 8-Neighbor Spatial Adjacency Matrix (with self-loops)
    4. Two-stage Graph Attention (GAT1: 4 heads concat, GAT2: 1 head average)
    5. Global Graph Mean Pooling -> Classifier Head -> 13 multi-label logits
    """

    def __init__(
        self,
        num_classes: int = 13,
        pretrained: bool = False,
        grid_size: int = 7,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_nodes = grid_size * grid_size

        # CNN feature extractor
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = tv_models.resnet18(weights=weights)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # Output: [B, 512, 7, 7]

        # Node feature projection
        self.projection = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )

        # Graph Attention Layers
        self.gat1 = GraphAttentionLayer(128, 32, heads=4, concat=True, dropout=0.1)  # 4 * 32 = 128
        self.norm1 = nn.LayerNorm(128)
        self.gat2 = GraphAttentionLayer(128, 128, heads=1, concat=False, dropout=0.1)  # 128
        self.norm2 = nn.LayerNorm(128)

        # Global classification head
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(128, num_classes)

        # Register 8-neighbor spatial adjacency matrix
        self.register_buffer("adjacency", self._build_spatial_adjacency(grid_size), persistent=False)

        # Runtime explanation storage
        self.last_attention_weights = None
        self.last_node_importance = None

    @staticmethod
    def _build_spatial_adjacency(side: int) -> torch.Tensor:
        """Constructs an 8-neighbor spatial grid adjacency matrix with self-loops."""
        total_nodes = side * side
        adj = torch.zeros(total_nodes, total_nodes, dtype=torch.bool)
        for y in range(side):
            for x in range(side):
                src_idx = y * side + x
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < side and 0 <= nx < side:
                            dst_idx = ny * side + nx
                            adj[src_idx, dst_idx] = True
        return adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor [B, 3, 224, 224]
        Returns:
            logits: Output logits tensor [B, num_classes]
        """
        # 1. CNN Feature extraction: [B, 512, 7, 7]
        features = self.cnn(x)

        # 2. Reshape into spatial nodes: [B, 49, 512]
        b = features.size(0)
        nodes = features.flatten(2).transpose(1, 2)
        nodes = self.projection(nodes)

        # 3. GAT Layer 1: message passing over 8-neighbor graph
        nodes1, attn1 = self.gat1(nodes, self.adjacency)
        nodes = self.norm1(nodes + nodes1)  # Residual connection

        # 4. GAT Layer 2: refinement
        nodes2, attn2 = self.gat2(nodes, self.adjacency)
        nodes = self.norm2(nodes + nodes2)

        # 5. Extract region attention map for explainability
        # attn2 shape: [B, 1, 49, 49] -> incoming attention received by each node
        incoming_attention = attn2.squeeze(1).mean(dim=1)  # [B, 49]
        self.last_node_importance = incoming_attention.view(b, self.grid_size, self.grid_size)
        self.last_attention_weights = attn2.detach()

        # 6. Global pooling & classification: [B, 128] -> [B, num_classes]
        graph_representation = nodes.mean(dim=1)
        logits = self.classifier(self.dropout(graph_representation))
        return logits

    def get_region_importance(self) -> torch.Tensor | None:
        """Returns the [7, 7] region attention importance map from the last forward pass."""
        if self.last_node_importance is not None:
            return self.last_node_importance[0].detach()
        return None
