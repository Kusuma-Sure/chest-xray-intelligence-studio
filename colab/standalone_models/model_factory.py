"""Unified Model Factory to instantiate any of the 4 independent architectures."""
import torch
import torch.nn as nn
from vit_b16 import CheXpertViT
from densenet121 import CheXpertDenseNet121
from capsnet import CapsNet
from gat import ImageGraphGAT

def build_model(model_name: str, pretrained: bool = False) -> nn.Module:
    name = model_name.strip().lower()
    if name == "vit":
        return CheXpertViT(pretrained=pretrained)
    elif name == "densenet":
        return CheXpertDenseNet121(pretrained=pretrained)
    elif name == "capsnet":
        return CapsNet(pretrained=pretrained)
    elif name == "gat":
        return ImageGraphGAT(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose: vit, densenet, capsnet, gat")

if __name__ == "__main__":
    for m in ["vit", "densenet", "capsnet", "gat"]:
        net = build_model(m)
        x = torch.randn(1, 3, 224, 224)
        out = net(x)
        params = sum(p.numel() for p in net.parameters())
        print(f"[+] Model: {m.upper():10} | Params: {params:,} | Output: {out.shape}")
