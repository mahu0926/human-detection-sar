"""Terrain gate: an ImageNet backbone with a 3-class head.

Class order is fixed and must match ``config.yaml``:
0 maritime, 1 urban, 2 vegetation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
    mobilenet_v3_large,
    resnet18,
)

BACKBONES = ("mobilenet_v3_large", "resnet18")


def build_gate(backbone: str = "mobilenet_v3_large", num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    if backbone == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if backbone == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unknown backbone {backbone!r}. Choose from {BACKBONES}.")


def load_gate(checkpoint_path: str, device: torch.device, pretrained_fallback: bool = False) -> tuple[nn.Module, dict]:
    """Load ``weights/gate_best.pt``. The head is built randomly, then overwritten."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = checkpoint.get("backbone", "mobilenet_v3_large")
    num_classes = len(checkpoint.get("class_names", ["maritime", "urban", "vegetation"]))
    model = build_gate(backbone, num_classes=num_classes, pretrained=pretrained_fallback)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict_proba(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    model.eval()
    logits = model(images)
    return torch.softmax(logits, dim=1)
