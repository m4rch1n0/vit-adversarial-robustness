"""ViT model loading from timm with configurable fine-tuning strategies."""

from __future__ import annotations

from typing import Literal

import timm
import torch
import torch.nn as nn


FinetuneStrategy = Literal["linear_probe", "full"]


def build_vit(
    model_name: str = "vit_small_patch16_224",
    num_classes: int = 10,
    pretrained: bool = True,
    strategy: FinetuneStrategy = "linear_probe",
) -> nn.Module:
    """Load a pretrained ViT and prepare it for fine-tuning on a downstream task.

    Strategies
    ----------
    linear_probe : freeze the entire backbone; only the classification head is trainable.
    full         : all parameters are trainable (use a lower LR for the backbone).
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    if strategy == "linear_probe":
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze the classification head
        for param in model.head.parameters():
            param.requires_grad = True
        # Re-initialise the head so it starts from scratch
        nn.init.trunc_normal_(model.head.weight, std=0.02)
        nn.init.zeros_(model.head.bias)

    elif strategy == "full":
        # Everything trainable; caller should set differential LR
        pass

    return model


def get_param_groups(
    model: nn.Module,
    lr_head: float = 1e-3,
    lr_backbone: float = 1e-5,
) -> list[dict]:
    """Split parameters into head vs backbone groups with different LRs.

    Useful for full fine-tuning where the backbone needs a much smaller LR.
    """
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    groups.append({"params": head_params, "lr": lr_head})
    return groups


def load_checkpoint(model: nn.Module, path: str | torch.serialization.FILE_LIKE) -> nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model
