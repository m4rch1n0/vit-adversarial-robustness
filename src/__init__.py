"""Unified model construction API.

Public entry points
-------------------
- ``build_model(cfg_model)`` — architecture-agnostic model factory
- ``get_head(model, arch)``  — return the classifier head regardless of arch
"""

from __future__ import annotations

import torch.nn as nn

from .cnn_models import build_cnn
from .models import build_vit, get_param_groups, load_checkpoint


_VIT_ARCHS = {"vit"}
_CNN_ARCHS = {"resnet18", "resnet50"}


def build_model(cfg_model: dict, *, pretrained: bool | None = None) -> nn.Module:
    """Construct a model from a config dict.

    Parameters
    ----------
    cfg_model : the ``model:`` section of a YAML config.
    pretrained : override the config's ``pretrained`` flag.
                 Pass ``False`` when loading from a checkpoint.
    """
    arch = cfg_model.get("arch", "vit")
    use_pretrained = pretrained if pretrained is not None else cfg_model["pretrained"]

    if arch in _VIT_ARCHS:
        return build_vit(
            model_name=cfg_model["name"],
            num_classes=cfg_model["num_classes"],
            pretrained=use_pretrained,
            strategy=cfg_model.get("strategy", "linear_probe"),
        )

    if arch in _CNN_ARCHS:
        freeze = cfg_model.get("strategy", "linear_probe") == "linear_probe"
        return build_cnn(
            arch=arch,
            num_classes=cfg_model["num_classes"],
            pretrained=use_pretrained,
            freeze_backbone=freeze,
        )

    raise ValueError(f"Unknown architecture: {arch!r}. Choose from {_VIT_ARCHS | _CNN_ARCHS}")


def get_head(model: nn.Module, arch: str) -> nn.Module:
    """Return the classifier head module, regardless of architecture.

    ViT (timm)  → ``model.head``
    ResNet (tv) → ``model.fc``
    """
    if arch in _CNN_ARCHS:
        return model.fc
    return model.head
