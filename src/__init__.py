"""Unified model construction API.

Public entry points
-------------------
- ``build_model(cfg_model)`` — architecture-agnostic model factory. Returns a
  ``NormalizedClassifier`` that takes [0, 1] pixels and applies the input
  normalisation of the backbone's pretraining.
- ``get_head(model)``        — return the classifier head regardless of arch
"""

from __future__ import annotations

import torch.nn as nn

from .cnn_models import build_cnn, cnn_normalization
from .models import build_vit, get_param_groups, load_checkpoint
from .wrapper import NormalizedClassifier


_VIT_ARCHS = {"vit"}
_CNN_ARCHS = {"resnet18", "resnet50"}


def build_model(cfg_model: dict, *, pretrained: bool | None = None) -> NormalizedClassifier:
    """Construct a model from a config dict.

    Parameters
    ----------
    cfg_model : the ``model:`` section of a YAML config.
    pretrained : override the config's ``pretrained`` flag.
                 Pass ``False`` when loading from a checkpoint.
    """
    arch = cfg_model.get("arch", "vit")
    use_pretrained = pretrained if pretrained is not None else cfg_model["pretrained"]
    strategy = cfg_model.get("strategy", "linear_probe")
    frozen = strategy == "linear_probe"

    if arch in _VIT_ARCHS:
        backbone = build_vit(
            model_name=cfg_model["name"],
            num_classes=cfg_model["num_classes"],
            pretrained=use_pretrained,
            strategy=strategy,
        )
        mean, std = backbone.pretrained_cfg["mean"], backbone.pretrained_cfg["std"]
        return NormalizedClassifier(backbone, "head", mean, std, frozen_backbone=frozen)

    if arch in _CNN_ARCHS:
        backbone = build_cnn(
            arch=arch,
            num_classes=cfg_model["num_classes"],
            pretrained=use_pretrained,
            freeze_backbone=frozen,
        )
        mean, std = cnn_normalization(arch)
        return NormalizedClassifier(backbone, "fc", mean, std, frozen_backbone=frozen)

    raise ValueError(f"Unknown architecture: {arch!r}. Choose from {_VIT_ARCHS | _CNN_ARCHS}")


def get_head(model: NormalizedClassifier) -> nn.Module:
    """Return the classifier head (``head`` for timm ViT, ``fc`` for torchvision ResNet)."""
    return model.head
