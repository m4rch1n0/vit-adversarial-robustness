"""CNN model loading for architectural comparison with ViT.

Supports ResNet-18 and ResNet-50 with ImageNet-1k V1 pretrained weights
(torchvision standard).  The linear-probe setup freezes the backbone and
replaces the final fully-connected layer with a fresh ``nn.Linear``.

Design note — weight selection
------------------------------
We deliberately use ``IMAGENET1K_V1`` (the classic supervised baseline) rather
than V2 (SWAG / semi-weakly supervised) to keep the pretraining regime
comparable to the ViT checkpoint (``orig_in21k_ft_in1k``).  Both families
were pretrained on ImageNet-scale data with standard supervision.
"""

from __future__ import annotations

from typing import Literal

import torch.nn as nn
from torchvision import models


CNNArch = Literal["resnet18", "resnet50"]

_WEIGHTS = {
    "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
    "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
}

_BUILDERS = {
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
}


def build_cnn(
    arch: CNNArch = "resnet50",
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Load a pretrained CNN and prepare it for downstream classification.

    Parameters
    ----------
    arch : ``"resnet18"`` or ``"resnet50"``
    num_classes : number of output classes (10 for Imagenette)
    pretrained : load ImageNet-1k V1 weights
    freeze_backbone : if True, only the final FC layer is trainable
                      (linear-probe regime, matching ViT setup)
    """
    if arch not in _BUILDERS:
        raise ValueError(f"Unknown CNN architecture: {arch!r}. Choose from {list(_BUILDERS)}")

    weights = _WEIGHTS[arch] if pretrained else None
    model = _BUILDERS[arch](weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
