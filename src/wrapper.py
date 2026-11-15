"""Pixel-space classifier wrapper.

Every model in this project takes images in [0, 1] and applies the input
normalisation of its own pretraining inside ``forward``. Data loaders,
attacks and transfer experiments therefore work in a single pixel space,
and each backbone still receives the inputs it was pretrained on:

- ViT-B/16 (timm ``orig_in21k_ft_in1k``): mean = std = 0.5, i.e. [-1, 1]
- ResNets (torchvision ``IMAGENET1K_V1``): ImageNet mean/std

With ``frozen_backbone=True`` (linear probe) the backbone is kept in eval
mode even when the wrapper is put in training mode, so that BatchNorm
running statistics of the CNNs are not updated during head training.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class NormalizedClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head_name: str,
        mean: Sequence[float],
        std: Sequence[float],
        frozen_backbone: bool,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head_name = head_name
        self.frozen_backbone = frozen_backbone
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

    @property
    def head(self) -> nn.Module:
        return getattr(self.backbone, self.head_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone((x - self.mean) / self.std)

    def train(self, mode: bool = True) -> "NormalizedClassifier":
        super().train(mode)
        if self.frozen_backbone:
            self.backbone.eval()
            self.head.train(mode)
        return self

    def backbone_state(self) -> dict[str, torch.Tensor]:
        """Copy of all backbone parameters and buffers except the head."""
        prefix = f"{self.head_name}."
        return {
            k: v.detach().clone()
            for k, v in self.backbone.state_dict().items()
            if not k.startswith(prefix)
        }
