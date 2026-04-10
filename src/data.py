"""Imagenette data loading with ImageNet-standard preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_loaders(
    data_path: str | Path,
    batch_size: int = 64,
    workers: int = 4,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, int]:
    """Return (train_loader, val_loader, num_classes) for Imagenette.

    Preprocessing matches ImageNet conventions used by timm ViT checkpoints.
    """
    data_path = Path(data_path)

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(str(data_path / "train"), transform=train_transform)
    val_ds = datasets.ImageFolder(str(data_path / "val"), transform=val_transform)
    num_classes = len(train_ds.classes)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader, num_classes


def denorm(x: torch.Tensor) -> torch.Tensor:
    """ImageNet de-normalisation: normalised tensor → [0, 1] pixel tensor."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return x * std + mean


def renorm(x: torch.Tensor) -> torch.Tensor:
    """ImageNet re-normalisation: [0, 1] pixel tensor → normalised tensor."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std
