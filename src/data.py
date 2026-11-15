"""Imagenette data loading.

Loaders return images in [0, 1] pixel space. Input normalisation is part of
each model (see ``src/wrapper.py``), so that every backbone receives the
normalisation of its own pretraining while attacks operate on pixels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_loaders(
    data_path: str | Path,
    batch_size: int = 64,
    workers: int = 4,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, int]:
    """Return (train_loader, val_loader, num_classes) for Imagenette.

    The geometric preprocessing (resize, crop) is shared by all models; images
    are returned in [0, 1] without normalisation.
    """
    data_path = Path(data_path)

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
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

