#!/usr/bin/env python3
"""Train a model (ViT or CNN) on Imagenette and save the checkpoint + clean metrics.

Architecture is determined by the ``model.arch`` field in the YAML config.
All other training parameters (optimizer, scheduler, epochs, seed, data) are
shared across architectures to guarantee a fair comparison.

Usage
-----
    python scripts/train.py --config configs/default.yaml      # ViT
    python scripts/train.py --config configs/resnet50.yaml      # ResNet-50
    python scripts/train.py --config configs/resnet18.yaml      # ResNet-18
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import build_model, get_head, get_param_groups
from src.data import get_loaders
from src.eval import evaluate
from src.utils import print_table, run_dir, save_csv, save_metadata, set_seed


def _count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
) -> float:
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        pbar.set_postfix(loss=f"{total_loss / n:.4f}")
    return total_loss / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = cfg["device"]

    train_loader, val_loader, num_classes = get_loaders(
        cfg["data"]["path"],
        batch_size=cfg["data"]["batch_size"],
        workers=cfg["data"]["workers"],
        image_size=cfg["data"]["image_size"],
    )

    cfg["model"]["num_classes"] = num_classes
    arch = cfg["model"].get("arch", "vit")
    strategy = cfg["model"].get("strategy", "linear_probe")

    model = build_model(cfg["model"]).to(device)
    trainable = _count_trainable(model)
    print(f"Architecture: {arch}  |  Strategy: {strategy}  |  Trainable params: {trainable:,}")

    head_module = get_head(model, arch)

    if strategy == "full" and arch == "vit":
        param_groups = get_param_groups(
            model,
            lr_head=cfg["training"]["lr"],
            lr_backbone=cfg["training"]["lr_backbone"],
        )
    else:
        param_groups = [{"params": head_module.parameters(), "lr": cfg["training"]["lr"]}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    loss_fn = nn.CrossEntropyLoss()

    strategy_short = "lp" if strategy == "linear_probe" else "ft"
    out = run_dir(f"train/{arch}_{strategy_short}", base=cfg["output_dir"])
    print(f"Run directory: {out}")

    epochs = cfg["training"]["epochs"]
    epoch_bar = tqdm(range(1, epochs + 1), desc="epochs", unit="ep")
    for epoch in epoch_bar:
        avg_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        scheduler.step()
        val_result = evaluate(model, val_loader, device=device, compute_calibration=True)
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", val_acc=f"{val_result['acc']:.4f}")
        tqdm.write(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  val_acc={val_result['acc']:.4f}")

    torch.save(model.state_dict(), out / "model.pt")
    print(f"Checkpoint saved: {out / 'model.pt'}")

    clean = evaluate(model, val_loader, device=device, compute_calibration=True)
    rows = [{"metric": k, "value": f"{v:.6f}"} for k, v in clean.items()]
    save_csv(out / "clean_metrics.csv", rows, fieldnames=["metric", "value"])
    print_table(rows)

    save_metadata(
        out / "metadata.json",
        seed=cfg["seed"],
        model=cfg["model"],
        training=cfg["training"],
        data_path=cfg["data"]["path"],
        clean_accuracy=clean["acc"],
        trainable_params=trainable,
        device=device,
        torch_version=torch.__version__,
    )


if __name__ == "__main__":
    main()
