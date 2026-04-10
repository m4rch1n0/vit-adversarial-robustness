#!/usr/bin/env python3
"""Attention rollout analysis: quantitative metrics + qualitative heatmaps.

Computes how adversarial perturbations shift the ViT's attention distribution,
measuring the effect with cosine similarity and KL divergence between the
clean and adversarial rollout vectors.

Outputs
-------
1. CSV with per-(attack, epsilon) attention shift metrics.
2. Heatmap grids showing rollout overlays on original images for visual
   inspection of where the model "looks" before and after perturbation.

Usage
-----
    python scripts/attention_analysis.py \\
        --config configs/default.yaml \\
        --checkpoint experiments/train/20260409_175532/model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import build_model, load_checkpoint
from src.attacks import ATTACKS
from src.attention import (
    attention_kl,
    attention_rollout,
    attention_shift,
    get_attention_maps,
    rollout_to_heatmap,
)
from src.data import denorm, get_loaders
from src.utils import run_dir, save_csv, save_metadata, set_seed


ATTACK_LABELS = {
    "fgsm": "FGSM",
    "pgd": "PGD-10",
    "patch_pgd": "Patch-PGD",
    "salient_patch_pgd": "Salient-patch PGD",
}


# ── Quantitative analysis ────────────────────────────────────────────────

def _compute_clean_rollouts(
    model: nn.Module,
    loader,
    device: str,
) -> torch.Tensor:
    """Compute rollout vectors for all clean images in *loader*.

    Returns ``(N_total, num_patches)`` tensor on CPU.
    """
    all_rollouts = []
    for x, _ in tqdm(loader, desc="clean rollout", leave=False):
        x = x.to(device)
        maps = get_attention_maps(model, x)
        rollout = attention_rollout(maps)
        all_rollouts.append(rollout)
    return torch.cat(all_rollouts, dim=0)


def _compute_adv_rollouts(
    model: nn.Module,
    loader,
    attack_fn,
    device: str,
    **attack_kwargs,
) -> torch.Tensor:
    """Generate adversarial examples and compute their rollout vectors.

    Returns ``(N_total, num_patches)`` tensor on CPU.
    """
    all_rollouts = []
    for x, y in tqdm(loader, desc="adv rollout", leave=False):
        x, y = x.to(device), y.to(device)
        with torch.enable_grad():
            x_adv = attack_fn(model, x, y, **attack_kwargs)
        maps = get_attention_maps(model, x_adv)
        rollout = attention_rollout(maps)
        all_rollouts.append(rollout)
    return torch.cat(all_rollouts, dim=0)


def quantitative_analysis(
    model: nn.Module,
    loader,
    cfg: dict,
    device: str,
) -> list[dict]:
    """Run attention shift analysis for all (attack, epsilon) combinations."""
    eps_list = cfg["attacks"]["eps_255"]
    attack_families = _build_attack_families(cfg)

    tqdm.write("Computing clean rollout (full val set)…")
    rollout_clean = _compute_clean_rollouts(model, loader, device)
    tqdm.write(f"  Clean rollout shape: {rollout_clean.shape}")

    rows = []
    for label, key, attack_fn, kwargs in attack_families:
        for eps_255 in eps_list:
            if eps_255 == 0:
                cos = attention_shift(rollout_clean, rollout_clean)
                kl = attention_kl(rollout_clean, rollout_clean)
            else:
                eps_pix = eps_255 / 255.0
                tqdm.write(f"  {label}  ε={eps_255}/255")
                rollout_adv = _compute_adv_rollouts(
                    model, loader, attack_fn, device,
                    eps_pix=eps_pix, **kwargs,
                )
                cos = attention_shift(rollout_clean, rollout_adv)
                kl = attention_kl(rollout_clean, rollout_adv)

            rows.append({
                "attack": key,
                "eps_255": eps_255,
                "mean_cos_sim": round(float(cos.mean()), 6),
                "std_cos_sim": round(float(cos.std()), 6),
                "mean_kl_div": round(float(kl.mean()), 6),
                "std_kl_div": round(float(kl.std()), 6),
                "attention_shift": round(float(1.0 - cos.mean()), 6),
            })
            tqdm.write(
                f"    cos_sim={cos.mean():.4f}±{cos.std():.4f}  "
                f"kl={kl.mean():.4f}±{kl.std():.4f}  "
                f"shift={1.0 - cos.mean():.4f}"
            )

    return rows


# ── Qualitative heatmap grid ─────────────────────────────────────────────

def _select_images(loader, n_per_class: int = 1) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Select *n_per_class* images per class from *loader* (deterministic).

    Returns (images, labels, indices_in_dataset).
    """
    targets = []
    for _, y in loader:
        targets.extend(y.tolist())
    targets = np.array(targets)

    num_classes = int(targets.max()) + 1
    selected_indices = []
    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        selected_indices.extend(class_indices[:n_per_class].tolist())

    xs, ys = [], []
    collected = set()
    idx = 0
    for x_batch, y_batch in loader:
        for i in range(x_batch.size(0)):
            if idx in selected_indices and idx not in collected:
                xs.append(x_batch[i])
                ys.append(y_batch[i])
                collected.add(idx)
            idx += 1
        if len(collected) == len(selected_indices):
            break

    return torch.stack(xs), torch.stack(ys), selected_indices


def _overlay_heatmap(
    img_np: np.ndarray,
    heatmap_np: np.ndarray,
    alpha: float = 0.5,
    cmap=plt.cm.jet,
) -> np.ndarray:
    """Overlay a heatmap on an image using the given colormap."""
    colored = cmap(heatmap_np)[:, :, :3]
    return (1 - alpha) * img_np + alpha * colored


def generate_heatmap_grid(
    model: nn.Module,
    loader,
    cfg: dict,
    device: str,
    out_dir: Path,
    eps_255: int = 4,
    n_per_class: int = 1,
    class_names: list[str] | None = None,
) -> None:
    """Generate a grid of attention rollout heatmaps.

    Rows = selected images.
    Columns = [Original, Clean rollout, FGSM rollout, PGD rollout].
    """
    image_size = cfg["data"]["image_size"]
    grid_size = image_size // 16  # ViT patch size

    tqdm.write("Selecting images for qualitative grid…")
    imgs, labels, _ = _select_images(loader, n_per_class=n_per_class)
    n_images = imgs.size(0)
    imgs, labels = imgs.to(device), labels.to(device)

    eps_pix = eps_255 / 255.0
    pgd_steps = cfg["attacks"]["pgd"]["steps"]

    # Compute rollouts
    tqdm.write("Computing clean rollout for selected images…")
    maps_clean = get_attention_maps(model, imgs)
    rollout_clean = attention_rollout(maps_clean, discard_ratio=0.1)
    heatmap_clean = rollout_to_heatmap(rollout_clean, grid_size, image_size)

    tqdm.write(f"Computing FGSM ε={eps_255}/255 rollout…")
    with torch.enable_grad():
        x_fgsm = ATTACKS["fgsm"](model, imgs, labels, eps_pix=eps_pix)
    maps_fgsm = get_attention_maps(model, x_fgsm)
    rollout_fgsm = attention_rollout(maps_fgsm, discard_ratio=0.1)
    heatmap_fgsm = rollout_to_heatmap(rollout_fgsm, grid_size, image_size)

    tqdm.write(f"Computing PGD ε={eps_255}/255 rollout…")
    with torch.enable_grad():
        x_pgd = ATTACKS["pgd"](model, imgs, labels, eps_pix=eps_pix, steps=pgd_steps)
    maps_pgd = get_attention_maps(model, x_pgd)
    rollout_pgd = attention_rollout(maps_pgd, discard_ratio=0.1)
    heatmap_pgd = rollout_to_heatmap(rollout_pgd, grid_size, image_size)

    # Convert to numpy for plotting
    def to_np(t: torch.Tensor) -> np.ndarray:
        return denorm(t).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()

    imgs_np = to_np(imgs)
    fgsm_np = to_np(x_fgsm)
    pgd_np = to_np(x_pgd)
    hm_clean_np = heatmap_clean.squeeze(1).cpu().numpy()
    hm_fgsm_np = heatmap_fgsm.squeeze(1).cpu().numpy()
    hm_pgd_np = heatmap_pgd.squeeze(1).cpu().numpy()

    col_titles = ["Original", "Clean Rollout", f"FGSM ε={eps_255}", f"PGD ε={eps_255}"]
    n_cols = len(col_titles)

    fig, axes = plt.subplots(n_images, n_cols, figsize=(3.5 * n_cols, 3 * n_images))
    if n_images == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_images):
        lbl = labels[i].item()
        class_str = class_names[lbl] if class_names else str(lbl)

        axes[i, 0].imshow(imgs_np[i])
        axes[i, 0].set_ylabel(class_str, fontsize=9, rotation=0, labelpad=60, va="center")

        axes[i, 1].imshow(_overlay_heatmap(imgs_np[i], hm_clean_np[i]))
        axes[i, 2].imshow(_overlay_heatmap(fgsm_np[i], hm_fgsm_np[i]))
        axes[i, 3].imshow(_overlay_heatmap(pgd_np[i], hm_pgd_np[i]))

        # Compute per-image shift
        cos_fgsm = float(torch.nn.functional.cosine_similarity(
            rollout_clean[i:i+1], rollout_fgsm[i:i+1], dim=1
        ))
        cos_pgd = float(torch.nn.functional.cosine_similarity(
            rollout_clean[i:i+1], rollout_pgd[i:i+1], dim=1
        ))
        axes[i, 2].set_xlabel(f"shift={1 - cos_fgsm:.3f}", fontsize=7)
        axes[i, 3].set_xlabel(f"shift={1 - cos_pgd:.3f}", fontsize=7)

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10, fontweight="bold")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Attention Rollout — Clean vs Adversarial (ε={eps_255}/255)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    path = out_dir / f"attention_rollout_grid_eps{eps_255}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"Saved: {path}")


def plot_attention_shift_curves(
    rows: list[dict],
    out_dir: Path,
) -> None:
    """Plot attention shift (1 - cos_sim) and KL divergence vs epsilon."""
    import pandas as pd

    df = pd.DataFrame(rows)
    attacks = df["attack"].unique()

    colors = {
        "fgsm": "#1f77b4",
        "pgd": "#d62728",
        "patch_pgd": "#2ca02c",
        "salient_patch_pgd": "#ff7f0e",
    }

    # Attention shift plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for atk in attacks:
        sub = df[df["attack"] == atk].sort_values("eps_255")
        label = ATTACK_LABELS.get(atk, atk)
        c = colors.get(atk, "gray")

        axes[0].errorbar(
            sub["eps_255"], sub["attention_shift"],
            yerr=sub["std_cos_sim"],
            marker="o", linewidth=2, capsize=3,
            label=label, color=c,
        )
        axes[1].errorbar(
            sub["eps_255"], sub["mean_kl_div"],
            yerr=sub["std_kl_div"],
            marker="s", linewidth=2, capsize=3,
            label=label, color=c,
        )

    axes[0].set_xlabel("ε (in /255)")
    axes[0].set_ylabel("Attention Shift (1 − cos_sim)")
    axes[0].set_title("Attention Shift vs Perturbation Strength")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(bottom=-0.01)

    axes[1].set_xlabel("ε (in /255)")
    axes[1].set_ylabel("KL Divergence")
    axes[1].set_title("KL Divergence of Rollout Distribution")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(bottom=-0.01)

    fig.tight_layout()
    path = out_dir / "attention_shift_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    tqdm.write(f"Saved: {path}")


# ── Attack family builder ────────────────────────────────────────────────

def _build_attack_families(cfg: dict) -> list[tuple]:
    """Build the list of (label, key, fn, kwargs) from config."""
    return [
        ("FGSM", "fgsm", ATTACKS["fgsm"], {}),
        ("PGD", "pgd", ATTACKS["pgd"], {"steps": cfg["attacks"]["pgd"]["steps"]}),
        ("Patch-PGD", "patch_pgd", ATTACKS["patch_pgd"], {
            "steps": cfg["attacks"]["patch_pgd"]["steps"],
            "patch_size": cfg["attacks"]["patch_pgd"]["patch_size"],
            "position": cfg["attacks"]["patch_pgd"]["position"],
        }),
        ("Salient-patch PGD", "salient_patch_pgd", ATTACKS["salient_patch_pgd"], {
            "steps": cfg["attacks"]["salient_patch_pgd"]["steps"],
            "vit_patch_size": cfg["attacks"]["salient_patch_pgd"]["vit_patch_size"],
            "top_k": cfg["attacks"]["salient_patch_pgd"]["top_k"],
        }),
    ]


# ── Imagenette class names ───────────────────────────────────────────────

IMAGENETTE_CLASSES = [
    "tench", "English springer", "cassette player", "chain saw",
    "church", "French horn", "garbage truck", "gas pump",
    "golf ball", "parachute",
]


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Attention rollout analysis")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--qual-eps", type=int, default=4,
                        help="ε (in /255) for qualitative heatmap grid")
    parser.add_argument("--n-per-class", type=int, default=1,
                        help="Images per class in the qualitative grid")
    parser.add_argument("--figures-dir", type=str, default="figures/attention")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = cfg["device"]

    _, val_loader, num_classes = get_loaders(
        cfg["data"]["path"],
        batch_size=cfg["data"]["batch_size"],
        workers=cfg["data"]["workers"],
        image_size=cfg["data"]["image_size"],
    )

    cfg["model"]["num_classes"] = num_classes
    model = build_model(cfg["model"], pretrained=False)
    load_checkpoint(model, args.checkpoint)
    model = model.to(device).eval()

    # Verify class names match dataset ordering
    ds_classes = val_loader.dataset.classes
    class_names = IMAGENETTE_CLASSES
    if len(ds_classes) != len(class_names):
        tqdm.write(f"Warning: dataset has {len(ds_classes)} classes, expected {len(class_names)}")
        class_names = ds_classes

    # ── Output directories ────────────────────────────────────────────
    strategy = cfg["model"].get("strategy", "linear_probe")
    strategy_short = "lp" if strategy == "linear_probe" else "ft"
    arch = cfg["model"].get("arch", "vit")
    out = run_dir(f"attention/{arch}_{strategy_short}", base=cfg["output_dir"])

    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Quantitative analysis ─────────────────────────────────────────
    tqdm.write("=" * 60)
    tqdm.write("  Quantitative attention shift analysis")
    tqdm.write("=" * 60)
    rows = quantitative_analysis(model, val_loader, cfg, device)

    fieldnames = [
        "attack", "eps_255", "mean_cos_sim", "std_cos_sim",
        "mean_kl_div", "std_kl_div", "attention_shift",
    ]
    save_csv(out / "attention_metrics.csv", rows, fieldnames=fieldnames)
    tqdm.write(f"\nMetrics saved: {out / 'attention_metrics.csv'}")

    # ── Shift curves plot ─────────────────────────────────────────────
    plot_attention_shift_curves(rows, fig_dir)

    # ── Qualitative heatmaps ──────────────────────────────────────────
    tqdm.write("\n" + "=" * 60)
    tqdm.write("  Qualitative attention rollout heatmaps")
    tqdm.write("=" * 60)
    generate_heatmap_grid(
        model, val_loader, cfg, device, fig_dir,
        eps_255=args.qual_eps,
        n_per_class=args.n_per_class,
        class_names=class_names,
    )

    # ── Metadata ──────────────────────────────────────────────────────
    save_metadata(
        out / "metadata.json",
        seed=cfg["seed"],
        checkpoint=args.checkpoint,
        model=cfg["model"],
        attacks_config=cfg["attacks"],
        qualitative_eps_255=args.qual_eps,
        n_per_class=args.n_per_class,
        device=device,
        torch_version=torch.__version__,
    )
    tqdm.write(f"\nAll outputs saved in: {out}")


if __name__ == "__main__":
    main()
