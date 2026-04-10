#!/usr/bin/env python3
"""Generate decay-profile plots, comparative overlays, and qualitative grids.

Single-model mode (original):
    python scripts/visualize.py --metrics experiments/eval/<ts>/metrics.csv

Comparative mode (Point A):
    python scripts/visualize.py \\
        --metrics experiments/eval/vit_lp/<ts>/metrics.csv \\
        --metrics-2 experiments/eval/resnet50_lp/<ts>/metrics.csv \\
        --label-1 "ViT-Base/16" --label-2 "ResNet-50"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attacks import ATTACKS
from src.data import denorm, get_loaders
from src import build_model, load_checkpoint

# ── Style constants ───────────────────────────────────────────────────────

ATTACK_LABELS = {
    "fgsm": "FGSM",
    "pgd": "PGD-10",
    "patch_pgd": "Patch-PGD (centre)",
    "salient_patch_pgd": "Salient-patch PGD",
}
ATTACK_COLORS = {
    "fgsm": "#1f77b4",
    "pgd": "#d62728",
    "patch_pgd": "#2ca02c",
    "salient_patch_pgd": "#ff7f0e",
}


# ── Metric helpers ────────────────────────────────────────────────────────

def compute_audc(eps_values: np.ndarray, metric_values: np.ndarray) -> float:
    """Area Under the Decay Curve via trapezoidal integration.

    Normalised to [0, 1] by dividing by the epsilon range so that AUDC
    represents the average metric value across the epsilon sweep.
    """
    if len(eps_values) < 2:
        return float(metric_values[0]) if len(metric_values) else 0.0
    area = float(np.trapezoid(metric_values, eps_values))
    span = float(eps_values[-1] - eps_values[0])
    return area / span if span > 0 else 0.0


# ── Single-model plots (unchanged API) ───────────────────────────────────

def plot_accuracy_decay(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for atk in df["attack"].unique():
        if atk == "clean":
            continue
        sub = df[df["attack"] == atk]
        ax.plot(
            sub["eps_255"], sub["acc"],
            marker="o", linewidth=2,
            label=ATTACK_LABELS.get(atk, atk),
            color=ATTACK_COLORS.get(atk),
        )

    clean_acc = df.loc[df["attack"] == "clean", "acc"].values
    if len(clean_acc):
        ax.axhline(clean_acc[0], color="gray", linestyle="--", linewidth=1, label="Clean")

    ax.set_xlabel("ε (in /255)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy decay under evasion attacks")
    ax.legend()
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_confidence_shift(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for atk in df["attack"].unique():
        if atk == "clean":
            continue
        sub = df[df["attack"] == atk]
        c = ATTACK_COLORS.get(atk)
        label = ATTACK_LABELS.get(atk, atk)
        axes[0].plot(sub["eps_255"], sub["avg_conf_correct"], marker="o", linewidth=2, color=c, label=label)
        axes[1].plot(sub["eps_255"], sub["avg_conf_wrong"], marker="s", linewidth=2, color=c, label=label)

    axes[0].set_title("Confidence on correct predictions")
    axes[1].set_title("Confidence on wrong predictions")
    for ax in axes:
        ax.set_xlabel("ε (in /255)")
        ax.set_ylabel("Mean confidence")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_confidence_gap(df: pd.DataFrame, out_path: Path) -> None:
    """Plot (avg_conf_wrong - avg_conf_correct) vs epsilon — overconfidence."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    clean = df[df["attack"] == "clean"]
    if not clean.empty:
        clean_gap = clean["avg_conf_wrong"].values[0] - clean["avg_conf_correct"].values[0]
        ax.axhline(clean_gap, color="gray", linestyle="--", linewidth=1, label="Clean")

    for atk in df["attack"].unique():
        if atk == "clean":
            continue
        sub = df[df["attack"] == atk]
        gap = sub["avg_conf_wrong"] - sub["avg_conf_correct"]
        ax.plot(
            sub["eps_255"], gap,
            marker="o", linewidth=2,
            label=ATTACK_LABELS.get(atk, atk),
            color=ATTACK_COLORS.get(atk),
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("ε (in /255)")
    ax.set_ylabel("conf_wrong − conf_correct")
    ax.set_title("Confidence gap: wrong vs correct predictions")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Comparative plots (Point A) ──────────────────────────────────────────

_MODEL_STYLES = {
    0: {"linestyle": "-",  "marker": "o"},
    1: {"linestyle": "--", "marker": "s"},
}


def _prepend_clean(df: pd.DataFrame, atk: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (eps_255, metric_values) for an attack, prepending the clean point at ε=0."""
    clean_row = df[df["attack"] == "clean"]
    atk_rows = df[df["attack"] == atk].sort_values("eps_255")
    if not clean_row.empty:
        eps = np.concatenate([[0], atk_rows["eps_255"].values])
        vals = np.concatenate([[clean_row[metric].values[0]], atk_rows[metric].values])
    else:
        eps = atk_rows["eps_255"].values
        vals = atk_rows[metric].values
    return eps, vals


def plot_comparative_decay(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    metric: str,
    out_path: Path,
    label_1: str = "Model 1",
    label_2: str = "Model 2",
    ylabel: str | None = None,
    title: str | None = None,
) -> None:
    """Overlay decay curves of two models, one subplot per attack.

    Each curve starts from ε=0 (clean accuracy) so the full decay profile
    from baseline to maximum perturbation is always visible.
    """
    attacks = [a for a in df1["attack"].unique() if a != "clean"]
    n = len(attacks)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)

    for idx, atk in enumerate(attacks):
        ax = axes[idx // cols][idx % cols]
        for i, (df, label) in enumerate([(df1, label_1), (df2, label_2)]):
            eps, vals = _prepend_clean(df, atk, metric)
            style = _MODEL_STYLES[i]
            ax.plot(
                eps, vals,
                linewidth=2, label=label,
                color=ATTACK_COLORS.get(atk, f"C{i}"),
                alpha=1.0 if i == 0 else 0.7,
                **style,
            )
        ax.set_title(ATTACK_LABELS.get(atk, atk))
        ax.set_xlabel("ε (in /255)")
        ax.set_ylabel(ylabel or metric)
        ax.legend(fontsize=8)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(title or f"Comparative {metric} decay", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_comparative_summary(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    out_path: Path,
    label_1: str = "Model 1",
    label_2: str = "Model 2",
) -> None:
    """Single plot: all attacks, both models — accuracy vs epsilon.

    Each curve starts from ε=0 (clean accuracy).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (df, label, ls) in enumerate([(df1, label_1, "-"), (df2, label_2, "--")]):
        marker = "o" if i == 0 else "s"
        for atk in df["attack"].unique():
            if atk == "clean":
                continue
            eps, vals = _prepend_clean(df, atk, "acc")
            ax.plot(
                eps, vals,
                linestyle=ls, marker=marker, markersize=5,
                linewidth=2, alpha=1.0 if i == 0 else 0.7,
                color=ATTACK_COLORS.get(atk),
                label=f"{ATTACK_LABELS.get(atk, atk)} ({label})",
            )

    ax.set_xlabel("ε (in /255)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy decay: {label_1} vs {label_2}")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_audc_table(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    out_path: Path,
    label_1: str = "Model 1",
    label_2: str = "Model 2",
) -> pd.DataFrame:
    """Compute AUDC for each (model, attack) pair and save as CSV."""
    rows = []
    for df, label in [(df1, label_1), (df2, label_2)]:
        for atk in df["attack"].unique():
            if atk == "clean":
                continue
            sub = df[df["attack"] == atk].sort_values("eps_255")
            audc_acc = compute_audc(sub["eps_255"].values, sub["acc"].values)
            audc_conf_wrong = compute_audc(sub["eps_255"].values, sub["avg_conf_wrong"].values)
            rows.append({
                "model": label,
                "attack": ATTACK_LABELS.get(atk, atk),
                "audc_accuracy": round(audc_acc, 4),
                "audc_conf_wrong": round(audc_conf_wrong, 4),
            })

    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(result.to_string(index=False))
    return result


# ── Qualitative grid ─────────────────────────────────────────────────────

def generate_qualitative_grid(
    cfg: dict,
    checkpoint: str,
    out_path: Path,
    n_images: int = 8,
    eps_255: int = 8,
) -> None:
    """Create a grid: rows = images, cols = clean / FGSM / PGD / patch-PGD."""
    device = cfg["device"]
    _, val_loader, num_classes = get_loaders(
        cfg["data"]["path"],
        batch_size=n_images,
        workers=0,
        image_size=cfg["data"]["image_size"],
    )

    cfg["model"]["num_classes"] = num_classes
    model = build_model(cfg["model"], pretrained=False)
    load_checkpoint(model, checkpoint)
    model = model.to(device).eval()

    x, y = next(iter(val_loader))
    x, y = x[:n_images].to(device), y[:n_images].to(device)
    eps = eps_255 / 255.0

    attack_keys = ["fgsm", "pgd", "patch_pgd"]
    attack_kwargs = {
        "fgsm": {"eps_pix": eps},
        "pgd": {"eps_pix": eps, "steps": cfg["attacks"]["pgd"]["steps"]},
        "patch_pgd": {
            "eps_pix": eps,
            "steps": cfg["attacks"]["patch_pgd"]["steps"],
            "patch_size": cfg["attacks"]["patch_pgd"]["patch_size"],
            "position": cfg["attacks"]["patch_pgd"]["position"],
        },
    }

    cols = ["Clean"] + [ATTACK_LABELS[k] for k in attack_keys]
    n_cols = len(cols)

    fig, axes = plt.subplots(n_images, n_cols, figsize=(3 * n_cols, 3 * n_images))
    if n_images == 1:
        axes = axes[np.newaxis, :]

    def to_numpy(t: torch.Tensor) -> np.ndarray:
        if t.dim() == 3:
            t = t.unsqueeze(0)
        img = denorm(t).squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        return img

    def annotate(ax, img_tensor, model, label_true):
        with torch.no_grad():
            logits = model(img_tensor.unsqueeze(0))
            prob = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            conf = prob[0, pred].item()
        colour = "green" if pred == label_true else "red"
        ax.set_title(f"pred={pred} ({conf:.2f})", fontsize=8, color=colour)

    for i in range(n_images):
        axes[i, 0].imshow(to_numpy(x[i]))
        axes[i, 0].set_ylabel(f"true={y[i].item()}", fontsize=8)
        annotate(axes[i, 0], x[i], model, y[i].item())

        for j, key in enumerate(attack_keys, start=1):
            x_adv = ATTACKS[key](model, x[i:i+1], y[i:i+1], **attack_kwargs[key])
            axes[i, j].imshow(to_numpy(x_adv[0]))
            annotate(axes[i, j], x_adv[0], model, y[i].item())

    for j, title in enumerate(cols):
        current = axes[0, j].get_title()
        axes[0, j].set_title(f"{title}\n{current}", fontsize=8)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Adversarial examples (ε = {eps_255}/255)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, required=True, help="Path to metrics.csv (model 1)")
    parser.add_argument("--metrics-2", type=str, default=None, help="Path to metrics.csv (model 2, for comparative plots)")
    parser.add_argument("--label-1", type=str, default="ViT-Base/16", help="Legend label for model 1")
    parser.add_argument("--label-2", type=str, default="ResNet-50", help="Legend label for model 2")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model.pt (for qualitative grid)")
    parser.add_argument("--out-dir", type=str, default="figures")
    parser.add_argument("--qual-eps", type=int, default=8, help="ε (in /255) for qualitative grid")
    parser.add_argument("--qual-n", type=int, default=8, help="Number of images in qualitative grid")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df1 = pd.read_csv(args.metrics)

    # Single-model plots (always generated)
    plot_accuracy_decay(df1, out / "decay_accuracy.png")
    plot_confidence_shift(df1, out / "decay_confidence.png")
    plot_confidence_gap(df1, out / "decay_confidence_gap.png")

    # Comparative plots (only if --metrics-2 provided)
    if args.metrics_2:
        df2 = pd.read_csv(args.metrics_2)
        comp_dir = out / "comparative"
        comp_dir.mkdir(parents=True, exist_ok=True)

        plot_comparative_summary(
            df1, df2, comp_dir / "comparative_accuracy_summary.png",
            label_1=args.label_1, label_2=args.label_2,
        )
        plot_comparative_decay(
            df1, df2, "acc", comp_dir / "comparative_accuracy_per_attack.png",
            label_1=args.label_1, label_2=args.label_2,
            ylabel="Accuracy", title="Accuracy decay by attack",
        )
        plot_comparative_decay(
            df1, df2, "avg_conf_wrong", comp_dir / "comparative_overconfidence.png",
            label_1=args.label_1, label_2=args.label_2,
            ylabel="Mean confidence (wrong predictions)",
            title="Overconfidence on wrong predictions",
        )
        generate_audc_table(
            df1, df2, comp_dir / "audc_table.csv",
            label_1=args.label_1, label_2=args.label_2,
        )

    if args.checkpoint:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        generate_qualitative_grid(
            cfg, args.checkpoint, out / "qualitative_grid.png",
            n_images=args.qual_n, eps_255=args.qual_eps,
        )


if __name__ == "__main__":
    main()
