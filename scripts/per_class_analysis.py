#!/usr/bin/env python3
"""Per-class analysis of adversarial robustness decay.

Computes per-class accuracy and confidence under FGSM and PGD attacks at
selected epsilon values, generates confusion-matrix heatmaps and per-class
bar charts to reveal which classes are most/least vulnerable.

Outputs
-------
1. CSV with per-(attack, epsilon, class) accuracy and confidence.
2. Confusion-matrix heatmaps for PGD at selected epsilon values.
3. Per-class accuracy bar charts.
4. Vulnerability ranking summary.

Usage
-----
    python scripts/per_class_analysis.py \\
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
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import build_model, load_checkpoint
from src.attacks import ATTACKS
from src.data import get_loaders
from src.eval import evaluate_per_class
from src.utils import run_dir, save_csv, save_metadata, set_seed


IMAGENETTE_CLASSES = [
    "tench", "English springer", "cassette player", "chain saw",
    "church", "French horn", "garbage truck", "gas pump",
    "golf ball", "parachute",
]

ATTACK_LABELS = {
    "fgsm": "FGSM",
    "pgd": "PGD-10",
}


# ── Plotting helpers ──────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    out_path: Path,
) -> None:
    """Confusion matrix heatmap with row-normalised percentages."""
    n = len(class_names)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction")

    ax.set(
        xticks=range(n),
        yticks=range(n),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            ax.text(
                j, i, f"{val:.0%}",
                ha="center", va="center",
                color="white" if val > 0.5 else "black",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"Saved: {out_path}")


def plot_per_class_accuracy_bars(
    rows: list[dict],
    class_names: list[str],
    eps_values: list[int],
    attack_key: str,
    out_path: Path,
) -> None:
    """Grouped bar chart of per-class accuracy across epsilon values."""
    n_classes = len(class_names)
    n_eps = len(eps_values)
    x = np.arange(n_classes)
    width = 0.8 / n_eps

    cmap = plt.cm.viridis
    colors = [cmap(i / max(n_eps - 1, 1)) for i in range(n_eps)]

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, eps in enumerate(eps_values):
        accs = []
        for c in range(n_classes):
            matching = [
                r for r in rows
                if r["attack"] == attack_key
                and r["eps_255"] == eps
                and r["class_idx"] == c
            ]
            accs.append(matching[0]["acc"] if matching else 0.0)

        offset = (i - n_eps / 2 + 0.5) * width
        bars = ax.bar(x + offset, accs, width, label=f"ε={eps}/255", color=colors[i])
        for bar, acc in zip(bars, accs):
            if acc < 0.5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{acc:.0%}", ha="center", va="bottom", fontsize=5.5,
                )

    attack_label = ATTACK_LABELS.get(attack_key, attack_key)
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-Class Accuracy under {attack_label}")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"Saved: {out_path}")


def plot_vulnerability_ranking(
    rows: list[dict],
    class_names: list[str],
    attack_key: str,
    eps_255: int,
    out_path: Path,
) -> None:
    """Horizontal bar chart ranking classes by accuracy drop (clean − attacked)."""
    clean_accs = {}
    atk_accs = {}
    for r in rows:
        if r["attack"] == attack_key and r["eps_255"] == 0:
            clean_accs[r["class_idx"]] = r["acc"]
        elif r["attack"] == attack_key and r["eps_255"] == eps_255:
            atk_accs[r["class_idx"]] = r["acc"]

    drops = []
    for c in range(len(class_names)):
        drop = clean_accs.get(c, 1.0) - atk_accs.get(c, 0.0)
        drops.append((class_names[c], drop, atk_accs.get(c, 0.0)))

    drops.sort(key=lambda t: t[1], reverse=True)
    names = [d[0] for d in drops]
    drop_vals = [d[1] for d in drops]
    remaining = [d[2] for d in drops]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(names))

    bars_drop = ax.barh(y_pos, drop_vals, color="#d62728", alpha=0.8, label="Accuracy drop")
    bars_rem = ax.barh(y_pos, remaining, left=drop_vals, color="#2ca02c", alpha=0.6, label="Remaining accuracy")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy")
    attack_label = ATTACK_LABELS.get(attack_key, attack_key)
    ax.set_title(f"Vulnerability Ranking — {attack_label} ε={eps_255}/255")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)

    for i, (drop, rem) in enumerate(zip(drop_vals, remaining)):
        ax.text(drop + rem + 0.01, i, f"{rem:.0%}", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"Saved: {out_path}")


# ── Main analysis ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Per-class adversarial robustness analysis")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eps-subset", type=int, nargs="+", default=[0, 2, 4, 8],
                        help="Subset of eps_255 values to analyse (default: 0 2 4 8)")
    parser.add_argument("--cm-eps", type=int, nargs="+", default=[4, 8],
                        help="Epsilon values for confusion matrix plots")
    parser.add_argument("--figures-dir", type=str, default="figures/per_class")
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

    ds_classes = val_loader.dataset.classes
    class_names = IMAGENETTE_CLASSES
    if len(ds_classes) != len(class_names):
        tqdm.write(f"Warning: dataset has {len(ds_classes)} classes, expected {len(class_names)}")
        class_names = list(ds_classes)

    # Count samples per class for reference
    targets = []
    for _, y in val_loader:
        targets.extend(y.tolist())
    class_counts = np.bincount(targets, minlength=num_classes)
    tqdm.write(f"Samples per class: {dict(zip(class_names, class_counts.tolist()))}")

    strategy = cfg["model"].get("strategy", "linear_probe")
    strategy_short = "lp" if strategy == "linear_probe" else "ft"
    arch = cfg["model"].get("arch", "vit")
    out = run_dir(f"per_class/{arch}_{strategy_short}", base=cfg["output_dir"])

    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    attack_families = [
        ("fgsm", ATTACKS["fgsm"], {}),
        ("pgd", ATTACKS["pgd"], {"steps": cfg["attacks"]["pgd"]["steps"]}),
    ]

    eps_subset = args.eps_subset
    all_rows: list[dict] = []
    confusion_matrices: dict[tuple[str, int], np.ndarray] = {}

    tqdm.write("=" * 60)
    tqdm.write("  Per-class adversarial robustness analysis")
    tqdm.write("=" * 60)

    for atk_key, atk_fn, atk_kwargs in attack_families:
        atk_label = ATTACK_LABELS.get(atk_key, atk_key)
        for eps_255 in eps_subset:
            eps_pix = eps_255 / 255.0
            tqdm.write(f"\n  {atk_label}  ε={eps_255}/255")

            if eps_255 == 0:
                result = evaluate_per_class(
                    model, val_loader, num_classes, device=device,
                )
            else:
                result = evaluate_per_class(
                    model, val_loader, num_classes, device=device,
                    attack=atk_fn, eps_pix=eps_pix, **atk_kwargs,
                )

            confusion_matrices[(atk_key, eps_255)] = result["confusion_matrix"]

            overall_acc = float((result["all_preds"] == result["all_labels"]).mean())
            tqdm.write(f"    Overall acc: {overall_acc:.4f}")

            for c in range(num_classes):
                row = {
                    "attack": atk_key,
                    "eps_255": eps_255,
                    "class_idx": c,
                    "class_name": class_names[c],
                    "acc": round(result["per_class_acc"][c], 6),
                    "mean_conf": round(result["per_class_conf"][c], 6),
                    "n_samples": int(class_counts[c]),
                }
                all_rows.append(row)

            # Print per-class summary
            accs = result["per_class_acc"]
            most_robust = class_names[int(np.argmax(accs))]
            most_vulnerable = class_names[int(np.argmin(accs))]
            tqdm.write(f"    Most robust:     {most_robust} ({max(accs):.4f})")
            tqdm.write(f"    Most vulnerable: {most_vulnerable} ({min(accs):.4f})")

    # ── Save CSV ──────────────────────────────────────────────────────
    fieldnames = ["attack", "eps_255", "class_idx", "class_name", "acc", "mean_conf", "n_samples"]
    save_csv(out / "per_class_metrics.csv", all_rows, fieldnames=fieldnames)
    tqdm.write(f"\nMetrics saved: {out / 'per_class_metrics.csv'}")

    # ── Confusion matrices ────────────────────────────────────────────
    tqdm.write("\n" + "=" * 60)
    tqdm.write("  Generating confusion matrix heatmaps")
    tqdm.write("=" * 60)

    for atk_key, _, _ in attack_families:
        atk_label = ATTACK_LABELS.get(atk_key, atk_key)
        for eps_255 in args.cm_eps:
            key = (atk_key, eps_255)
            if key in confusion_matrices:
                plot_confusion_matrix(
                    confusion_matrices[key],
                    class_names,
                    f"Confusion Matrix — {atk_label} ε={eps_255}/255",
                    fig_dir / f"confusion_matrix_{atk_key}_eps{eps_255}.png",
                )

    # Also generate clean confusion matrix for reference
    clean_key = ("fgsm", 0)
    if clean_key in confusion_matrices:
        plot_confusion_matrix(
            confusion_matrices[clean_key],
            class_names,
            "Confusion Matrix — Clean (no attack)",
            fig_dir / "confusion_matrix_clean.png",
        )

    # ── Per-class accuracy bar charts ─────────────────────────────────
    tqdm.write("\n" + "=" * 60)
    tqdm.write("  Generating per-class bar charts")
    tqdm.write("=" * 60)

    for atk_key, _, _ in attack_families:
        plot_per_class_accuracy_bars(
            all_rows, class_names, eps_subset, atk_key,
            fig_dir / f"per_class_accuracy_{atk_key}.png",
        )

    # ── Vulnerability ranking ─────────────────────────────────────────
    for atk_key, _, _ in attack_families:
        for eps_255 in args.cm_eps:
            plot_vulnerability_ranking(
                all_rows, class_names, atk_key, eps_255,
                fig_dir / f"vulnerability_ranking_{atk_key}_eps{eps_255}.png",
            )

    # ── Per-class decay curves (accuracy vs epsilon per class) ────────
    tqdm.write("\n" + "=" * 60)
    tqdm.write("  Generating per-class decay curves")
    tqdm.write("=" * 60)

    for atk_key, _, _ in attack_families:
        atk_label = ATTACK_LABELS.get(atk_key, atk_key)
        fig, ax = plt.subplots(figsize=(10, 6))

        cmap = plt.cm.tab10
        for c in range(num_classes):
            class_rows = [
                r for r in all_rows
                if r["attack"] == atk_key and r["class_idx"] == c
            ]
            class_rows.sort(key=lambda r: r["eps_255"])
            eps_vals = [r["eps_255"] for r in class_rows]
            acc_vals = [r["acc"] for r in class_rows]
            ax.plot(eps_vals, acc_vals, marker="o", linewidth=1.5,
                    color=cmap(c / num_classes), label=class_names[c])

        ax.set_xlabel("ε (in /255)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Per-Class Accuracy Decay — {atk_label}")
        ax.legend(fontsize=7, ncol=2, loc="lower left")
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.02, 1.05)

        path = fig_dir / f"per_class_decay_{atk_key}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        tqdm.write(f"Saved: {path}")

    # ── Metadata ──────────────────────────────────────────────────────
    save_metadata(
        out / "metadata.json",
        seed=cfg["seed"],
        checkpoint=args.checkpoint,
        model=cfg["model"],
        attacks_analysed=list(ATTACK_LABELS.keys()),
        eps_subset=eps_subset,
        cm_eps=args.cm_eps,
        class_names=class_names,
        samples_per_class=class_counts.tolist(),
        device=device,
        torch_version=torch.__version__,
    )
    tqdm.write(f"\nAll outputs saved in: {out}")


if __name__ == "__main__":
    main()
