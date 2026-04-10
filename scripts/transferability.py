#!/usr/bin/env python3
"""Cross-architecture adversarial transferability analysis.

Generates adversarial examples with a source model and evaluates them on a
target model, measuring how well attacks transfer across architectures.

The key metric is *effective transfer rate*: of the examples that fool the
source model, what fraction also fools the target?  This isolates the
transferability signal from the baseline attack success rate.

Outputs
-------
1. CSV with per-(source, target, attack, epsilon) transfer metrics.
2. Summary table printed to stdout.

Usage
-----
    python scripts/transferability.py \\
        --config configs/default.yaml \\
        --models vit:configs/default.yaml:experiments/train/20260409_175532/model.pt \\
                 resnet50:configs/resnet50.yaml:<resnet50_checkpoint> \\
                 resnet18:configs/resnet18.yaml:<resnet18_checkpoint>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import build_model, load_checkpoint
from src.attacks import ATTACKS
from src.data import get_loaders
from src.utils import print_table, run_dir, save_csv, save_metadata, set_seed


def evaluate_transferability(
    source_model: nn.Module,
    target_model: nn.Module,
    loader,
    attack_fn,
    device: str = "cuda",
    **attack_kwargs,
) -> dict:
    """Generate adversarial examples with source_model, evaluate on target_model.

    Returns dict with:
        source_acc     : accuracy of source model on adversarial examples
        target_acc     : accuracy of target model on adversarial examples
        transfer_rate  : 1 - target_acc (raw)
        effective_transfer : fraction of source-fooled examples that also fool target
        n_source_fooled: count of examples that fool the source
        n_total        : total examples evaluated
    """
    source_model.eval()
    target_model.eval()

    correct_source = 0
    correct_target = 0
    both_fooled = 0
    source_fooled_total = 0
    n = 0

    for x, y in tqdm(loader, desc="transfer", leave=False):
        x, y = x.to(device), y.to(device)
        with torch.enable_grad():
            x_adv = attack_fn(source_model, x, y, **attack_kwargs)

        with torch.no_grad():
            pred_s = source_model(x_adv).argmax(dim=1)
            pred_t = target_model(x_adv).argmax(dim=1)

            s_correct = (pred_s == y)
            t_correct = (pred_t == y)

            correct_source += s_correct.sum().item()
            correct_target += t_correct.sum().item()

            s_fooled = ~s_correct
            t_fooled = ~t_correct
            both_fooled += (s_fooled & t_fooled).sum().item()
            source_fooled_total += s_fooled.sum().item()

            n += y.size(0)

    source_acc = correct_source / n
    target_acc = correct_target / n
    effective = both_fooled / source_fooled_total if source_fooled_total > 0 else 0.0

    return {
        "source_acc": round(source_acc, 6),
        "target_acc": round(target_acc, 6),
        "transfer_rate": round(1.0 - target_acc, 6),
        "effective_transfer": round(effective, 6),
        "n_source_fooled": source_fooled_total,
        "n_total": n,
    }


MODEL_LABELS = {
    "vit": "ViT-Base/16",
    "resnet50": "ResNet-50",
    "resnet18": "ResNet-18",
}


def _parse_model_spec(spec: str) -> tuple[str, str, str]:
    """Parse 'name:config_path:checkpoint_path' into (name, config, ckpt)."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Model spec must be 'name:config:checkpoint', got: {spec!r}"
        )
    return parts[0], parts[1], parts[2]


def _load_model(config_path: str, checkpoint_path: str, device: str) -> nn.Module:
    """Load a model from config + checkpoint."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    _, val_loader, num_classes = get_loaders(
        cfg["data"]["path"],
        batch_size=1, workers=0,
        image_size=cfg["data"]["image_size"],
    )
    cfg["model"]["num_classes"] = num_classes
    model = build_model(cfg["model"], pretrained=False)
    load_checkpoint(model, checkpoint_path)
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-architecture transferability")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Base config (for data/attacks/seed)")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="Model specs: name:config:checkpoint (at least 2)")
    parser.add_argument("--eps-subset", type=int, nargs="+", default=[2, 4, 8],
                        help="Epsilon values (/255) to test")
    parser.add_argument("--figures-dir", type=str, default="figures/transferability")
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

    model_specs = [_parse_model_spec(s) for s in args.models]
    tqdm.write(f"Loading {len(model_specs)} models…")
    models: dict[str, nn.Module] = {}
    for name, config_path, ckpt_path in model_specs:
        tqdm.write(f"  {name}: {ckpt_path}")
        models[name] = _load_model(config_path, ckpt_path, device)

    attack_families = [
        ("FGSM", "fgsm", ATTACKS["fgsm"], {}),
        ("PGD",  "pgd",  ATTACKS["pgd"],  {"steps": cfg["attacks"]["pgd"]["steps"]}),
    ]

    out = run_dir("transferability", base=cfg["output_dir"])
    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    tqdm.write("\n" + "=" * 70)
    tqdm.write("  Cross-architecture adversarial transferability")
    tqdm.write("=" * 70)

    model_names = list(models.keys())
    for source_name in model_names:
        for target_name in model_names:
            if source_name == target_name:
                continue

            source_label = MODEL_LABELS.get(source_name, source_name)
            target_label = MODEL_LABELS.get(target_name, target_name)

            for atk_label, atk_key, atk_fn, atk_kwargs in attack_families:
                for eps_255 in args.eps_subset:
                    eps_pix = eps_255 / 255.0
                    tqdm.write(
                        f"\n  {source_label} → {target_label}  "
                        f"{atk_label}  ε={eps_255}/255"
                    )

                    result = evaluate_transferability(
                        models[source_name],
                        models[target_name],
                        val_loader,
                        atk_fn,
                        device=device,
                        eps_pix=eps_pix,
                        **atk_kwargs,
                    )

                    row = {
                        "source": source_name,
                        "target": target_name,
                        "attack": atk_key,
                        "eps_255": eps_255,
                        **result,
                    }
                    all_rows.append(row)

                    tqdm.write(
                        f"    source_acc={result['source_acc']:.4f}  "
                        f"target_acc={result['target_acc']:.4f}  "
                        f"eff_transfer={result['effective_transfer']:.4f}"
                    )

    # ── Save results ──────────────────────────────────────────────────
    fieldnames = [
        "source", "target", "attack", "eps_255",
        "source_acc", "target_acc", "transfer_rate",
        "effective_transfer", "n_source_fooled", "n_total",
    ]
    save_csv(out / "transferability.csv", all_rows, fieldnames=fieldnames)
    tqdm.write(f"\nResults saved: {out / 'transferability.csv'}")

    tqdm.write("\n" + "=" * 70)
    tqdm.write("  Summary Table")
    tqdm.write("=" * 70)
    print_table(all_rows, fieldnames=fieldnames)

    # ── Metadata ──────────────────────────────────────────────────────
    model_info = {
        name: {"config": cfg_path, "checkpoint": ckpt}
        for name, cfg_path, ckpt in model_specs
    }
    save_metadata(
        out / "metadata.json",
        seed=cfg["seed"],
        models=model_info,
        attacks=["fgsm", "pgd"],
        eps_subset=args.eps_subset,
        device=device,
        torch_version=torch.__version__,
    )
    tqdm.write(f"\nAll outputs saved in: {out}")


if __name__ == "__main__":
    main()
