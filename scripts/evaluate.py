#!/usr/bin/env python3
"""Run evasion attacks over an epsilon grid and record the decay profile.

Architecture is determined by the ``model.arch`` field in the YAML config.
All attack parameters, epsilon grid, and metrics are shared across
architectures so that results are directly comparable.

Usage
-----
    python scripts/evaluate.py --config configs/default.yaml \\
        --checkpoint experiments/train/vit_lp/<ts>/model.pt

    python scripts/evaluate.py --config configs/resnet50.yaml \\
        --checkpoint experiments/train/resnet50_lp/<ts>/model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import build_model, load_checkpoint
from src.attacks import ATTACKS
from src.data import get_loaders
from src.eval import evaluate, evaluate_decay
from src.utils import print_table, run_dir, save_csv, save_metadata, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model.pt")
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
    arch = cfg["model"].get("arch", "vit")
    strategy = cfg["model"].get("strategy", "linear_probe")

    model = build_model(cfg["model"], pretrained=False)
    load_checkpoint(model, args.checkpoint)
    model = model.to(device)

    eps_list = [e / 255.0 for e in cfg["attacks"]["eps_255"]]
    cal = cfg["eval"].get("compute_calibration", False)

    all_rows = []

    tqdm.write("─── Clean baseline ───")
    clean = evaluate(model, val_loader, device=device, compute_calibration=cal)
    clean_row = {"attack": "clean", "eps_255": 0, "eps": 0.0, **clean}
    all_rows.append(clean_row)
    tqdm.write(f"  acc={clean['acc']:.4f}  nll={clean['nll']:.4f}")

    attack_families = [
        ("FGSM",               "fgsm",              ATTACKS["fgsm"],              {}),
        ("PGD",                "pgd",               ATTACKS["pgd"],               {"steps": cfg["attacks"]["pgd"]["steps"]}),
        ("Patch-PGD (centre)", "patch_pgd",         ATTACKS["patch_pgd"],         {"steps": cfg["attacks"]["patch_pgd"]["steps"], "patch_size": cfg["attacks"]["patch_pgd"]["patch_size"], "position": cfg["attacks"]["patch_pgd"]["position"]}),
        ("Salient-patch PGD",  "salient_patch_pgd", ATTACKS["salient_patch_pgd"], {"steps": cfg["attacks"]["salient_patch_pgd"]["steps"], "vit_patch_size": cfg["attacks"]["salient_patch_pgd"]["vit_patch_size"], "top_k": cfg["attacks"]["salient_patch_pgd"]["top_k"]}),
    ]

    family_bar = tqdm(attack_families, desc="attack families", unit="family", position=0)
    for label, key, attack_fn, kwargs in family_bar:
        family_bar.set_description(f"family: {label}")
        tqdm.write(f"\n─── {label} ───")
        rows = evaluate_decay(
            model, val_loader, attack_fn, eps_list[1:], key,
            device=device, compute_calibration=cal,
            **kwargs,
        )
        all_rows.extend(rows)

    strategy_short = "lp" if strategy == "linear_probe" else "ft"
    out = run_dir(f"eval/{arch}_{strategy_short}", base=cfg["output_dir"])
    fieldnames = ["attack", "eps_255", "eps", "acc", "avg_conf_correct", "avg_conf_wrong", "nll"]
    if cal:
        fieldnames.append("ece")
    save_csv(out / "metrics.csv", all_rows, fieldnames=fieldnames)
    print(f"\nResults saved: {out / 'metrics.csv'}")
    print_table(all_rows, fieldnames=fieldnames)

    save_metadata(
        out / "metadata.json",
        seed=cfg["seed"],
        checkpoint=args.checkpoint,
        model=cfg["model"],
        attacks_config=cfg["attacks"],
        device=device,
        torch_version=torch.__version__,
    )


if __name__ == "__main__":
    main()
