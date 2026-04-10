"""Seed control, I/O helpers, and lightweight metrics utilities."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import csv


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def run_dir(domain: str, base: str = "./experiments") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base) / domain / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def save_metadata(path: Path, **kwargs: Any) -> None:
    data = {"timestamp": datetime.now().isoformat(), **kwargs}
    save_json(path, data)


def save_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: List[Dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    """Print a list of dicts as an aligned table to stdout."""
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    col_widths = {k: max(len(k), *(len(str(r.get(k, ""))) for r in rows)) for k in fieldnames}
    header = "  ".join(k.ljust(col_widths[k]) for k in fieldnames)
    print(header)
    print("  ".join("-" * col_widths[k] for k in fieldnames))
    for r in rows:
        print("  ".join(str(r.get(k, "")).ljust(col_widths[k]) for k in fieldnames))


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute ECE from parallel arrays of per-sample confidence and correctness."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() / len(confidences) * abs(bin_acc - bin_conf)
    return float(ece)
