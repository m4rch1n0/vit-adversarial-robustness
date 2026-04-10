"""Evaluation loop with decay-profile metrics.

Computes accuracy, confidence statistics (correct vs wrong predictions), and
optional calibration metrics for each (attack, epsilon) pair.

Also provides ``evaluate_per_class`` for per-class breakdown with confusion
matrix — used by the Point C per-class analysis.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import expected_calibration_error


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    attack: Optional[Callable] = None,
    compute_calibration: bool = False,
    **attack_kwargs: Any,
) -> Dict[str, float]:
    """Run a full pass over *loader*, optionally applying *attack*.

    Returns a dict with:
        acc                - top-1 accuracy
        avg_conf_correct   - mean predicted-class confidence on correct examples
        avg_conf_wrong     - mean predicted-class confidence on wrong examples
        nll                - mean negative log-likelihood
        ece                - expected calibration error (only if compute_calibration)
    """
    model.eval()

    all_confs: list[float] = []
    all_correct: list[bool] = []
    total_nll = 0.0
    n = 0

    ctx = torch.enable_grad() if attack is not None else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, desc="eval", leave=False):
            x, y = x.to(device), y.to(device)

            if attack is not None:
                x = attack(model, x, y, **attack_kwargs)

            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                pred = logits.argmax(dim=1)

                conf = probs.gather(1, pred.unsqueeze(1)).squeeze(1)
                correct = (pred == y)

                total_nll += F.cross_entropy(logits, y, reduction="sum").item()

                all_confs.extend(conf.cpu().tolist())
                all_correct.extend(correct.cpu().tolist())
                n += y.size(0)

    confs = np.array(all_confs)
    correct = np.array(all_correct)

    acc = correct.mean()
    avg_conf_correct = float(confs[correct].mean()) if correct.any() else 0.0
    avg_conf_wrong = float(confs[~correct].mean()) if (~correct).any() else 0.0

    result: Dict[str, float] = {
        "acc": float(acc),
        "avg_conf_correct": avg_conf_correct,
        "avg_conf_wrong": avg_conf_wrong,
        "nll": total_nll / n,
    }

    if compute_calibration:
        result["ece"] = expected_calibration_error(confs, correct.astype(float))

    return result


def evaluate_decay(
    model: nn.Module,
    loader: DataLoader,
    attack_fn: Callable,
    eps_list: List[float],
    attack_name: str,
    device: str = "cuda",
    compute_calibration: bool = False,
    **extra_attack_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Run *attack_fn* over a grid of epsilon values, returning one row per epsilon."""
    rows: List[Dict[str, Any]] = []
    eps_bar = tqdm(eps_list, desc=attack_name, unit="ε", leave=True)
    for eps in eps_bar:
        eps_255 = round(eps * 255)
        eps_bar.set_description(f"{attack_name}  ε={eps_255}/255")

        if eps == 0.0:
            result = evaluate(model, loader, device=device, compute_calibration=compute_calibration)
        else:
            result = evaluate(
                model, loader, device=device,
                attack=attack_fn, eps_pix=eps,
                compute_calibration=compute_calibration,
                **extra_attack_kwargs,
            )

        row = {
            "attack": attack_name,
            "eps_255": eps_255,
            "eps": round(eps, 6),
            **result,
        }
        rows.append(row)
        eps_bar.set_postfix(acc=f"{result['acc']:.4f}", conf_ok=f"{result['avg_conf_correct']:.4f}", conf_err=f"{result['avg_conf_wrong']:.4f}")
        tqdm.write(f"  {attack_name}  ε={eps_255}/255  acc={result['acc']:.4f}  conf_ok={result['avg_conf_correct']:.4f}  conf_err={result['avg_conf_wrong']:.4f}")

    return rows


# ---------------------------------------------------------------------------
# Per-class evaluation
# ---------------------------------------------------------------------------

def evaluate_per_class(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: str = "cuda",
    attack: Optional[Callable] = None,
    **attack_kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate with per-class breakdown.

    Returns a dict with:
        per_class_acc     : list of float, length ``num_classes``
        per_class_conf    : list of float, mean confidence per class
        confusion_matrix  : ``(num_classes, num_classes)`` ndarray
        all_preds         : flat array of predictions
        all_labels        : flat array of true labels
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_confs: list[float] = []

    ctx = torch.enable_grad() if attack is not None else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, desc="eval_per_class", leave=False):
            x, y = x.to(device), y.to(device)
            if attack is not None:
                x = attack(model, x, y, **attack_kwargs)
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                pred = logits.argmax(dim=1)
                conf = probs.gather(1, pred.unsqueeze(1)).squeeze(1)

                all_preds.extend(pred.cpu().tolist())
                all_labels.extend(y.cpu().tolist())
                all_confs.extend(conf.cpu().tolist())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    confs = np.array(all_confs)

    per_class_acc: list[float] = []
    per_class_conf: list[float] = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class_acc.append(float((preds[mask] == c).mean()))
            per_class_conf.append(float(confs[mask].mean()))
        else:
            per_class_acc.append(0.0)
            per_class_conf.append(0.0)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(labels, preds):
        cm[true_label][pred_label] += 1

    return {
        "per_class_acc": per_class_acc,
        "per_class_conf": per_class_conf,
        "confusion_matrix": cm,
        "all_preds": preds,
        "all_labels": labels,
    }
