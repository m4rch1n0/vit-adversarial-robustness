"""Evasion attacks: FGSM, PGD, and patch-based variants.

All attacks follow the same callable protocol:

    attack(model, x_norm, y, *, eps_pix, **kwargs) -> x_adv_norm

where:
    model    - the target classifier (in eval mode)
    x_norm   - batch of ImageNet-normalised inputs
    y        - ground-truth labels
    eps_pix  - L∞ budget in [0, 1] pixel scale
    returns  - adversarial examples, ImageNet-normalised

Perturbations are computed in pixel space [0, 1] (denorm → perturb → clip →
renorm), consistent with the prior project's convention.

Extension point for training-time attacks
-----------------------------------------
Training-time attacks (poisoning, backdoor) modify the *dataset*, not the test
input.  They do not share this function signature and belong in a separate
module (e.g. ``poisoning.py``) that wraps or subclasses the dataset.  The config
schema already reserves a ``poisoning`` key for this purpose.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .data import denorm, renorm


# ---------------------------------------------------------------------------
# FGSM
# ---------------------------------------------------------------------------

def fgsm(
    model: nn.Module,
    x_norm: torch.Tensor,
    y: torch.Tensor,
    *,
    eps_pix: float,
) -> torch.Tensor:
    """Fast Gradient Sign Method (Goodfellow et al., 2014). Single-step, L∞."""
    model.eval()
    x_pix = denorm(x_norm).detach()
    x_pix.requires_grad_(True)

    logits = model(renorm(x_pix))
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()

    x_adv = torch.clamp(x_pix + eps_pix * x_pix.grad.sign(), 0.0, 1.0)
    return renorm(x_adv).detach()


# ---------------------------------------------------------------------------
# PGD
# ---------------------------------------------------------------------------

def pgd(
    model: nn.Module,
    x_norm: torch.Tensor,
    y: torch.Tensor,
    *,
    eps_pix: float,
    steps: int = 10,
    alpha_pix: Optional[float] = None,
) -> torch.Tensor:
    """Projected Gradient Descent (Madry et al., 2017). Multi-step, L∞.

    Defaults: alpha = eps / 4, random start within the L∞ ball.
    """
    if alpha_pix is None:
        alpha_pix = eps_pix / 4.0

    model.eval()
    x0 = denorm(x_norm).detach()

    # random start
    x = x0 + torch.empty_like(x0).uniform_(-eps_pix, eps_pix)
    x = torch.clamp(x, 0.0, 1.0)

    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        logits = model(renorm(x))
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        x = x.detach() + alpha_pix * x.grad.sign()
        x = torch.max(torch.min(x, x0 + eps_pix), x0 - eps_pix)
        x = torch.clamp(x, 0.0, 1.0)

    return renorm(x).detach()


# ---------------------------------------------------------------------------
# Patch-based perturbation (region-constrained PGD)
# ---------------------------------------------------------------------------

def _make_spatial_mask(
    h: int, w: int,
    patch_size: int,
    position: str = "center",
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Return a (1, 1, H, W) binary mask with 1s inside the patch region."""
    mask = torch.zeros(1, 1, h, w, device=device)
    if position == "center":
        cy, cx = h // 2, w // 2
        y0 = max(cy - patch_size // 2, 0)
        x0 = max(cx - patch_size // 2, 0)
        mask[:, :, y0:y0 + patch_size, x0:x0 + patch_size] = 1.0
    else:
        raise ValueError(f"Unknown position: {position}")
    return mask


def patch_pgd(
    model: nn.Module,
    x_norm: torch.Tensor,
    y: torch.Tensor,
    *,
    eps_pix: float,
    steps: int = 10,
    alpha_pix: Optional[float] = None,
    patch_size: int = 64,
    position: str = "center",
) -> torch.Tensor:
    """PGD constrained to a fixed spatial patch (L∞ within the patch, zero outside)."""
    if alpha_pix is None:
        alpha_pix = eps_pix / 4.0

    model.eval()
    x0 = denorm(x_norm).detach()
    _, _, h, w = x0.shape
    mask = _make_spatial_mask(h, w, patch_size, position=position, device=x0.device)

    x = x0 + mask * torch.empty_like(x0).uniform_(-eps_pix, eps_pix)
    x = torch.clamp(x, 0.0, 1.0)

    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        logits = model(renorm(x))
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        grad = x.grad * mask
        x = x.detach() + alpha_pix * grad.sign()
        x = torch.max(torch.min(x, x0 + eps_pix * mask), x0 - eps_pix * mask)
        # keep unmasked pixels identical to the original
        x = mask * x + (1 - mask) * x0
        x = torch.clamp(x, 0.0, 1.0)

    return renorm(x).detach()


def _gradient_patch_indices(
    model: nn.Module,
    x_norm: torch.Tensor,
    y: torch.Tensor,
    vit_patch_size: int = 16,
    top_k: int = 20,
) -> torch.Tensor:
    """Select top-k ViT patches by gradient magnitude (per image in the batch).

    Returns a (B, 1, H, W) binary mask in pixel space.
    """
    model.eval()
    x_pix = denorm(x_norm).detach().requires_grad_(True)
    logits = model(renorm(x_pix))
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()

    grad = x_pix.grad.detach()
    b, c, h, w = grad.shape
    ph = h // vit_patch_size
    pw = w // vit_patch_size

    # reshape to (B, C, ph, ps, pw, ps) then take L2 norm per patch
    grad_patches = grad[:, :, :ph * vit_patch_size, :pw * vit_patch_size]
    grad_patches = grad_patches.reshape(b, c, ph, vit_patch_size, pw, vit_patch_size)
    patch_norms = (grad_patches ** 2).sum(dim=(1, 3, 5)).sqrt()  # (B, ph, pw)

    # top-k per image
    flat = patch_norms.view(b, -1)
    _, topk_idx = flat.topk(min(top_k, flat.shape[1]), dim=1)

    mask = torch.zeros(b, 1, h, w, device=x_pix.device)
    for i in range(b):
        for idx in topk_idx[i]:
            py = (idx // pw) * vit_patch_size
            px = (idx % pw) * vit_patch_size
            mask[i, :, py:py + vit_patch_size, px:px + vit_patch_size] = 1.0

    return mask


def salient_patch_pgd(
    model: nn.Module,
    x_norm: torch.Tensor,
    y: torch.Tensor,
    *,
    eps_pix: float,
    steps: int = 10,
    alpha_pix: Optional[float] = None,
    vit_patch_size: int = 16,
    top_k: int = 20,
) -> torch.Tensor:
    """PGD restricted to the top-k most gradient-salient ViT patches."""
    if alpha_pix is None:
        alpha_pix = eps_pix / 4.0

    model.eval()
    mask = _gradient_patch_indices(model, x_norm, y, vit_patch_size=vit_patch_size, top_k=top_k)
    x0 = denorm(x_norm).detach()

    x = x0 + mask * torch.empty_like(x0).uniform_(-eps_pix, eps_pix)
    x = torch.clamp(x, 0.0, 1.0)

    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        logits = model(renorm(x))
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        grad = x.grad * mask
        x = x.detach() + alpha_pix * grad.sign()
        x = torch.max(torch.min(x, x0 + eps_pix * mask), x0 - eps_pix * mask)
        x = mask * x + (1 - mask) * x0
        x = torch.clamp(x, 0.0, 1.0)

    return renorm(x).detach()


# ---------------------------------------------------------------------------
# Registry for config-driven dispatch
# ---------------------------------------------------------------------------

ATTACKS = {
    "fgsm": fgsm,
    "pgd": pgd,
    "patch_pgd": patch_pgd,
    "salient_patch_pgd": salient_patch_pgd,
}
