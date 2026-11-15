"""Evasion attacks: FGSM, PGD, and patch-based variants.

All attacks follow the same callable protocol:

    attack(model, x, y, *, eps_pix, **kwargs) -> x_adv

where:
    model    - the target classifier (in eval mode), taking [0, 1] pixels
    x        - batch of images in [0, 1] pixel space
    y        - ground-truth labels
    eps_pix  - L∞ budget in [0, 1] pixel scale
    returns  - adversarial examples in [0, 1] pixel space

Each model applies the input normalisation of its own pretraining inside its
forward pass (``src/wrapper.py``), so the budget is defined in the same pixel
space for every architecture. Gradients are taken with respect to the input
only (``torch.autograd.grad``), leaving the model parameters untouched.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _input_grad(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gradient of the cross-entropy loss with respect to the input pixels."""
    x = x.detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    (grad,) = torch.autograd.grad(loss, x)
    return grad.detach()


def _pgd_masked(
    model: nn.Module,
    x0: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    eps_pix: float,
    steps: int,
    alpha_pix: float,
) -> torch.Tensor:
    """L∞ PGD with random start, restricted to the region where ``mask == 1``.

    With an all-ones mask this is standard PGD (Madry et al., 2017).
    """
    x = x0 + mask * torch.empty_like(x0).uniform_(-eps_pix, eps_pix)
    x = torch.clamp(x, 0.0, 1.0)

    for _ in range(steps):
        grad = _input_grad(model, x, y) * mask
        x = x + alpha_pix * grad.sign()
        x = torch.max(torch.min(x, x0 + eps_pix * mask), x0 - eps_pix * mask)
        x = mask * x + (1 - mask) * x0
        x = torch.clamp(x, 0.0, 1.0)

    return x.detach()


# ---------------------------------------------------------------------------
# FGSM
# ---------------------------------------------------------------------------

def fgsm(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps_pix: float,
) -> torch.Tensor:
    """Fast Gradient Sign Method (Goodfellow et al., 2014). Single-step, L∞."""
    model.eval()
    x0 = x.detach()
    grad = _input_grad(model, x0, y)
    return torch.clamp(x0 + eps_pix * grad.sign(), 0.0, 1.0).detach()


# ---------------------------------------------------------------------------
# PGD
# ---------------------------------------------------------------------------

def pgd(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps_pix: float,
    steps: int = 10,
    alpha_pix: Optional[float] = None,
) -> torch.Tensor:
    """Projected Gradient Descent (Madry et al., 2017). Multi-step, L∞.

    Defaults: alpha = eps / 4, random start within the L∞ ball.
    """
    model.eval()
    x0 = x.detach()
    mask = torch.ones_like(x0[:, :1])
    return _pgd_masked(model, x0, y, mask, eps_pix, steps, alpha_pix or eps_pix / 4.0)


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
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps_pix: float,
    steps: int = 10,
    alpha_pix: Optional[float] = None,
    patch_size: int = 64,
    position: str = "center",
) -> torch.Tensor:
    """PGD constrained to a fixed spatial patch (L∞ within the patch, zero outside)."""
    model.eval()
    x0 = x.detach()
    _, _, h, w = x0.shape
    mask = _make_spatial_mask(h, w, patch_size, position=position, device=x0.device)
    return _pgd_masked(model, x0, y, mask, eps_pix, steps, alpha_pix or eps_pix / 4.0)


def _gradient_patch_mask(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    vit_patch_size: int = 16,
    top_k: int = 20,
) -> torch.Tensor:
    """Select the top-k cells of a ``vit_patch_size`` grid by input-gradient norm.

    The gradient is computed once on the clean image, with the attacked
    model's own gradient. Returns a (B, 1, H, W) binary mask in pixel space.
    """
    grad = _input_grad(model, x, y)
    b, c, h, w = grad.shape
    ph = h // vit_patch_size
    pw = w // vit_patch_size

    # reshape to (B, C, ph, ps, pw, ps) then take L2 norm per cell
    cells = grad[:, :, :ph * vit_patch_size, :pw * vit_patch_size]
    cells = cells.reshape(b, c, ph, vit_patch_size, pw, vit_patch_size)
    cell_norms = (cells ** 2).sum(dim=(1, 3, 5)).sqrt()  # (B, ph, pw)

    _, topk_idx = cell_norms.view(b, -1).topk(min(top_k, ph * pw), dim=1)

    cell_mask = torch.zeros(b, ph * pw, device=x.device)
    cell_mask.scatter_(1, topk_idx, 1.0)
    cell_mask = cell_mask.view(b, 1, ph, pw)
    mask = torch.zeros(b, 1, h, w, device=x.device)
    mask[:, :, :ph * vit_patch_size, :pw * vit_patch_size] = cell_mask.repeat_interleave(
        vit_patch_size, dim=2
    ).repeat_interleave(vit_patch_size, dim=3)
    return mask


def salient_patch_pgd(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps_pix: float,
    steps: int = 10,
    alpha_pix: Optional[float] = None,
    vit_patch_size: int = 16,
    top_k: int = 20,
) -> torch.Tensor:
    """PGD restricted to the top-k most gradient-salient 16x16 cells."""
    model.eval()
    x0 = x.detach()
    mask = _gradient_patch_mask(model, x0, y, vit_patch_size=vit_patch_size, top_k=top_k)
    return _pgd_masked(model, x0, y, mask, eps_pix, steps, alpha_pix or eps_pix / 4.0)


# ---------------------------------------------------------------------------
# Registry for config-driven dispatch
# ---------------------------------------------------------------------------

ATTACKS = {
    "fgsm": fgsm,
    "pgd": pgd,
    "patch_pgd": patch_pgd,
    "salient_patch_pgd": salient_patch_pgd,
}
