"""Attention rollout for timm ViT models.

Provides tools to extract, aggregate, and quantify attention flow
from the CLS token to spatial patches across all transformer layers.

Key functions
-------------
- ``get_attention_maps``  — extract per-layer attention matrices with
  forward hooks on timm's explicit (non-fused) attention path.
- ``attention_rollout``   — multiplicative rollout with residual
  identity mixing (Abnar et al., 2020).
- ``rollout_to_heatmap``  — reshape rollout vector to a spatial heatmap
  at image resolution.
- ``attention_shift``     — cosine similarity between clean/adversarial
  rollout vectors (quantitative metric).
- ``attention_kl``        — KL divergence between clean/adversarial
  rollout distributions (complementary metric).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import torch
import torch.nn as nn
import torch.nn.functional as F


def _vit_core(model: nn.Module) -> nn.Module:
    """Return the timm ViT inside a ``NormalizedClassifier`` (or the model itself)."""
    return getattr(model, "backbone", model)


@contextmanager
def _captured_attention(model: nn.Module) -> Generator[list[torch.Tensor], None, None]:
    """Capture the softmax attention weights of every block.

    timm computes attention with fused SDPA by default, which never
    materialises the weights. Inside this context every block falls back to
    timm's own explicit path (``fused_attn = False``) and a forward hook on
    ``attn_drop`` (identity at eval time) records its input, i.e. the
    post-softmax attention ``(B, H, N, N)``. The model computation is
    otherwise unchanged; the original settings are restored on exit.
    """
    storage: list[torch.Tensor] = []
    saved: list[tuple[nn.Module, bool]] = []
    handles = []

    for block in _vit_core(model).blocks:
        attn_module = block.attn
        saved.append((attn_module, attn_module.fused_attn))
        attn_module.fused_attn = False
        handles.append(attn_module.attn_drop.register_forward_hook(
            lambda _m, inputs, _out: storage.append(inputs[0].detach().cpu())
        ))

    try:
        yield storage
    finally:
        for h in handles:
            h.remove()
        for module, fused in saved:
            module.fused_attn = fused


def get_attention_maps(
    model: nn.Module,
    x: torch.Tensor,
) -> list[torch.Tensor]:
    """Forward pass that captures attention weights from all transformer layers.

    Parameters
    ----------
    model : a timm ViT, or a ``NormalizedClassifier`` wrapping one.
    x     : input tensor; [0, 1] pixels when *model* is a ``NormalizedClassifier``.

    Returns
    -------
    list of ``(B, H, N, N)`` tensors (one per block, on CPU).
    """
    with _captured_attention(model) as storage:
        with torch.no_grad():
            _ = model(x)
    return storage


def attention_rollout(
    attention_maps: list[torch.Tensor],
    head_reduction: str = "mean",
    discard_ratio: float = 0.0,
) -> torch.Tensor:
    """Compute attention rollout from CLS token to input patches.

    Parameters
    ----------
    attention_maps : list of ``(B, H, N, N)`` tensors, one per layer.
    head_reduction : ``"mean"`` (average across heads) or ``"max"``.
    discard_ratio  : fraction of lowest-attention entries to zero before
                     re-normalising (improves sharpness; Abnar et al. 2020).

    Returns
    -------
    ``(B, N-1)`` tensor — attention from CLS to each spatial patch.
    """
    result = None
    for attn in attention_maps:
        if head_reduction == "mean":
            attn_heads = attn.mean(dim=1)
        elif head_reduction == "max":
            attn_heads = attn.max(dim=1).values
        else:
            raise ValueError(f"Unknown head_reduction: {head_reduction!r}")

        if discard_ratio > 0:
            flat = attn_heads.view(attn_heads.size(0), -1)
            threshold = flat.quantile(discard_ratio, dim=1, keepdim=True)
            attn_heads = attn_heads * (attn_heads >= threshold.unsqueeze(-1)).float()
            attn_heads = attn_heads / (attn_heads.sum(dim=-1, keepdim=True) + 1e-9)

        # Residual connection: mix with identity (Section 3, Abnar et al.)
        I = torch.eye(attn_heads.size(-1), device=attn_heads.device).unsqueeze(0)
        attn_heads = 0.5 * attn_heads + 0.5 * I
        attn_heads = attn_heads / (attn_heads.sum(dim=-1, keepdim=True) + 1e-9)

        result = attn_heads if result is None else result @ attn_heads

    # CLS row, drop CLS-to-CLS entry
    return result[:, 0, 1:]


def rollout_to_heatmap(
    rollout: torch.Tensor,
    grid_size: int = 14,
    image_size: int = 224,
) -> torch.Tensor:
    """Reshape rollout vector to a spatial heatmap at image resolution.

    Parameters
    ----------
    rollout    : ``(B, num_patches)`` from ``attention_rollout``.
    grid_size  : sqrt of num_patches (14 for ViT-Base/16 on 224×224).
    image_size : target spatial resolution for the heatmap.

    Returns
    -------
    ``(B, 1, image_size, image_size)`` heatmap normalised to [0, 1].
    """
    B = rollout.size(0)
    heatmap = rollout.reshape(B, 1, grid_size, grid_size).float()
    heatmap = F.interpolate(
        heatmap, size=(image_size, image_size),
        mode="bilinear", align_corners=False,
    )
    hmin = heatmap.flatten(1).min(dim=1).values.view(B, 1, 1, 1)
    hmax = heatmap.flatten(1).max(dim=1).values.view(B, 1, 1, 1)
    return (heatmap - hmin) / (hmax - hmin + 1e-9)


# ── Quantitative metrics ─────────────────────────────────────────────────

def attention_shift(
    rollout_clean: torch.Tensor,
    rollout_adv: torch.Tensor,
) -> torch.Tensor:
    """Per-image cosine similarity between clean and adversarial rollout.

    Returns ``(B,)`` tensor in [0, 1].  1.0 = identical focus, 0.0 = orthogonal.
    """
    return F.cosine_similarity(rollout_clean, rollout_adv, dim=1)


def attention_kl(
    rollout_clean: torch.Tensor,
    rollout_adv: torch.Tensor,
) -> torch.Tensor:
    """Per-image KL divergence from clean to adversarial rollout distribution.

    Both inputs are normalised to sum-to-one before computing KL.
    Returns ``(B,)`` tensor (non-negative).
    """
    p = rollout_clean / (rollout_clean.sum(dim=1, keepdim=True) + 1e-9)
    q = rollout_adv / (rollout_adv.sum(dim=1, keepdim=True) + 1e-9)
    # Clamp to avoid log(0)
    p = p.clamp(min=1e-9)
    q = q.clamp(min=1e-9)
    return (p * (p.log() - q.log())).sum(dim=1)
