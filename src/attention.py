"""Attention rollout for timm ViT models.

Provides tools to extract, aggregate, and quantify attention flow
from the CLS token to spatial patches across all transformer layers.

Key functions
-------------
- ``get_attention_maps``  — extract per-layer attention matrices via
  monkey-patched forward (bypasses fused SDPA in timm >=1.0).
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

import types
from contextlib import contextmanager
from typing import Generator

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextmanager
def _patched_attention(model: nn.Module) -> Generator[list[torch.Tensor], None, None]:
    """Context manager that monkey-patches every ``Attention.forward`` in
    *model* to capture the softmax attention weights, then restores the
    originals on exit.

    Yields a list that gets populated with one ``(B, H, N, N)`` tensor
    per block during the next forward pass.
    """
    storage: list[torch.Tensor] = []
    originals: list[tuple[nn.Module, object]] = []

    for block in model.blocks:
        attn_module = block.attn
        originals.append((attn_module, attn_module.forward))

        def _make_patched(store: list[torch.Tensor]):
            def patched_forward(self, x: torch.Tensor, **_kwargs) -> torch.Tensor:
                B, N, C = x.shape
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                q, k = self.q_norm(q), self.k_norm(k)

                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                store.append(attn.detach().cpu())
                attn = self.attn_drop(attn)

                x = (attn @ v).transpose(1, 2).reshape(B, N, self.attn_dim)
                x = self.norm(x)
                x = self.proj(x)
                x = self.proj_drop(x)
                return x

            return patched_forward

        attn_module.forward = types.MethodType(
            _make_patched(storage), attn_module
        )

    try:
        yield storage
    finally:
        for module, orig_fwd in originals:
            module.forward = orig_fwd


def get_attention_maps(
    model: nn.Module,
    x: torch.Tensor,
) -> list[torch.Tensor]:
    """Forward pass that captures attention weights from all transformer layers.

    Parameters
    ----------
    model : a timm ViT model (must have ``model.blocks``).
    x     : input tensor, ImageNet-normalised, shape ``(B, 3, H, W)``.

    Returns
    -------
    list of ``(B, H, N, N)`` tensors (one per block, on CPU).
    """
    with _patched_attention(model) as storage:
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
