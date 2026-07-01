"""MixUp and CutMix batch augmentations for QAT training."""
from __future__ import annotations

import numpy as np
import torch


def _rand_bbox(W: int, H: int, lam: float):
    cut_ratio = (1.0 - lam) ** 0.5
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    return max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0), min(cx + cut_w // 2, W), min(cy + cut_h // 2, H)


def mixup_batch(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.2):
    """Returns (mixed_inputs, targets_a, targets_b, lam)."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(inputs.size(0), device=inputs.device)
    mixed = lam * inputs + (1.0 - lam) * inputs[idx]
    return mixed, targets, targets[idx], lam


def cutmix_batch(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0):
    """Returns (mixed_inputs, targets_a, targets_b, lam)."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(inputs.size(0), device=inputs.device)
    W, H = inputs.size(-1), inputs.size(-2)
    x1, y1, x2, y2 = _rand_bbox(W, H, lam)
    mixed = inputs.clone()
    mixed[:, :, y1:y2, x1:x2] = inputs[idx, :, y1:y2, x1:x2]
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed, targets, targets[idx], lam


def apply_mixup_cutmix(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
):
    """Randomly apply MixUp or CutMix (50/50 when both are enabled).

    Returns (inputs, targets_a, targets_b, lam).  When neither is enabled,
    returns the inputs and targets unchanged with lam=1.0.
    """
    has_mixup = mixup_alpha > 0
    has_cutmix = cutmix_alpha > 0
    if not has_mixup and not has_cutmix:
        return inputs, targets, targets, 1.0
    use_cutmix = has_cutmix and (not has_mixup or np.random.rand() < 0.5)
    if use_cutmix:
        return cutmix_batch(inputs, targets, cutmix_alpha)
    return mixup_batch(inputs, targets, mixup_alpha)
