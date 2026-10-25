"""
schedulers.py — Learning rate scheduler and model-state helpers.

Provides:
  WarmupCosineScheduler  — linear warmup then cosine annealing
  freeze_bn / collect_scale_factors / _set_quant_enabled — used by QATTrainer
"""

from __future__ import annotations

import math

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

# ---------------------------------------------------------------------------
# WarmupCosineScheduler
# ---------------------------------------------------------------------------

class WarmupCosineScheduler(LRScheduler):
    """
    Learning rate scheduler with a linear warmup phase followed by
    cosine annealing.

    Warmup: LR increases linearly from `warmup_start_lr` to `base_lr`
            over `warmup_steps` steps.
    Cosine: LR decreases from `base_lr` to `eta_min` following a cosine
            curve over the remaining steps.

    Call scheduler.step() once per **optimizer step** (not per epoch).

    Usage::

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=500,
            total_steps=10_000,
            eta_min=1e-6,
        )

        for step in range(total_steps):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 0.0,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        self.warmup_start_lr = warmup_start_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        base_lrs = self.base_lrs

        if step < self.warmup_steps:
            # Linear warmup
            alpha = step / max(1, self.warmup_steps)
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                for base_lr in base_lrs
            ]

        # Cosine annealing
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_factor
            for base_lr in base_lrs
        ]


# ---------------------------------------------------------------------------
# Brevitas model state helpers
# ------------------------------------------------------------------

def _set_quant_enabled(model: nn.Module, enabled: bool) -> None:
    """
    Toggle quantization on Brevitas modules.

    Brevitas exposes `disable_quant` attributes on quant proxies
    (e.g., `QuantLinear.weight_quant`, `QuantIdentity.input_quant`).
    We walk the model and flip them.
    """
    for module in model.modules():
        if hasattr(module, "disable_quant"):
            module.disable_quant = not enabled


def freeze_bn(model: nn.Module) -> None:
    """
    Freeze all BatchNorm layers: fix running stats and disable training_harness mode.

    After freezing, BN layers act as fixed affine transforms, which
    prevents quantization ranges from shifting in late-stage QAT.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
            module.weight.requires_grad_(False)
            module.bias.requires_grad_(False)


def collect_scale_factors(model: nn.Module) -> dict[str, float]:
    """
    Walk a Brevitas model and collect current quantization scale factors.

    Returns a dict mapping layer_name → scale (as a Python float).
    Handles both per-tensor and per-channel scales (returns the mean for
    per-channel to keep things scalar).

    Only reads scales from quantizers that have already calibrated
    (`search_done=True`); see pitfall #19 in
    docs/llm/pitfalls/brevitas_pitfalls.md. A Brevitas `WeightQuantProxyFromInjector`
    whose weight hasn't been quantized yet (`proxy._cached_weight is None`)
    computes `.scale()` by calling `self.__call__(weight)` -- i.e. it runs a
    REAL forward pass through our `BaseQuantizer.forward()` on the model's
    live weight tensor, right here, as a side effect of merely reading the
    scale. Since this is called once per epoch as soon as QAT is active
    (`QATTrainer`, in eval mode, outside the normal gated
    training loop), it used to make ANY quantizer whose staggered gate
    threshold happens to be 0 (the first one reached in forward order)
    calibrate, open its gate, and start annealing on the very first epoch
    QAT activates -- regardless of `quantization_start_gap` -- because this
    call reaches it before any real training batch does. Skipping
    uncalibrated quantizers here instead leaves them to calibrate exactly
    when their own gate legitimately opens during real training, and they
    simply don't contribute a scale entry until then.
    """
    from quantizers.base_quantizer import BaseQuantizer

    scales: dict[str, float] = {}
    for name, module in model.named_modules():
        # Weight quantizer
        for attr in ("weight_quant", "input_quant", "output_quant", "act_quant"):
            proxy = getattr(module, attr, None)
            if proxy is None:
                continue
            tensor_quant = getattr(proxy, "tensor_quant", None)
            if isinstance(tensor_quant, BaseQuantizer) and not tensor_quant.search_done_value:
                continue
            try:
                scale = proxy.scale()
                if scale is not None:
                    val = float(scale.abs().mean().item())
                    key = f"{name}.{attr}.scale"
                    scales[key] = val
            except Exception:
                pass
    return scales
