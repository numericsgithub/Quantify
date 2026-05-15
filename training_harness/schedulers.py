"""
schedulers.py — Learning rate and quantization warmup schedulers.

Provides:
  WarmupCosineScheduler  — linear warmup then cosine annealing
  QATWarmupScheduler     — float warmup → calibration → fake-quant
"""

from __future__ import annotations

import math
from typing import Optional

import torch
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
# QATWarmupScheduler
# ---------------------------------------------------------------------------

class QATWarmupScheduler:
    """
    Controls the quantization state of a Brevitas model across training_harness.

    Training phases
    ---------------
    1. **Float warmup**  (epochs 0 … float_warmup_epochs-1)
       The model trains in full precision. Fake-quantization modules are
       present but disabled. The model learns good floating-point weights
       before locking in quantization ranges.

    2. **Calibration**   (triggered at the end of float warmup)
       A short pass over a few batches sets the initial clipping ranges
       (via PTQ-style calibration). This gives QAT a much better starting
       point than random initialisation.

    3. **QAT**           (epochs float_warmup_epochs … end)
       Fake-quantization is enabled. Scales/zero-points are learned.

    4. **BN freeze**     (optional, epoch freeze_bn_after_epoch)
       BatchNorm statistics are frozen to stabilise ranges in late QAT.

    Usage::

        qat_sched = QATWarmupScheduler(
            model=model,
            float_warmup_epochs=5,
            freeze_bn_after_epoch=20,
        )

        for epoch in range(total_epochs):
            qat_sched.step(epoch)          # update quant state
            train_one_epoch(model, ...)
    """

    def __init__(
        self,
        model: nn.Module,
        float_warmup_epochs: int = 5,
        freeze_bn_after_epoch: Optional[int] = None,
    ):
        self.model = model
        self.float_warmup_epochs = float_warmup_epochs
        self.freeze_bn_after_epoch = freeze_bn_after_epoch

        self._quant_enabled: bool = False
        self._bn_frozen: bool = False

        # Start with quant disabled
        disable_quant(model)

    # ------------------------------------------------------------------

    def step(self, epoch: int) -> None:
        """Call at the start of each epoch to update the model's quant state."""

        # Phase transition: float → QAT
        if epoch == self.float_warmup_epochs and not self._quant_enabled:
            enable_quant(self.model)
            self._quant_enabled = True
            print(f"[qat_sched] Epoch {epoch}: fake-quantization ENABLED ✓")

        # BN freeze
        if (
            self.freeze_bn_after_epoch is not None
            and epoch >= self.freeze_bn_after_epoch
            and not self._bn_frozen
        ):
            freeze_bn(self.model)
            self._bn_frozen = True
            print(f"[qat_sched] Epoch {epoch}: BatchNorm statistics FROZEN ✓")

    @property
    def in_float_warmup(self) -> bool:
        """True while training_harness in full precision (before fake-quant is on)."""
        return not self._quant_enabled

    @property
    def in_qat(self) -> bool:
        """True once fake-quantization has been enabled."""
        return self._quant_enabled


# ---------------------------------------------------------------------------
# Brevitas model state helpers
# ------------------------------------------------------------------

def enable_quant(model: nn.Module) -> None:
    """
    Enable fake-quantization on all Brevitas quantized modules.
    """
    _set_quant_enabled(model, enabled=True)


def disable_quant(model: nn.Module) -> None:
    """Disable fake-quantization on all Brevitas quantized modules."""
    _set_quant_enabled(model, enabled=False)


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
    """
    scales: dict[str, float] = {}
    for name, module in model.named_modules():
        # Weight quantizer
        for attr in ("weight_quant", "input_quant", "output_quant", "act_quant"):
            proxy = getattr(module, attr, None)
            if proxy is None:
                continue
            try:
                scale = proxy.scale()
                if scale is not None:
                    import torch
                    val = float(scale.abs().mean().item())
                    key = f"{name}.{attr}.scale"
                    scales[key] = val
            except Exception:
                pass
    return scales
