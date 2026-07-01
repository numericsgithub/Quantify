"""Exponential Moving Average of model parameters."""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


class EMAModel:
    """
    Maintains an EMA shadow copy of a model's parameters.

    Parameters are exponentially averaged; buffers (BatchNorm running stats,
    quantizer calibration state, etc.) are copied directly so calibration is
    not blurred across iterations.

    The EMA model is kept in eval mode and its parameters have requires_grad=False.
    Use parameter_swap() + restore_params() to temporarily run validation with EMA
    weights while keeping the training model's quantization proxy state intact.

    Usage::

        ema = EMAModel(model, decay=0.9999)
        ema.to(device)
        # ... inside training loop after optimizer.step() ...
        ema.update(model)
        # ... at validation time ...
        stash = ema.apply_to(model)   # swap EMA params in
        run_val(model)
        ema.restore(model, stash)     # restore training params
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self._shadow = deepcopy(model)
        self._shadow.eval()
        for p in self._shadow.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend training model parameters into the shadow."""
        for sp, mp in zip(self._shadow.parameters(), model.parameters()):
            sp.mul_(self.decay).add_(mp.data, alpha=1.0 - self.decay)
        # Sync buffers directly (BN stats, quantizer calibration buffers, etc.)
        for sb, mb in zip(self._shadow.buffers(), model.buffers()):
            sb.copy_(mb)

    # ------------------------------------------------------------------
    # Temporary parameter swap for validation
    # ------------------------------------------------------------------

    def apply_to(self, model: nn.Module) -> list:
        """
        Copy EMA parameters into model for validation.

        Returns a stash list that must be passed to restore() afterwards.
        Only parameters are swapped; buffers (quantization state, BN stats)
        remain those of the training model so quantization inference is correct.
        """
        stash = [(p, p.data.clone()) for p in model.parameters()]
        for (p, _), sp in zip(stash, self._shadow.parameters()):
            p.data.copy_(sp.data)
        return stash

    def restore(self, model: nn.Module, stash: list) -> None:
        """Restore the training parameters saved by apply_to()."""
        for p, saved in stash:
            p.data.copy_(saved)

    # ------------------------------------------------------------------
    # Device / state management
    # ------------------------------------------------------------------

    def to(self, device) -> "EMAModel":
        self._shadow.to(device)
        return self

    def state_dict(self) -> dict:
        """State dict compatible with the original model (for checkpointing)."""
        return self._shadow.state_dict()

    def load_state_dict(self, sd: dict, **kwargs) -> None:
        self._shadow.load_state_dict(sd, **kwargs)
