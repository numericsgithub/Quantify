"""
calibration.py — Post-float-warmup calibration for Brevitas QAT.

Before enabling fake-quantization, we run a short calibration pass
to set sensible initial clipping ranges from real data statistics.
This is essentially PTQ (Post-Training Quantization) used as an
initialiser for QAT.

Public API
----------
run_calibration(model, data_loader, n_batches, device)
    High-level: enable calibration mode, run N batches, restore state.

enable_quant(model) / disable_quant(model)
    Toggle fake-quantization on Brevitas modules.
    Also exported from schedulers.py for convenience.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# High-level calibration entry point
# ---------------------------------------------------------------------------

def run_calibration(
    model: nn.Module,
    data_loader,
    n_batches: int = 100,
    device: str = "cpu",
    forward_fn: Optional[Callable] = None,
    verbose: bool = True,
) -> None:
    """
    Run a calibration pass over `n_batches` batches of data.

    During calibration, Brevitas records the min/max (or percentile)
    of activations and weights to initialise quantization scales.

    After this function returns, the model is left in its original
    train/eval state with calibration disabled.

    Args:
        model:      A Brevitas model with at least one QuantLayer.
        data_loader: Any iterable yielding (inputs, targets) batches.
        n_batches:  Number of batches to pass through the model.
        device:     Device string for moving data.
        forward_fn: Optional custom forward function with signature
                    `forward_fn(model, inputs) -> outputs`.
                    Defaults to `model(inputs)`.
        verbose:    Print progress.
    """
    if verbose:
        print(f"[calibration] Starting calibration over {n_batches} batches …")

    original_training = model.training
    model.eval()

    with _calibration_context(model):
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= n_batches:
                    break

                # Unpack batch — handle (inputs,), (inputs, targets), etc.
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch

                inputs = inputs.to(device)

                if forward_fn is not None:
                    forward_fn(model, inputs)
                else:
                    model(inputs)

                if verbose and (i + 1) % max(1, n_batches // 5) == 0:
                    print(f"  [calibration] {i + 1}/{n_batches} batches done")

    # Restore original training state
    model.train(original_training)

    if verbose:
        print("[calibration] Done. Quantization ranges initialised ✓")


# ---------------------------------------------------------------------------
# Calibration context manager
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _calibration_context(model: nn.Module):
    """
    Enable Brevitas calibration mode for the duration of the block.

    In calibration mode, Brevitas QuantLayers observe activations to
    compute scale factors without injecting quantization noise.
    """
    _set_calibration(model, enabled=True)
    try:
        yield
    finally:
        _set_calibration(model, enabled=False)


def _set_calibration(model: nn.Module, enabled: bool) -> None:
    """
    Toggle Brevitas calibration mode.

    Brevitas ≥ 0.8 exposes `_calibration_enabled` on quant proxies.
    Older versions may use different APIs — we try a few approaches.
    """
    # Approach 1: Brevitas ≥ 0.8 context manager API
    try:
        from brevitas.core.quant import QuantType
        from brevitas.inject.enum import QuantType as QT
    except ImportError:
        pass

    try:
        import brevitas.nn as qnn
        for module in model.modules():
            # Try the standard Brevitas enable_act_quantization API
            if hasattr(module, "_calibration_enabled"):
                module._calibration_enabled = enabled
    except Exception:
        pass

    # Approach 2: Walk all modules looking for calibration handles
    for module in model.modules():
        for attr in ("_calibration_enabled", "calibration_enabled"):
            if hasattr(module, attr):
                setattr(module, attr, enabled)


# ---------------------------------------------------------------------------
# enable / disable quant (re-exported for convenience)
# ---------------------------------------------------------------------------

def enable_quant(model: nn.Module) -> None:
    """Enable fake-quantization on all Brevitas quantized modules."""
    from .schedulers import _set_quant_enabled
    _set_quant_enabled(model, enabled=True)


def disable_quant(model: nn.Module) -> None:
    """Disable fake-quantization on all Brevitas quantized modules."""
    from .schedulers import _set_quant_enabled
    _set_quant_enabled(model, enabled=False)


# ---------------------------------------------------------------------------
# Quantization range analysis
# ---------------------------------------------------------------------------

def inspect_quant_ranges(model: nn.Module) -> dict:
    """
    Return a summary of current quantization ranges for all layers.

    Useful for debugging after calibration to see if ranges are sane.

    Returns::

        {
          "conv1.weight_quant": {"scale": 0.0042, "zero_point": 0},
          "conv1.input_quant":  {"scale": 0.0031, "zero_point": 0},
          ...
        }
    """
    ranges = {}
    for name, module in model.named_modules():
        for attr in ("weight_quant", "input_quant", "output_quant", "act_quant"):
            proxy = getattr(module, attr, None)
            if proxy is None:
                continue
            entry = {}
            try:
                scale = proxy.scale()
                if scale is not None:
                    entry["scale"] = float(scale.abs().mean().item())
            except Exception:
                pass
            try:
                zp = proxy.zero_point()
                if zp is not None:
                    entry["zero_point"] = float(zp.mean().item())
            except Exception:
                pass
            if entry:
                ranges[f"{name}.{attr}"] = entry
    return ranges


def print_quant_ranges(model: nn.Module) -> None:
    """Print a formatted table of quantization ranges."""
    ranges = inspect_quant_ranges(model)
    if not ranges:
        print("[calibration] No quantization ranges found — is this a Brevitas model?")
        return

    col_w = max(len(k) for k in ranges) + 2
    print(f"\n{'Layer':<{col_w}}  {'Scale':>12}  {'Zero-Point':>12}")
    print("-" * (col_w + 28))
    for name, info in sorted(ranges.items()):
        scale = f"{info.get('scale', 'N/A'):>12.6f}" if "scale" in info else f"{'N/A':>12}"
        zp    = f"{info.get('zero_point', 'N/A'):>12.4f}" if "zero_point" in info else f"{'N/A':>12}"
        print(f"{name:<{col_w}}  {scale}  {zp}")
    print()
