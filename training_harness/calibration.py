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
    reset_calibration: bool = True,
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
        reset_calibration: If True, resets lazy calibration buffers (e.g., `search_done`)
                           before starting the calibration pass to ensure fresh statistics.
    """
    if verbose:
        print(f"[calibration] Starting calibration over {n_batches} batches …")

    if reset_calibration:
        print("[calibration] Resetting calibration buffers …")
        for module in model.modules():
            for name, buffer in module.named_buffers():
                if "search_done" in name or "calibration_done" in name:
                    buffer.fill_(False)

    original_training = model.training
    model.eval()

    # Use Brevitas' official calibration context manager
    from brevitas.graph.calibrate import calibration_mode
    with calibration_mode(model):
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

    # Restore original training_harness state
    model.train(original_training)

    if verbose:
        print("[calibration] Done. Quantization ranges initialised ✓")


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
# ------------------------------------------------------------------

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
