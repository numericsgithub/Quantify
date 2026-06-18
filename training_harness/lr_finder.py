"""
lr_finder.py — LR Range Test for QAT fine-tuning.

Two-phase routine:

  Phase 1: Calibration pre-pass
      Uses the model as-is (whatever weights the caller loaded), configures the
      quantizer manager for immediate single-quantizer activation (gap=20,
      annealing_steps=1), and runs `calibration_steps` training steps at near-zero
      LR (1e-10 by default).  This calibrates the first quantizer under realistic
      forward-pass conditions without meaningfully moving the weights.

  Phase 2: LR Range Sweep (Leslie Smith's method)
      Sweeps LR geometrically from sweep_start_lr to sweep_end_lr, recording
      batch loss at each step with EMA smoothing.  After the sweep the model and
      optimizer are restored to the post-Phase-1 calibrated state.  A plot is
      saved and the suggested starting LR is returned.

Usage::

    from training_harness.lr_finder import find_lr

    result = find_lr(
        model=model,          # already loaded with pretrained / checkpoint weights
        optimizer=optimizer,
        train_loader=train_loader,
        loss_fn=nn.CrossEntropyLoss(),
        out_dir="output/lr_finder",
    )
    print(f"Suggested LR: {result.suggested_lr:.2e}")
"""

from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from quantizers.manager import QuantizerManager
from .trainer_v2 import _reset_and_register, _unpack_batch


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class LRFindResult:
    """Results from an LR range test."""
    lrs: List[float]
    losses: List[float]        # EMA-smoothed losses (bias-corrected)
    raw_losses: List[float]    # unsmoothed per-step batch losses
    steep_lr: float            # LR at steepest negative slope  ← primary recommendation
    min_loss_lr: float         # LR at minimum smoothed loss
    conservative_lr: float     # min_loss_lr / 10
    suggested_lr: float        # = steep_lr
    plot_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def find_lr(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    *,
    device: str = "auto",
    # Phase 1
    calibration_steps: int = 10,
    calibration_lr: float = 1e-10,
    calibration_qat_gap: int = 20,
    # Phase 2
    sweep_start_lr: float = 1e-8,
    sweep_end_lr: float = 1e-2,
    sweep_steps: int = 100,
    ema_beta: float = 0.98,
    diverge_factor: float = 5.0,
    # Output
    out_dir: str = "output/lr_finder",
    grad_clip_norm: Optional[float] = None,
) -> LRFindResult:
    """
    Two-phase LR Range Test for QAT fine-tuning.

    The model is used as-is — no checkpoint loading occurs inside this function.
    Load pretrained weights (or a QAT checkpoint) before calling find_lr.

    Args:
        model:               Quantized model with the weights to start from.
        optimizer:           Optimizer linked to model.parameters().
        train_loader:        DataLoader for training data (iterated cyclically).
        loss_fn:             Loss function.
        device:              "auto", "cuda", or "cpu".
        calibration_steps:   Number of training steps for Phase 1 (default 10).
        calibration_lr:      LR during Phase 1 — effectively zero updates
                             (default 1e-10).
        calibration_qat_gap: quantization_start_gap for Phase 1 (default 20).
        sweep_start_lr:      Lowest LR in the sweep (default 1e-8).
        sweep_end_lr:        Highest LR in the sweep (default 1e-2).
        sweep_steps:         Number of sweep steps (default 100).
        ema_beta:            EMA smoothing factor (default 0.98).
        diverge_factor:      Stop sweep if smoothed loss > factor × best smoothed loss
                             (default 5.0).
        out_dir:             Directory for plot output.
        grad_clip_norm:      Gradient clipping norm (None = disabled).

    Returns:
        LRFindResult with recommended LRs, loss curves, and plot path.
    """
    device_ = _resolve_device(device)
    os.makedirs(out_dir, exist_ok=True)
    model.to(device_)

    # Save per-group base LRs for proportional scaling in Phase 2.
    # These are the LRs the caller configured, not the calibration LR.
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    # ── Phase 1: Calibration pre-pass ────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  LR Finder — Phase 1: Calibration pre-pass")
    print(f"  Steps      : {calibration_steps}  at LR={calibration_lr:.1e}")
    print(f"{'─'*60}")

    # Re-register quantizers so the manager coordinates the current model's
    # quantizers with fresh counters.
    _reset_and_register(model)

    mgr = QuantizerManager()
    # annealing_steps=1 → alpha_step=1.0 → quantizer reaches alpha=1.0 in one pass
    mgr.set_annealing_for_n_inferences(1)
    mgr.quantization_start_gap = calibration_qat_gap

    _set_all_lrs(optimizer, calibration_lr)
    model.train()

    loader_iter = _inf_loader(train_loader)
    for step in range(calibration_steps):
        inputs, targets = _to_device(next(loader_iter), device_)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        print(f"  calib {step+1:2d}/{calibration_steps}  loss={loss.item():.5f}")

    print("[lr_finder] Phase 1 complete — first quantizer calibrated ✓")

    # Snapshot calibrated state (in-memory; no checkpoint file written)
    calib_model_state = copy.deepcopy(model.state_dict())
    calib_optim_state = copy.deepcopy(optimizer.state_dict())
    calib_quant_attrs = _save_quant_attrs(model)
    calib_mgr_seq_id  = mgr._inference_sequence_id_counter

    # ── Phase 2: LR range sweep ───────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  LR Finder — Phase 2: LR sweep")
    print(f"  Range      : {sweep_start_lr:.1e} → {sweep_end_lr:.1e}")
    print(f"  Steps      : {sweep_steps}")
    print(f"{'─'*60}")

    # The quantizer manager state from Phase 1 is preserved: the first quantizer
    # is calibrated and fully quantized; subsequent quantizers gate in as their
    # counter thresholds are crossed during the sweep.
    lrs: List[float] = []
    raw_losses: List[float] = []
    smooth_losses: List[float] = []

    lr_mult = (sweep_end_lr / sweep_start_lr) ** (1.0 / (sweep_steps - 1))
    current_lr = sweep_start_lr
    smoothed = 0.0
    best_smoothed = float("inf")

    loader_iter = _inf_loader(train_loader)
    model.train()

    for step in range(sweep_steps):
        _set_lrs_proportional(optimizer, current_lr, base_lrs)

        inputs, targets = _to_device(next(loader_iter), device_)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        raw = loss.item()
        smoothed = ema_beta * smoothed + (1.0 - ema_beta) * raw
        debiased = smoothed / (1.0 - ema_beta ** (step + 1))

        lrs.append(current_lr)
        raw_losses.append(raw)
        smooth_losses.append(debiased)

        best_smoothed = min(best_smoothed, debiased)

        print(
            f"  sweep {step+1:3d}/{sweep_steps}  lr={current_lr:.2e}"
            f"  loss={raw:.5f}  smooth={debiased:.5f}"
        )

        if not math.isfinite(debiased) or debiased > diverge_factor * best_smoothed:
            print(
                f"[lr_finder] Loss diverged at step {step+1} "
                f"(smooth={debiased:.4f} > {diverge_factor}× best={best_smoothed:.4f}). "
                f"Stopping sweep early."
            )
            break

        current_lr *= lr_mult

    # ── Derive suggested LRs ─────────────────────────────────────────────────
    steep_lr, min_loss_lr = _suggest_lr(lrs, smooth_losses)
    conservative_lr = min_loss_lr / 10.0
    suggested_lr = steep_lr

    # ── Restore post-Phase-1 calibrated state ─────────────────────────────────
    print("\n[lr_finder] Restoring post-Phase-1 calibrated state …")
    model.load_state_dict(calib_model_state)
    optimizer.load_state_dict(calib_optim_state)
    # optimizer.load_state_dict restores the saved LRs (which were 1e-10 from
    # Phase 1); put the original caller-configured LRs back.
    for g, lr in zip(optimizer.param_groups, base_lrs):
        g["lr"] = lr
    # Restore per-quantizer non-buffer attributes and manager counter
    _restore_quant_attrs(model, calib_quant_attrs)
    mgr._inference_sequence_id_counter = calib_mgr_seq_id
    print("[lr_finder] State restored ✓")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_path = _save_plot(
        lrs=lrs,
        raw_losses=raw_losses,
        smooth_losses=smooth_losses,
        steep_lr=steep_lr,
        min_loss_lr=min_loss_lr,
        out_dir=out_dir,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  LR Finder Summary")
    print(f"{'─'*60}")
    print(f"  Steepest descent LR  : {steep_lr:.2e}   ← primary recommendation")
    print(f"  Min-loss LR          : {min_loss_lr:.2e}")
    print(f"  Conservative pick    : {conservative_lr:.2e}   (min-loss / 10)")
    print(f"  Plot saved to        : {plot_path}")
    print(f"{'═'*60}\n")

    return LRFindResult(
        lrs=lrs,
        losses=smooth_losses,
        raw_losses=raw_losses,
        steep_lr=steep_lr,
        min_loss_lr=min_loss_lr,
        conservative_lr=conservative_lr,
        suggested_lr=suggested_lr,
        plot_path=plot_path,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)



def _set_all_lrs(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def _set_lrs_proportional(
    optimizer: torch.optim.Optimizer,
    target_lr: float,
    base_lrs: List[float],
) -> None:
    """
    Set each param group's LR proportionally to its base LR relative to the
    first group.  If all groups had the same base LR (the common case), every
    group is simply set to target_lr.
    """
    ref = base_lrs[0] if base_lrs[0] != 0.0 else 1.0
    for g, base in zip(optimizer.param_groups, base_lrs):
        g["lr"] = target_lr * (base / ref)


def _inf_loader(loader: DataLoader) -> Iterator:
    """Cycle through a DataLoader indefinitely."""
    while True:
        for batch in loader:
            yield batch


def _to_device(batch, device: torch.device):
    inputs, targets = _unpack_batch(batch)
    return inputs.to(device), targets.to(device)


def _save_quant_attrs(model: nn.Module) -> Dict[str, dict]:
    """Save non-buffer per-quantizer attributes that state_dict does not capture."""
    from quantizers.base_quantizer import BaseQuantizer
    attrs = {}
    for name, mod in model.named_modules():
        if isinstance(mod, BaseQuantizer):
            attrs[name] = {
                "inference_counter":    mod.inference_counter,
                "inference_sequence_id": mod.inference_sequence_id,
                "annealing_alpha_step": mod.annealing_alpha_step,
                "_calibration_count":   mod._calibration_count,
                "_was_annealing":       mod._was_annealing,
                "_post_annealing_fired": mod._post_annealing_fired,
                "_last_snapshot_seen":  mod._last_snapshot_seen,
            }
    return attrs


def _restore_quant_attrs(model: nn.Module, attrs: Dict[str, dict]) -> None:
    from quantizers.base_quantizer import BaseQuantizer
    for name, mod in model.named_modules():
        if isinstance(mod, BaseQuantizer) and name in attrs:
            a = attrs[name]
            mod.inference_counter      = a["inference_counter"]
            mod.inference_sequence_id  = a["inference_sequence_id"]
            mod.annealing_alpha_step   = a["annealing_alpha_step"]
            mod._calibration_count     = a["_calibration_count"]
            mod._was_annealing         = a["_was_annealing"]
            mod._post_annealing_fired  = a["_post_annealing_fired"]
            mod._last_snapshot_seen    = a["_last_snapshot_seen"]


def _suggest_lr(
    lrs: List[float],
    smooth_losses: List[float],
) -> Tuple[float, float]:
    """
    Return (steep_lr, min_loss_lr).

    steep_lr:     LR at the point of steepest negative gradient in the
                  smoothed loss curve (d_loss / d_log10_lr most negative).
    min_loss_lr:  LR at the minimum smoothed loss.
    """
    if len(smooth_losses) < 2:
        lr = lrs[0] if lrs else 1e-4
        return lr, lr

    log_lrs = [math.log10(lr) for lr in lrs]

    # Gradient: d(smooth_loss) / d(log10_lr) between consecutive steps
    grads = [
        (smooth_losses[i + 1] - smooth_losses[i]) / max(log_lrs[i + 1] - log_lrs[i], 1e-12)
        for i in range(len(smooth_losses) - 1)
    ]
    steep_idx = min(range(len(grads)), key=lambda i: grads[i])
    steep_lr  = lrs[steep_idx]

    min_loss_idx = min(range(len(smooth_losses)), key=lambda i: smooth_losses[i])
    min_loss_lr  = lrs[min_loss_idx]

    return steep_lr, min_loss_lr


def _save_plot(
    *,
    lrs: List[float],
    raw_losses: List[float],
    smooth_losses: List[float],
    steep_lr: float,
    min_loss_lr: float,
    out_dir: str,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(lrs, raw_losses,    color="lightsteelblue", linewidth=0.8,
            alpha=0.6, label="Raw batch loss")
    ax.plot(lrs, smooth_losses, color="steelblue",      linewidth=2.0,
            label="Smoothed loss (EMA)")

    # Suggested LR markers
    ymin, ymax = ax.get_ylim()
    ax.axvline(steep_lr,   color="orangered", linestyle="--", linewidth=1.5,
               label=f"Steepest descent  {steep_lr:.2e}  ← recommended")
    ax.axvline(min_loss_lr, color="green",    linestyle=":",  linewidth=1.5,
               label=f"Min-loss LR  {min_loss_lr:.2e}")
    ax.axvline(min_loss_lr / 10.0, color="darkgreen", linestyle="-.", linewidth=1.2,
               label=f"Conservative (min/10)  {min_loss_lr/10:.2e}")

    ax.set_xscale("log")
    ax.set_xlabel("Learning rate (log scale)", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("LR Range Test — loss vs learning rate", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    ts = time.strftime("%Y%m%d_%H%M%S")
    base = Path(out_dir) / f"lr_finder_{ts}"
    svg_path = str(base.with_suffix(".svg"))
    png_path = str(base.with_suffix(".png"))
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

    return png_path
