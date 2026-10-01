"""
Quantizer diagnostics: per-event metrics, text log, and matplotlib plot.

Two consumers share the same `compute_metrics()` call:

  1. Lifecycle-event logging (`BaseQuantizer._log_lifecycle_event`, called
     from `forward()`): a rich metrics dict is attached to the
     "calibration_completed"/"calibration_rerun" and "annealing_complete"
     log records (`record.metrics`), via the standard `logging` module --
     no file I/O, nothing written to disk. This always runs (cheap,
     one-shot; never on the per-forward-call hot path) whenever the
     quantizer subclass supports it (`_get_diagnostics_params()` returns
     non-None).
  2. The file-writing path below (`run_diagnostics`, gated behind
     `QuantizerManager.diagnostics_dir`): a text log line + a two-panel SVG/PNG
     plot per event, at three trigger points:
       - calibration_N : first time (or Nth forced recalibration) search_done becomes True
       - post_annealing: alpha first reaches 1.0 after having been < 1.0
       - snapshot_NNNN : on-demand via QuantizerManager.request_snapshot()

All diagnostics measure the *ideal* quantized tensor (no annealing blend), so
they reflect the real quantizer error at full strength.

Activation tensors during ImageNet training can reach hundreds of millions of
elements (batch=1024 × spatial × channels). To avoid moving huge tensors to
CPU:
  - All scalar reductions (including the clip counts and the float-input
    histogram, via `torch.histc`) run on the tensor's original device.
  - Only the resulting small arrays move to CPU: the histogram (`n_hist_bins`
    entries, default 256) and the EXACT per-code quantized-value counts
    (`torch.unique(..., return_counts=True)`, at most `2**bit_width` entries
    -- small enough to transfer in full, so the quantized-value bar chart is
    exact, not an approximation from a random subsample like this module
    used to produce).

Use `plot_quantizer_metrics(metrics)` to turn any metrics dict (from a log
record's `record.metrics`, or a direct `compute_metrics()` call) into a
matplotlib figure yourself -- e.g. from a custom logging.Handler, a
notebook, or after loading metrics you saved some other way.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Metric computation  (runs on original device — no large CPU transfer)
# ---------------------------------------------------------------------------

def compute_metrics(
    x: torch.Tensor,
    quantized: torch.Tensor,
    lsb: int,
    bit_width: int,
    signed: bool,
    quantizer_role: str = "unknown",
    input_shape: Optional[Tuple[int, ...]] = None,
    n_hist_bins: int = 256,
) -> Dict[str, Any]:
    """Compute a rich, self-contained metrics dict for one quantizer event
    (calibration, annealing-complete, or an on-demand snapshot).

    Safe to call on arbitrarily large tensors -- every reduction (including
    the histogram) runs on `x`'s original device; only the small resulting
    arrays (histogram bins, and the quantized tensor's exact per-code value
    counts -- at most `2**bit_width` entries) are moved to CPU/numpy.

    Returns a plain dict (json/pickle-friendly except for the two numpy
    array pairs `hist_counts`/`hist_edges` and `quant_values`/`quant_counts`),
    including:
      - Grid/scale info: `bit_width`, `lsb`, `step`, `signed`, `q_min`, `q_max`,
        `n_representable`.
      - Clipping: `n_clipped`, `n_unclipped` (exact counts) plus
        `clip_low_pct`/`clip_high_pct`/`total_clip_pct`.
      - Coverage: `n_unique` (distinct quantized codes actually used) and
        `coverage_pct` (`n_unique / n_representable`).
      - Range: `input_min`/`input_max` (unquantized) vs `q_min`/`q_max`
        (representable grid).
      - Error: `mae`, `max_ae`, `mse`, `sqnr_db`.
      - Plot data: `hist_counts`/`hist_edges` (float-input histogram,
        `n_hist_bins` bins over `[min(input_min, q_min), max(input_max, q_max)]`)
        and `quant_values`/`quant_counts` (exact quantized-output histogram).
    """
    x_f = x.float()
    q_f = quantized.float()

    step = 2.0 ** lsb

    if signed:
        q_min = -(2 ** (bit_width - 1)) * step
        q_max = (2 ** (bit_width - 1) - 1) * step
    else:
        q_min = 0.0
        q_max = (2 ** bit_width - 1) * step
    n_representable = 2 ** bit_width

    n_elements = x_f.numel()
    if input_shape is None:
        input_shape = tuple(x.shape)

    # Exact per-code quantized value counts -- at most 2**bit_width entries,
    # always safe to move to CPU in full (no subsampling/approximation).
    quant_values, quant_counts = torch.unique(q_f.ravel(), return_counts=True)
    n_unique = quant_values.numel()

    clip_low  = int((x_f < q_min).sum().item())
    clip_high = int((x_f > q_max).sum().item())
    n_clipped = clip_low + clip_high
    clip_low_pct  = 100.0 * clip_low  / max(n_elements, 1)
    clip_high_pct = 100.0 * clip_high / max(n_elements, 1)

    err = x_f - q_f
    mae    = err.abs().mean().item()
    max_ae = err.abs().max().item()
    mse    = (err ** 2).mean().item()

    signal_power = (x_f ** 2).mean().item()
    if mse > 1e-30 and signal_power > 0:
        sqnr_db = 10.0 * math.log10(signal_power / mse)
    elif mse <= 1e-30:
        sqnr_db = float("inf")
    else:
        sqnr_db = float("-inf")

    input_min = x_f.min().item()
    input_max = x_f.max().item()

    # Histogram of the (unquantized) float input, computed on-device via
    # torch.histc where possible (works directly on CUDA -- no CPU transfer
    # of raw data). torch.histc has no deterministic CUDA kernel, though, so
    # it raises under `torch.use_deterministic_algorithms(True)` (common in
    # reproducibility-focused training runs/tests) -- fall back to a CPU
    # numpy histogram in that case. This is a rare, one-shot event (at most
    # a couple of times per quantizer over an entire run), so the fallback's
    # CPU transfer is an acceptable, infrequent cost -- not a hot-path one.
    hist_lo = min(input_min, q_min)
    hist_hi = max(input_max, q_max)
    if hist_hi <= hist_lo:
        hist_hi = hist_lo + max(abs(step), 1e-12)
    try:
        hist_counts_t = torch.histc(x_f, bins=n_hist_bins, min=hist_lo, max=hist_hi)
        hist_counts = hist_counts_t.cpu().numpy()
    except RuntimeError:
        hist_counts, _ = np.histogram(
            x_f.detach().cpu().numpy(), bins=n_hist_bins, range=(hist_lo, hist_hi)
        )
    hist_edges = np.linspace(hist_lo, hist_hi, n_hist_bins + 1)

    return {
        "bit_width":       bit_width,
        "lsb":             lsb,
        "step":            step,
        "signed":          signed,
        "n_representable": n_representable,
        "q_min":           q_min,
        "q_max":           q_max,
        "n_elements":      n_elements,
        "input_shape":     input_shape,
        "quantizer_role":  quantizer_role,
        "n_unique":        n_unique,
        "coverage_pct":    100.0 * n_unique / n_representable,
        "n_clipped":       n_clipped,
        "n_unclipped":     n_elements - n_clipped,
        "clip_low_pct":    clip_low_pct,
        "clip_high_pct":   clip_high_pct,
        "total_clip_pct":  clip_low_pct + clip_high_pct,
        "mae":             mae,
        "max_ae":          max_ae,
        "mse":             mse,
        "sqnr_db":         sqnr_db,
        "input_mean":      x_f.mean().item(),
        "input_std":       x_f.std().item(),
        "input_min":       input_min,
        "input_max":       input_max,
        "hist_counts":     hist_counts,
        "hist_edges":      hist_edges,
        "quant_values":    quant_values.cpu().numpy(),
        "quant_counts":    quant_counts.cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Text log
# ---------------------------------------------------------------------------

def _append_log(log_path: Path, quant_id: str, trigger: str, m: Dict[str, Any]) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sqnr_str = f"{m['sqnr_db']:.2f} dB" if math.isfinite(m["sqnr_db"]) else str(m["sqnr_db"])

    n_total = m["n_elements"]
    shape_str = "×".join(str(d) for d in m["input_shape"])

    lines = [
        "",
        f"{'='*60}",
        f"  Quantizer : {quant_id}   Event : {trigger}   {ts}",
        f"{'='*60}",
        f"  Role               : {m['quantizer_role']}",
        f"  Input shape        : ({shape_str})   elements: {n_total:,}",
        f"  Bit width          : {m['bit_width']}b  ({'signed' if m['signed'] else 'unsigned'})",
        f"  LSB position       : {m['lsb']}   step = {m['step']:.6e}",
        f"  Representable range: [{m['q_min']:.6e}, {m['q_max']:.6e}]",
        f"  Representable codes: {m['n_representable']}",
        f"  Unique quant vals  : {m['n_unique']} / {m['n_representable']}"
        f"  ({m['coverage_pct']:.1f}% coverage)",
        f"  Clipped values     : {m['n_clipped']:,} / {n_total:,}  ({m['total_clip_pct']:.2f}%)"
        f"  [low: {m['clip_low_pct']:.2f}%  high: {m['clip_high_pct']:.2f}%]",
        f"  Unclipped values   : {m['n_unclipped']:,} / {n_total:,}",
        f"  MAE                : {m['mae']:.6e}",
        f"  Max AE             : {m['max_ae']:.6e}",
        f"  MSE                : {m['mse']:.6e}",
        f"  SQNR               : {sqnr_str}",
        f"  Input mean         : {m['input_mean']:.6e}",
        f"  Input std          : {m['input_std']:.6e}",
        f"  Input range        : [{m['input_min']:.6e}, {m['input_max']:.6e}]",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _info_box_text(m: Dict[str, Any], quant_id: str, trigger: str) -> str:
    sqnr_str = f"{m['sqnr_db']:.1f} dB" if math.isfinite(m["sqnr_db"]) else str(m["sqnr_db"])
    n_total = m["n_elements"]
    shape_str = "×".join(str(d) for d in m["input_shape"])
    return (
        f"ID: {quant_id}  |  Role: {m['quantizer_role']}  |  Event: {trigger}\n"
        f"Input shape: ({shape_str})   elements: {n_total:,}\n"
        f"Bit width : {m['bit_width']}b {'S' if m['signed'] else 'U'}  "
        f"LSB={m['lsb']}  step={m['step']:.3e}\n"
        f"Range     : [{m['q_min']:.3e}, {m['q_max']:.3e}]  "
        f"({m['n_representable']} codes)\n"
        f"Unique    : {m['n_unique']} / {m['n_representable']}"
        f"  ({m['coverage_pct']:.1f}% coverage)\n"
        f"Clipped   : {m['n_clipped']:,} / {n_total:,}  ({m['total_clip_pct']:.2f}%)"
        f"  (↓{m['clip_low_pct']:.2f}%  ↑{m['clip_high_pct']:.2f}%)\n"
        f"Unclipped : {m['n_unclipped']:,} / {n_total:,}\n"
        f"MAE={m['mae']:.2e}  MaxAE={m['max_ae']:.2e}  SQNR={sqnr_str}\n"
        f"Input  μ={m['input_mean']:.2e}  σ={m['input_std']:.2e}"
        f"  [{m['input_min']:.2e}, {m['input_max']:.2e}]"
    )


def plot_quantizer_metrics(
    metrics: Dict[str, Any],
    *,
    ax=None,
    log_scale: bool = False,
    title: Optional[str] = None,
    quant_id: str = "",
    trigger: str = "",
    info_box: bool = True,
):
    """Plot one quantizer metrics dict (from `compute_metrics()`, or
    `record.metrics` off a "calibration_completed"/"calibration_rerun"/
    "annealing_complete" log record) as a histogram of the float input with
    the quantized output's exact value counts overlaid, plus the
    representable range as vertical markers.

    This is the "helper function" to turn a metrics dict into a figure
    yourself -- e.g. from a custom `logging.Handler`, a notebook, or a
    metrics dict loaded back from wherever you saved it:

        import logging
        from utils.quantizer_diagnostics import plot_quantizer_metrics

        class PlotOnCalibration(logging.Handler):
            def emit(self, record):
                if getattr(record, "event", None) == "calibration_completed":
                    fig = plot_quantizer_metrics(record.metrics, quant_id=record.quant_id)
                    fig.savefig(f"{record.quant_id}_calibration.png")

        logging.getLogger("quantizers").addHandler(PlotOnCalibration())

    Parameters
    ----------
    metrics : the dict returned by `compute_metrics()`.
    ax : an existing `matplotlib.axes.Axes` to draw into; a new figure (with
        one axes) is created if omitted.
    log_scale : log-scale the Y axis (useful when a few codes dominate the
        count and rare-but-important codes would otherwise be invisible).
    title : overrides the default `"{quant_id} [{trigger}]"` title.
    quant_id, trigger : only used for the default title/info-box text.
    info_box : draw the metrics summary (LSB, clipping, SQNR, ...) as a text
        box in the corner.

    Returns
    -------
    The `matplotlib.figure.Figure` the axes belongs to.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    m = metrics
    hist_counts = np.asarray(m["hist_counts"])
    hist_edges  = np.asarray(m["hist_edges"])
    centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
    bw = hist_edges[1] - hist_edges[0]

    q_pos     = np.asarray(m["quant_values"])
    q_heights = np.asarray(m["quant_counts"])
    bar_w = max(abs(m["step"]) * 0.55, bw * 0.5)

    float_label = f"Float input  ({m['n_elements']:,} values)"
    q_label = f"Quantized  ({m['n_unique']} unique / {m['n_representable']} representable)"

    if log_scale:
        hist_mask = hist_counts > 0
        q_mask    = q_heights > 0
        y_max = max(
            int(hist_counts[hist_mask].max()) if hist_mask.any() else 1,
            int(q_heights[q_mask].max()) if q_mask.any() else 1,
        )
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5, top=y_max * 3)
        ax.bar(centers[hist_mask], hist_counts[hist_mask], width=bw,
               color="steelblue", alpha=0.55, label=float_label)
        ax.bar(q_pos[q_mask], q_heights[q_mask], width=bar_w,
               color="orangered", alpha=0.80, label=q_label)
    else:
        ax.bar(centers, hist_counts, width=bw,
               color="steelblue", alpha=0.55, label=float_label)
        ax.bar(q_pos, q_heights, width=bar_w,
               color="orangered", alpha=0.80, label=q_label)

    ax.axvline(m["q_min"], color="crimson", linestyle="--", linewidth=1.2,
               alpha=0.75, label="Quant range")
    ax.axvline(m["q_max"], color="crimson", linestyle="--", linewidth=1.2, alpha=0.75)

    ax.set_xlabel("Value")
    ax.set_ylabel("Count (log)" if log_scale else "Count")
    ax.legend(loc="upper right", fontsize=8)
    if title is not None:
        ax.set_title(title)
    elif quant_id or trigger:
        ax.set_title(f"{quant_id} [{trigger}]")

    if info_box:
        ax.text(
            0.01, 0.98, _info_box_text(m, quant_id, trigger),
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.90),
        )

    return fig


def plot_quantizer_metrics_grid(metrics: Dict[str, Any], quant_id: str = "", trigger: str = ""):
    """Side-by-side linear + log-scale version of `plot_quantizer_metrics()`
    -- the two-panel view used for the on-disk diagnostics plots
    (`run_diagnostics`), also handy directly: rare, saturated, or
    near-empty codes are easy to miss on a linear axis alone."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(20, 6))
    plot_quantizer_metrics(metrics, ax=ax_lin, log_scale=False, quant_id=quant_id,
                           trigger=trigger, info_box=True)
    plot_quantizer_metrics(metrics, ax=ax_log, log_scale=True, quant_id=quant_id,
                           trigger=trigger, info_box=False)
    ax_lin.set_title("Linear Y axis")
    ax_log.set_title("Log Y axis")

    role = metrics.get("quantizer_role", "unknown")
    fig.suptitle(f"Quantizer Diagnostics — {quant_id}  [{role}]  [{trigger}]", fontsize=11)
    plt.tight_layout()
    return fig


def _save_plot(
    plot_path: Path,
    quant_id: str,
    trigger: str,
    m: Dict[str, Any],
) -> None:
    fig = plot_quantizer_metrics_grid(m, quant_id=quant_id, trigger=trigger)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, format="svg", bbox_inches="tight")
    fig.savefig(plot_path.with_suffix(".png"), dpi=400, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# LSB search diagnostic plot
# ---------------------------------------------------------------------------

def _save_search_plot(
    *,
    search_records: list,
    best_lsb: int,
    quant_id: str,
    trigger: str,
    quantizer_role: str,
    bit_width: int,
    out_dir: Path,
) -> None:
    """Dual-axis plot of the LSB search: SAD bars (left) + unique-count line (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not search_records:
        return

    from matplotlib.patches import Patch

    # Sort low → high for a natural left-to-right x-axis
    records   = sorted(search_records, key=lambda r: r[0])
    lsb_vals  = [r[0] for r in records]
    n_uniq    = [r[1] for r in records]
    sad_vals  = [r[2] for r in records]
    n_max     = 2 ** bit_width

    global_max_unique = max(n_uniq)
    min_sad           = min(sad_vals)
    min_sad_lsb       = lsb_vals[sad_vals.index(min_sad)]

    fig, ax_sad = plt.subplots(figsize=(14, 5))
    ax_uniq = ax_sad.twinx()

    # ── SAD bars ─────────────────────────────────────────────────────────────
    # orangered = selected, gold = lowest SAD, steelblue = everything else.
    # If selected and lowest-SAD coincide, orangered wins.
    def _bar_color(lsb):
        if lsb == best_lsb:    return "orangered"
        if lsb == min_sad_lsb: return "gold"
        return "steelblue"

    ax_sad.bar(lsb_vals, sad_vals, color=[_bar_color(l) for l in lsb_vals],
               alpha=0.75, width=0.6)
    ax_sad.set_xlabel("LSB position")
    ax_sad.set_ylabel("SAD", color="steelblue")
    ax_sad.tick_params(axis="y", labelcolor="steelblue")

    # ── Unique-values line + markers ─────────────────────────────────────────
    # Line connecting all points, then two scatter series:
    #   • circles  for positions that did NOT reach global max unique count
    #   • stars (★) for positions that DID reach global max unique count
    ax_uniq.plot(lsb_vals, n_uniq, color="green", linewidth=1.5, zorder=2)

    non_max_x = [lsb_vals[i] for i, u in enumerate(n_uniq) if u < global_max_unique]
    non_max_y = [u           for u in n_uniq                if u < global_max_unique]
    if non_max_x:
        ax_uniq.scatter(non_max_x, non_max_y, color="green", marker="o",
                        s=20, zorder=3)

    max_x = [lsb_vals[i] for i, u in enumerate(n_uniq) if u == global_max_unique]
    max_y = [u           for u in n_uniq                if u == global_max_unique]
    ax_uniq.scatter(max_x, max_y, color="darkgreen", marker="*",
                    s=180, zorder=4)

    ax_uniq.axhline(n_max, color="green", linestyle=":", linewidth=1.0, alpha=0.6)
    ax_uniq.set_ylabel(f"Unique values  (max {n_max})", color="green")
    ax_uniq.tick_params(axis="y", labelcolor="green")
    ax_uniq.set_ylim(bottom=0, top=n_max * 1.12)

    # ── Selected LSB marker ───────────────────────────────────────────────────
    ax_sad.axvline(best_lsb, color="red", linestyle="--", linewidth=1.5)

    # ── Integer x-ticks ──────────────────────────────────────────────────────
    ax_sad.set_xticks(lsb_vals)
    ax_sad.tick_params(axis="x", rotation=45)

    # ── Manual legend (mix of bar patches, line, and scatter markers) ─────────
    from matplotlib.lines import Line2D
    selected_label = f"Selected  LSB={best_lsb}"
    if best_lsb == min_sad_lsb:
        selected_label += "  (also min SAD)"
    legend_handles = [
        Patch(facecolor="orangered", alpha=0.75, label=selected_label),
        Patch(facecolor="gold",      alpha=0.75, label=f"Min SAD  LSB={min_sad_lsb}"),
        Patch(facecolor="steelblue", alpha=0.75, label="SAD"),
        Line2D([0], [0], color="green", linewidth=1.5, label="Unique values"),
        Line2D([0], [0], color="darkgreen", marker="*", markersize=9,
               linestyle="None", label=f"Max unique ({global_max_unique}) — {len(max_x)} positions"),
        Line2D([0], [0], color="green", linestyle=":", linewidth=1.0,
               label=f"Max representable ({n_max})"),
    ]
    ax_sad.legend(handles=legend_handles, loc="upper left", fontsize=8)

    rule = ("highest LSB with max unique values"
            if quantizer_role == "activation"
            else "max unique values, SAD tie-break")
    fig.suptitle(
        f"LSB Search — {quant_id}  [{quantizer_role}]  [{trigger}]\n"
        f"Rule: {rule}",
        fontsize=10,
    )
    plt.tight_layout()

    safe = trigger.replace(" ", "_")
    base = out_dir / f"quantizer_{quant_id}_{safe}_lsb_search"
    fig.savefig(base.with_suffix(".svg"), format="svg", bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def _append_search_log(
    log_path: Path,
    quant_id: str,
    trigger: str,
    search_records: list,
    best_lsb: int,
    quantizer_role: str,
) -> None:
    """Append a compact summary of the LSB search to the quantizer's log file."""
    if not search_records:
        return
    records = sorted(search_records, key=lambda r: r[0])
    lsb_vals = [r[0] for r in records]
    rule = ("highest LSB with max unique values"
            if quantizer_role == "activation"
            else "max unique values, SAD tie-break")
    lines = [
        f"  ── LSB Search ({'activation' if quantizer_role == 'activation' else 'weight'} rule) ──",
        f"  Positions tested : LSB {lsb_vals[0]} to {lsb_vals[-1]}  ({len(records)} positions)",
        f"  Selection rule   : {rule}",
        f"  Selected LSB     : {best_lsb}",
        f"  {'LSB':>5}  {'Unique':>7}  {'SAD':>14}",
        f"  {'───':>5}  {'──────':>7}  {'─────────────':>14}",
    ]
    for lsb, n_uniq, sad in records:
        marker = " ◄" if lsb == best_lsb else ""
        lines.append(f"  {lsb:>5}  {n_uniq:>7}  {sad:>14.4e}{marker}")
    with open(log_path, "a") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_diagnostics(
    *,
    quant_id: str,
    x: torch.Tensor,
    quantized: torch.Tensor,
    lsb: int,
    bit_width: int,
    signed: bool,
    quantizer_role: str = "unknown",
    trigger: str,
    out_dir: Path,
    search_records: list = None,
) -> None:
    """
    Compute metrics on the full tensor (on its original device -- see
    `compute_metrics()`), then write the text log line and the two-panel
    plot. No raw tensor data ever reaches CPU/numpy; only the histogram and
    the exact per-code quantized-value counts do.
    """
    x_d = x.detach()
    q_d = quantized.detach()

    with torch.no_grad():
        m = compute_metrics(x_d, q_d, lsb, bit_width, signed, quantizer_role)

    log_path  = Path(out_dir) / f"quantizer_{quant_id}.txt"
    safe      = trigger.replace(" ", "_")
    plot_path = Path(out_dir) / f"quantizer_{quant_id}_{safe}.svg"

    _append_log(log_path, quant_id, trigger, m)
    _save_plot(plot_path, quant_id, trigger, m)

    if search_records:
        _append_search_log(log_path, quant_id, trigger, search_records, lsb, quantizer_role)
        _save_search_plot(
            search_records=search_records,
            best_lsb=lsb,
            quant_id=quant_id,
            trigger=trigger,
            quantizer_role=quantizer_role,
            bit_width=bit_width,
            out_dir=Path(out_dir),
        )
