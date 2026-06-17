"""
Quantizer diagnostics: per-event metrics, text log, and matplotlib plot.

Called from BaseQuantizer at three trigger points:
  - calibration_N : first time (or Nth forced recalibration) search_done becomes True
  - post_annealing: alpha first reaches 1.0 after having been < 1.0
  - snapshot_NNNN : on-demand via QuantizerManager.request_snapshot()

All diagnostics measure the *ideal* quantized tensor (no annealing blend), so
they reflect the real quantizer error at full strength.

Activation tensors during ImageNet training can reach hundreds of millions of
elements (batch=1024 × spatial × channels). To avoid moving huge tensors to
CPU and passing them to numpy/matplotlib:
  - All scalar metrics are computed via PyTorch on the original device.
  - Only a random subsample of at most MAX_PLOT_SAMPLES elements is moved to
    CPU for the histogram and bar chart.
  - Metrics in the log and info box always reflect the FULL tensor.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import numpy as np

# Maximum number of individual values sent to numpy/matplotlib for plotting.
# Metrics are always computed on the full tensor regardless of this limit.
MAX_PLOT_SAMPLES = 100_000


# ---------------------------------------------------------------------------
# Metric computation  (runs on original device — no large CPU transfer)
# ---------------------------------------------------------------------------

def _compute_metrics(
    x: torch.Tensor,
    quantized: torch.Tensor,
    lsb: int,
    bit_width: int,
    signed: bool,
    input_shape: Tuple[int, ...],
    quantizer_role: str,
) -> Dict[str, Any]:
    """All heavy reductions stay on the tensor's original device."""
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

    # unique_vals has at most 2^bit_width entries — safe to transfer to CPU
    unique_vals = torch.unique(q_f.ravel())
    n_unique = unique_vals.numel()

    clip_low  = int((x_f < q_min).sum().item())
    clip_high = int((x_f > q_max).sum().item())
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
        "clip_low_pct":    clip_low_pct,
        "clip_high_pct":   clip_high_pct,
        "total_clip_pct":  clip_low_pct + clip_high_pct,
        "mae":             mae,
        "max_ae":          max_ae,
        "mse":             mse,
        "sqnr_db":         sqnr_db,
        "input_mean":      x_f.mean().item(),
        "input_std":       x_f.std().item(),
        "input_min":       x_f.min().item(),
        "input_max":       x_f.max().item(),
        "unique_vals":     unique_vals.cpu(),  # small: ≤ 2^bit_width entries
    }


# ---------------------------------------------------------------------------
# Text log
# ---------------------------------------------------------------------------

def _append_log(log_path: Path, quant_id: str, trigger: str, m: Dict[str, Any]) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sqnr_str = f"{m['sqnr_db']:.2f} dB" if math.isfinite(m["sqnr_db"]) else str(m["sqnr_db"])

    n_total = m["n_elements"]
    n_plot  = m.get("n_plot_samples", n_total)
    sample_line = (f"  Plot sample        : {n_plot:,} / {n_total:,} (random subsample)\n"
                   if n_plot < n_total else "")

    shape_str = "×".join(str(d) for d in m["input_shape"])

    lines = [
        "",
        f"{'='*60}",
        f"  Quantizer : {quant_id}   Event : {trigger}   {ts}",
        f"{'='*60}",
        f"  Role               : {m['quantizer_role']}",
        f"  Input shape        : ({shape_str})   elements: {n_total:,}",
        sample_line +
        f"  Bit width          : {m['bit_width']}b  ({'signed' if m['signed'] else 'unsigned'})",
        f"  LSB position       : {m['lsb']}   step = {m['step']:.6e}",
        f"  Representable range: [{m['q_min']:.6e}, {m['q_max']:.6e}]",
        f"  Representable codes: {m['n_representable']}",
        f"  Unique quant vals  : {m['n_unique']} / {m['n_representable']}"
        f"  ({m['coverage_pct']:.1f}% coverage)",
        f"  Clipping (low)     : {m['clip_low_pct']:.2f}%",
        f"  Clipping (high)    : {m['clip_high_pct']:.2f}%",
        f"  Total clipping     : {m['total_clip_pct']:.2f}%",
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

def _draw_axes(
    ax,
    centers: np.ndarray,
    counts: np.ndarray,
    bw: float,
    q_pos: np.ndarray,
    q_heights: np.ndarray,
    bar_w: float,
    m: Dict[str, Any],
    log_scale: bool,
) -> None:
    """Draw histogram + quantized bars + range markers onto one axes.

    For log scale: yscale is set BEFORE drawing and zero-count bins are
    filtered out. Drawing bars with count=0 and then switching to log scale
    causes matplotlib to render them as mirrored floor-stubs.
    """
    n_plot = m.get("n_plot_samples", int(counts.sum()))
    float_label = (f"Float input  ({n_plot:,} sampled)" if n_plot < m["n_elements"]
                   else f"Float input  ({n_plot:,} values)")
    q_label = f"Quantized  ({m['n_unique']} unique / {m['n_representable']} representable)"

    if log_scale:
        # Set scale FIRST so bars are drawn into the correct coordinate system.
        # Also set explicit ylim top so the axis doesn't auto-expand to the
        # matplotlib default (which can add several extra decades of empty space).
        hist_mask = counts > 0
        q_mask    = q_heights > 0
        y_max = max(
            int(counts[hist_mask].max()) if hist_mask.any() else 1,
            int(q_heights[q_mask].max()) if q_mask.any()   else 1,
        )
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5, top=y_max * 3)
        # Filter zero counts — log(0) = -inf renders as downward stubs
        ax.bar(centers[hist_mask], counts[hist_mask], width=bw,
               color="steelblue", alpha=0.55, label=float_label)
        ax.bar(q_pos[q_mask], q_heights[q_mask], width=bar_w,
               color="orangered", alpha=0.80, label=q_label)
    else:
        ax.bar(centers, counts, width=bw,
               color="steelblue", alpha=0.55, label=float_label)
        ax.bar(q_pos, q_heights, width=bar_w,
               color="orangered", alpha=0.80, label=q_label)

    ax.axvline(m["q_min"], color="crimson", linestyle="--", linewidth=1.2,
               alpha=0.75, label="Quant range")
    ax.axvline(m["q_max"], color="crimson", linestyle="--", linewidth=1.2,
               alpha=0.75)

    ax.set_xlabel("Value")
    ax.set_ylabel("Count (log)" if log_scale else "Count")
    ax.set_title("Log Y axis" if log_scale else "Linear Y axis")
    ax.legend(loc="upper right", fontsize=8)


def _save_plot(
    plot_path: Path,
    x_cpu: torch.Tensor,
    q_cpu: torch.Tensor,
    quant_id: str,
    trigger: str,
    m: Dict[str, Any],
) -> None:
    """x_cpu and q_cpu are already on CPU and already subsampled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_np = x_cpu.float().numpy().ravel()
    q_np = q_cpu.float().numpy().ravel()
    step = m["step"]

    n_bins = 2 ** (m["bit_width"] + 2)
    x_lo = min(float(x_np.min()), m["q_min"])
    x_hi = max(float(x_np.max()), m["q_max"])
    pad  = (x_hi - x_lo) * 0.05 if x_hi > x_lo else abs(step)
    plot_range = (x_lo - pad, x_hi + pad)

    counts, edges = np.histogram(x_np, bins=n_bins, range=plot_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bw = edges[1] - edges[0]

    # q_counts: iterate over the sampled q_np (≤ MAX_PLOT_SAMPLES entries)
    q_counts: Dict[float, int] = {}
    for v in q_np:
        q_counts[float(v)] = q_counts.get(float(v), 0) + 1
    q_pos     = np.array(sorted(q_counts.keys()))
    q_heights = np.array([q_counts[v] for v in q_pos])
    bar_w = max(step * 0.55, bw * 0.5)

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(20, 6))
    _draw_axes(ax_lin, centers, counts, bw, q_pos, q_heights, bar_w, m, log_scale=False)
    _draw_axes(ax_log, centers, counts, bw, q_pos, q_heights, bar_w, m, log_scale=True)

    # Info box on the left subplot only — metrics are always from the FULL tensor
    sqnr_str = f"{m['sqnr_db']:.1f} dB" if math.isfinite(m["sqnr_db"]) else str(m["sqnr_db"])
    n_total = m["n_elements"]
    n_plot  = m.get("n_plot_samples", n_total)
    sample_line = (f"\nPlot sample: {n_plot:,} / {n_total:,} (random)"
                   if n_plot < n_total else "")
    shape_str = "×".join(str(d) for d in m["input_shape"])

    info = (
        f"ID: {quant_id}  |  Role: {m['quantizer_role']}  |  Event: {trigger}\n"
        f"Input shape: ({shape_str})   elements: {n_total:,}{sample_line}\n"
        f"Bit width : {m['bit_width']}b {'S' if m['signed'] else 'U'}  "
        f"LSB={m['lsb']}  step={m['step']:.3e}\n"
        f"Range     : [{m['q_min']:.3e}, {m['q_max']:.3e}]  "
        f"({m['n_representable']} codes)\n"
        f"Unique    : {m['n_unique']} / {m['n_representable']}"
        f"  ({m['coverage_pct']:.1f}% coverage)\n"
        f"Clipping  : {m['total_clip_pct']:.2f}%"
        f"  (↓{m['clip_low_pct']:.2f}%  ↑{m['clip_high_pct']:.2f}%)\n"
        f"MAE={m['mae']:.2e}  MaxAE={m['max_ae']:.2e}  SQNR={sqnr_str}\n"
        f"Input  μ={m['input_mean']:.2e}  σ={m['input_std']:.2e}"
        f"  [{m['input_min']:.2e}, {m['input_max']:.2e}]"
    )
    ax_lin.text(
        0.01, 0.98, info,
        transform=ax_lin.transAxes, fontsize=8, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.90),
    )

    role = m.get("quantizer_role", "unknown")
    fig.suptitle(
        f"Quantizer Diagnostics — {quant_id}  [{role}]  [{trigger}]",
        fontsize=11,
    )
    plt.tight_layout()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, format="svg", bbox_inches="tight")
    fig.savefig(plot_path.with_suffix(".png"), dpi=400, bbox_inches="tight")
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
    Compute metrics on the full tensor (on its original device), then
    subsample down to MAX_PLOT_SAMPLES before touching numpy/matplotlib.
    """
    x_d = x.detach()
    q_d = quantized.detach()
    input_shape = tuple(x_d.shape)

    with torch.no_grad():
        m = _compute_metrics(x_d, q_d, lsb, bit_width, signed, input_shape, quantizer_role)

    # Subsample for plotting — the only large CPU transfer
    n_total = x_d.numel()
    n_plot  = min(n_total, MAX_PLOT_SAMPLES)
    m["n_plot_samples"] = n_plot

    if n_total > MAX_PLOT_SAMPLES:
        idx   = torch.randperm(n_total, device=x_d.device)[:MAX_PLOT_SAMPLES]
        x_cpu = x_d.ravel()[idx].cpu()
        q_cpu = q_d.ravel()[idx].cpu()
    else:
        x_cpu = x_d.cpu()
        q_cpu = q_d.cpu()

    log_path  = Path(out_dir) / f"quantizer_{quant_id}.txt"
    safe      = trigger.replace(" ", "_")
    plot_path = Path(out_dir) / f"quantizer_{quant_id}_{safe}.svg"

    _append_log(log_path, quant_id, trigger, m)
    _save_plot(plot_path, x_cpu, q_cpu, quant_id, trigger, m)

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
