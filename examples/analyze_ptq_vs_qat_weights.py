"""
analyze_ptq_vs_qat_weights.py — compare the PTQ and QAT checkpoints layer by
layer: how did QAT move the weights, and would a different LSB position now make
more sense?

Paths are hardcoded (this is an analysis/understanding tool, not a CLI):
  PTQ = the post-training-quantization search result (weights = pretrained,
        LSBs = the radius-7 search) — the ancestor the QAT chain started from.
  QAT = the current best checkpoint of the weight-only QAT chain.

Both must carry the SAME quantizer settings (LSB / signedness); that is asserted
as a sanity check up front, so any difference in the plots is caused by the
weights moving, not by a different grid.

For every quantized tensor (each layer's weight, plus the classifier bias) one
figure is written with 2 rows x 3 columns:

  row 1 = the quantizer's actual LSB
  row 2 = LSB - 1 (one more fractional bit: finer step, HALF the range) — this
          is the "would a shifted LSB be better now?" experiment

  col 1: unquantized weight distributions, PTQ vs QAT overlaid (transparent),
         with the quantizer's representable range marked.
  col 2: the QUANTIZED values as a bar graph of the real grid values (never
         binned — binning quantized values would smear the grid), with the
         unquantized histograms behind at low alpha as context.
  col 3: PTQ-vs-QAT bars for the error/meta metrics (clipping error, rounding
         error, rounding error excluding clipping, and more).

Figures are written to <OUT_ROOT>/PDF/<layer>_<kind>.pdf and
<OUT_ROOT>/SVG/<layer>_<kind>.svg.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Hardcoded config
# --------------------------------------------------------------------------
PTQ_PATH = "output/pretrained_qat_cache/mobilenetv2_W8_A8_B8_r7.pt"
QAT_PATH = "output/mnv2_wonly_best.pt"
OUT_ROOT = "output/analysis_ptq_vs_qat"

BIT_WIDTH = 8          # weight/bias bit width for this chain (extra.role_bit_widths)
WEIGHT_BINS = 1024
BIAS_BINS = 128
NARROW_RANGE = False   # FixedPointPerTensor{Weight,Bias}Quant default

C_PTQ = "#1f77b4"      # blue
C_QAT = "#ff7f0e"      # orange
C_RANGE = "#2ca02c"    # green


# --------------------------------------------------------------------------
# Quantization math (mirrors quantize_fixed_point_with_integers, numpy side)
# --------------------------------------------------------------------------

def _int_range(bit_width: int, signed: bool, narrow_range: bool = NARROW_RANGE) -> Tuple[int, int]:
    if signed:
        imin = -(2 ** (bit_width - 1))
        if narrow_range:
            imin += 1
        imax = 2 ** (bit_width - 1) - 1
    else:
        imin, imax = 0, 2 ** bit_width - 1
    return imin, imax


def quantize(w: np.ndarray, lsb: int, bit_width: int, signed: bool):
    """Round-to-nearest then clamp onto the fixed-point grid (RoundingMode.ROUND
    == floor(x + 0.5), which is what the weight/bias quantizers use)."""
    step = 2.0 ** lsb
    imin, imax = _int_range(bit_width, signed)
    ints = np.floor(w / step + 0.5)
    ints = np.clip(ints, imin, imax)
    q = ints * step
    return q, ints, imin * step, imax * step, step


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

# Bars in col 3 (all share "weight units" or are small positives -> log axis)
BAR_METRICS = [
    "mean |err| all",
    "mean |err| in-range",
    "mean |clip err|",
    "sum |clip err|",
    "sum |err| all",
    "max |err|",
    "RMSE",
    "max |w|",
    "std(w)",
]


def metrics(w: np.ndarray, lsb: int, bit_width: int, signed: bool) -> Dict[str, float]:
    q, ints, qmin, qmax, step = quantize(w, lsb, bit_width, signed)
    err = q - w
    abs_err = np.abs(err)
    # "clipping error" = the error of values that fell OUTSIDE the representable
    # range (they were clamped). A value inside the range that merely rounds is
    # NOT a clipping error.
    outside = (w < qmin) | (w > qmax)
    inside = ~outside

    sig_pow = float(np.mean(w ** 2))
    err_pow = float(np.mean(err ** 2))
    uniq = int(np.unique(ints).size)
    max_abs_w = float(np.abs(w).max())
    span = max(abs(qmax), abs(qmin))

    m: Dict[str, float] = {
        "mean |err| all": float(abs_err.mean()),
        "mean |err| in-range": float(abs_err[inside].mean()) if inside.any() else 0.0,
        "mean |clip err|": float(abs_err[outside].mean()) if outside.any() else 0.0,
        "sum |clip err|": float(abs_err[outside].sum()) if outside.any() else 0.0,
        "sum |err| all": float(abs_err.sum()),
        "max |err|": float(abs_err.max()),
        "RMSE": float(math.sqrt(err_pow)),
        "max |w|": max_abs_w,
        "std(w)": float(w.std()),
        # --- extras (reported in the text box, not as bars) ---
        "_clipped_pct": 100.0 * float(outside.mean()),
        "_n_clipped": int(outside.sum()),
        "_sqnr_db": (10.0 * math.log10(sig_pow / err_pow)) if err_pow > 0 else float("inf"),
        "_codes_used": uniq,
        "_codes_pct": 100.0 * uniq / (2 ** bit_width),
        "_range_util_pct": 100.0 * max_abs_w / span if span > 0 else 0.0,
        "_qmin": qmin,
        "_qmax": qmax,
        "_step": step,
        "_n": int(w.size),
    }
    return m


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def _hist(ax, w, bins, color, label, alpha, rng):
    ax.hist(w, bins=bins, range=rng, color=color, alpha=alpha, label=label,
            histtype="stepfilled", linewidth=0)


def _plot_unquantized(ax, w_ptq, w_qat, bins, qmin, qmax, lsb, title):
    lo = float(min(w_ptq.min(), w_qat.min(), qmin))
    hi = float(max(w_ptq.max(), w_qat.max(), qmax))
    pad = 0.02 * (hi - lo + 1e-12)
    rng = (lo - pad, hi + pad)
    _hist(ax, w_ptq, bins, C_PTQ, "PTQ (unquantized)", 0.55, rng)
    _hist(ax, w_qat, bins, C_QAT, "QAT (unquantized)", 0.55, rng)
    ax.axvline(qmin, color=C_RANGE, ls="--", lw=1.8,
               label=f"quantizer range [{qmin:.4g}, {qmax:.4g}]")
    ax.axvline(qmax, color=C_RANGE, ls="--", lw=1.8)
    ax.axvspan(qmin, qmax, color=C_RANGE, alpha=0.05)
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    ax.legend(fontsize=7, loc="upper right")


def _plot_quantized(ax, w_ptq, w_qat, bins, lsb, bit_width, signed, title):
    qp, ip, qmin, qmax, step = quantize(w_ptq, lsb, bit_width, signed)
    qq, iq, _, _, _ = quantize(w_qat, lsb, bit_width, signed)

    # Background context: the unquantized distributions, barely visible, on a
    # hidden twin axis so their counts don't rescale the bar axis. Not legended.
    lo = float(min(w_ptq.min(), w_qat.min(), qmin))
    hi = float(max(w_ptq.max(), w_qat.max(), qmax))
    axbg = ax.twinx()
    axbg.hist(w_ptq, bins=bins, range=(lo, hi), color=C_PTQ, alpha=0.2,
              histtype="stepfilled", linewidth=0)
    axbg.hist(w_qat, bins=bins, range=(lo, hi), color=C_QAT, alpha=0.2,
              histtype="stepfilled", linewidth=0)
    axbg.set_yticks([])
    axbg.grid(False)

    # The real grid values — a bar per occupied code. NEVER binned.
    for ints, color, label in ((ip, C_PTQ, "PTQ (quantized)"), (iq, C_QAT, "QAT (quantized)")):
        vals, counts = np.unique(ints, return_counts=True)
        ax.bar(vals * step, counts, width=step * 0.9, color=color, alpha=0.6,
               label=label, align="center")

    ax.axvline(qmin, color=C_RANGE, ls="--", lw=1.8, label="quantizer range")
    ax.axvline(qmax, color=C_RANGE, ls="--", lw=1.8)
    ax.set_title(title)
    ax.set_xlabel(f"quantized value (grid step = 2^{lsb} = {step:.3g})")
    ax.set_ylabel("count")
    ax.set_zorder(axbg.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.legend(fontsize=7, loc="upper right")


def _plot_metrics(ax, m_ptq, m_qat, lsb, bit_width, title):
    names = BAR_METRICS
    x = np.arange(len(names))
    wbar = 0.38
    vp = np.array([m_ptq[k] for k in names], dtype=float)
    vq = np.array([m_qat[k] for k in names], dtype=float)

    # log axis: clamp zeros to a floor so bars remain drawable
    pos = np.concatenate([vp[vp > 0], vq[vq > 0]])
    floor = (pos.min() * 1e-2) if pos.size else 1e-12
    pp = np.where(vp > 0, vp, floor)
    pq = np.where(vq > 0, vq, floor)

    b1 = ax.bar(x - wbar / 2, pp, wbar, color=C_PTQ, alpha=0.85, label="PTQ")
    b2 = ax.bar(x + wbar / 2, pq, wbar, color=C_QAT, alpha=0.85, label="QAT")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("value (log)")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")

    for rects, vals in ((b1, vp), (b2, vq)):
        for r, v in zip(rects, vals):
            ax.annotate(f"{v:.3g}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=5.5, rotation=90)

    def pct(a, b):
        return "n/a" if a == 0 else f"{(b - a) / abs(a) * 100:+.1f}%"

    txt = (
        f"LSB={lsb}  step={m_ptq['_step']:.4g}  range=[{m_ptq['_qmin']:.4g}, {m_ptq['_qmax']:.4g}]  n={m_ptq['_n']}\n"
        f"{'':22s}{'PTQ':>12s}{'QAT':>12s}{'Δ':>10s}\n"
        f"{'clipped':22s}{m_ptq['_n_clipped']:>12d}{m_qat['_n_clipped']:>12d}\n"
        f"{'clipped %':22s}{m_ptq['_clipped_pct']:>12.3f}{m_qat['_clipped_pct']:>12.3f}\n"
        f"{'SQNR dB':22s}{m_ptq['_sqnr_db']:>12.2f}{m_qat['_sqnr_db']:>12.2f}"
        f"{m_qat['_sqnr_db'] - m_ptq['_sqnr_db']:>+10.2f}\n"
        f"{'codes used':22s}{m_ptq['_codes_used']:>12d}{m_qat['_codes_used']:>12d}\n"
        f"{'codes used %':22s}{m_ptq['_codes_pct']:>12.1f}{m_qat['_codes_pct']:>12.1f}\n"
        f"{'range util %':22s}{m_ptq['_range_util_pct']:>12.1f}{m_qat['_range_util_pct']:>12.1f}\n"
        f"{'RMSE change':22s}{'':>12s}{'':>12s}{pct(m_ptq['RMSE'], m_qat['RMSE']):>10s}"
    )
    ax.text(1.02, 0.5, txt, transform=ax.transAxes, va="center", ha="left",
            family="monospace", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#F7F7F7", ec="#BBBBBB"))


def make_figure(layer: str, kind: str, w_ptq: np.ndarray, w_qat: np.ndarray,
                lsb: int, signed: bool, bit_width: int) -> plt.Figure:
    bins = WEIGHT_BINS if kind == "weight" else BIAS_BINS
    fig, axes = plt.subplots(2, 3, figsize=(34, 13))

    for row, use_lsb in enumerate((lsb, lsb - 1)):
        tag = f"LSB={use_lsb}" + ("  (original)" if row == 0 else f"  (shifted: {lsb} → {lsb-1}, finer step / half range)")
        _, _, qmin, qmax, _ = quantize(w_ptq, use_lsb, bit_width, signed)
        mp = metrics(w_ptq, use_lsb, bit_width, signed)
        mq = metrics(w_qat, use_lsb, bit_width, signed)

        _plot_unquantized(axes[row, 0], w_ptq, w_qat, bins, qmin, qmax, use_lsb,
                          f"Unquantized distribution — {tag}")
        _plot_quantized(axes[row, 1], w_ptq, w_qat, bins, use_lsb, bit_width, signed,
                        f"Quantized values (real grid) — {tag}")
        _plot_metrics(axes[row, 2], mp, mq, use_lsb, bit_width,
                      f"Quantization metrics PTQ vs QAT — {tag}")

    fig.suptitle(
        f"{layer}  [{kind}]   bit_width={bit_width}  signed={signed}   "
        f"PTQ={os.path.basename(PTQ_PATH)}  vs  QAT={os.path.basename(QAT_PATH)}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 0.98, 0.97))
    return fig


# --------------------------------------------------------------------------
# Checkpoint plumbing
# --------------------------------------------------------------------------

def _state(path: str) -> dict:
    p = torch.load(path, map_location="cpu")
    return p.get("model_state_dict", p)


def _quantized_tensors(sd: dict) -> List[Tuple[str, str, str, str, str]]:
    """-> [(layer, kind, param_key, lsb_key, signed_key)] for every quantizer
    that has a matching parameter in the state dict."""
    out = []
    for k in sd:
        if not k.endswith("search_result_lsb"):
            continue
        if ".weight_quant.tensor_quant." in k:
            kind, prefix = "weight", k.split(".weight_quant.tensor_quant.")[0]
            param = f"{prefix}.weight"
        elif ".bias_quant.tensor_quant." in k:
            kind, prefix = "bias", k.split(".bias_quant.tensor_quant.")[0]
            param = f"{prefix}.bias"
        else:
            continue
        if param not in sd:
            continue  # e.g. a Brevitas "ghost" quantizer with no real parameter
        out.append((prefix, kind, param, k, k.replace("search_result_lsb",
                                                      "search_result_is_signed")))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def main() -> None:
    sd_ptq, sd_qat = _state(PTQ_PATH), _state(QAT_PATH)
    items = _quantized_tensors(sd_qat)
    print(f"PTQ: {PTQ_PATH}\nQAT: {QAT_PATH}")
    print(f"quantized tensors in QAT: {len(items)}")

    # ---- sanity check: identical quantizer settings -----------------------
    mismatches = []
    for layer, kind, param, lsb_k, sgn_k in items:
        if lsb_k not in sd_ptq:
            mismatches.append((layer, kind, "missing in PTQ", "", ""))
            continue
        a, b = int(sd_ptq[lsb_k].item()), int(sd_qat[lsb_k].item())
        sa, sb = bool(sd_ptq[sgn_k].item()), bool(sd_qat[sgn_k].item())
        if a != b or sa != sb:
            mismatches.append((layer, kind, "LSB/signed differ", f"{a}/{sa}", f"{b}/{sb}"))
    if mismatches:
        print(f"\n[SANITY CHECK FAILED] {len(mismatches)} quantizer setting mismatch(es):")
        for m in mismatches[:20]:
            print("   ", m)
        raise SystemExit(
            "PTQ and QAT must share identical quantizer settings, otherwise the "
            "comparison conflates 'weights moved' with 'different grid'."
        )
    print(f"[sanity] OK — all {len(items)} quantizers have identical LSB/signedness "
          f"in both checkpoints.\n")

    pdf_dir, svg_dir = os.path.join(OUT_ROOT, "PDF"), os.path.join(OUT_ROOT, "SVG")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    summary = []
    for i, (layer, kind, param, lsb_k, sgn_k) in enumerate(items, 1):
        lsb = int(sd_qat[lsb_k].item())
        signed = bool(sd_qat[sgn_k].item())
        w_ptq = sd_ptq[param].detach().float().numpy().ravel()
        w_qat = sd_qat[param].detach().float().numpy().ravel()

        fig = make_figure(layer, kind, w_ptq, w_qat, lsb, signed, BIT_WIDTH)
        name = f"{layer.replace('.', '_')}_{kind}"
        fig.savefig(os.path.join(pdf_dir, name + ".pdf"), format="pdf", bbox_inches="tight")
        fig.savefig(os.path.join(svg_dir, name + ".svg"), format="svg", bbox_inches="tight")
        plt.close(fig)

        m0p, m0q = metrics(w_ptq, lsb, BIT_WIDTH, signed), metrics(w_qat, lsb, BIT_WIDTH, signed)
        m1p, m1q = metrics(w_ptq, lsb - 1, BIT_WIDTH, signed), metrics(w_qat, lsb - 1, BIT_WIDTH, signed)
        summary.append((layer, kind, lsb, m0p, m0q, m1p, m1q))
        print(f"[{i:>2}/{len(items)}] {name:<44s} LSB={lsb:>3}  "
              f"SQNR PTQ {m0p['_sqnr_db']:6.2f} → QAT {m0q['_sqnr_db']:6.2f} dB   "
              f"clipped {m0p['_clipped_pct']:5.2f}% → {m0q['_clipped_pct']:5.2f}%")

    # ---- roll-up: would shifting the LSB help the QAT weights? -----------
    print(f"\nWrote {len(items)} figures to {pdf_dir} and {svg_dir}\n")
    print("=" * 100)
    print("Would LSB-1 (finer step, half the range) be better for the QAT weights?")
    print("=" * 100)
    print(f"{'layer':<40s}{'kind':<8s}{'LSB':>4s}{'SQNR@LSB':>10s}{'SQNR@LSB-1':>12s}{'Δ dB':>8s}  verdict")
    better = 0
    for layer, kind, lsb, m0p, m0q, m1p, m1q in summary:
        d = m1q["_sqnr_db"] - m0q["_sqnr_db"]
        verdict = "LSB-1 BETTER" if d > 0 else "keep LSB"
        better += d > 0
        print(f"{layer:<40s}{kind:<8s}{lsb:>4d}{m0q['_sqnr_db']:>10.2f}"
              f"{m1q['_sqnr_db']:>12.2f}{d:>+8.2f}  {verdict}")
    print(f"\n{better}/{len(summary)} tensors would get a better SQNR from LSB-1 "
          f"on the QAT weights.")


if __name__ == "__main__":
    main()
