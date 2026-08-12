"""
analyze_lsb_methods.py — which rule should pick the LSB position?

Takes the pretrained float MobileNetV2 (BatchNorm fused, because that is what
actually gets quantized — and what creates the pathological weight
distributions), and applies every LSB-selection rule below to each weight
tensor. No QAT checkpoint is involved: this compares ways of *creating* a PTQ
grid from float weights, so each histogram carries a single series.

The rules (numbering kept from the design discussion; 1 was dropped because the
shipped rule turned out to BE rule 6):

  2  max unique count -> biggest range (highest LSB)
        == today's ACTIVATION rule (find_optimal_lsb(prefer_high_lsb=True))
  3  lowest MSE                                     (unconstrained)
  4  lowest SAD (sum of absolute errors)            (unconstrained)
  5  max unique count -> lowest MSE
  6  max unique count -> lowest SAD
        == today's WEIGHT rule (find_optimal_lsb(prefer_high_lsb=False))
  8  cover p99.99 of |w|   ] the percentile ladder: each step trims harder,
  7  cover p99.9  of |w|   ] from clipping 0.01% of the weights down to 2%.
 11  cover p99.5  of |w|   ] The LSB is a power of two, so the range can only
 12  cover p99    of |w|   ] halve between candidates — neighbouring rungs will
 13  cover p98    of |w|   ] collapse onto one LSB where the tail is short.
  9  cover +/-4 * robust sigma (MAD)
 10  lowest MSE over the core only (outliers get no vote)

Rules 2 and 6 are asserted against the real find_optimal_lsb, which validates
the whole harness: if our reimplementation disagrees with the shipped function,
every other row is suspect too.

Every rule scans the SAME candidate set (the range find_optimal_lsb uses), or
the comparison would be rigged.

A caveat worth keeping in mind when reading the output: ranking these rules by
an error metric is circular, because whichever metric you rank by crowns its own
optimiser (SQNR is MSE wearing a hat, so rule 3 wins any SQNR contest by
construction). The rules are reported, not scored.

Output, per weight quantizer, named by forward-execution order so 01_* is the
conv that processes the input image:

  <OUT_ROOT>/{PDF,SVG}/<NN>_<layer>.*         one row per rule, 3 cols
  <OUT_ROOT>/sweep/{PDF,SVG}/<NN>_<layer>.*   one row per LSB: starts at the
        finest LSB that clips NOTHING, then -1 (range halves) per row — the
        clipping trade-off itself, independent of any rule

  col 1: unquantized weights, with the representable range marked
  col 2: the QUANTIZED values on the real grid (bars, never binned — binning
         quantized values smears the grid that is the whole point)
  col 3: the error decomposition, split by cause

plus <OUT_ROOT>/lsb_methods.csv and <OUT_ROOT>/{PDF,SVG}/00_summary.*
"""

from __future__ import annotations

import csv
import math
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.analyze_ptq_vs_qat_weights import _int_range, quantize

# --------------------------------------------------------------------------
# Hardcoded config (analysis tool, not a CLI)
# --------------------------------------------------------------------------
MODEL = "mobilenetv2"
BIT_WIDTH = 8
OUT_ROOT = "output/analysis_lsb_methods"

WEIGHT_BINS = 1024
NARROW_RANGE = False    # FixedPointPerTensorWeightQuant default
SEARCH_PAD = 12         # find_optimal_lsb scans floor(ideal)-12 .. ceil(ideal)+12

C_W = "#1f77b4"         # weights
C_Q = "#d62728"         # quantized grid
C_RANGE = "#2ca02c"     # representable range


# --------------------------------------------------------------------------
# Candidate scan — one pass, every rule selects from these records
# --------------------------------------------------------------------------

def candidate_range(w: np.ndarray, bit_width: int, signed: bool) -> List[int]:
    """The exact candidate set find_optimal_lsb uses (fixedpoint_per_tensor.py:192)."""
    abs_max = float(max(abs(w.min()), abs(w.max())))
    if abs_max == 0.0:
        return [0]
    n_pos = (2 ** (bit_width - 1) - 1) if signed else (2 ** bit_width - 1)
    n_pos = max(n_pos, 1)
    ideal = math.log2(abs_max / n_pos)
    return list(range(math.floor(ideal) - SEARCH_PAD, math.ceil(ideal) + SEARCH_PAD + 1))


CORE_PCT = 99.9     # what counts as "the core" of the distribution (methods 7, 10)


def scan(w: np.ndarray, bit_width: int, signed: bool) -> List[Dict]:
    """Every candidate LSB with its stats. Ordered high -> low, like find_optimal_lsb."""
    aw = np.abs(w)
    # The core: everything except the most extreme CORE_PCT tail. Used by the
    # outlier-blind rules -- mse_core deliberately does not see what clipping
    # does to the tail.
    core = aw <= np.percentile(aw, CORE_PCT)
    sig = float((w ** 2).sum())

    recs: List[Dict] = []
    for lsb in reversed(candidate_range(w, bit_width, signed)):
        q, ints, qmin, qmax, step = quantize(w, lsb, bit_width, signed)
        err = w - q
        aerr = np.abs(err)
        clipped = (w < qmin) | (w > qmax)
        n_clip = int(clipped.sum())
        noise = float((err ** 2).sum())
        recs.append(dict(
            lsb=int(lsb), step=float(step), qmin=float(qmin), qmax=float(qmax),
            n_unique=int(np.unique(q).size),
            # --- error split by cause (these add up exactly) ---
            clip_err=float(aerr[clipped].sum()) if n_clip else 0.0,
            round_err=float(aerr[~clipped].sum()),
            sad_total=float(aerr.sum()),            # SAD: sum of |w-q| over EVERY weight
            mse=float((err ** 2).mean()),
            # --- outlier-blind: error over the core only ---
            mse_core=float((err[core] ** 2).mean()),
            # --- context ---
            n_clipped=n_clip,
            pct_clipped=100.0 * n_clip / w.size,
            sqnr=10.0 * math.log10(sig / noise) if noise > 0 else float("inf"),
        ))
    return recs


# --------------------------------------------------------------------------
# The rules. Each takes (scan records, weights) and returns one lsb.
# Ties broken toward the HIGHER lsb (wider range) for determinism.
#
# 2-6 are the original six (1 was dropped: the shipped rule IS 6).
# 7-10 trim outliers: the LSB is really a choice of clipping threshold
# (range = 2^lsb * 2^(bits-1)), and 2-6 sit at the two extremes -- max-unique
# clips ~12.9% (far too aggressive), lowest-MSE clips ~0.008% because squaring
# makes a single 12-sigma weight dominate the objective, so the range ends up
# sized by the most extreme value in the tensor. 7-10 aim at the middle.
# --------------------------------------------------------------------------

def _max_unique(recs: List[Dict]) -> List[Dict]:
    top = max(r["n_unique"] for r in recs)
    return [r for r in recs if r["n_unique"] == top]


def _finest_covering(recs: List[Dict], thr: float) -> int:
    """Finest (smallest) lsb whose representable range still covers +/-thr.

    For signed, |qmin| > qmax, so qmax is the binding side. Falls back to the
    coarsest candidate if thr exceeds every candidate's range.
    """
    ok = [r for r in recs if r["qmax"] >= thr]
    if not ok:
        return max(recs, key=lambda r: r["lsb"])["lsb"]
    return min(ok, key=lambda r: r["lsb"])["lsb"]


# NOTE on the one-liners below: `recs` is the list of CANDIDATE LSBs (~25 of
# them), so min(recs, key=...) is an argmin over candidates -- it picks WHICH
# LSB scores best. The summing already happened in scan(): sad_total is
# sum(|w-q|) over every weight, mse is mean((w-q)^2) over every weight.

def m2_unique_then_range(recs, w):  return max(_max_unique(recs), key=lambda r: r["lsb"])["lsb"]
def m3_lowest_mse(recs, w):         return min(recs, key=lambda r: (r["mse"], -r["lsb"]))["lsb"]
def m4_lowest_sad(recs, w):         return min(recs, key=lambda r: (r["sad_total"], -r["lsb"]))["lsb"]
def m5_unique_then_mse(recs, w):    return min(_max_unique(recs), key=lambda r: (r["mse"], -r["lsb"]))["lsb"]
def m6_unique_then_sad(recs, w):    return min(_max_unique(recs), key=lambda r: (r["sad_total"], -r["lsb"]))["lsb"]


def _percentile_method(p: float) -> Callable:
    """Rule: cover the p-th percentile of |w|, deliberately clipping the (100-p)%
    tail above it. Lower p -> tighter range -> more trimming."""
    def rule(recs, w, _p=p):
        return _finest_covering(recs, float(np.percentile(np.abs(w), _p)))
    rule.__name__ = f"m_percentile_{str(p).replace('.', '_')}"
    return rule


# The percentile ladder. Each step down trims harder; p99.99 clips 0.01% of the
# weights, p98 clips 2%. Note the LSB is a power of two, so the range can only
# halve between candidates -- several of these will collapse onto the same LSB
# on layers whose tail is short, and that collapsing is itself informative.
m8_percentile_9999 = _percentile_method(99.99)
m7_percentile_999  = _percentile_method(99.9)
m11_percentile_995 = _percentile_method(99.5)
m12_percentile_99  = _percentile_method(99.0)
m13_percentile_98  = _percentile_method(98.0)


def m9_robust_sigma(recs, w):
    """Cover +/-4 robust sigma, where sigma = 1.4826 * MAD.

    MAD is computed from the middle of the distribution, so unlike std it cannot
    be inflated by the tail at all -- a lone 23.0 outlier does not move it. This
    is the purest form of 'size the range from the core, ignore outliers'.
    """
    mad = float(np.median(np.abs(w - np.median(w))))
    sigma = 1.4826 * mad
    if sigma <= 0.0:                       # degenerate (e.g. >50% exact zeros)
        return m7_percentile_999(recs, w)
    return _finest_covering(recs, 4.0 * sigma)


def m10_trimmed_mse(recs, w):
    """Lowest MSE measured over the core only -- blind to what clipping does to
    the tail. Same criterion as method 3, but with the outliers' vote removed."""
    return min(recs, key=lambda r: (r["mse_core"], -r["lsb"]))["lsb"]


METHODS: List[Tuple[int, str, Callable]] = [
    (2,  "max-unique → biggest range", m2_unique_then_range),
    (3,  "lowest MSE", m3_lowest_mse),
    (4,  "lowest SAD", m4_lowest_sad),
    (5,  "max-unique → lowest MSE", m5_unique_then_mse),
    (6,  "max-unique → lowest SAD  [current]", m6_unique_then_sad),
    (8,  "cover p99.99 of |w|", m8_percentile_9999),
    (7,  "cover p99.9 of |w|", m7_percentile_999),
    (11, "cover p99.5 of |w|", m11_percentile_995),
    (12, "cover p99 of |w|", m12_percentile_99),
    (13, "cover p98 of |w|", m13_percentile_98),
    (9,  "cover ±4·robust σ (MAD)", m9_robust_sigma),
    (10, "lowest MSE over core only", m10_trimmed_mse),
]


# --------------------------------------------------------------------------
# Model: pretrained float MobileNetV2 with BN fused
# --------------------------------------------------------------------------

_PROXY_WEIGHT_SUFFIXES = (".weight_quant.tensor_quant", ".weight_quant")


def build_fused_pretrained() -> Tuple[nn.Module, List]:
    """-> (model, [weight quantizers in forward-execution order])"""
    from argparse import Namespace

    from quantizers import FixedPointPerTensorWeightQuant
    from quantizers.base_quantizer import BaseQuantizer
    from quantizers.manager import QuantizerManager
    from utils.bn_fusion import fuse_bn_into_conv
    from examples.train_imagenet_qat import _build_model, _load_pretrained

    class WQuant(FixedPointPerTensorWeightQuant):
        bit_width = BIT_WIDTH

    args = Namespace(model=MODEL, num_classes=1000)
    # act/bias quant off: this study is weights-only.
    model = _build_model(args, WQuant, None, None)
    model = _load_pretrained(model, args)
    n_fused = fuse_bn_into_conv(model)
    print(f"[model] fused {n_fused} BatchNorm layers into their convs")

    # A train-mode forward calibrates the quantizers (needed before any eval-mode
    # forward), and the eval forward then establishes inference_sequence_id.
    # No optimizer.step() runs, so the weights are untouched.
    x = torch.randn(2, 3, 224, 224)
    model.train()
    for _ in range(12):
        model(x)
    model.eval()
    model(x)

    mgr = QuantizerManager()
    ordered = [q for q in mgr.quantizers_in_execution_order()
               if getattr(q, "quantizer_role", None) == "weight"]
    print(f"[model] {len(ordered)} weight quantizers in the forward path")
    return model, ordered


def weight_of(model: nn.Module, quantizer) -> Optional[Tuple[str, np.ndarray]]:
    """Map a weight quantizer back to the nn.Parameter it quantizes.

    QuantizerManager is a flat registry with no back-reference to owning modules,
    so go through named_modules() and strip the proxy suffix. Brevitas registers
    'ghost' quantizers that no forward reaches and that own no parameter; those
    return None.
    """
    for path, module in model.named_modules():
        if module is not quantizer:
            continue
        for suffix in _PROXY_WEIGHT_SUFFIXES:
            if path.endswith(suffix):
                parent = path[: -len(suffix)]
                try:
                    owner = model.get_submodule(parent)
                except AttributeError:
                    return None
                w = getattr(owner, "weight", None)
                if w is None:
                    return None
                return parent, w.detach().float().cpu().numpy()
    return None


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def _plot_unquantized(ax, w, rec):
    ax.hist(w.ravel(), bins=WEIGHT_BINS, color=C_W, alpha=0.85)
    ax.axvspan(rec["qmin"], rec["qmax"], color=C_RANGE, alpha=0.16, zorder=0)
    for b in (rec["qmin"], rec["qmax"]):
        ax.axvline(b, color=C_RANGE, lw=1.2, ls="--")
    ax.set_yscale("log")
    ax.set_title(f"weights  |  range [{rec['qmin']:.4g}, {rec['qmax']:.4g}]", fontsize=8)
    ax.tick_params(labelsize=7)


def _plot_quantized(ax, w, rec, signed):
    q, _, qmin, qmax, step = quantize(w, rec["lsb"], BIT_WIDTH, signed)
    vals, counts = np.unique(q, return_counts=True)
    width = max(step * 0.9, (vals.max() - vals.min()) / 400 if vals.size > 1 else step)
    ax.bar(vals, counts, width=width, color=C_Q)
    ax.set_yscale("log")
    ax.set_title(f"quantized grid  |  {rec['n_unique']} / {2**BIT_WIDTH} codes used",
                 fontsize=8)
    ax.tick_params(labelsize=7)


def _plot_metrics(ax, rec):
    names = ["clip err\n(saturated)", "round err\n(in-range)", "total err\n(=SAD)", "MSE"]
    vals = [rec["clip_err"], rec["round_err"], rec["sad_total"], rec["mse"]]
    colors = ["#d62728", "#1f77b4", "#7f7f7f", "#9467bd"]
    plot_vals = [max(v, 1e-30) for v in vals]
    bars = ax.bar(range(len(vals)), plot_vals, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=6)
    ax.tick_params(axis="y", labelsize=7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3g}",
                ha="center", va="bottom", fontsize=6)
    ax.set_title(f"error split  |  clipped {rec['pct_clipped']:.2f}%  |  "
                 f"SQNR {rec['sqnr']:.1f} dB", fontsize=8)


def make_figure(layer: str, shape: tuple, w: np.ndarray, signed: bool,
                chosen: Dict[int, Dict], order: int) -> plt.Figure:
    n = len(METHODS)
    fig, axes = plt.subplots(n, 3, figsize=(16, 3.0 * n))
    for row, (num, label, _) in enumerate(METHODS):
        rec = chosen[num]
        _plot_unquantized(axes[row][0], w, rec)
        _plot_quantized(axes[row][1], w, rec, signed)
        _plot_metrics(axes[row][2], rec)
        axes[row][0].set_ylabel(f"({num}) {label}\nLSB={rec['lsb']}  step={rec['step']:.3g}",
                                fontsize=8)
    fig.suptitle(f"[{order}]  {layer}   {tuple(shape)}   "
                 f"{'signed' if signed else 'unsigned'} {BIT_WIDTH}b   "
                 f"|w|max={np.abs(w).max():.4g}  std={w.std():.4g}",
                 fontsize=11, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    return fig


def save(fig, name: str, group: str = "") -> None:
    for sub, ext in (("PDF", "pdf"), ("SVG", "svg")):
        d = os.path.join(OUT_ROOT, group, sub) if group else os.path.join(OUT_ROOT, sub)
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# The LSB sweep figure: how much clipping is actually optimal?
# --------------------------------------------------------------------------

SWEEP_ROWS = 6


def sweep_lsbs(recs: List[Dict], w: np.ndarray) -> List[int]:
    """Start at the finest LSB that still clips NOTHING, then step down.

    Each -1 halves the representable range, so clipping grows monotonically down
    the rows. Row 1 is where lowest-MSE lands (0% clipped); by the last row we
    are near where max-unique sits (~13% clipped). The sweep therefore spans
    exactly the gap between the two families of rules.
    """
    start = _finest_covering(recs, float(np.abs(w).max()))
    have = {r["lsb"] for r in recs}
    out = []
    lsb = start
    while len(out) < SWEEP_ROWS and lsb in have:
        out.append(lsb)
        lsb -= 1
    return out


def make_sweep_figure(layer: str, w: np.ndarray, signed: bool,
                      recs_by_lsb: Dict[int, Dict], lsbs: List[int],
                      order: int) -> plt.Figure:
    n = len(lsbs)
    fig, axes = plt.subplots(n, 3, figsize=(16, 3.0 * n))
    for row, lsb in enumerate(lsbs):
        rec = recs_by_lsb[lsb]
        _plot_unquantized(axes[row][0], w, rec)
        _plot_quantized(axes[row][1], w, rec, signed)
        _plot_metrics(axes[row][2], rec)
        tag = "  ← no clipping" if row == 0 else ""
        axes[row][0].set_ylabel(f"LSB = {lsb}{tag}\nstep={rec['step']:.3g}",
                                fontsize=8)
    fig.suptitle(f"[{order}]  {layer}   {tuple(w.shape)}   LSB sweep "
                 f"(row 1 = finest LSB with zero clipping, then -1 per row)   "
                 f"|w|max={np.abs(w).max():.4g}  std={w.std():.4g}",
                 fontsize=11, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    return fig


def summary_figure(rows: List[Dict]) -> plt.Figure:
    """Chosen LSB and resulting SQNR per method, across all layers."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, len(rows) * 0.30), 9))
    x = np.arange(len(rows))
    labels = [r["layer"] for r in rows]
    for num, label, _ in METHODS:
        ax1.plot(x, [r[f"lsb_{num}"] for r in rows], marker="o", ms=3, lw=1,
                 label=f"({num}) {label}")
        ax2.plot(x, [r[f"sqnr_{num}"] for r in rows], marker="o", ms=3, lw=1,
                 label=f"({num}) {label}")
    for ax, ylab, title in ((ax1, "chosen LSB", "LSB chosen per layer, by rule"),
                            (ax2, "SQNR (dB)", "Resulting weight SQNR per layer, by rule")):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    from quantizers.fixedpoint_per_tensor import find_optimal_lsb

    model, ordered = build_fused_pretrained()
    os.makedirs(OUT_ROOT, exist_ok=True)

    rows: List[Dict] = []
    mismatches: List[str] = []

    for order, q in enumerate(ordered, start=1):
        got = weight_of(model, q)
        if got is None:
            print(f"  [{order:02d}] <ghost quantizer, no parameter> — skipped")
            continue
        layer, w = got
        signed = bool(q.search_result_is_signed.item())
        assert not getattr(q, "narrow_range", False), \
            f"{layer}: narrow_range=True, but this script's quantize() assumes False"

        recs = scan(w, BIT_WIDTH, signed)
        by_lsb = {r["lsb"]: r for r in recs}
        chosen = {num: by_lsb[fn(recs, w)] for num, _lbl, fn in METHODS}

        # Harness validation: our rules 2 and 6 must reproduce the shipped function.
        wt = torch.from_numpy(w)
        for num, prefer_high in ((2, True), (6, False)):
            ref, _, _ = find_optimal_lsb(wt, BIT_WIDTH, signed, q.rounding_mode,
                                         narrow_range=False, prefer_high_lsb=prefer_high)
            if int(ref) != chosen[num]["lsb"]:
                mismatches.append(
                    f"{layer}: method {num} gave {chosen[num]['lsb']}, "
                    f"find_optimal_lsb(prefer_high_lsb={prefer_high}) gave {int(ref)}")

        # The split must be exact.
        for num, rec in chosen.items():
            assert abs(rec["clip_err"] + rec["round_err"] - rec["sad_total"]) < 1e-3 * max(
                rec["sad_total"], 1e-9), f"{layer} m{num}: error split does not add up"

        fig = make_figure(layer, w.shape, w, signed, chosen, order)
        save(fig, f"{order:02d}_{layer}")

        sw = sweep_lsbs(recs, w)
        save(make_sweep_figure(layer, w, signed, by_lsb, sw, order),
             f"{order:02d}_{layer}", group="sweep")

        row = dict(order=order, layer=layer, shape=str(tuple(w.shape)), n=w.size,
                   signed=signed, absmax=float(np.abs(w).max()), std=float(w.std()))
        for num, _lbl, _fn in METHODS:
            r = chosen[num]
            row[f"lsb_{num}"] = r["lsb"]
            row[f"sqnr_{num}"] = round(r["sqnr"], 2)
            row[f"clip%_{num}"] = round(r["pct_clipped"], 3)
            row[f"codes_{num}"] = r["n_unique"]
        rows.append(row)
        print(f"  [{order:02d}] {layer:<24} {str(tuple(w.shape)):>18}  " +
              "  ".join(f"m{n}:LSB={chosen[n]['lsb']:>3}/{chosen[n]['sqnr']:5.1f}dB"
                        for n, _l, _f in METHODS))

    # ---- roll-up -------------------------------------------------------
    csv_path = os.path.join(OUT_ROOT, "lsb_methods.csv")
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    save(summary_figure(rows), "00_summary")

    print("\n" + "=" * 78)
    print(f"{'rule':<34}{'mean SQNR':>11}{'min SQNR':>10}{'mean clip%':>12}{'mean codes':>12}")
    print("-" * 78)
    for num, label, _ in METHODS:
        s = np.array([r[f"sqnr_{num}"] for r in rows])
        c = np.array([r[f"clip%_{num}"] for r in rows])
        u = np.array([r[f"codes_{num}"] for r in rows])
        print(f"({num}) {label:<29}{s.mean():>11.2f}{s.min():>10.2f}"
              f"{c.mean():>12.3f}{u.mean():>12.1f}")
    print("=" * 78)

    if mismatches:
        print("\n[FAIL] harness validation — our rules disagree with find_optimal_lsb:")
        for m in mismatches:
            print("   " + m)
    else:
        print("\n[ok] methods 2 and 6 reproduce find_optimal_lsb exactly on all "
              f"{len(rows)} layers")

    print(f"\nfigures : {OUT_ROOT}/PDF/  and  {OUT_ROOT}/SVG/")
    print(f"table   : {csv_path}")


if __name__ == "__main__":
    main()
