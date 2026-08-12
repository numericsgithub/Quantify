"""
analyze_ptq_checkpoint.py — inspect the grid every quantizer in a PTQ checkpoint
actually chose, weights AND biases.

Unlike analyze_lsb_methods.py (which compares candidate RULES on the pretrained
float weights), this reads a real checkpoint and asks: what grid did each
quantizer end up with, and is it sane?

Biases are included. They were never part of the rule study — that covered 53
weight tensors only — and folded biases have quite different statistics from
weights, so they are the obvious place for a surprise to hide.

Per quantizer, one figure. Rows are an LSB sweep: row 1 is the finest LSB that
clips NOTHING, then -1 per row (each step halves the representable range, so
clipping grows monotonically downward). The row the checkpoint ACTUALLY chose is
marked "<<< CHOSEN" — so you see the choice in the context of its alternatives
rather than on its own.

  col 1: the float tensor, with the representable range marked
  col 2: the QUANTIZED values on the real grid (bars, never binned)
  col 3: the error decomposition, split by cause

Files are named by order of appearance, weights and biases in separate folders:
  <OUT_ROOT>/weight/{PDF,SVG}/<NN>_<layer>.*
  <OUT_ROOT>/bias/{PDF,SVG}/<NN>_<layer>.*
plus a summary table on stdout and <OUT_ROOT>/ptq_checkpoint.csv
"""

from __future__ import annotations

import csv
import math
import os
from typing import Dict, List

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.analyze_ptq_vs_qat_weights import quantize
from examples.analyze_lsb_methods import (
    _finest_covering,
    _plot_metrics,
    _plot_quantized,
    _plot_unquantized,
    scan,
)

CKPT = "output/ptq/mobilenetv2_W8_Anone_B8_singlepass.pt"
OUT_ROOT = "output/analysis_ptq_singlepass"
BIT_WIDTH = 8
SWEEP_ROWS = 6


def _save(fig, name: str, group: str) -> None:
    for sub, ext in (("PDF", "pdf"), ("SVG", "svg")):
        d = os.path.join(OUT_ROOT, group, sub)
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def _rows_for(recs: List[Dict], w: np.ndarray, chosen: int) -> List[int]:
    """Sweep from the finest zero-clipping LSB downward, always including the
    LSB the checkpoint actually chose (it may sit outside the sweep)."""
    have = {r["lsb"] for r in recs}
    start = _finest_covering(recs, float(np.abs(w).max()))
    rows, lsb = [], start
    while len(rows) < SWEEP_ROWS and lsb in have:
        rows.append(lsb)
        lsb -= 1
    if chosen not in rows and chosen in have:
        rows.append(chosen)
    return sorted(set(rows), reverse=True)


_PROXY_SUFFIXES = (
    (".weight_quant.tensor_quant", "weight"),
    (".weight_quant", "weight"),
    (".bias_quant.tensor_quant", "bias"),
    (".bias_quant", "bias"),
)


def _load_in_execution_order():
    """-> [(layer, kind, tensor, chosen_lsb, signed)] in true forward order.

    State-dict key order is alphabetical ('classifier' < 'features', and
    'features.10' < 'features.2'), which is NOT execution order. Order has to
    come from a real forward pass: inference_sequence_id is only assigned on a
    quantizer's first forward, which is what quantizers_in_execution_order()
    sorts by. That also drops Brevitas ghost quantizers no forward reaches.
    """
    from argparse import Namespace
    import torch.nn as nn
    from examples.train_imagenet_qat import _build_model, _load_ptq_checkpoint
    from quantizers import FixedPointPerTensorBiasQuant, FixedPointPerTensorWeightQuant
    from quantizers.manager import QuantizerManager

    QuantizerManager().reset()
    args = Namespace(model="mobilenetv2", num_classes=1000)
    model = _build_model(args, FixedPointPerTensorWeightQuant, None,
                         FixedPointPerTensorBiasQuant)
    model, _ = _load_ptq_checkpoint(model, CKPT)   # fuses BN, then loads
    model.eval()
    with torch.no_grad():
        model(torch.randn(2, 3, 224, 224))        # establishes execution order

    mgr = QuantizerManager()
    owner_of = {}
    for path, module in model.named_modules():
        for suffix, kind in _PROXY_SUFFIXES:
            if path.endswith(suffix):
                owner_of[id(module)] = (path[: -len(suffix)], kind)
                break

    out = []
    for q in mgr.quantizers_in_execution_order():
        if q.quantizer_role not in ("weight", "bias"):
            continue
        got = owner_of.get(id(q))
        if got is None:
            continue
        parent, kind = got
        try:
            owner = model.get_submodule(parent)
        except AttributeError:
            continue
        param = getattr(owner, kind, None)
        if param is None:
            continue
        out.append((parent, kind, param.detach().float().cpu().numpy(),
                    int(q.search_result_lsb.item()),
                    bool(q.search_result_is_signed.item())))
    return out


def main() -> None:
    items = _load_in_execution_order()
    print(f"[ckpt] {CKPT}")
    print(f"[ckpt] {sum(1 for i in items if i[1]=='weight')} weight + "
          f"{sum(1 for i in items if i[1]=='bias')} bias quantizers, "
          f"in forward-execution order\n")

    rows_csv: List[Dict] = []
    order = {"weight": 0, "bias": 0}

    for layer, kind, t, chosen, signed in items:
        order[kind] += 1
        idx = order[kind]

        recs = scan(t, BIT_WIDTH, signed)
        by_lsb = {r["lsb"]: r for r in recs}
        if chosen not in by_lsb:      # chosen sits outside the scanned window
            q, _, qmin, qmax, step = quantize(t, chosen, BIT_WIDTH, signed)
            err = t - q
            aerr = np.abs(err)
            clipped = (t < qmin) | (t > qmax)
            sig, noise = float((t ** 2).sum()), float((err ** 2).sum())
            by_lsb[chosen] = dict(
                lsb=chosen, step=float(step), qmin=float(qmin), qmax=float(qmax),
                n_unique=int(np.unique(q).size),
                clip_err=float(aerr[clipped].sum()), round_err=float(aerr[~clipped].sum()),
                sad_total=float(aerr.sum()), mse=float((err ** 2).mean()),
                mse_core=0.0, n_clipped=int(clipped.sum()),
                pct_clipped=100.0 * int(clipped.sum()) / t.size,
                sqnr=10 * math.log10(sig / noise) if noise > 0 else float("inf"))

        rec = by_lsb[chosen]
        sweep = _rows_for(recs, t, chosen)

        fig, axes = plt.subplots(len(sweep), 3, figsize=(16, 3.0 * len(sweep)))
        if len(sweep) == 1:
            axes = np.array([axes])
        for r, lsb in enumerate(sweep):
            rr = by_lsb[lsb]
            _plot_unquantized(axes[r][0], t, rr)
            _plot_quantized(axes[r][1], t, rr, signed)
            _plot_metrics(axes[r][2], rr)
            tag = "  <<< CHOSEN" if lsb == chosen else ("  (no clipping)" if r == 0 else "")
            axes[r][0].set_ylabel(f"LSB = {lsb}{tag}\nstep={rr['step']:.3g}", fontsize=8,
                                  color=("red" if lsb == chosen else "black"))
        fig.suptitle(f"[{idx}] {layer}  [{kind}]  {tuple(t.shape)}  "
                     f"{'signed' if signed else 'unsigned'} {BIT_WIDTH}b  |  "
                     f"CHOSEN LSB={chosen}  range=[{rec['qmin']:.4g},{rec['qmax']:.4g}]  "
                     f"|t|max={np.abs(t).max():.4g}  clipped={rec['pct_clipped']:.2f}%  "
                     f"SQNR={rec['sqnr']:.1f}dB", fontsize=11, y=0.998)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        _save(fig, f"{idx:02d}_{layer}", kind)

        # range utilisation: how much of the grid the data actually spans
        util = 100.0 * float(np.abs(t).max()) / max(rec["qmax"], 1e-30)
        rows_csv.append(dict(
            order=idx, layer=layer, kind=kind, shape=str(tuple(t.shape)),
            lsb=chosen, signed=signed, step=rec["step"],
            qmin=round(rec["qmin"], 8), qmax=round(rec["qmax"], 8),
            absmax=round(float(np.abs(t).max()), 6), std=round(float(t.std()), 6),
            pct_clipped=round(rec["pct_clipped"], 4),
            codes=rec["n_unique"], sqnr=round(rec["sqnr"], 2),
            range_util_pct=round(util, 2),
        ))

    with open(os.path.join(OUT_ROOT, "ptq_checkpoint.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows_csv[0].keys()))
        wr.writeheader()
        wr.writerows(rows_csv)

    for kind in ("weight", "bias"):
        sel = [r for r in rows_csv if r["kind"] == kind]
        print("=" * 108)
        print(f"{kind.upper()}  ({len(sel)} tensors)")
        print(f"{'#':>3} {'layer':<24}{'shape':>18}{'lsb':>5}{'range':>22}"
              f"{'|t|max':>10}{'clip%':>8}{'codes':>7}{'SQNR':>8}{'util%':>9}")
        print("-" * 108)
        for r in sel:
            print(f"{r['order']:>3} {r['layer']:<24}{r['shape']:>18}{r['lsb']:>5}"
                  f"  [{r['qmin']:>8.4g},{r['qmax']:>8.4g}]{r['absmax']:>10.4g}"
                  f"{r['pct_clipped']:>8.2f}{r['codes']:>7}{r['sqnr']:>8.1f}"
                  f"{r['range_util_pct']:>9.1f}")
        s = np.array([r["sqnr"] for r in sel])
        c = np.array([r["pct_clipped"] for r in sel])
        u = np.array([r["range_util_pct"] for r in sel])
        n = np.array([r["codes"] for r in sel])
        print(f"\n  mean SQNR {s.mean():6.2f} dB | min {s.min():6.2f} dB | "
              f"mean clip {c.mean():.2f}% | max clip {c.max():.2f}% | "
              f"mean codes {n.mean():5.1f} | mean range-util {u.mean():.1f}%")
        worst = sorted(sel, key=lambda r: r["sqnr"])[:5]
        print("  worst by SQNR: " +
              ", ".join(f"{r['layer']}({r['sqnr']:.1f}dB/{r['pct_clipped']:.1f}%clip)"
                        for r in worst))
        print()

    print(f"figures: {OUT_ROOT}/weight/PDF/  and  {OUT_ROOT}/bias/PDF/")


if __name__ == "__main__":
    main()
