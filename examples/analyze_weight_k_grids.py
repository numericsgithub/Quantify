"""
analyze_weight_k4_vs_k16.py — what does the robust-sigma k actually do to each
weight tensor's grid?

k=4 is what the PTQ checkpoint currently ships and it scores 0.000% weight-only
top-1. k=16 scores 40.117%. Same rule, same tensors, one constant apart. This
plots both so the difference is visible per layer rather than inferred from a
single aggregate number.

One figure per weight quantizer, in forward-execution order (01_ = the conv that
sees the input image), 2 rows x 3 columns:

  row 1: k=4   (currently shipping, 0.000%)
  row 2: k=16  (40.117%)

  col 1: the float weights, with the representable range marked
  col 2: the QUANTIZED values on the real grid (bars, never binned)
  col 3: the error decomposition, split by cause

Deliberately NOT ranked by SQNR anywhere. SQNR has been wrong every time it has
been consulted here — it scores the 0.000% grid 10 dB BETTER than a 56.7% one.
The dB figures are printed as context, not as a verdict.

  <OUT_ROOT>/{PDF,SVG}/<NN>_<layer>.*
  <OUT_ROOT>/k4_vs_k16.csv
"""

from __future__ import annotations

import csv
import os
from argparse import Namespace
from typing import List

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import brevitas.nn as qnn

from examples.analyze_lsb_methods import _plot_metrics, _plot_quantized, _plot_unquantized, scan
from quantizers import FixedPointPerTensorWeightQuant
from quantizers.fixedpoint_per_tensor import RoundingMode, find_optimal_lsb
from quantizers.manager import QuantizerManager
from utils.bn_fusion import fuse_bn_into_conv

OUT_ROOT = "output/analysis_weight_k_grids"
BIT_WIDTH = 8
# Measured weight-only top-1 (sweep_weight_k.py, 40 val batches). k=12 is the
# shipping value: it trims outliers deliberately, trading PTQ accuracy for finer
# resolution on the bulk that QAT can then adapt to.
KS = [(4,  "k=4    [ 0.00% top-1]"),
      (8,  "k=8    [ 2.25% top-1]"),
      (12, "k=12   [45.90% top-1]  <- SHIPPING"),
      (16, "k=16   [39.96% top-1]")]


def _save(fig, name):
    for sub, ext in (("PDF", "pdf"), ("SVG", "svg")):
        d = os.path.join(OUT_ROOT, sub)
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def _weights_in_execution_order():
    from examples.train_imagenet_qat import _build_model, _load_pretrained
    from examples.find_perfect_lsbs_imagenet_ptq import _assign_descriptive_ids

    QuantizerManager().reset()
    args = Namespace(model="mobilenetv2", num_classes=1000)
    model = _build_model(args, FixedPointPerTensorWeightQuant, None, None)
    model = _load_pretrained(model, args)
    fuse_bn_into_conv(model)
    QuantizerManager().quantization_start_gap = 0
    model.train()
    with torch.no_grad():
        model(torch.randn(2, 3, 224, 224))
    _assign_descriptive_ids(model)

    owner = {}
    for path, mod in model.named_modules():
        for suffix in (".weight_quant.tensor_quant", ".weight_quant"):
            if path.endswith(suffix):
                owner[id(mod)] = path[: -len(suffix)]
                break
    out = []
    for q in QuantizerManager().quantizers_in_execution_order():
        if q.quantizer_role != "weight" or id(q) not in owner:
            continue
        parent = owner[id(q)]
        m = model.get_submodule(parent)
        if getattr(m, "weight", None) is None:
            continue
        out.append((parent, m.weight.detach().float().cpu().numpy(),
                    bool(q.search_result_is_signed.item())))
    return out


def main():
    items = _weights_in_execution_order()
    print(f"{len(items)} weight quantizers, forward-execution order\n")
    os.makedirs(OUT_ROOT, exist_ok=True)
    rows: List[dict] = []

    for idx, (layer, w, signed) in enumerate(items, start=1):
        recs = scan(w, BIT_WIDTH, signed)
        by_lsb = {r["lsb"]: r for r in recs}

        chosen = {}
        for k, _label in KS:
            lsb, _, _ = find_optimal_lsb(torch.from_numpy(w), BIT_WIDTH, signed,
                                         RoundingMode.ROUND, False,
                                         robust_sigma_k=float(k))
            chosen[k] = by_lsb[lsb]

        fig, axes = plt.subplots(len(KS), 3, figsize=(16, 3.0 * len(KS)))
        for r, (k, label) in enumerate(KS):
            rec = chosen[k]
            _plot_unquantized(axes[r][0], w, rec)
            _plot_quantized(axes[r][1], w, rec, signed)
            _plot_metrics(axes[r][2], rec)
            axes[r][0].set_ylabel(f"{label}\nLSB={rec['lsb']}  step={rec['step']:.3g}",
                                  fontsize=8, color=("#2ca02c" if k == 12 else "#d62728" if k == 4 else "black"),
                                  fontweight=("bold" if k == 12 else "normal"))
        same = len({chosen[k]["lsb"] for k, _ in KS}) == 1
        fig.suptitle(
            f"[{idx}] {layer}   {tuple(w.shape)}   {'signed' if signed else 'unsigned'} "
            f"{BIT_WIDTH}b   |w|max={np.abs(w).max():.4g}  std={w.std():.4g}"
            f"{'    (k=4 and k=16 AGREE here)' if same else ''}",
            fontsize=11, y=0.998)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        _save(fig, f"{idx:02d}_{layer}")

        row = dict(order=idx, layer=layer, shape=str(tuple(w.shape)),
                   absmax=round(float(np.abs(w).max()), 5),
                   std=round(float(w.std()), 6))
        for k, _l in KS:
            r_ = chosen[k]
            row[f"lsb_k{k}"] = r_["lsb"]
            row[f"qmax_k{k}"] = round(r_["qmax"], 6)
            row[f"clip%_k{k}"] = round(r_["pct_clipped"], 3)
            row[f"codes_k{k}"] = r_["n_unique"]
            row[f"sqnr_k{k}"] = round(r_["sqnr"], 2)
        row["lsb_delta"] = chosen[16]["lsb"] - chosen[4]["lsb"]
        rows.append(row)
        print(f"[{idx:02d}] {layer:<24}{str(tuple(w.shape)):>18}  " +
              "  ".join(f"k{k}: LSB={chosen[k]['lsb']:>3}/{chosen[k]['pct_clipped']:>5.2f}%"
                        for k, _ in KS))

    with open(os.path.join(OUT_ROOT, "k4_vs_k16.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    d = np.array([r["lsb_delta"] for r in rows])
    print("\n" + "=" * 78)
    print(f"layers where k=4 and k=16 pick the SAME LSB : {(d == 0).sum()}/{len(d)}")
    print(f"k=16 picks a COARSER LSB (wider range)      : {(d > 0).sum()}/{len(d)}")
    print(f"LSB shift distribution: " +
          ", ".join(f"+{v}:{(d == v).sum()}" for v in sorted(set(d.tolist()))))
    for k, _l in KS:
        c = np.array([r[f"clip%_k{k}"] for r in rows])
        n = np.array([r[f"codes_k{k}"] for r in rows])
        print(f"  k={k:<3} mean clip {c.mean():6.3f}%  max clip {c.max():6.2f}%  "
              f"mean codes {n.mean():6.1f}")
    print("=" * 78)
    print(f"figures: {OUT_ROOT}/PDF/  and  {OUT_ROOT}/SVG/")


if __name__ == "__main__":
    main()
