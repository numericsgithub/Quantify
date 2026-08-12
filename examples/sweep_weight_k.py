"""
sweep_weight_k.py — weight-only top-1 vs the robust-sigma k, finely sampled.

Why: a coarse sweep gave 0.000% at k=4, 40.117% at k=16, 4.609% at k=32 and
0.391% at k=64. A sharp peak like that is suspicious — a smooth criterion ought
to produce a smooth accuracy curve. This samples k densely around 16 and, for
every k, ALSO records the LSB each layer chose, so we can see whether the peak is
real or an artefact.

The likely explanation to test: the LSB is an integer (the range can only halve
between candidates), so accuracy is a STEP function of k, not a smooth one. Each
layer flips its LSB at its own k. That would make the "peak" a plateau whose
edges are wherever a critical layer happens to flip.

Accuracy is the only arbiter used here. SQNR has been wrong every time it has
been consulted — it rated the 0.000% grid 10 dB BETTER than a 56.7% one.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from argparse import Namespace

import torch
import torch.nn as nn

import quantizers.fixedpoint_per_tensor as fp
from quantizers import FixedPointPerTensorWeightQuant
from quantizers.manager import QuantizerManager
from utils.bn_fusion import fuse_bn_into_conv

OUT = "output/sweep_weight_k"
KS = [4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 28, 32, 48, 64]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="/home/th/tmp/datasets/imagenet")
    p.add_argument("--eval-batches", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    a = p.parse_args()
    a.model = "mobilenetv2"
    a.num_classes = 1000
    a.dali_threads = 4
    a.randaugment_n = 0
    a.randaugment_m = 0
    return a


def build_at_k(args, k, device):
    """Fresh model, pretrained, BN fused, all weight quantizers calibrated at this k."""
    from examples.train_imagenet_qat import _build_model, _load_pretrained
    fp.ROBUST_SIGMA_K_WEIGHT = float(k)
    QuantizerManager().reset()
    model = _build_model(args, FixedPointPerTensorWeightQuant, None, None)
    model = _load_pretrained(model, args)
    fuse_bn_into_conv(model)
    model = model.to(device)
    mgr = QuantizerManager()
    mgr.quantization_start_gap = 0
    model.train()
    with torch.no_grad():
        model(torch.randn(8, 3, 224, 224, device=device))
    reached = mgr.quantizers_in_execution_order()
    for q in reached:
        q.annealing_alpha.data.fill_(1.0)
        q.annealing_alpha_step = 0.0
    return model, reached


def main():
    args = parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from examples.train_imagenet_qat import _build_dataloaders
    from examples.find_perfect_lsbs_imagenet_ptq import _evaluate, _assign_descriptive_ids
    _, val = _build_dataloaders(args)
    loss_fn = nn.CrossEntropyLoss()

    rows = []
    lsb_map = {}
    for k in KS:
        model, reached = build_at_k(args, k, device)
        _assign_descriptive_ids(model)
        lsbs = {q.display_name: int(q.search_result_lsb.item()) for q in reached}
        model.eval()
        loss, acc = _evaluate(model, val, loss_fn, device, args.eval_batches, label=f"k={k}")
        rows.append(dict(k=k, acc=round(acc, 4), loss=round(loss, 4)))
        lsb_map[k] = lsbs
        print(f"SWEEP  k={k:>4}   weight-only top-1 = {acc:7.3f}%   loss={loss:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(OUT, "sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["k", "acc", "loss"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "lsbs_per_k.json"), "w") as fh:
        json.dump(lsb_map, fh, indent=1)

    # How many layers change their LSB between consecutive k? If accuracy moves
    # only where LSBs flip, the curve is a step function and the "peak" is a
    # plateau, not a resonance.
    print("\n=== LSB churn between consecutive k ===")
    print(f"{'k':>5}{'acc%':>9}{'layers whose LSB changed vs previous k':>42}")
    prev = None
    for r in rows:
        k = r["k"]
        if prev is None:
            print(f"{k:>5}{r['acc']:>9.3f}{'-':>42}")
        else:
            changed = [n for n in lsb_map[k] if lsb_map[k][n] != lsb_map[prev][n]]
            print(f"{k:>5}{r['acc']:>9.3f}{len(changed):>42}")
        prev = k

    best = max(rows, key=lambda r: r["acc"])
    print(f"\nBEST: k={best['k']}  ->  {best['acc']:.3f}%")
    print(f"csv : {OUT}/sweep.csv     lsbs: {OUT}/lsbs_per_k.json")


if __name__ == "__main__":
    main()
