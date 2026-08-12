"""
create_ptq_checkpoint.py — build a weights+biases PTQ checkpoint in ONE forward pass.

What this replaces and why
--------------------------
The old greedy search (examples/find_perfect_lsbs_imagenet_ptq.py) calibrated one
quantizer per forward pass, walking the network upstream->downstream and running a
val_loss sweep at every step — ~4h38m for MobileNetV2.

That structure is unnecessary for weights and biases. A weight quantizer calibrates
against the WEIGHT TENSOR, and a bias quantizer against the BIAS TENSOR — neither
looks at activations, so neither depends on any other quantizer's decision. There is
nothing to sequence. With quantization_start_gap=0 a single training forward
calibrates all of them simultaneously, in seconds.

(Activations are a different matter — an activation quantizer does see the tensor
flowing into it, so its calibration depends on what upstream quantizers did. They are
deliberately NOT handled here; act_quant is left off entirely.)

Expect the resulting validation accuracy to be poor. That is expected and fine: this
is an un-adapted PTQ starting point for QAT, not a finished model.

Pipeline
--------
  1. build the model with weight + bias quantizers, NO activation quantizers
  2. load the float weights (timm pretrained by default, or --init-from)
  3. fuse BatchNorm  (QAT trains the BN-folded deployment graph, and folding is what
     creates conv.bias in the first place — so it must happen before calibration)
  4. ONE train-mode forward on a real training batch -> every weight/bias quantizer
     calibrates at once
  5. verify every one of them actually calibrated
  6. evaluate on validation (reported for the record; expected to be bad)
  7. save a checkpoint --init-from-ptq can consume

Usage
-----
  python -m examples.create_ptq_checkpoint --data-dir /path/to/imagenet
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import torch
import torch.nn as nn

from quantizers import (
    FixedPointPerTensorBiasQuant,
    FixedPointPerTensorWeightQuant,
)
from quantizers.manager import QuantizerManager
from utils.bn_fusion import fuse_bn_into_conv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a weights+biases PTQ checkpoint in a single forward pass.")
    p.add_argument("--data-dir", type=str, default=os.environ.get("IMAGENET_DALI_PATH"),
                   help="ImageNet root for the DALI loaders. Defaults to $IMAGENET_DALI_PATH.")
    p.add_argument("--model", type=str, default="mobilenetv2")
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--weight-bits", type=int, default=8)
    p.add_argument("--bias-bits", type=int, default=8)
    p.add_argument("--init-from", type=str, default=None, metavar="CKPT",
                   help="Float checkpoint to start from. Default: timm pretrained "
                        "weights for --model (the best float model available).")
    p.add_argument("--output", type=str, default=None, metavar="PATH",
                   help="Where to write the checkpoint. Default: "
                        "output/ptq/<model>_W<w>_Anone_B<b>_singlepass.pt")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--weight-sigma-k", type=float, default=None, metavar="K",
                   help="Override robust-sigma k for WEIGHT LSB calibration "
                        "(default 12). MobileNetV1 needs ~20 (see pitfall #15).")
    p.add_argument("--dali-threads", type=int, default=4)
    p.add_argument("--eval-batches", type=int, default=None,
                   help="Limit validation to N batches (default: full val set).")
    p.add_argument("--skip-eval", action="store_true",
                   help="Skip the validation pass (it is only for the record).")
    args = p.parse_args()
    if not args.data_dir:
        p.error("--data-dir is required (or set $IMAGENET_DALI_PATH)")
    # _build_dali_loaders wants these; no augmentation is wanted for calibration.
    args.randaugment_n = 0
    args.randaugment_m = 0
    return args


def _default_output(args) -> str:
    tag = f"{args.model}_W{args.weight_bits}_Anone_B{args.bias_bits}_singlepass"
    return os.path.join("output", "ptq", f"{tag}.pt")


def main() -> None:
    args = parse_args()
    out_path = args.output or _default_output(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.weight_sigma_k is not None:
        import quantizers.fixedpoint_per_tensor as fp
        fp.ROBUST_SIGMA_K_WEIGHT = float(args.weight_sigma_k)
        print(f"[calib] ROBUST_SIGMA_K_WEIGHT overridden -> {args.weight_sigma_k}")

    from examples.train_imagenet_qat import _build_model, _load_pretrained, _load_ptq_checkpoint
    from examples.find_perfect_lsbs_imagenet_ptq import _assign_descriptive_ids, _evaluate

    QuantizerManager().reset()

    # ---- 1. model: weights + biases only, no activation quantizers ----------
    class WQuant(FixedPointPerTensorWeightQuant):
        bit_width = args.weight_bits

    class BQuant(FixedPointPerTensorBiasQuant):
        bit_width = args.bias_bits

    print(f"[1/7] Building {args.model} (W{args.weight_bits} / B{args.bias_bits}, "
          f"activations FLOAT)")
    model = _build_model(args, WQuant, None, BQuant)

    # ---- 2. float weights --------------------------------------------------
    if args.init_from:
        print(f"[2/7] Loading float checkpoint: {args.init_from}")
        model, _ = _load_ptq_checkpoint(model, args.init_from)
    else:
        print(f"[2/7] Loading timm pretrained weights for {args.model}")
        model = _load_pretrained(model, args)

    # ---- 3. fuse BN --------------------------------------------------------
    # Must precede calibration twice over: it rewrites the weight distribution the
    # weight quantizers calibrate against, and it CREATES conv.bias (assigning it is
    # what makes Brevitas wire up each bias quantizer at all).
    n_fused = fuse_bn_into_conv(model)
    print(f"[3/7] Fused {n_fused} BatchNorm layer(s) into the preceding conv/linear")

    model = model.to(device)
    mgr = QuantizerManager()
    roles = Counter(q.quantizer_role for q in mgr.quantizers.values())
    # Counts here are inflated by Brevitas ghost objects (registered, never reached
    # by forward). The real per-role counts are reported in step 5, after the pass.
    print(f"      quantizer objects registered: {dict(roles)} (incl. unreached ghosts)")
    if roles.get("activation"):
        raise RuntimeError(f"{roles['activation']} activation quantizers were built; "
                           "this script is weights+biases only")

    # ---- 4. calibrate: ONE forward -----------------------------------------
    # gap=0 disables the staggered cascade, so every quantizer calibrates on this
    # single pass instead of one per forward. Sound here precisely because weight
    # and bias quantizers read their own parameter tensor, never the activations —
    # so no quantizer's calibration depends on another's.
    mgr.quantization_start_gap = 0
    mgr.force_recalibration = False

    from examples.train_imagenet_qat import _build_dataloaders
    train_loader, val_loader = _build_dataloaders(args)
    images, labels = next(iter(train_loader))
    images = images.to(device)
    print(f"[4/7] Calibrating ALL {roles['weight'] + roles['bias']} weight/bias "
          f"quantizers in a single forward (batch {tuple(images.shape)}) …")

    model.train()
    for m in model.modules():                     # BN is gone, but be explicit:
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
    with torch.no_grad():
        model(images)

    # ---- 5. verify ---------------------------------------------------------
    # Only quantizers the forward actually REACHES matter. Brevitas registers
    # "ghost" quantizer objects that no forward ever calls; they never calibrate,
    # and that is correct — they quantize nothing. quantizers_in_execution_order()
    # drops them (it sorts by inference_sequence_id, which is only assigned on a
    # quantizer's first real forward), which is why the pass above had to run first.
    reached = mgr.quantizers_in_execution_order()
    ghosts = len(mgr.quantizers) - len(reached)

    uncalibrated = [q for q in reached if not q.search_done.item()]
    if uncalibrated:
        by_role = Counter(q.quantizer_role for q in uncalibrated)
        for q in uncalibrated[:10]:
            print(f"        UNCALIBRATED: {getattr(q, 'display_name', q.quant_id)} "
                  f"({q.quantizer_role})")
        raise RuntimeError(
            f"{len(uncalibrated)} reached quantizer(s) did not calibrate in the single "
            f"pass {dict(by_role)}. A quantizer whose tensor holds a single unique value "
            "never sets search_done (see _save_calibration).")

    # Activate ONLY the ones that really calibrated. Deliberately do NOT touch the
    # ghosts: forcing search_done/alpha on an uncalibrated quantizer is exactly the
    # mistake that corrupted the previous search (train_imagenet_qat.py:688-690 marked
    # every quantizer "calibrated" without calibrating it, leaving LSB=0).
    for q in reached:
        q.annealing_alpha.data.fill_(1.0)         # fully quantized, no annealing left
        q.annealing_alpha_step = 0.0

    reached_roles = Counter(q.quantizer_role for q in reached)
    print(f"[5/7] All {len(reached)} reached quantizers calibrated and active "
          f"{dict(reached_roles)}   ({ghosts} unreached ghost objects ignored)")

    _assign_descriptive_ids(model)
    lsbs = sorted(int(q.search_result_lsb.item()) for q in reached)
    print(f"      LSB range: {lsbs[0]} .. {lsbs[-1]}   median {lsbs[len(lsbs)//2]}")

    # ---- 6. evaluate (for the record; expected to be poor) -----------------
    val_acc = None
    if not args.skip_eval:
        print("[6/7] Evaluating (un-adapted PTQ — a poor number here is expected) …")
        model.eval()
        loss_fn = nn.CrossEntropyLoss()
        val_loss, val_acc = _evaluate(model, val_loader, loss_fn, device,
                                      args.eval_batches, label="ptq")
        print(f"      val_loss={val_loss:.4f}  val_acc={val_acc:.3f}%")
    else:
        print("[6/7] Evaluation skipped (--skip-eval)")

    # ---- 7. save -----------------------------------------------------------
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "metrics": {"val_acc": val_acc} if val_acc is not None else {},
        "config": {
            "model": args.model,
            "weight_bits": args.weight_bits,
            "act_bits": None,
            "bias_bits": args.bias_bits,
        },
        "extra": {
            "pretrained_qat": True,
            "fuse_bn": True,
            "calibration": "single_pass_weights_biases",
            "role_bit_widths": {"weight": args.weight_bits, "bias": args.bias_bits},
        },
    }, out_path)
    print(f"[7/7] Saved -> {out_path}")
    print(f"\nUse it with:  --init-from-ptq {out_path} --no-act-quant")


if __name__ == "__main__":
    main()
