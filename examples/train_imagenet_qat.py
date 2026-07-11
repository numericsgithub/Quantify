"""
train_imagenet_qat.py — ImageNet QAT with the V2 harness.

Supports ResNet-18, ResNet-50, MobileNetV1, MobileNetV2 with configurable
fixed-point or coefficient-based weight quantization.

Dataset is loaded from Hugging Face (ILSVRC/imagenet-1k by default).

Usage examples
--------------
# ResNet-18, 8-bit weights + activations, load torchvision pretrained:
python examples/train_imagenet_qat.py \\
    --model resnet18 --act-bits 8 --weight-bits 8 --bias-bits 8 --pretrained

# ResNet-50, coefficient weights from a file:
python examples/train_imagenet_qat.py \\
    --model resnet50 \\
    --act-bits 8 --weight-coeffs /path/to/coefficients.txt --bias-bits 8 --pretrained

# MobileNetV2 4-bit:
python examples/train_imagenet_qat.py \\
    --model mobilenetv2 --act-bits 4 --weight-bits 4 --bias-bits 8 --pretrained
"""

from __future__ import annotations

import argparse
import os  # still used for os.path.splitext in experiment name
import warnings

warnings.filterwarnings("ignore", message="Corrupt EXIF data", category=UserWarning)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.resnet_quant import QuantResNet18, QuantResNet50
from models.mobilenetv1_quant import QuantMobileNetV1
from models.mobilenetv2_quant import QuantMobileNetV2
from quantizers.fixedpoint_per_tensor import (
    FixedPointPerTensorWeightQuant,
    FixedPointPerTensorActivationQuant,
    FixedPointPerTensorBiasQuant,
)
from quantizers.coefficient_per_tensor_weights import CoefficientPerTensorWeightQuant
from training_harness.trainer_v2 import QATTrainerV2
from training_harness.config_v2 import TrainerConfigV2, QATScheduleConfigV2
from training_harness.config import CheckpointConfig
from training_harness.schedulers import WarmupCosineScheduler
from training_harness.lr_finder import find_lr
from utils.weight_mapping import load_timm_weights
from utils.bn_fusion import fuse_bn_into_conv
from utils.run_utils import env_default, next_run_dir, setup_output_tee


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _print_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Print every parsed argument, grouped the same way --help groups them."""
    print(f"\n{'='*70}")
    print("  train_imagenet_qat.py — arguments")
    print(f"{'='*70}")

    seen: set = set()
    for group in parser._action_groups:
        rows = []
        for action in group._group_actions:
            # --flag/--no-flag pairs (e.g. --mixed-precision/--no-mixed-precision)
            # share one dest and both land in the same group; only list it once.
            if action.dest == "help" or action.dest in seen:
                continue
            seen.add(action.dest)
            rows.append((action.dest, getattr(args, action.dest)))
        if not rows:
            continue

        print(f"\n  [{group.title}]")
        width = max(len(dest) for dest, _ in rows)
        for dest, value in rows:
            print(f"    {dest:<{width}} : {value}")

    print(f"\n{'='*70}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ImageNet QAT — model and quantization selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Model -------------------------------------------------------------
    p.add_argument(
        "--model",
        choices=["resnet18", "resnet50", "mobilenetv1", "mobilenetv2"],
        default="resnet18",
        help="Network architecture",
    )
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument(
        "--pretrained",
        action="store_true",
        help="Load torchvision pretrained float weights before QAT "
             "(supported for resnet18, resnet50, mobilenetv2)",
    )

    # ---- Data --------------------------------------------------------------
    d = p.add_argument_group("data")
    d.add_argument(
        "--data-dir",
        type=str,
        default=env_default("IMAGENET_DALI_PATH"),
        metavar="PATH",
        help="Path to ImageFolder dataset (train/ and val/ subdirs). "
             "When set, uses NVIDIA DALI instead of the HuggingFace dataloader. "
             "Extract with: python scripts/extract_imagenet.py --output-dir PATH. "
             "Defaults to $IMAGENET_DALI_PATH if set.",
    )
    d.add_argument(
        "--hf-dataset",
        type=str,
        default="ILSVRC/imagenet-1k",
        help="Hugging Face dataset name (ignored when --data-dir is set)",
    )
    d.add_argument(
        "--num-workers", type=int, default=20,
        help="HuggingFace DataLoader workers (ignored when --data-dir is set)",
    )
    d.add_argument(
        "--dali-threads", type=int, default=4,
        help="DALI CPU preprocessing threads (used when --data-dir is set). "
             "DALI offloads most work to GPU so 4 is usually enough.",
    )
    d.add_argument("--randaugment-n", type=int, default=2,
                   help="Number of RandAugment transforms per image")
    d.add_argument("--randaugment-m", type=int, default=7,
                   help="RandAugment magnitude")

    # ---- Quantization ------------------------------------------------------
    q = p.add_argument_group("quantization")
    q.add_argument("--act-bits", type=int, default=8, help="Activation bit width")

    wq = q.add_mutually_exclusive_group()
    wq.add_argument(
        "--weight-bits",
        type=int,
        default=8,
        help="Weight bit width — uses FixedPointPerTensorWeightQuant",
    )
    wq.add_argument(
        "--weight-coeffs",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to coefficient file — uses CoefficientPerTensorWeightQuant "
             "(mutually exclusive with --weight-bits)",
    )
    q.add_argument("--bias-bits", type=int, default=8, help="Bias bit width")
    q.add_argument(
        "--weight-lsb-subtract",
        type=int,
        default=0,
        metavar="N",
        help="After loading --init-from-ptq, subtract N from every weight quantizer's "
             "LSB position (finer grid). Implicitly disables all activation quantizers. "
             "A before/after table is printed as a sanity check.",
    )

    # ---- Training ----------------------------------------------------------
    t = p.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=150)
    t.add_argument("--batch-size", type=int, default=1024)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument(
        "--mixed-precision",
        action="store_true",
        default=True,
        help="Enable AMP (autocast + GradScaler). Disable if Brevitas fake-quant "
             "ops cause NaN losses during QAT (use --no-mixed-precision).",
    )
    t.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    t.add_argument(
        "--prefetch-factor",
        type=int,
        default=3,
        help="DataLoader prefetch factor (batches queued per worker ahead of GPU)",
    )
    t.add_argument(
        "--mixup",
        type=float,
        default=0.1,
        help="MixUp Beta distribution alpha (0 = off)",
    )
    t.add_argument(
        "--cutmix",
        type=float,
        default=1.0,
        help="CutMix Beta distribution alpha (0 = off)",
    )
    t.add_argument(
        "--mixup-prob",
        type=float,
        default=1.0,
        help="Probability of applying mixup or cutmix per batch",
    )
    t.add_argument(
        "--mixup-switch-prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix are enabled",
    )
    t.add_argument(
        "--smoothing",
        type=float,
        default=0.1,
        help="Label smoothing: via Mixup when active, else LabelSmoothingCE (0 = off)",
    )
    t.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        help="Random Erasing probability (0 = off)",
    )
    t.add_argument(
        "--ema-decay",
        type=float,
        default=0.9999,
        help="EMA decay for shadow model (validation uses EMA weights). Set 0 to disable.",
    )
    t.add_argument(
        "--repeat-aug",
        type=int,
        default=1,
        metavar="N",
        help="Repeated augmentation: each image appears N times per epoch with different "
             "augmentations (HuggingFace dataloader only; N=1 = off).",
    )

    # ---- QAT schedule ------------------------------------------------------
    s = p.add_argument_group("qat schedule")
    s.add_argument(
        "--float-warmup-epochs",
        type=int,
        default=30,
        help="Epochs of float-only training. QAT starts when val_loss plateaus "
             "for --plateau-patience epochs OR this epoch limit is reached.",
    )
    s.add_argument(
        "--plateau-patience",
        type=int,
        default=10,
        help="Epochs of no val_loss improvement before QAT is triggered",
    )
    s.add_argument(
        "--annealing-steps",
        type=int,
        default=20,
        help="Forward passes over which each quantizer anneals 0→1",
    )
    s.add_argument(
        "--qat-gap",
        type=int,
        default=300,
        help="Forward passes between successive quantizer activations (staggered cascade)",
    )

    # ---- Output ------------------------------------------------------------
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory. Defaults to output/imagenet_qat_<model> "
             "(e.g. output/imagenet_qat_resnet18).",
    )
    p.add_argument(
        "--new-run-dir",
        action="store_true",
        help="Auto-increment the output directory if it already exists "
             "(output/imagenet_qat_<model> → output/imagenet_qat_<model>_1 → …). "
             "Useful for keeping each run's checkpoints separate.",
    )
    p.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override the auto-generated experiment name",
    )

    # ---- Dry-run (for quick config validation) -----------------------------
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only --dry-run-batches batches per epoch (for fast config testing)",
    )
    p.add_argument(
        "--dry-run-batches",
        type=int,
        default=10,
        help="Number of batches per epoch when --dry-run is active",
    )

    # ---- LR Finder ---------------------------------------------------------
    lr = p.add_argument_group("lr finder")
    lr.add_argument(
        "--find-lr",
        action="store_true",
        help=(
            "Run the two-phase LR Range Test instead of normal training. "
            "Uses whichever model weights are active at startup "
            "(--pretrained or default init)."
        ),
    )
    lr.add_argument(
        "--find-lr-sweep-start", type=float, default=1e-8,
        help="Start of the LR sweep (default: 1e-8)",
    )
    lr.add_argument(
        "--find-lr-sweep-end", type=float, default=1e-2,
        help="End of the LR sweep (default: 1e-2)",
    )
    lr.add_argument(
        "--find-lr-steps", type=int, default=100,
        help="Number of steps in the LR sweep (default: 100)",
    )
    lr.add_argument(
        "--find-lr-calib-steps", type=int, default=10,
        help="Calibration pre-pass steps in Phase 1 (default: 10)",
    )

    # ---- Reduce LR on plateau -----------------------------------------------
    rlr = p.add_argument_group("reduce lr on plateau")
    rlr.add_argument(
        "--cosine-lr",
        action="store_true",
        default=False,
        help="Use a per-step linear-warmup + cosine-annealing LR schedule "
             "instead of ReduceLROnPlateau. Mutually exclusive with the "
             "--reduce-lr-* options: cosine steps every batch and would "
             "overwrite any plateau-triggered reduction, so ReduceLROnPlateau "
             "is disabled when this is set.",
    )
    rlr.add_argument(
        "--cosine-warmup-frac", type=float, default=0.1,
        help="Fraction of total steps spent in linear warmup (--cosine-lr only)",
    )
    rlr.add_argument(
        "--cosine-eta-min", type=float, default=1e-6,
        help="Final LR at the end of cosine annealing (--cosine-lr only)",
    )
    rlr.add_argument("--reduce-lr-patience", type=int, default=20,
                     help="Epochs of no improvement before reducing LR (default: 5)")
    rlr.add_argument("--reduce-lr-factor", type=float, default=0.5,
                     help="Multiplicative factor applied to LR on plateau (default: 0.5)")
    rlr.add_argument("--reduce-lr-min-lr", type=float, default=1e-8,
                     help="Lower bound on LR (default: 1e-8)")
    rlr.add_argument(
        "--reduce-lr-metric", type=str, default="val_loss",
        help="Metric monitored by ReduceLROnPlateau (e.g. val_loss, val_acc)",
    )

    # ---- Init from a PTQ checkpoint -----------------------------------------
    ptq = p.add_argument_group("ptq init")
    ptq.add_argument(
        "--init-from-ptq",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a checkpoint produced by "
            "examples/find_perfect_lsbs_imagenet_ptq.py — typically the "
            "activations-mode run, chained from a weights-mode run via that "
            "script's --init-from-ckpt so both roles are calibrated. Loaded "
            "with strict=False after model construction (and after "
            "--pretrained, if both are given — the checkpoint's weights win). "
            "Automatically sets preserve_calibrated_quantizers=True so the "
            "PTQ-found LSBs survive the float-warmup -> QAT transition instead "
            "of being reset and re-derived from scratch. Consider pairing with "
            "--float-warmup-epochs 0 since the model is already calibrated."
        ),
    )

    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = f"output/imagenet_qat_{args.model}"
    _print_args(p, args)
    return args


# ---------------------------------------------------------------------------
# Quantizer factories
# ---------------------------------------------------------------------------

def _make_weight_quant(args: argparse.Namespace):
    if args.weight_coeffs:
        fp = args.weight_coeffs
        class WeightQuant(CoefficientPerTensorWeightQuant):
            filepath = fp
        return WeightQuant

    bw = args.weight_bits
    class WeightQuant(FixedPointPerTensorWeightQuant):
        bit_width = bw
    return WeightQuant


def _make_act_quant(args: argparse.Namespace):
    bw = args.act_bits
    class ActQuant(FixedPointPerTensorActivationQuant):
        bit_width = bw
    return ActQuant


def _make_bias_quant(args: argparse.Namespace):
    bw = args.bias_bits
    class BiasQuant(FixedPointPerTensorBiasQuant):
        bit_width = bw
    return BiasQuant


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _build_model(args, weight_quant, act_quant, bias_quant) -> nn.Module:
    nc = args.num_classes
    if args.model == "resnet18":
        return QuantResNet18(nc, weight_quant, act_quant, bias_quant)
    if args.model == "resnet50":
        return QuantResNet50(nc, weight_quant, act_quant, bias_quant)
    if args.model == "mobilenetv1":
        return QuantMobileNetV1(nc, weight_quant, act_quant, bias_quant)
    if args.model == "mobilenetv2":
        return QuantMobileNetV2(nc, weight_quant=weight_quant, act_quant=act_quant, bias_quant=bias_quant)
    raise ValueError(f"Unknown model: {args.model}")


_TIMM_NAMES = {
    "resnet18":    "resnet18.a1_in1k",
    "resnet50":    "resnet50.a1_in1k",
    "mobilenetv1": "mobilenetv1_100.ra4_e3600_r224_in1k",
    "mobilenetv2": "mobilenetv2_100.ra_in1k",
}


def _load_pretrained(model: nn.Module, args) -> nn.Module:
    import timm
    timm_name = _TIMM_NAMES.get(args.model)
    if timm_name is None:
        print(f"[pretrained] No timm weights configured for {args.model}, skipping.")
        return model
    print(f"[pretrained] Loading timm {timm_name} …")
    float_model = timm.create_model(timm_name, pretrained=True)
    float_model.eval()
    return load_timm_weights(model, float_model, args.model)


def _load_ptq_checkpoint(model: nn.Module, ckpt_path: str) -> tuple[nn.Module, bool]:
    """
    Load a checkpoint produced by examples/find_perfect_lsbs_imagenet_ptq.py,
    typically the activations-mode run chained from a weights-mode run via
    that script's --init-from-ckpt so both roles are calibrated.

    Uses strict=False and does NOT reset calibration buffers — search_done /
    search_result_lsb / annealing_alpha are loaded as-is so the PTQ-found
    LSBs are what QAT starts from. Missing/unexpected keys are reported but
    not fatal: a checkpoint produced with a different --mode / model variant
    than the one being constructed here will legitimately have mismatched
    quantizer buffers for the role that wasn't searched.

    If the checkpoint was produced with --fuse-bn (or was itself a QAT
    checkpoint saved from such a run), its model_state_dict has BatchNorm
    folded into the preceding conv/linear (conv gained a bias, BatchNorm
    became Identity) — loading that into a freshly built model that still
    has separate, randomly-initialized BatchNorm layers would leave BatchNorm
    untrained and silently produce garbage output. Detect this via
    extra.fuse_bn and fuse this model's BatchNorm the same way before
    loading, so the module structures match.

    Returns (model, bn_fused) so the caller can propagate fuse_bn=True into
    subsequent checkpoint saves, allowing further chained runs to work.
    """
    print(f"[init-from-ptq] Loading {ckpt_path} …")
    payload = torch.load(ckpt_path, map_location="cpu")
    bn_fused = False
    if payload.get("extra", {}).get("fuse_bn"):
        n_fused = fuse_bn_into_conv(model)
        bn_fused = True
        print(f"[init-from-ptq] Checkpoint was produced with --fuse-bn; fused "
              f"{n_fused} BatchNorm layer(s) into preceding conv/linear weights "
              f"to match its module structure.")
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
    if incompatible.missing_keys:
        print(f"[init-from-ptq] Missing keys: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"[init-from-ptq] Unexpected keys: {incompatible.unexpected_keys}")
    metrics = payload.get("metrics", {})
    if metrics:
        print(f"[init-from-ptq] Checkpoint metrics: {metrics}")
    return model, bn_fused


def _disable_act_quant_proxies(model: nn.Module) -> None:
    """Set disable_quant=True on all activation proxies (leaves weight/bias proxies alone)."""
    from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector, BiasQuantProxyFromInjector
    for m in model.modules():
        if hasattr(m, "disable_quant") and not isinstance(
            m, (WeightQuantProxyFromInjector, BiasQuantProxyFromInjector)
        ):
            m.disable_quant = True


def _apply_weight_lsb_subtract(model: nn.Module, delta: int) -> None:
    """
    Subtract `delta` from every weight quantizer's search_result_lsb buffer,
    then disable all activation quantizer proxies.

    Prints a before/after table for each adjusted quantizer so the caller can
    verify the shift is correct before training starts.
    """
    from quantizers.fixedpoint_per_tensor import FixedPointPerTensorQuantizer

    col = 62
    print(f"\n[weight-lsb-subtract] Subtracting {delta} from all weight quantizer LSBs")
    print(f"  {'Module path':<{col}}  {'Before':>6}  {'After':>6}  Check")
    print(f"  {'-'*col}  {'-'*6}  {'-'*6}  -----")

    n = 0
    for name, module in model.named_modules():
        if not isinstance(module, FixedPointPerTensorQuantizer):
            continue
        if "weight_quant" not in name:
            continue

        before = int(module.search_result_lsb.item())
        after  = before - delta
        module.search_result_lsb.fill_(after)
        readback = int(module.search_result_lsb.item())
        ok = "OK" if readback == after else f"MISMATCH (got {readback})"
        print(f"  {name:<{col}}  {before:>6}  {after:>6}  {ok}")
        n += 1

    if n == 0:
        print("  WARNING: no weight quantizers found — load a PTQ checkpoint first.")
    else:
        print(f"\n  {n} quantizer(s) adjusted.")

    _disable_act_quant_proxies(model)
    print("  Activation quantizer proxies disabled for this run.\n")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class HFDatasetWrapper(Dataset):
    """Wraps a Hugging Face dataset for use with PyTorch DataLoader."""
    def __init__(self, hf_dataset, preprocess):
        self.hf_dataset = hf_dataset
        self.preprocess = preprocess

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        img = self.preprocess(item["image"].convert("RGB"))
        return img, item["label"]


class RepeatAugSampler(torch.utils.data.Sampler):
    """
    Repeated augmentation sampler: each unique image appears n_repeats times
    in consecutive slots so that, with an appropriate batch_size, every batch
    contains batch_size // n_repeats unique images each seen n_repeats times
    with independent random augmentations.

    Only meaningful when batch_size is a multiple of n_repeats.
    """

    def __init__(self, dataset: Dataset, n_repeats: int = 2, shuffle: bool = True) -> None:
        self._n = len(dataset)
        self.n_repeats = n_repeats
        self.shuffle = shuffle

    def __len__(self) -> int:
        return self._n * self.n_repeats

    def __iter__(self):
        import random
        indices = list(range(self._n))
        if self.shuffle:
            random.shuffle(indices)
        for idx in indices:
            for _ in range(self.n_repeats):
                yield idx


def _build_dataloaders(args):
    if args.data_dir:
        return _build_dali_loaders(args)
    return _build_hf_loaders(args)


def _build_dali_loaders(args):
    from utils.dali_pipeline import build_dali_loaders, norm_for_model
    mean, std = norm_for_model(args.model)
    print(f"Building DALI loaders from {args.data_dir} …")
    print(f"  normalization for {args.model}: mean={mean} std={std}")
    train_loader, val_loader = build_dali_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_threads=args.dali_threads,
        randaugment_n=args.randaugment_n,
        randaugment_m=args.randaugment_m,
        mean=mean,
        std=std,
    )
    print(f"  train: {len(train_loader):,} batches   val: {len(val_loader):,} batches")
    return train_loader, val_loader



def _build_hf_loaders(args):
    raise Exception("Deprecated. Use dali instead.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Build quantizer injector classes
    weight_quant = _make_weight_quant(args)
    act_quant    = _make_act_quant(args)
    bias_quant   = _make_bias_quant(args)

    # Resolve output directory (auto-increment if --new-run-dir is set)
    if args.new_run_dir:
        args.output_dir = next_run_dir(args.output_dir)

    setup_output_tee(args.output_dir)

    # Derive a descriptive experiment name if not provided
    weight_desc = (
        f"coeffs_{os.path.splitext(os.path.basename(args.weight_coeffs))[0]}"
        if args.weight_coeffs
        else f"W{args.weight_bits}"
    )
    exp_name = args.experiment_name or f"{args.model}_{weight_desc}_A{args.act_bits}_B{args.bias_bits}"

    print(f"\n{'═'*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Model      : {args.model}")
    print(f"  Weight Q   : {weight_desc}")
    print(f"  Act Q      : A{args.act_bits}")
    print(f"  Bias Q     : B{args.bias_bits}")
    print(f"  Pretrained : {args.pretrained}")
    print(f"  AMP        : {args.mixed_precision}")
    if args.data_dir:
        print(f"  Data       : DALI  ({args.data_dir})  threads={args.dali_threads}")
    else:
        print(f"  Data       : HuggingFace ({args.hf_dataset})  workers={args.num_workers}")
    print(f"{'═'*60}\n")

    # Build model
    model = _build_model(args, weight_quant, act_quant, bias_quant)
    if args.pretrained:
        model = _load_pretrained(model, args)
    bn_fused = False
    if args.init_from_ptq:
        model, bn_fused = _load_ptq_checkpoint(model, args.init_from_ptq)

    if args.weight_lsb_subtract:
        _apply_weight_lsb_subtract(model, args.weight_lsb_subtract)

    # Data
    train_loader, val_loader = _build_dataloaders(args)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # ── LR Finder mode ───────────────────────────────────────────────────────
    if args.find_lr:
        find_lr(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            loss_fn=nn.CrossEntropyLoss(),
            device="auto",
            calibration_steps=args.find_lr_calib_steps,
            sweep_start_lr=args.find_lr_sweep_start,
            sweep_end_lr=args.find_lr_sweep_end,
            sweep_steps=args.find_lr_steps,
            out_dir=os.path.join(args.output_dir, "lr_finder"),
            grad_clip_norm=1.0,
        )
        return
    # ─────────────────────────────────────────────────────────────────────────
    # By default ReduceLROnPlateau manages the LR epoch-by-epoch inside the
    # harness. A per-step cosine scheduler would override every plateau-triggered
    # reduction on the very next batch, so the two cannot coexist; --cosine-lr
    # swaps to cosine and disables the plateau scheduler below.
    scheduler = None
    if args.cosine_lr:
        total_steps = len(train_loader) * args.epochs
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=int(total_steps * args.cosine_warmup_frac),
            total_steps=total_steps,
            eta_min=args.cosine_eta_min,
        )
        print(f"[lr-schedule] Cosine: total_steps={total_steps:,} "
              f"warmup={int(total_steps * args.cosine_warmup_frac):,} "
              f"eta_min={args.cosine_eta_min} (ReduceLROnPlateau disabled)")

    # V2 harness config
    config = TrainerConfigV2(
        experiment_name=exp_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=1.0,
        num_workers=args.num_workers,
        mixed_precision=args.mixed_precision,

        dry_run=args.dry_run,
        dry_run_batches=args.dry_run_batches,

        qat=QATScheduleConfigV2(
            float_warmup_epochs=args.float_warmup_epochs,
            plateau_metric="val_loss",
            plateau_patience=args.plateau_patience,
            plateau_min_delta=1e-4,
            annealing_steps=args.annealing_steps,
            quantization_start_gap=args.qat_gap,
            freeze_bn_at_qat=True,
            track_scale_factors=True,
            preserve_calibrated_quantizers=bool(args.init_from_ptq),
        ),

        checkpoint=CheckpointConfig(
            monitor_metric="val_acc",
            monitor_mode="max",
            top_k=3,
            save_last=True,
        ),

        early_stopping_patience=None,
        reduce_lr_on_plateau=not args.cosine_lr,
        reduce_lr_patience=args.reduce_lr_patience,
        reduce_lr_factor=args.reduce_lr_factor,
        reduce_lr_min_lr=args.reduce_lr_min_lr,
        reduce_lr_metric=args.reduce_lr_metric,

        mixup=args.mixup,
        cutmix=args.cutmix,
        mixup_prob=args.mixup_prob,
        mixup_switch_prob=args.mixup_switch_prob,
        smoothing=args.smoothing,
        reprob=args.reprob,
        num_classes=args.num_classes,
        ema_decay=args.ema_decay,
    )

    trainer = QATTrainerV2(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=nn.CrossEntropyLoss(),
        scheduler=scheduler,
        onnx_dummy_input=torch.zeros(1, 3, 224, 224),
        extra_checkpoint_fields={"fuse_bn": True} if bn_fused else None,
    )

    print("\nPre-training evaluation (eval mode, quantization disabled):")
    trainer.evaluate(val_loader,   label="val  ")
    trainer.evaluate(train_loader, label="train")
    print()

    # When --weight-lsb-subtract is active, re-disable activation proxies after
    # every epoch so QAT activation (which re-enables all proxies) can't undo it.
    epoch_hook = None
    if args.weight_lsb_subtract:
        def epoch_hook(trainer, epoch, snap):
            _disable_act_quant_proxies(trainer.model)

    tracker = trainer.fit(after_epoch_hook=epoch_hook)

    best_acc = tracker.best_value("val_acc", "max")
    print(f"\nDone. Best val_acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
