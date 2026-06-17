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
import torchvision.transforms as T
from datasets import load_dataset

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
from utils.weight_mapping import load_pretrained_weights


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

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
        default=None,
        metavar="PATH",
        help="Path to ImageFolder dataset (train/ and val/ subdirs). "
             "When set, uses NVIDIA DALI instead of the HuggingFace dataloader. "
             "Extract with: python scripts/extract_imagenet.py --output-dir PATH",
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

    # ---- Training ----------------------------------------------------------
    t = p.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=150)
    t.add_argument("--batch-size", type=int, default=1024)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--weight-decay", type=float, default=1e-5)
    t.add_argument(
        "--label-smoothing", type=float, default=0.1,
        help="Label smoothing for CrossEntropyLoss (0 = off)",
    )
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
    p.add_argument("--output-dir", type=str, default="output/imagenet_qat")
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

    return p.parse_args()


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


def _load_pretrained(model: nn.Module, args) -> nn.Module:
    from torchvision.models import (
        resnet18, ResNet18_Weights,
        resnet50, ResNet50_Weights,
        mobilenet_v2, MobileNet_V2_Weights,
    )

    if args.model == "resnet18":
        print(f"[pretrained] Loading torchvision resnet18 (IMAGENET1K_V1) …")
        float_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif args.model == "resnet50":
        print(f"[pretrained] Loading torchvision resnet50 (IMAGENET1K_V2) …")
        float_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    elif args.model == "mobilenetv2":
        print(f"[pretrained] Loading torchvision mobilenet_v2 pretrained weights …")
        float_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    else:
        print(f"[pretrained] No pretrained weights for {args.model}, skipping.")
        return model

    print(f"[pretrained] Mapping weights to quantized model …")
    return load_pretrained_weights(model, float_model)


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


def _build_dataloaders(args):
    if args.data_dir:
        return _build_dali_loaders(args)
    return _build_hf_loaders(args)


def _build_dali_loaders(args):
    from utils.dali_pipeline import build_dali_loaders
    print(f"Building DALI loaders from {args.data_dir} …")
    train_loader, val_loader = build_dali_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_threads=args.dali_threads,
    )
    print(f"  train: {len(train_loader):,} batches   val: {len(val_loader):,} batches")
    return train_loader, val_loader



def _build_hf_loaders(args):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Standard torchvision recipe (matches IMAGENET1K_V1/V2 pretrained weights):
    #   train: RandomResizedCrop(224) + RandomHorizontalFlip
    #   val:   Resize(256) + CenterCrop(224)
    train_preprocess = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    val_preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])

    print(f"Loading ImageNet datasets from Hugging Face ({args.hf_dataset})...")
    hf_train = load_dataset(args.hf_dataset, split="train")
    hf_val   = load_dataset(args.hf_dataset, split="validation")

    persistent = args.num_workers > 0
    prefetch   = args.prefetch_factor if args.num_workers > 0 else None
    train_loader = DataLoader(
        HFDatasetWrapper(hf_train, train_preprocess),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=persistent, prefetch_factor=prefetch,
    )
    val_loader = DataLoader(
        HFDatasetWrapper(hf_val, val_preprocess),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=persistent, prefetch_factor=prefetch,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Build quantizer injector classes
    weight_quant = _make_weight_quant(args)
    act_quant    = _make_act_quant(args)
    bias_quant   = _make_bias_quant(args)

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
            loss_fn=nn.CrossEntropyLoss(label_smoothing=0.1),
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
    total_steps = len(train_loader) * args.epochs
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=total_steps // 10,
        total_steps=total_steps,
        eta_min=1e-6,
    )

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
        ),

        checkpoint=CheckpointConfig(
            monitor_metric="val_acc",
            monitor_mode="max",
            top_k=3,
            save_last=True,
        ),

        early_stopping_patience=20,
        early_stopping_min_delta=1e-4,
    )

    trainer = QATTrainerV2(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),
        scheduler=scheduler,
        onnx_dummy_input=torch.zeros(1, 3, 224, 224),
    )

    print("\nPre-training evaluation (eval mode, quantization disabled):")
    trainer.evaluate(val_loader,   label="val  ")
    trainer.evaluate(train_loader, label="train")
    print()

    tracker = trainer.fit()

    best_acc = tracker.best_value("val_acc", "max")
    print(f"\nDone. Best val_acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
