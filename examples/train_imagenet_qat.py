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
        "--hf-dataset",
        type=str,
        default="ILSVRC/imagenet-1k",
        help="Hugging Face dataset name",
    )
    d.add_argument("--num-workers", type=int, default=4)

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
    t.add_argument("--epochs", type=int, default=90)
    t.add_argument("--batch-size", type=int, default=1024)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--weight-decay", type=float, default=1e-5)
    t.add_argument(
        "--label-smoothing", type=float, default=0.1,
        help="Label smoothing for CrossEntropyLoss (0 = off)",
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
        default=500,
        help="Forward passes over which each quantizer anneals 0→1",
    )
    s.add_argument(
        "--qat-gap",
        type=int,
        default=200,
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
    import timm
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

    if args.model in ("resnet18", "resnet50"):
        # timm provides better-trained weights than torchvision V1:
        #   resnet18 → ~73.3% top-1  (vs torchvision V1 69.8%)
        #   resnet50 → ~80.9% top-1  (vs torchvision V1 76.1%)
        # timm's default ResNet uses the same parameter naming as torchvision,
        # so the existing name-based weight mapping works without changes.
        print(f"[pretrained] Loading timm/{args.model} pretrained weights …")
        float_model = timm.create_model(args.model, pretrained=True)
    elif args.model == "mobilenetv2":
        # timm's MobileNetV2 uses different layer naming; torchvision naming
        # matches our QuantMobileNetV2 module structure directly.
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


def _build_dataloaders(args) -> tuple[DataLoader, DataLoader]:
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    bicubic = T.InterpolationMode.BICUBIC

    # RSB-aligned recipe (matches timm resnet18/50 a1_in1k pretraining):
    #   train: bicubic crop + RandAugment(2,9) + RandomErasing
    #   val:   Resize(236, bicubic) + CenterCrop(224)  [crop_pct=0.95]
    # Using a mismatched val pipeline would hide ~3-4 pp accuracy from the
    # pretrained baseline (standard 256→224 bilinear only yields ~69.8%).
    train_preprocess = T.Compose([
        T.RandomResizedCrop(224, interpolation=bicubic),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9, interpolation=bicubic),
        T.ToTensor(),
        normalize,
        T.RandomErasing(p=0.25),
    ])
    val_preprocess = T.Compose([
        T.Resize(236, interpolation=bicubic),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])

    print(f"Loading ImageNet datasets from Hugging Face ({args.hf_dataset})...")
    hf_train = load_dataset(args.hf_dataset, split="train",      trust_remote_code=True)
    hf_val   = load_dataset(args.hf_dataset, split="validation", trust_remote_code=True)

    train_loader = DataLoader(
        HFDatasetWrapper(hf_train, train_preprocess),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        HFDatasetWrapper(hf_val, val_preprocess),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
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
    print(f"{'═'*60}\n")

    # Build model
    model = _build_model(args, weight_quant, act_quant, bias_quant)
    if args.pretrained:
        model = _load_pretrained(model, args)

    # Data
    train_loader, val_loader = _build_dataloaders(args)

    # Optimizer + cosine LR with warmup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
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

    tracker = trainer.fit()

    best_acc = tracker.best_value("val_acc", "max")
    print(f"\nDone. Best val_acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
