"""
train_imagenet_float.py — Float-only ImageNet fine-tuning (no quantization).

Refine a pretrained model in full float32 precision before QAT.  Uses the
same Brevitas-wrapped model classes as train_imagenet_qat.py, but all
quantizers are permanently disabled, so the checkpoint format is identical
and can be loaded directly via:

    python -m examples.train_imagenet_qat \\
        --init-from-ptq output/imagenet_float/<exp>/checkpoints/last.pt \\
        --float-warmup-epochs 0 \\
        ...

Usage examples
--------------
# Fine-tune torchvision ResNet-18 with DALI:
python -m examples.train_imagenet_float \\
    --model resnet18 --pretrained \\
    --data-dir /home/th/tmp/datasets/imagenet \\
    --epochs 90 --lr 3e-4

# Continue from a previous float checkpoint:
python -m examples.train_imagenet_float \\
    --model resnet18 --init-from-ckpt output/imagenet_float/checkpoints/last.pt \\
    --data-dir /home/th/tmp/datasets/imagenet \\
    --epochs 30 --lr 1e-4
"""

from __future__ import annotations

import argparse
import os
import warnings

warnings.filterwarnings("ignore", message="Corrupt EXIF data", category=UserWarning)
# Brevitas warns about AMP + fake-quant interaction; irrelevant here since quant is disabled.
warnings.filterwarnings("ignore", message="Mixed precision.*Brevitas", category=UserWarning)

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
from training_harness.trainer_v2 import QATTrainerV2
from training_harness.config_v2 import TrainerConfigV2, QATScheduleConfigV2
from training_harness.config import CheckpointConfig
from training_harness.schedulers import WarmupCosineScheduler
from utils.weight_mapping import load_timm_weights
from utils.run_utils import env_default, next_run_dir, setup_output_tee


# ---------------------------------------------------------------------------
# Fixed 8-bit placeholder quantizers (constructed but never activated)
# ---------------------------------------------------------------------------

class _W8(FixedPointPerTensorWeightQuant):
    bit_width = 8

class _A8(FixedPointPerTensorActivationQuant):
    bit_width = 8

class _B8(FixedPointPerTensorBiasQuant):
    bit_width = 8


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _print_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    print(f"\n{'='*70}")
    print("  train_imagenet_float.py — arguments")
    print(f"{'='*70}")
    seen: set = set()
    for group in parser._action_groups:
        rows = []
        for action in group._group_actions:
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
        description="ImageNet float fine-tuning — no quantization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Model -------------------------------------------------------------
    p.add_argument(
        "--model",
        choices=["resnet18", "resnet50", "mobilenetv1", "mobilenetv2"],
        default="resnet18",
    )
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument(
        "--pretrained",
        action="store_true",
        help="Load torchvision pretrained float weights as starting point",
    )

    # ---- Init from checkpoint ----------------------------------------------
    p.add_argument(
        "--init-from-ckpt",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume or initialise from any checkpoint (float or QAT). "
             "Loaded with strict=False after model construction.",
    )

    # ---- Data --------------------------------------------------------------
    d = p.add_argument_group("data")
    d.add_argument(
        "--data-dir",
        type=str,
        default=env_default("IMAGENET_DALI_PATH"),
        metavar="PATH",
        help="ImageFolder root (train/ and val/). Uses DALI when set. "
             "Defaults to $IMAGENET_DALI_PATH if set.",
    )
    d.add_argument("--hf-dataset", type=str, default="ILSVRC/imagenet-1k")
    d.add_argument("--num-workers", type=int, default=20)
    d.add_argument("--dali-threads", type=int, default=4)
    d.add_argument("--randaugment-n", type=int, default=2,
                   help="Number of RandAugment transforms per image")
    d.add_argument("--randaugment-m", type=int, default=7,
                   help="RandAugment magnitude")

    # ---- Training ----------------------------------------------------------
    t = p.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=290)
    t.add_argument("--batch-size", type=int, default=1024)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument(
        "--mixed-precision",
        action="store_true",
        default=True,
        help="Enable AMP (autocast + GradScaler)",
    )
    t.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    t.add_argument("--prefetch-factor", type=int, default=3)
    t.add_argument("--mixup", type=float, default=0.1,
                   help="MixUp Beta alpha (0 = off)")
    t.add_argument("--cutmix", type=float, default=1.0,
                   help="CutMix Beta alpha (0 = off)")
    t.add_argument("--mixup-prob", type=float, default=1.0,
                   help="Probability of applying mixup or cutmix per batch")
    t.add_argument("--mixup-switch-prob", type=float, default=0.5,
                   help="Probability of switching to cutmix when both are enabled")
    t.add_argument("--smoothing", type=float, default=0.1,
                   help="Label smoothing: via Mixup when active, else LabelSmoothingCE (0 = off)")
    t.add_argument("--reprob", type=float, default=0.25,
                   help="Random Erasing probability (0 = off)")
    t.add_argument("--ema-decay", type=float, default=0.9999,
                   help="EMA shadow model decay (0 = off)")
    t.add_argument("--freeze-bn", action="store_true", default=False,
                   help="Freeze BatchNorm running stats and affine params throughout "
                        "training. Prevents BN stat drift from heavy augmentation "
                        "causing val_acc regression when fine-tuning at low LR.")
    t.add_argument(
        "--repeat-aug",
        type=int,
        default=1,
        metavar="N",
        help="Repeated augmentation: N views per image per epoch (HuggingFace only)",
    )
    t.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        metavar="N",
        help="Stop early if val_acc does not improve for N epochs (default: off)",
    )

    # ---- LR schedule -------------------------------------------------------
    s = p.add_argument_group("lr schedule")
    s.add_argument(
        "--cosine-lr",
        action="store_true",
        default=False,
        help="Use a per-step linear-warmup + cosine-annealing LR schedule "
             "instead of ReduceLROnPlateau. Mutually exclusive with the "
             "--reduce-lr-* options: cosine steps every batch and would "
             "overwrite any plateau-triggered reduction, so ReduceLROnPlateau "
             "is disabled when this is set.",
    )
    s.add_argument(
        "--cosine-warmup-frac", type=float, default=0.1,
        help="Fraction of total steps spent in linear warmup (--cosine-lr only)",
    )
    s.add_argument(
        "--cosine-eta-min", type=float, default=1e-6,
        help="Final LR at the end of cosine annealing (--cosine-lr only)",
    )
    s.add_argument(
        "--reduce-lr-patience", type=int, default=20,
        help="ReduceLROnPlateau: epochs of no improvement before reducing LR",
    )
    s.add_argument("--reduce-lr-factor", type=float, default=0.5)
    s.add_argument("--reduce-lr-min-lr", type=float, default=1e-8)
    s.add_argument(
        "--reduce-lr-metric", type=str, default="val_loss",
        help="Metric monitored by ReduceLROnPlateau (e.g. val_loss, val_acc)",
    )

    # ---- Output ------------------------------------------------------------
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory. Defaults to output/imagenet_float_<model> "
             "(e.g. output/imagenet_float_resnet18).",
    )
    p.add_argument(
        "--new-run-dir",
        action="store_true",
        help="Auto-increment the output directory if it already exists "
             "(output/imagenet_float_<model> → output/imagenet_float_<model>_1 → …).",
    )
    p.add_argument("--experiment-name", type=str, default=None)

    # ---- Dry-run -----------------------------------------------------------
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dry-run-batches", type=int, default=10)

    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = f"output/imagenet_float_{args.model}"
    _print_args(p, args)
    return args


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _build_model(args) -> nn.Module:
    nc = args.num_classes
    if args.model == "resnet18":
        return QuantResNet18(nc, _W8, _A8, _B8)
    if args.model == "resnet50":
        return QuantResNet50(nc, _W8, _A8, _B8)
    if args.model == "mobilenetv1":
        return QuantMobileNetV1(nc, _W8, _A8, _B8)
    if args.model == "mobilenetv2":
        return QuantMobileNetV2(nc, weight_quant=_W8, act_quant=_A8, bias_quant=_B8)
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


def _load_checkpoint(model: nn.Module, ckpt_path: str) -> nn.Module:
    print(f"[init-from-ckpt] Loading {ckpt_path} …")
    payload = torch.load(ckpt_path, map_location="cpu")
    sd = payload.get("model_state_dict", payload)
    incompatible = model.load_state_dict(sd, strict=False)
    if incompatible.missing_keys:
        print(f"  Missing keys: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"  Unexpected keys: {incompatible.unexpected_keys}")
    metrics = payload.get("metrics", {})
    if metrics:
        print(f"  Checkpoint metrics: {metrics}")
    return model


# ---------------------------------------------------------------------------
# Data loading  (same loaders as train_imagenet_qat.py)
# ---------------------------------------------------------------------------

class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, preprocess):
        self.hf_dataset = hf_dataset
        self.preprocess = preprocess

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        return self.preprocess(item["image"].convert("RGB")), item["label"]


class RepeatAugSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, n_repeats: int = 2, shuffle: bool = True) -> None:
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

    if args.new_run_dir:
        args.output_dir = next_run_dir(args.output_dir)

    setup_output_tee(args.output_dir)

    exp_name = args.experiment_name or f"{args.model}_float"

    print(f"\n{'═'*60}")
    print(f"  Experiment  : {exp_name}")
    print(f"  Model       : {args.model}  (float, no quantization)")
    print(f"  Pretrained  : {args.pretrained}")
    print(f"  AMP         : {args.mixed_precision}")
    print(f"  Mixup α     : {args.mixup}  CutMix α: {args.cutmix}  reprob: {args.reprob}  smoothing: {args.smoothing}")
    print(f"  EMA decay   : {args.ema_decay}")
    if args.data_dir:
        print(f"  Data        : DALI ({args.data_dir})")
    else:
        print(f"  Data        : HuggingFace ({args.hf_dataset})")
    print(f"{'═'*60}\n")

    model = _build_model(args)
    if args.pretrained:
        model = _load_pretrained(model, args)
    if args.init_from_ckpt:
        model = _load_checkpoint(model, args.init_from_ckpt)

    train_loader, val_loader = _build_dataloaders(args)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # A per-step cosine schedule and epoch-level ReduceLROnPlateau cannot
    # coexist: the cosine scheduler steps every batch and would overwrite any
    # plateau-triggered reduction on the next batch (see commit 0265040). When
    # --cosine-lr is set we build the scheduler and turn the plateau one off.
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

    # QATScheduleConfigV2 with astronomically large warmup_epochs and patience
    # so the plateau detector never fires and QAT is never activated.
    # The trainer starts by calling _fully_disable_quantization(), so the
    # Brevitas layers are pure pass-through throughout the entire run.
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
            float_warmup_epochs=10_000_000,
            plateau_patience=10_000_000,
            annealing_steps=1,
            quantization_start_gap=1,
            freeze_bn_at_qat=False,
            track_scale_factors=False,
        ),

        checkpoint=CheckpointConfig(
            monitor_metric="val_acc",
            monitor_mode="max",
            top_k=3,
            save_last=True,
        ),

        early_stopping_patience=args.early_stopping_patience,

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
        freeze_bn=args.freeze_bn,
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
    )

    print(f"Checkpoint output: {os.path.abspath(config.checkpoint_dir)}")
    print(f"  Load in QAT script via:  --init-from-ptq {config.checkpoint_dir}/last.pt\n")

    print("─" * 60)
    print("  Pre-training evaluation (pretrained weights, quant off)")
    print("─" * 60)
    trainer.evaluate(val_loader,   label="val")
    trainer.evaluate(train_loader, label="train")
    print("─" * 60)

    tracker = trainer.fit()

    best_acc = tracker.best_value("val_acc", "max")
    print(f"\nDone. Best val_acc: {best_acc:.4f}")
    print(f"  Load into QAT: python -m examples.train_imagenet_qat "
          f"--init-from-ptq {config.checkpoint_dir}/last.pt --float-warmup-epochs 0 ...")


if __name__ == "__main__":
    main()
