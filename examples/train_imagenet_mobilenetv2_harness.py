"""
ImageNet Fine-tuning for Quantized MobileNetV2 using Fixed-Point Weights & Activations.

Uses the training harness for QAT, checkpointing, logging, and calibration.

Run
---
    python examples/training/train_imagenet_mobilenetv2_harness.py --workdir ./runs/imagenet_fixedpoint_harness --weight-bits 8 --act-bits 8
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from datasets import load_dataset

from training_harness.trainer import Trainer
from training_harness.config import TrainerConfig, CheckpointConfig, LoggingConfig, QuantScheduleConfig
from training_harness.schedulers import WarmupCosineScheduler
from models.mobilenetv2_quant import QuantMobileNetV2
from quantizers.fixedpoint_per_tensor import FixedPointPerTensorWeightQuant, FixedPointPerTensorActivationQuant
from utils import add_workspace_args, workspace_from_args
from utils.weight_mapping import load_pretrained_weights


class HFDatasetWrapper(Dataset):
    """Wrapper to make a Hugging Face dataset compatible with PyTorch DataLoader."""
    def __init__(self, hf_dataset, preprocess):
        self.hf_dataset = hf_dataset
        self.preprocess = preprocess

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        img = self.preprocess(item["image"].convert("RGB"))
        return img, item["label"]


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Quantized MobileNetV2 on ImageNet (Fixed-Point QAT)")
    add_workspace_args(p, name="imagenet_mobilenetv2_harness")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-bits", type=int, default=8)
    p.add_argument("--act-bits", type=int, default=8)
    return p.parse_args()


def main(args):
    ws = workspace_from_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Workspace: {ws.root}")
    print(f"Device:    {device}")

    # ---------------- Data ----------
    weights = MobileNet_V2_Weights.DEFAULT
    preprocess = weights.transforms()

    print("Loading ImageNet-1k datasets from Hugging Face...")
    hf_train_dataset = load_dataset("ILSVRC/imagenet-1k", split="train", trust_remote_code=True)
    hf_val_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", trust_remote_code=True)

    train_set = HFDatasetWrapper(hf_train_dataset, preprocess)
    val_set = HFDatasetWrapper(hf_val_dataset, preprocess)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ---------------- Model ----------
    print(f"Initializing Quantized MobileNetV2 (W{args.weight_bits}A{args.act_bits})...")
    model = QuantMobileNetV2(
        num_classes=1000,
        weight_bit_width=args.weight_bits,
        act_bit_width=args.act_bits,
        act_quant=FixedPointPerTensorActivationQuant
    ).to(device)

    print("Loading pretrained floating-point weights...")
    float_model = mobilenet_v2(weights=weights).to(device)
    model = load_pretrained_weights(model, float_model)
    del float_model

    # ---------------- Optimizer & Scheduler ----------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=total_steps // 10,
        total_steps=total_steps,
        eta_min=1e-6
    )

    # ---------------- Harness Config ----------
    config = TrainerConfig(
        experiment_name="mobilenetv2_fixedpoint_qat",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=str(device),
        checkpoint=CheckpointConfig(
            save_dir="checkpoints",
            top_k=3,
            monitor_metric="val_loss",
            monitor_mode="min",
            save_last=True,
        ),
        logging=LoggingConfig(
            log_dir="logs",
            csv_log=True,
            log_every_n_steps=10,
            plot_dir="plots",
            save_plots=True,
        ),
        quant_schedule=QuantScheduleConfig(
            float_warmup_epochs=3,
            calibration_batches=50,
            freeze_bn_after_epoch=None,
            track_scale_factors=True,
        ),
        early_stopping_patience=10,
        early_stopping_min_delta=1e-4,
    )

    # ---------------- Trainer ----------
    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=criterion,
        scheduler=scheduler,
    )

    print(f"Starting QAT training for {args.epochs} epochs...")
    tracker = trainer.fit()

    print(f"\nDone. Best validation loss: {tracker.best_value('val_loss', 'min'):.4f}")
    print(f"Best validation accuracy: {tracker.best_value('val_acc', 'max'):.2f}%")


if __name__ == "__main__":
    main(parse_args())
