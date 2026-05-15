"""
ImageNet Fine-tuning for MobileNetV2.

This script loads a pretrained MobileNetV2 model and fine-tunes it on the 
ImageNet training_harness dataset loaded from Hugging Face.

Run
---
    python examples/train_imagenet_mobilenetv2.py --workdir ./runs/imagenet_train --batch-size 64
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from datasets import load_dataset

from utils import add_workspace_args, workspace_from_args
from utils.logging import CSVLogger
from tqdm import tqdm


class HFDatasetWrapper(Dataset):
    """
    Wrapper to make a Hugging Face dataset compatible with PyTorch DataLoader.
    """
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        # The transform applied via set_transform adds 'pixel_values'
        return item["pixel_values"], item["label"]


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(loader, desc="Training")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += batch_size
        
        # Update progress bar with current metrics
        pbar.set_postfix({
            "loss": f"{running_loss / total:.4f}",
            "acc": f"{100.0 * correct / total:.2f}%"
        })
        
    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss_sum += criterion(outputs, targets).item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += inputs.size(0)
        
    return loss_sum / total, 100.0 * correct / total


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune pretrained MobileNetV2 on ImageNet")
    add_workspace_args(p, name="imagenet_mobilenetv2_train")
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--num-workers",  type=int,   default=4)
    return p.parse_args()


def main(args):
    ws = workspace_from_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Workspace: {ws.root}")
    print(f"Device:    {device}")

    # ---------------- data ----------------
    weights = MobileNet_V2_Weights.DEFAULT
    preprocess = weights.transforms()

    print("Loading ImageNet-1k datasets from Hugging Face...")
    # Load training_harness and validation splits
    hf_train_dataset = load_dataset("ILSVRC/imagenet-1k", split="train", trust_remote_code=True)
    hf_val_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", trust_remote_code=True)

    def transform_fn(batch):
        # Ensure RGB conversion to prevent shape mismatch for grayscale images
        batch["pixel_values"] = [preprocess(img.convert("RGB")) for img in batch["image"]]
        return batch

    hf_train_dataset.set_transform(transform_fn)
    hf_val_dataset.set_transform(transform_fn)
    
    train_set = HFDatasetWrapper(hf_train_dataset)
    val_set = HFDatasetWrapper(hf_val_dataset)
    
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )

    # ---------------- model ----------------
    print("Loading pretrained MobileNetV2...")
    model = mobilenet_v2(weights=weights).to(device)

    # ---------------- optimization ----------------
    criterion = nn.CrossEntropyLoss()
    # Adam is typically used with a smaller learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---------------- training_harness loop ----------------
    best_acc = 0.0
    best_ckpt = ws.checkpoints / "best.pt"
    last_ckpt = ws.checkpoints / "last.pt"
    log_path  = ws.logs / "training_log.csv"
    
    print(f"Starting training_harness for {args.epochs} epochs...")
    
    with CSVLogger(log_path, fieldnames=["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"]) as log:
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            
            lr_now = scheduler.get_last_lr()[0]
            scheduler.step()

            torch.save(model.state_dict(), last_ckpt)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_ckpt)

            log.log(
                epoch=epoch, 
                lr=f"{lr_now:.6f}", 
                train_loss=f"{tr_loss:.4f}", 
                train_acc=f"{tr_acc:.2f}", 
                val_loss=f"{val_loss:.4f}", 
                val_acc=f"{val_acc:.2f}"
            )
            
            print(f"[{epoch:3d}/{args.epochs}] "
                  f"lr={lr_now:.6f} "
                  f"train loss={tr_loss:.3f} acc={tr_acc:5.2f}% | "
                  f"val loss={val_loss:.3f} acc={val_acc:5.2f}% "
                  f"(best {best_acc:5.2f}%)")

    print(f"\nDone. Best validation accuracy: {best_acc:.2f}%")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Training log:    {log_path}")


if __name__ == "__main__":
    main(parse_args())
