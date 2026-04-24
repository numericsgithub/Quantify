"""
ImageNet Fine-tuning for Quantized MobileNetV2 using Fixed-Point Weights.

This script loads a pretrained floating-point MobileNetV2, maps the weights 
to a quantized version, and fine-tunes it on ImageNet.

Run
---
    python examples/train_imagenet_mobilenetv2_fixedpoint.py --workdir ./runs/imagenet_fixedpoint --weight-bits 4 --act-bits 4
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from datasets import load_dataset
import logging

from utils import add_workspace_args, workspace_from_args
from utils.logging import CSVLogger
from utils.weight_mapping import load_pretrained_weights
from models.mobilenetv2_quant import QuantMobileNetV2
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)

import torch
import torch.nn as nn


def check_all_devices(model, inputs=None):
    print("=" * 60)
    print("MODEL PARAMETERS & BUFFERS")
    print("=" * 60)

    devices_found = set()
    issues = []

    # 1. Parameters
    for name, param in model.named_parameters():
        dev = str(param.device)
        devices_found.add(dev)
        print(f"{dev}  [PARAM]  {name}: shape={tuple(param.shape)}")

    # 2. Buffers (BatchNorm stats, positional encodings, etc.)
    for name, buf in model.named_buffers():
        dev = str(buf.device)
        devices_found.add(dev)
        print(f"{dev}:  [BUFFER] {name}: shape={tuple(buf.shape)}")

    # 3. Check every module's __dict__ for any tensor attribute
    print("\n" + "=" * 60)
    print("TENSOR ATTRIBUTES IN MODULE __dict__")
    print("=" * 60)
    for mod_name, module in model.named_modules():
        for attr_name, attr_val in module.__dict__.items():
            if isinstance(attr_val, torch.Tensor):
                dev = str(attr_val.device)
                devices_found.add(dev)
                label = f"{mod_name}.{attr_name}" if mod_name else attr_name
                print(f"{dev}:  [ATTR]   {label}: shape={tuple(attr_val.shape)}")
                if dev not in ("cuda:0",):  # adjust expected device
                    issues.append(label)

    # 4. Check inputs if provided
    if inputs is not None:
        print("\n" + "=" * 60)
        print("INPUTS")
        print("=" * 60)
        if isinstance(inputs, torch.Tensor):
            inputs = {"input": inputs}
        if isinstance(inputs, (list, tuple)):
            inputs = {f"input[{i}]": v for i, v in enumerate(inputs)}
        for name, val in inputs.items():
            if isinstance(val, torch.Tensor):
                dev = str(val.device)
                devices_found.add(dev)
                print(f"{dev}  [INPUT]  {name}: shape={tuple(val.shape)}")

    # 5. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Devices found: {devices_found}")
    if len(devices_found) > 1:
        print(f"  ⚠️  MULTIPLE DEVICES DETECTED — likely cause of your error!")
    else:
        print(f"  ✅ All tensors on same device.")

    return devices_found

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
        
        pbar.set_postfix({
            "loss": f"{running_loss / total:.4f}",
            "acc": f"{100.0 * correct / total:.2f}%"
        })
        
    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    model = model.to("cpu")
    model = model.to("cuda")

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss_sum += criterion(outputs, targets).item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += inputs.size(0)
        
    return loss_sum / total, 100.0 * correct / total


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Quantized MobileNetV2 on ImageNet (Fixed-Point)")
    add_workspace_args(p, name="imagenet_mobilenetv2_fixedpoint")
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--weight-bits",  type=int,   default=8)
    p.add_argument("--act-bits",     type=int,   default=8)
    return p.parse_args()


def main(args):
    ws = workspace_from_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # This does not! (I have a gpu, so it will choose "cuda")
    # device = torch.device("cpu") # This works! No problems!

    print(f"Workspace: {ws.root}")
    print(f"Device:    {device}")

    # ---------------- data ----------------
    weights = MobileNet_V2_Weights.DEFAULT
    preprocess = weights.transforms()

    print("Loading ImageNet-1k datasets from Hugging Face...")
    hf_train_dataset = load_dataset("ILSVRC/imagenet-1k", split="train", trust_remote_code=True)
    hf_val_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", trust_remote_code=True)

    def transform_fn(batch):
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
    print(f"Initializing Quantized MobileNetV2 (W{args.weight_bits}A{args.act_bits})...")
    model = QuantMobileNetV2(
        num_classes=1000, 
        weight_bit_width=args.weight_bits, 
        act_bit_width=args.act_bits
    ).to("cpu")#.to(device)

    print("Loading pretrained floating-point weights...")
    float_model = mobilenet_v2(weights=weights).to("cpu")#.to(device)
    model = load_pretrained_weights(model, float_model)
    model = model.to(device)


    # Debug device locations for all modules, buffers, and custom quantizer internals
    check_all_devices(model)
    
    # Clean up float model to save memory
    del float_model

    # ---------------- optimization ----------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---------------- training loop ----------------
    best_acc = 0.0
    best_ckpt = ws.checkpoints / "best.pt"
    last_ckpt = ws.checkpoints / "last.pt"
    log_path  = ws.logs / "training_log.csv"
    
    print(f"Starting training for {args.epochs} epochs...")
    
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
