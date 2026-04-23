"""
ImageNet Validation for pretrained MobileNetV2.

This script loads a pretrained MobileNetV2 model and validates it on the 
ImageNet validation dataset loaded from Hugging Face.

Run
---
    python examples/validate_imagenet_mobilenetv2.py --workdir ./runs/imagenet_val
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from datasets import load_dataset

from utils import add_workspace_args, workspace_from_args
from utils.logging import CSVLogger


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


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += inputs.size(0)
        
    return 100.0 * correct / total


def parse_args():
    p = argparse.ArgumentParser(description="Validate pretrained MobileNetV2 on ImageNet")
    add_workspace_args(p, name="imagenet_mobilenetv2_val")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main(args):
    ws = workspace_from_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Workspace: {ws.root}")
    print(f"Device:    {device}")

    # ---------------- data ----------------
    # MobileNetV2 pretrained weights expect specific normalization and resizing
    weights = MobileNet_V2_Weights.DEFAULT
    preprocess = weights.transforms()

    print("Loading ImageNet-1k validation set from Hugging Face...")
    # Note: This requires huggingface-cli login and access to the imagenet-1k dataset
    hf_val_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", trust_remote_code=True)

    def transform_fn(batch):
        # Apply the torchvision preprocess to each image in the batch.
        # We call .convert("RGB") to ensure grayscale images are converted to 3 channels,
        # preventing shape mismatch errors during normalization.
        batch["pixel_values"] = [preprocess(img.convert("RGB")) for img in batch["image"]]
        return batch

    hf_val_dataset.set_transform(transform_fn)
    
    val_set = HFDatasetWrapper(hf_val_dataset)
    
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
    model.eval()

    # ---------------- evaluation ----------------
    log_path = ws.logs / "validation_log.csv"
    
    with CSVLogger(log_path, fieldnames=["top1_acc"]) as log:
        acc = evaluate(model, val_loader, device)
        log.log(top1_acc=f"{acc:.2f}")
        print(f"\nValidation Top-1 Accuracy: {acc:.2f}%")

    print(f"Results logged to: {log_path}")


if __name__ == "__main__":
    main(parse_args())
