"""
CIFAR-10 PTQ with Brevitas — small MobileNet-style CNN.

This script performs Post-Training Quantization (PTQ). It loads a pretrained 
floating-point model, initializes a quantized version of the architecture, 
loads the weights, calibrates activation statistics using a small subset 
of the training data, and evaluates the result.

Run
---
    python examples/ptq_cifar10.py
    python examples/ptq_cifar10.py --weight-bits 4 --act-bits 4 \\
                               --workdir ./runs/cifar10_ptq_4bit
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import brevitas.nn as qnn
from brevitas.inject.enum import FloatToIntImplType

from utils import add_workspace_args, workspace_from_args, summarize_parameters
from models.cifar10_quant import QuantMobileNetCIFAR


# Mapping from CLI string to Brevitas FloatToIntImplType
ROUNDING_MAP = {
    "round": FloatToIntImplType.ROUND,
    "floor": FloatToIntImplType.FLOOR,
    "ceil": FloatToIntImplType.CEIL,
    "round_to_zero": FloatToIntImplType.ROUND_TO_ZERO,
    "dpu": FloatToIntImplType.DPU,
    "learned_round": FloatToIntImplType.LEARNED_ROUND,
    "stochastic_round": FloatToIntImplType.STOCHASTIC_ROUND,
}


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def set_rounding_mode(model, rounding_mode):
    """
    Iterate through all modules in the model and set the rounding mode
    for any quantizers found.
    """
    for m in model.modules():
        if hasattr(m, 'weight_quant') and m.weight_quant is not None:
            m.weight_quant.float_to_int_impl_type = rounding_mode
        if hasattr(m, 'act_quant') and m.act_quant is not None:
            m.act_quant.float_to_int_impl_type = rounding_mode


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


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Brevitas PTQ on CIFAR-10")
    # --workdir (+ QATLAB_WORKDIR env-var fallback) handled here:
    add_workspace_args(p, name="cifar10_ptq")
    p.add_argument("--batch-size",   type=int,   default=512)
    p.add_argument("--weight-bits",  type=int,   default=8)
    p.add_argument("--act-bits",     type=int,   default=8)
    p.add_argument("--num-workers",  type=int,   default=2)
    p.add_argument("--pretrained",   type=str,   default=None,
                   help="Path to pretrained floating-point model checkpoint")
    p.add_argument("--calib-batches", type=int,  default=10,
                   help="Number of batches to use for activation calibration")
    p.add_argument("--rounding",     type=str,   default="round",
                   choices=list(ROUNDING_MAP.keys()),
                   help="Rounding technique for quantization")
    return p.parse_args()


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main(args):
    ws = workspace_from_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Workspace: {ws.root}")
    print(f"Device:    {device}")

    # ---------------- data ----------------
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=str(ws.datasets), train=True,  download=True,
        transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(
        root=str(ws.datasets), train=False, download=True,
        transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    # ---------------- model ----------------
    model = QuantMobileNetCIFAR(num_classes=10,
                     weight_bit_width=args.weight_bits,
                     act_bit_width=args.act_bits).to(device)

    # Apply selected rounding mode
    rounding_mode = ROUNDING_MAP[args.rounding]
    set_rounding_mode(model, rounding_mode)
    print(f"Rounding mode: {args.rounding}")

    if args.pretrained:
        pretrained_path = args.pretrained
    else:
        pretrained_path = ws.checkpoints / "best_float.pt"
        pretrained_path = "/home/th/tmp/quanttests/cifar10_vgg_float/checkpoints/best_float.pt"  # ws.checkpoints / "best_float.pt"

    print(f"Loading pretrained weights from: {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location=device)
    # strict=False is used because the float model lacks quantization parameters
    # and has slightly different attribute names (e.g. self.inp vs self.quant_inp)
    model.load_state_dict(state_dict, strict=False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model:     {n_params / 1e6:.2f}M params "
          f"(W{args.weight_bits}A{args.act_bits})")
    summarize_parameters(model)

    # ---------------- calibration ----------------
    # PTQ requires a calibration phase to determine the scale factors for 
    # activation quantizers that rely on runtime statistics.
    print(f"Calibrating activations using {args.calib_batches} batches...")
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(device)
            model(inputs)
            if i + 1 >= args.calib_batches:
                break
    print("Calibration complete.")

    # ---------------- evaluation ----------------
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\nPTQ Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.2f}%")


if __name__ == "__main__":
    main(parse_args())
