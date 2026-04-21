"""
CIFAR-10 Validation script for Brevitas QAT.

This script loads a trained QuantVGG model from a checkpoint and evaluates
its performance on the CIFAR-10 test set.

Run
---
    python examples/validate_cifar10.py --workdir ./runs/cifar10_4bit --weight-bits 4 --act-bits 4
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import brevitas.nn as qnn

from utils.workspace import add_workspace_args, workspace_from_args


# --------------------------------------------------------------------
# Model (Must match the architecture in train_cifar10.py)
# --------------------------------------------------------------------
class QuantVGG(nn.Module):
    """
    Small VGG-style quantized CNN for CIFAR-10.
    """

    def __init__(self,
                 num_classes: int = 10,
                 weight_bit_width: int = 8,
                 act_bit_width: int = 8):
        super().__init__()

        self.quant_inp = qnn.QuantIdentity(bit_width=act_bit_width,
                                           return_quant_tensor=False)

        self.features = nn.Sequential(
            *self._conv_block(3,   64,  weight_bit_width, act_bit_width),
            *self._conv_block(64,  64,  weight_bit_width, act_bit_width),
            nn.MaxPool2d(2),                        # 32 -> 16
            *self._conv_block(64,  128, weight_bit_width, act_bit_width),
            *self._conv_block(128, 128, weight_bit_width, act_bit_width),
            nn.MaxPool2d(2),                        # 16 -> 8
            *self._conv_block(128, 256, weight_bit_width, act_bit_width),
            *self._conv_block(256, 256, weight_bit_width, act_bit_width),
            nn.AdaptiveAvgPool2d(1),                #  8 -> 1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            qnn.QuantLinear(256, 256, bias=False,
                            weight_bit_width=weight_bit_width),
            nn.BatchNorm1d(256),
            qnn.QuantReLU(bit_width=act_bit_width),
            qnn.QuantLinear(256, num_classes, bias=True,
                            weight_bit_width=weight_bit_width),
        )

    @staticmethod
    def _conv_block(in_ch, out_ch, w_bits, a_bits):
        return [
            qnn.QuantConv2d(in_ch, out_ch, kernel_size=3, padding=1,
                            bias=False, weight_bit_width=w_bits),
            nn.BatchNorm2d(out_ch),
            qnn.QuantReLU(bit_width=a_bits),
        ]

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.features(x)
        x = self.classifier(x)
        return x


# --------------------------------------------------------------------
# Evaluation helper
# --------------------------------------------------------------------
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
    p = argparse.ArgumentParser(description="Validate Brevitas QAT model on CIFAR-10")
    add_workspace_args(p, name="cifar10_vgg")
    p.add_argument("--weight-bits",  type=int,   default=8)
    p.add_argument("--act-bits",     type=int,   default=8)
    p.add_argument("--num-workers",  type=int,   default=2)
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
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_set = torchvision.datasets.CIFAR10(
        root=str(ws.data), train=False, download=True,
        transform=transform_test)

    test_loader = DataLoader(test_set,  batch_size=512,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    # ---------------- model ----------------
    model = QuantVGG(num_classes=10,
                     weight_bit_width=args.weight_bits,
                     act_bit_width=args.act_bits).to(device)

    # Load the best checkpoint
    best_ckpt = ws.checkpoints / "best.pt"
    if not best_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found at {best_ckpt}")

    print(f"Loading checkpoint: {best_ckpt}")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()

    # ---------------- evaluation ----------------
    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, test_loader, criterion, device)

    print(f"\nResults for W{args.weight_bits}A{args.act_bits}:")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Acc:  {acc:.2f}%")


if __name__ == "__main__":
    main(parse_args())
