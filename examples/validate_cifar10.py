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
# Model
# --------------------------------------------------------------------
class DepthwiseSeparableBlock(nn.Module):
    """Depthwise 3x3 (BN, ReLU) followed by pointwise 1x1 (BN, ReLU).

    Stride is applied on the depthwise conv, which is also where
    spatial downsampling happens.

    Note on quantization
    --------------------
    Depthwise convs in particular often benefit noticeably from
    per-channel weight quantization (each depthwise kernel is
    independent), whereas the default here is per-tensor. For an
    8-bit run the difference is small; at 4 bits it becomes visible.
    Swap ``weight_quant=Int8WeightPerChannelFloat`` into the depthwise
    ``QuantConv2d`` if you want to try it.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int,
                 weight_bit_width: int, act_bit_width: int):
        super().__init__()

        # Depthwise: one 3x3 filter per input channel (groups == in_ch).
        self.dw = qnn.QuantConv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1,
            groups=in_ch, bias=False,
            weight_bit_width=weight_bit_width)
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.relu_dw = qnn.QuantReLU(bit_width=act_bit_width)

        # Pointwise: 1x1 conv mixing the channels.
        self.pw = qnn.QuantConv2d(
            in_ch, out_ch, kernel_size=1, bias=False,
            weight_bit_width=weight_bit_width)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        self.relu_pw = qnn.QuantReLU(bit_width=act_bit_width)

    def forward(self, x):
        x = self.relu_dw(self.bn_dw(self.dw(x)))
        x = self.relu_pw(self.bn_pw(self.pw(x)))
        return x

class QuantMobileNetCIFAR(nn.Module):
    """Small MobileNet-style quantized CNN for CIFAR-10.

    Stem (3x3 conv, stride 1) keeps 32x32 resolution, then six DS
    blocks with two stride-2 downsamples bring us to 8x8 x 256.
    Global average pooling and a single QuantLinear produce the logits.
    """

    # (out_channels, stride) for each depthwise-separable block.
    BLOCK_CFG = [
        (16, 1),
        (32, 2), # 32 -> 16
        (32, 1),
        (64, 2), # 16 -> 8
        (128, 2),  # 8 ->  4
        (128, 2),  # 4 ->  2
        (256, 2),  # 4 ->  1
        #(256, 1),
        #(256, 1),
    ]

    BLOCK_CFG = [
        (16, 1),
        (32, 2), # 32 -> 16
        (32, 1),
        (64, 2), # 16 -> 8
        (128, 2),  # 8 ->  4
        (128, 2),  # 4 ->  2
        (64, 2),  # 4 ->  1
        #(256, 1),
        #(256, 1),
    ]

    def __init__(self,
                 num_classes: int = 10,
                 weight_bit_width: int = 8,
                 act_bit_width: int = 8):
        super().__init__()

        self.quant_inp = qnn.QuantIdentity(bit_width=act_bit_width,
                                           return_quant_tensor=False)

        # Standard 3x3 stem conv — first layer is usually kept dense
        # (no depthwise) because it has only 3 input channels.
        self.stem = nn.Sequential(
            qnn.QuantConv2d(3, 32, kernel_size=3, padding=1, bias=False,
                            weight_bit_width=weight_bit_width),
            nn.BatchNorm2d(32),
            qnn.QuantReLU(bit_width=act_bit_width),
        )

        blocks = []
        in_ch = 32
        for out_ch, stride in self.BLOCK_CFG:
            blocks.append(DepthwiseSeparableBlock(
                in_ch, out_ch, stride, weight_bit_width, act_bit_width))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            qnn.QuantLinear(in_ch, num_classes, bias=True,
                            weight_bit_width=weight_bit_width),
        )

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

class QuantVGG(nn.Module):
    """
    Small VGG-style quantized CNN for CIFAR-10.

    Brevitas cheat sheet
    --------------------
    * ``qnn.QuantIdentity``  – fake-quantizes the (float) network input.
    * ``qnn.QuantConv2d`` /
      ``qnn.QuantLinear``    – standard conv/linear with fake-quant
                               applied to weights during forward.
    * ``qnn.QuantReLU``      – ReLU whose output is fake-quantized to
                               an unsigned integer range.
    * ``nn.BatchNorm*``      – kept in float, can be folded into the
                               preceding conv/linear later.
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
    model = QuantMobileNetCIFAR(num_classes=10,
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
