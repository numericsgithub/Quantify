"""
CIFAR-10 QAT with Brevitas — small VGG-style CNN.

Uses qatlab's shared Workspace/CSVLogger so all artifacts land in
a consistent directory layout under ``--workdir``.

Run
---
    python examples/cifar10_vgg.py
    python examples/cifar10_vgg.py --weight-bits 4 --act-bits 4 \\
                               --workdir ./runs/cifar10_4bit
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import brevitas.nn as qnn

from utils import add_workspace_args, workspace_from_args, summarize_parameters
from utils.logging import CSVLogger


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
# Training / evaluation helpers
# --------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += inputs.size(0)

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


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Brevitas QAT on CIFAR-10 (small VGG)")
    # --workdir (+ QATLAB_WORKDIR env-var fallback) handled here:
    add_workspace_args(p, name="cifar10_vgg")
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch-size",   type=int,   default=512)
    p.add_argument("--lr",           type=float, default=0.05)
    p.add_argument("--momentum",     type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--weight-bits",  type=int,   default=8)
    p.add_argument("--act-bits",     type=int,   default=8)
    p.add_argument("--num-workers",  type=int,   default=2)
    p.add_argument("--pretrained",   type=str,   default=None,
                   help="Path to pretrained floating-point model checkpoint")
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
        root=str(ws.data), train=True,  download=True,
        transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(
        root=str(ws.data), train=False, download=True,
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

    if args.pretrained:
        print(f"Loading pretrained weights from: {args.pretrained}")
        state_dict = torch.load(args.pretrained, map_location=device)
        # strict=False is used because the float model lacks quantization parameters
        # and has slightly different attribute names (e.g. self.inp vs self.quant_inp)
        model.load_state_dict(state_dict, strict=False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model:     {n_params / 1e6:.2f}M params "
          f"(W{args.weight_bits}A{args.act_bits})")
    summarize_parameters(model)

    # ---------------- optimizer ----------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=args.epochs)

    # ---------------- training loop ----------------
    best_acc = 0.0
    best_ckpt = ws.checkpoints / "best.pt"
    last_ckpt = ws.checkpoints / "last.pt"
    log_path  = ws.logs / "training_log.csv"

    with CSVLogger(log_path,
                   fieldnames=["epoch", "lr",
                               "train_loss", "train_acc",
                               "test_loss",  "test_acc"]) as log:
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device)
            te_loss, te_acc = evaluate(
                model, test_loader, criterion, device)
            lr_now = scheduler.get_last_lr()[0]
            scheduler.step()

            torch.save(model.state_dict(), last_ckpt)
            if te_acc > best_acc:
                best_acc = te_acc
                torch.save(model.state_dict(), best_ckpt)

            log.log(epoch=epoch, lr=f"{lr_now:.6f}",
                    train_loss=f"{tr_loss:.4f}", train_acc=f"{tr_acc:.2f}",
                    test_loss=f"{te_loss:.4f}", test_acc=f"{te_acc:.2f}")

            print(f"[{epoch:3d}/{args.epochs}] "
                  f"lr={lr_now:.4f}  "
                  f"train loss={tr_loss:.3f} acc={tr_acc:5.2f}%  | "
                  f"test loss={te_loss:.3f} acc={te_acc:5.2f}%  "
                  f"(best {best_acc:5.2f}%)")

    print(f"\nDone. Best test accuracy: {best_acc:.2f}%")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Last checkpoint: {last_ckpt}")
    print(f"Training log:    {log_path}")


if __name__ == "__main__":
    main(parse_args())
