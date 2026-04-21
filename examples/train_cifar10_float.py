"""
CIFAR-10 Floating-Point training — small VGG-style CNN.

Uses qatlab's shared Workspace/CSVLogger so all artifacts land in
a consistent directory layout under ``--workdir``.

Run
---
    python examples/train_cifar10_float.py
    python examples/train_cifar10_float.py --workdir ./runs/cifar10_float
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from utils import add_workspace_args, workspace_from_args, summarize_parameters
from utils.logging import CSVLogger


# --------------------------------------------------------------------
# Model
# --------------------------------------------------------------------
class DepthwiseSeparableBlock(nn.Module):
    """Depthwise 3x3 (BN, ReLU) followed by pointwise 1x1 (BN, ReLU).

    Stride is applied on the depthwise conv, which is also where
    spatial downsampling happens.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()

        # Depthwise: one 3x3 filter per input channel (groups == in_ch).
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1,
            groups=in_ch, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.relu_dw = nn.ReLU()

        # Pointwise: 1x1 conv mixing the channels.
        self.pw = nn.Conv2d(
            in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        self.relu_pw = nn.ReLU()

    def forward(self, x):
        x = self.relu_dw(self.bn_dw(self.dw(x)))
        x = self.relu_pw(self.bn_pw(self.pw(x)))
        return x

class MobileNetCIFAR(nn.Module):
    """Small MobileNet-style CNN for CIFAR-10 (Floating Point).

    Stem (3x3 conv, stride 1) keeps 32x32 resolution, then six DS
    blocks with two stride-2 downsamples bring us to 8x8 x 256.
    Global average pooling and a single Linear produce the logits.
    """

    # (out_channels, stride) for each depthwise-separable block.
    BLOCK_CFG = [
        (16, 1),
        (32, 2), # 32 -> 16
        (32, 1),
        (64, 2), # 16 -> 8
        (128, 2),  # 8 ->  4
        (128, 2),  # 4 ->  2
        (64, 2),  # 4 ->  1
    ]

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.inp = nn.Identity()

        # Standard 3x3 stem conv — first layer is usually kept dense
        # (no depthwise) because it has only 3 input channels.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        blocks = []
        in_ch = 32
        for out_ch, stride in self.BLOCK_CFG:
            blocks.append(DepthwiseSeparableBlock(
                in_ch, out_ch, stride))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, num_classes, bias=True),
        )

    def forward(self, x):
        x = self.inp(x)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

class VGG(nn.Module):
    """
    Small VGG-style CNN for CIFAR-10 (Floating Point).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.inp = nn.Identity()

        self.features = nn.Sequential(
            *self._conv_block(3,   64),
            *self._conv_block(64,  64),
            nn.MaxPool2d(2),                        # 32 -> 16
            *self._conv_block(64,  128),
            *self._conv_block(128, 128),
            nn.MaxPool2d(2),                        # 16 -> 8
            *self._conv_block(128, 256),
            *self._conv_block(256, 256),
            nn.AdaptiveAvgPool2d(1),                #  8 -> 1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes, bias=True),
        )

    @staticmethod
    def _conv_block(in_ch, out_ch):
        return [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        ]

    def forward(self, x):
        x = self.inp(x)
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
    p = argparse.ArgumentParser(description="Floating Point training on CIFAR-10")
    # Use a different name to avoid overwriting quantized results
    add_workspace_args(p, name="cifar10_vgg_float")
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch-size",   type=int,   default=512)
    p.add_argument("--lr",           type=float, default=0.05)
    p.add_argument("--momentum",     type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
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
    model = MobileNetCIFAR(num_classes=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model:     {n_params / 1e6:.2f}M params (Float32)")
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
