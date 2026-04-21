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
from models.cifar10_float import MobileNetCIFAR


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
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=2048)
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
    best_ckpt = ws.checkpoints / "best_float.pt"
    last_ckpt = ws.checkpoints / "last_float.pt"
    log_path  = ws.logs / "training_log_float.csv"

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
