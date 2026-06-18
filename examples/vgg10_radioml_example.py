#!/usr/bin/env python3
"""
Reproducing "RadioML Meets FINN: Enabling Future RF Applications
With FPGA Streaming Architectures" (Jentzsch et al., IEEE Micro 2022)

Implements VGG10 / VGG10-S with Brevitas quantization-aware training (QAT)
on the RadioML 2018.01A dataset.

Architecture recap from the paper:
  - 7 x Conv1d(kernel=3) + BatchNorm + ReLU + MaxPool(2) [halves feature map each time]
  - 2 x Linear + BatchNorm + ReLU
  - 1 x Linear (classifier, 24 classes)
  - VGG10:   Fc=64 filters, Fd=128 dense units, 4-bit W/A (5-bit first conv), 8-bit input
  - VGG10-S: Fc=32 filters, same otherwise — smaller but competitive accuracy

Quantization scheme (paper Table 1 / Section "Models and Training"):
  Input         →  8-bit fixed-point
  First conv    →  5-bit weights, 4-bit activations
  Other convs   →  4-bit weights, 4-bit activations
  Dense layers  →  4-bit weights, 4-bit activations
  Output layer  →  full precision (for training stability)

Dataset:
  RadioML 2018.01A — 24 modulation types, SNR -20..+30 dB, 1024-sample frames
  HuggingFace: rfml/deepsig_radioml_2018_01a  (primary)
  Fallback:    local HDF5 (GOLD_XYZ_OSC.0001_1024.hdf5 from DeepSig)

Usage:
  pip install brevitas datasets torch tqdm h5py

  # Download automatically via HuggingFace:
  python train_radioml.py --model vgg10s

  # Or point to a local HDF5 file:
  python train_radioml.py --model vgg10 --hdf5 /path/to/GOLD_XYZ_OSC.0001_1024.hdf5

  # High-SNR only (matches paper's evaluation focus):
  python train_radioml.py --model vgg10s --min-snr 6
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

# ── Brevitas imports ──────────────────────────────────────────────────────────
try:
    import brevitas.nn as qnn
    from brevitas.quant.scaled_int import (
        Int8ActPerTensorFloat,
        Int8WeightPerTensorFloat,
    )
except ImportError:
    sys.exit(
        "Brevitas not found. Install with:\n"
        "  pip install brevitas"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Custom quantizers (extend base classes, just override bit_width)
# ─────────────────────────────────────────────────────────────────────────────

class W4(Int8WeightPerTensorFloat):
    """4-bit signed weight quantizer."""
    bit_width = 4

class W5(Int8WeightPerTensorFloat):
    """5-bit signed weight quantizer (paper uses this for the first conv layer)."""
    bit_width = 5

class A8(Int8ActPerTensorFloat):
    """8-bit activation quantizer — applied to the raw I/Q input."""
    bit_width = 8

class A4(Int8ActPerTensorFloat):
    """4-bit activation quantizer — all intermediate activations."""
    bit_width = 4


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Model definition
# ─────────────────────────────────────────────────────────────────────────────

def _conv_block(in_ch: int, out_ch: int, weight_quant, first: bool = False) -> nn.Sequential:
    """One VGG conv block: QuantConv1d → BN → ReLU → QuantIdentity → MaxPool."""
    return nn.Sequential(
        qnn.QuantConv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            padding=1,
            bias=False,
            weight_quant=weight_quant,
        ),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
        # Quantize activations *after* BN+ReLU so the scale is learned on the
        # normalised, rectified distribution — mirrors the FINN multithreshold.
        qnn.QuantIdentity(act_quant=A4, return_quant_tensor=False),
        nn.MaxPool1d(kernel_size=2, stride=2),
    )


def _dense_block(in_features: int, out_features: int) -> nn.Sequential:
    """Dense block: QuantLinear → BN → ReLU → QuantIdentity."""
    return nn.Sequential(
        qnn.QuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            weight_quant=W4,
        ),
        nn.BatchNorm1d(out_features),
        nn.ReLU(inplace=True),
        qnn.QuantIdentity(act_quant=A4, return_quant_tensor=False),
    )


class QuantVGG10(nn.Module):
    """
    Quantised VGG10 / VGG10-S for automatic modulation classification.

    Args:
        num_classes: number of output classes (24 for RadioML 2018.01A)
        Fc: convolutional filter count  (64 → VGG10, 32 → VGG10-S)
        Fd: dense layer width           (128 for both variants in the paper)
    """

    def __init__(self, num_classes: int = 24, Fc: int = 64, Fd: int = 128):
        super().__init__()

        # ── Input quantiser: fixed 8-bit range learned from dataset stats ────
        self.input_quant = qnn.QuantIdentity(
            act_quant=A8,
            return_quant_tensor=False,
        )

        # ── 7 convolutional blocks ────────────────────────────────────────────
        # First layer: 5-bit weights (more sensitive to quantisation).
        # After 7 × MaxPool(2): time dimension  1024 → 8.
        conv_layers = []
        in_ch = 2  # I and Q channels
        for i in range(7):
            wq = W5 if i == 0 else W4
            conv_layers.append(_conv_block(in_ch, Fc, wq, first=(i == 0)))
            in_ch = Fc
        self.features = nn.Sequential(*conv_layers)

        # ── Two hidden dense blocks ───────────────────────────────────────────
        flat_size = Fc * 8   # Fc channels × 8 time steps
        self.dense = nn.Sequential(
            _dense_block(flat_size, Fd),
            _dense_block(Fd, Fd),
        )

        # ── Classifier (full-precision for stable softmax during training) ────
        self.classifier = nn.Linear(Fd, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 1024)  — I/Q channels first
        x = self.input_quant(x)   # → 8-bit
        x = self.features(x)      # → (B, Fc, 8)
        x = x.flatten(1)          # → (B, Fc*8)
        x = self.dense(x)         # → (B, Fd)
        return self.classifier(x) # → (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

# Canonical modulation order in RadioML 2018.01A
MODULATIONS = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
]
NUM_CLASSES = len(MODULATIONS)   # 24


def load_from_hdf5(path: str, min_snr: int = -20) -> tuple:
    """
    Load RadioML 2018.01A from a local HDF5 file.

    Expected keys (as shipped by DeepSig):
      X  — (2555904, 1024, 2)   float32  I/Q samples
      Y  — (2555904, 24)        float32  one-hot labels
      Z  — (2555904,)           float32  SNR in dB

    Returns (X, labels, snrs) with X transposed to (N, 2, 1024).
    """
    try:
        import h5py
    except ImportError:
        sys.exit("h5py not installed. Run: pip install h5py")

    print(f"Loading HDF5 from {path} …")
    with h5py.File(path, "r") as f:
        X = f["X"][:]       # (N, 1024, 2)
        Y = f["Y"][:]       # (N, 24)  one-hot
        Z = f["Z"][:]       # (N,)

    mask = Z >= min_snr
    X, Y, Z = X[mask], Y[mask], Z[mask]

    X = X.transpose(0, 2, 1).astype(np.float32)   # (N, 2, 1024)
    labels = np.argmax(Y, axis=1).astype(np.int64)
    return X, labels, Z.astype(np.float32)


def load_from_huggingface(min_snr: int = -20) -> tuple:
    """
    Load RadioML 2018.01A from the HuggingFace Hub.

    Dataset: rfml/deepsig_radioml_2018_01a
    Expected columns: 'iq'  (list[float], length 2048 — I then Q interleaved, or 2×1024),
                      'label' (int),
                      'snr'   (int)

    If the column layout differs on the version you download, adjust the
    reshaping logic below.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("HuggingFace `datasets` not installed. Run: pip install datasets")

    # Primary candidate — change if the dataset moves:
    HF_DATASET = "rfml/deepsig_radioml_2018_01a"
    print(f"Downloading {HF_DATASET} from HuggingFace Hub …")
    print("(This may take a while on first run — dataset is ~25 GB)")

    ds = load_dataset(HF_DATASET, split="train")

    # Filter by SNR
    if min_snr > -20:
        ds = ds.filter(lambda ex: ex["snr"] >= min_snr)

    # Convert to numpy — HF columns: 'iq' flat list of 2048 floats (I‖Q)
    iq    = np.array(ds["iq"],    dtype=np.float32)   # (N, 2048)
    labels = np.array(ds["label"], dtype=np.int64)
    snrs   = np.array(ds["snr"],   dtype=np.float32)

    # Reshape: (N, 2048) → (N, 2, 1024)
    X = iq.reshape(-1, 2, 1024)
    return X, labels, snrs


def build_dataloaders(
    X: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 1024,
    val_split: float = 0.1,
    seed: int = 42,
) -> tuple:
    """Split into train/val and return DataLoaders."""
    X_t      = torch.from_numpy(X)
    labels_t = torch.from_numpy(labels)

    dataset  = TensorDataset(X_t, labels_t)
    n_val    = int(len(dataset) * val_split)
    n_train  = len(dataset) - n_val
    gen      = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple:
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in tqdm(loader, desc="  train", leave=False):
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(y_batch)
        correct      += (logits.argmax(1) == y_batch).sum().item()
        total        += len(y_batch)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in tqdm(loader, desc="  val  ", leave=False):
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        logits = model(X_batch)
        loss   = criterion(logits, y_batch)

        running_loss += loss.item() * len(y_batch)
        correct      += (logits.argmax(1) == y_batch).sum().item()
        total        += len(y_batch)

    return running_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train quantised VGG10/VGG10-S on RadioML 2018.01A (Brevitas QAT)"
    )
    p.add_argument(
        "--model", choices=["vgg10", "vgg10s"], default="vgg10s",
        help="vgg10 → Fc=64 (larger, ~94%% acc); vgg10s → Fc=32 (smaller, ~91%% acc)",
    )
    p.add_argument(
        "--hdf5", type=str, default=None,
        help="Path to local HDF5 file. If omitted, downloads from HuggingFace.",
    )
    p.add_argument("--min-snr",  type=int,   default=6,
                   help="Minimum SNR (dB) for training/eval. Paper focuses on ≥6 dB.")
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--batch",    type=int,   default=1024)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--wd",       type=float, default=1e-4,  help="Weight decay.")
    p.add_argument("--patience", type=int,   default=10,
                   help="ReduceLROnPlateau patience (epochs).")
    p.add_argument("--out",      type=str,   default="checkpoints",
                   help="Directory to save best model.")
    p.add_argument("--seed",     type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = (
        "cuda"  if torch.cuda.is_available()  else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    if args.hdf5:
        X, labels, snrs = load_from_hdf5(args.hdf5, min_snr=args.min_snr)
    else:
        X, labels, snrs = load_from_huggingface(min_snr=args.min_snr)

    print(f"Dataset: {len(X):,} samples  |  "
          f"SNR range: {snrs.min():.0f}..{snrs.max():.0f} dB  |  "
          f"Classes: {NUM_CLASSES}")

    train_loader, val_loader = build_dataloaders(
        X, labels, batch_size=args.batch, seed=args.seed
    )
    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # ── Build model ───────────────────────────────────────────────────────────
    Fc = 64 if args.model == "vgg10" else 32
    Fd = 128
    model = QuantVGG10(num_classes=NUM_CLASSES, Fc=Fc, Fd=Fd).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model.upper()}  |  Fc={Fc}, Fd={Fd}  |  "
          f"Params: {n_params:,}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=args.patience, verbose=True,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    best_val_acc = 0.0
    best_path    = os.path.join(args.out, f"{args.model}_best.pt")

    print(f"\n{'Epoch':>5}  {'Train loss':>10}  {'Train acc':>9}  "
          f"{'Val loss':>9}  {'Val acc':>8}  {'LR':>8}")
    print("─" * 62)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(vl_acc)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>5}  {tr_loss:>10.4f}  {tr_acc:>8.2%}  "
              f"{vl_loss:>9.4f}  {vl_acc:>7.2%}  {lr_now:>8.2e}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(
                {
                    "epoch":     epoch,
                    "model":     args.model,
                    "Fc":        Fc,
                    "Fd":        Fd,
                    "state":     model.state_dict(),
                    "val_acc":   vl_acc,
                    "min_snr":   args.min_snr,
                },
                best_path,
            )
            print(f"  ✓ Best model saved  (val acc = {vl_acc:.2%})")

    print(f"\nTraining complete.  Best val acc: {best_val_acc:.2%}")
    print(f"Checkpoint: {best_path}")

    # ── Paper target (Table 1) ────────────────────────────────────────────────
    target = 94.1 if args.model == "vgg10" else 91.0
    print(f"\nPaper target @ 30 dB SNR: {target}%  "
          f"(this run: min_snr≥{args.min_snr} dB)")
    print("Note: the paper trains/tests on high-SNR data only; "
          "to match exactly, set --min-snr 6 and evaluate at 30 dB.")


if __name__ == "__main__":
    main()