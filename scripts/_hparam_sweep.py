"""
Hyperparameter sweep for the MNIST QAT example.

Each config trains a fresh SimpleMNISTNet for a short horizon (default 15 epochs)
and we record best/final val_acc. The goal is to identify configurations that
keep the model alive after annealing finishes, rather than locking up at random
accuracy.

Knobs swept: learning rate, warmup epochs, bit width, optimizer, batch size,
grad-clip norm.

To keep iteration fast we monkey-patch the checkpoint manager to skip ONNX
export per save (that's ~1-2s/epoch overhead we don't need for a sweep).
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, ".")

import brevitas.nn as qnn
from quantizers.fixedpoint_per_tensor import (
    FixedPointPerTensorActivationQuant,
    FixedPointPerTensorBiasQuant,
    FixedPointPerTensorWeightQuant,
    RoundingMode,
)
from quantizers.manager import QuantizerManager
from training_harness import checkpointing as ckpt_module
from training_harness.config import QuantScheduleConfig, TrainerConfig
from training_harness.engine_utils import set_seed
from training_harness.trainer import Trainer


# ---------------------------------------------------------------------------
# Speed: skip ONNX export per checkpoint
# ---------------------------------------------------------------------------
def _noop_save(self, *args, **kwargs):
    return
ckpt_module.CheckpointManager.save = _noop_save
def _noop_load_best(self, *args, **kwargs):
    return None
ckpt_module.CheckpointManager.load_best = _noop_load_best


# ---------------------------------------------------------------------------
# Model factory — lets us override bit_width per config
# ---------------------------------------------------------------------------
def make_quantizers(bit_width: int):
    # Brevitas Injectors are frozen after class creation; bake bit_width into
    # the class body via type() so the Injector machinery sees it at __init__.
    WQ = type(f"WQ{bit_width}", (FixedPointPerTensorWeightQuant,), {"bit_width": bit_width})
    AQ = type(f"AQ{bit_width}", (FixedPointPerTensorActivationQuant,), {"bit_width": bit_width})
    BQ = type(f"BQ{bit_width}", (FixedPointPerTensorBiasQuant,), {"bit_width": bit_width})
    return WQ, AQ, BQ


def build_model(bit_width: int) -> nn.Module:
    WQ, AQ, BQ = make_quantizers(bit_width)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_quant = qnn.QuantIdentity(act_quant=AQ)
            self.conv1 = qnn.QuantConv2d(1, 16, 3, stride=2, bias=True,
                                         bias_quant=BQ, weight_quant=WQ, output_quant=AQ)
            self.relu1 = nn.ReLU()
            self.conv2 = qnn.QuantConv2d(16, 8, 3, stride=2,
                                         bias_quant=BQ, weight_quant=WQ, output_quant=AQ)
            self.relu2 = nn.ReLU()
            self.conv3 = qnn.QuantConv2d(4, 6, 3, stride=2,
                                         bias_quant=BQ, weight_quant=WQ, output_quant=AQ)
            self.conv4 = qnn.QuantConv2d(4, 6, 3, stride=2,
                                         bias_quant=BQ, weight_quant=WQ, output_quant=AQ)
            self.flatten = nn.Flatten()
            self.fc = qnn.QuantLinear(12 * 2 * 2, 10,
                                     weight_quant=WQ, output_quant=AQ)

        def forward(self, x):
            x = self.input_quant(x)
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            a, b = torch.split(x, 4, dim=1)
            x = torch.cat((self.conv3(a), self.conv4(b)), 1)
            x = self.flatten(x)
            x = self.fc(x)
            return x

    return Net()


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class HParam:
    name: str
    lr: float = 1e-3
    warmup_epochs: int = 5
    bit_width: int = 8
    batch_size: int = 256
    optimizer: str = "adam"          # 'adam' or 'sgd'
    grad_clip: Optional[float] = 1.0
    epochs: int = 15


# ---------------------------------------------------------------------------
# Trainer driver for one config
# ---------------------------------------------------------------------------
def run_one(hp: HParam, device: str, data_root: str = "./data") -> dict:
    set_seed(42, deterministic=False)
    QuantizerManager().reset()  # critical: fresh quantizer registry per run

    tx = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(data_root, train=True, download=True, transform=tx)
    val_ds = datasets.MNIST(data_root, train=False, transform=tx)
    train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=hp.batch_size, shuffle=False, num_workers=2)

    model = build_model(hp.bit_width).to(device)
    if hp.optimizer == "adam":
        opt = optim.Adam(model.parameters(), lr=hp.lr)
    elif hp.optimizer == "sgd":
        opt = optim.SGD(model.parameters(), lr=hp.lr, momentum=0.9)
    else:
        raise ValueError(hp.optimizer)

    cfg = TrainerConfig(
        experiment_name=f"sweep_{hp.name}",
        output_dir=f"logs/_sweep/{hp.name}",
        epochs=hp.epochs,
        batch_size=hp.batch_size,
        learning_rate=hp.lr,
        grad_clip_norm=hp.grad_clip,
        device=device,
        quant_schedule=QuantScheduleConfig(
            float_warmup_epochs=hp.warmup_epochs,
            calibration_batches=10,
            track_scale_factors=False,
        ),
    )
    cfg.logging.save_plots = False
    cfg.logging.csv_log = True
    cfg.checkpoint.top_k = 0
    cfg.checkpoint.save_last = False

    trainer = Trainer(config=cfg, model=model, optimizer=opt,
                      train_loader=train_loader, val_loader=val_loader,
                      loss_fn=nn.CrossEntropyLoss())

    t0 = time.time()
    tracker = trainer.fit()
    elapsed = time.time() - t0

    # Pull per-epoch val_acc trajectory from tracker.history
    val_accs = [snap.metrics["val_acc"] for snap in tracker.history
                if "val_acc" in snap.metrics]
    return {
        "name": hp.name,
        "best_val_acc": max(val_accs) if val_accs else 0.0,
        "final_val_acc": val_accs[-1] if val_accs else 0.0,
        "val_acc_trajectory": [round(a, 4) for a in val_accs],
        "elapsed_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Sweep definition
# ---------------------------------------------------------------------------
def build_sweep() -> list[HParam]:
    return [
        # Baseline — same as the current default
        HParam(name="baseline_lr1e-3_w5_b8"),

        # Learning-rate ladder
        HParam(name="lr_1e-4", lr=1e-4),
        HParam(name="lr_3e-4", lr=3e-4),
        HParam(name="lr_3e-3", lr=3e-3),
        HParam(name="lr_1e-2", lr=1e-2),

        # Longer warmup (anneal slower)
        HParam(name="warmup_10", warmup_epochs=10, epochs=20),
        HParam(name="warmup_2", warmup_epochs=2),

        # Sanity check: more bits — does QAT work at all if grid is finer?
        HParam(name="bits_16", bit_width=16),
        HParam(name="bits_12", bit_width=12),

        # Larger LR + more bits combined
        HParam(name="bits_12_lr_3e-3", bit_width=12, lr=3e-3),

        # SGD with momentum
        HParam(name="sgd_lr_1e-2", optimizer="sgd", lr=1e-2),

        # Smaller batch (more updates / epoch)
        HParam(name="batch_64", batch_size=64),

        # Tighter grad clip + larger LR
        HParam(name="clip_0.5_lr_3e-3", grad_clip=0.5, lr=3e-3),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sweep] device={device}\n")
    sweep = build_sweep()
    results = []
    for i, hp in enumerate(sweep):
        print(f"[sweep] ({i+1}/{len(sweep)}) {hp.name}: "
              f"lr={hp.lr} warmup={hp.warmup_epochs} bw={hp.bit_width} "
              f"bs={hp.batch_size} opt={hp.optimizer} clip={hp.grad_clip} epochs={hp.epochs}",
              flush=True)
        try:
            res = run_one(hp, device=device)
        except Exception as e:
            res = {"name": hp.name, "error": str(e), "best_val_acc": 0.0,
                   "final_val_acc": 0.0, "val_acc_trajectory": [], "elapsed_s": 0.0}
        results.append(res)
        print(f"    → best={res['best_val_acc']:.4f}  final={res['final_val_acc']:.4f}  "
              f"({res['elapsed_s']:.1f}s)", flush=True)

    print("\n" + "=" * 90)
    print(f"{'Config':<28} {'Best val':>10} {'Final val':>10} {'Δ':>8} {'Trajectory (val_acc)':>30}")
    print("=" * 90)
    for r in sorted(results, key=lambda x: -x["best_val_acc"]):
        traj = r["val_acc_trajectory"]
        if len(traj) > 6:
            traj_str = " ".join(f"{a:.2f}" for a in traj[:3]) + " … " + " ".join(f"{a:.2f}" for a in traj[-3:])
        else:
            traj_str = " ".join(f"{a:.2f}" for a in traj)
        delta = r["final_val_acc"] - r["best_val_acc"]
        print(f"{r['name']:<28} {r['best_val_acc']:>10.4f} {r['final_val_acc']:>10.4f} {delta:>+8.4f}  {traj_str}")
    print("=" * 90)

    out_path = "logs/_sweep/results.json"
    os.makedirs("logs/_sweep", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[sweep] results → {out_path}")


if __name__ == "__main__":
    main()
