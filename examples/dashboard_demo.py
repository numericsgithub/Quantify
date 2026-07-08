"""
dashboard_demo.py — Small, CPU-friendly QAT run with the live monitoring
API enabled, for exercising the training dashboard.

This is NOT a real training recipe. It trains a tiny CNN on a subset of
MNIST for a handful of epochs so the whole thing finishes in ~1-2 minutes
on a laptop without a GPU. Its only purpose is to make the read-only
monitoring API live so you can watch the dashboard update.

Usage (two terminals):

    # Terminal 1 — start the backend (API goes live on port 8765)
    python examples/dashboard_demo.py

    # Terminal 2 — start the UI, pointed at the API
    python dashboard/serve.py --api http://127.0.0.1:8765

Then open http://127.0.0.1:8080/ in a browser. You'll see the float
warmup -> QAT phase transition, live train loss, validation accuracy,
and the top-K checkpoints. When the run finishes, /status flips to
"finished"; the script then lingers a short while (so you can look at the
final state) before exiting, at which point the UI shows "disconnected".

Note: MNIST here only exercises the dashboard plumbing — it says nothing
about real training quality, so don't read anything into the accuracy.
"""

import os
import sys
import time

# Make the repo root importable so `examples.*` resolves regardless of how
# this script is launched (`python examples/dashboard_demo.py` puts only the
# examples/ folder on sys.path, which hides the top-level `examples` package).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from examples.simple_mnist_qat import SimpleMNISTNet
from training_harness.trainer import Trainer
from training_harness.config import TrainerConfig, QuantScheduleConfig, LoggingConfig
from training_harness.engine_utils import set_seed

# ── Knobs (kept small on purpose) ──────────────────────────────────────
SEED = 42
BATCH_SIZE = 256
EPOCHS = 12
LR = 1e-3
TRAIN_SUBSET = 4000        # of 60,000 — keeps CPU epochs to a few seconds
VAL_SUBSET = 1000          # of 10,000
API_PORT = 8765
LINGER_SECONDS = 90        # keep API up after training so you can look around


def main() -> None:
    set_seed(SEED, deterministic=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = Subset(
        datasets.MNIST("./data", train=True, download=True, transform=transform),
        range(TRAIN_SUBSET),
    )
    val_ds = Subset(
        datasets.MNIST("./data", train=False, download=True, transform=transform),
        range(VAL_SUBSET),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = SimpleMNISTNet()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    config = TrainerConfig(
        experiment_name="dashboard_demo",
        output_dir="logs/dashboard_demo",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        api_port=API_PORT,        # ← this is what makes the API live
        logging=LoggingConfig(log_every_n_steps=1, save_plots=False),
        quant_schedule=QuantScheduleConfig(
            float_warmup_epochs=3,   # phase switch happens early so you see it
            calibration_batches=10,
            track_scale_factors=True,
        ),
    )

    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=nn.CrossEntropyLoss(),
    )

    print(f"\nStart the UI in another terminal:")
    print(f"    python dashboard/serve.py --api http://127.0.0.1:{API_PORT}")
    print(f"then open http://127.0.0.1:8080/\n")

    trainer.fit()

    print(f"\n[demo] Training done. Keeping the API live for {LINGER_SECONDS}s "
          f"so you can inspect the final state (Ctrl+C to quit now).")
    print("[demo] /status now reports \"finished\"; when this process exits the "
          "UI will show \"disconnected\".")
    try:
        time.sleep(LINGER_SECONDS)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
