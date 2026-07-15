"""
dashboard_demo.py — Small, CPU-friendly QAT run on the **V2** harness with the
live monitoring API enabled, for exercising the training dashboard.

This is NOT a real training recipe. It trains a tiny fixed-point CNN on a
subset of MNIST for a handful of epochs so the whole thing finishes in ~1-2
minutes on a laptop without a GPU. Its only purpose is to make the monitoring
API live so you can watch the dashboard update — and, now, poke at the
quantizer role histogram.

Why V2: QATTrainerV2 (training_harness/trainer_v2.py) is the migration target
for the live-control work. It runs the correct float-warmup → gradual QAT
cascade for the project's custom quantizers, so the dashboard shows the
quantizers activating and annealing one by one.

Usage (two terminals):

    # Terminal 1 — start the backend (API goes live on port 8765)
    python examples/dashboard_demo.py

    # Terminal 2 — start the UI, pointed at the API
    python dashboard/serve.py --api http://127.0.0.1:8765

Then open http://127.0.0.1:8080/ in a browser. You'll see the float
warmup -> QAT phase transition, live train loss, validation accuracy, the
top-K checkpoints, and the quantizer cascade (fully-quantized count climbing).
When the run finishes, /status flips to "finished"; the script then lingers a
short while before exiting, at which point the UI shows "disconnected".

Handy endpoint to try while it runs:
    curl http://127.0.0.1:8765/api/v1/quantizers/roles
    -> {"pid":..., "histogram":{"weight":3,"bias":2,"activation":4,"unknown":0,"total":9}}

NOTE (control endpoints): on V2 the API is currently **read-only** — the
control layer (LR changes, group anneal/disable, etc.) is still bound to V1 and
has not been migrated to V2 yet (that migration is the next phase). Write
endpoints will return 503 until then. This demo is for the read/monitoring
side and the role histogram.

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

from examples.mnist_qat_v2 import MNISTQuantNet
from training_harness.trainer_v2 import QATTrainerV2
from training_harness.config_v2 import TrainerConfigV2, QATScheduleConfigV2
from training_harness.config import CheckpointConfig, LoggingConfig
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

    model = MNISTQuantNet()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    config = TrainerConfigV2(
        experiment_name="dashboard_demo_v2",
        output_dir="output/dashboard_demo_v2",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        num_classes=10,
        smoothing=0.0,            # avoid the timm LabelSmoothing import (keep deps light)
        # A live scheduler so you can see a manual LR change suspend it
        # (/status shows scheduler_suspended=true after POST /control/hyperparams).
        reduce_lr_on_plateau=True,
        reduce_lr_patience=3,
        reduce_lr_metric="val_loss",
        api_port=API_PORT,        # ← this is what makes the API live
        logging=LoggingConfig(log_every_n_steps=1, save_plots=False),
        qat=QATScheduleConfigV2(
            # Short warmup so the float→QAT switch happens early and is visible.
            # QAT starts when val_loss plateaus, or at float_warmup_epochs latest.
            float_warmup_epochs=3,
            plateau_metric="val_loss",
            plateau_patience=3,
            plateau_min_delta=1e-3,
            # Small so the cascade (9 quantizers activating + annealing 0→1)
            # completes within the epoch budget and you can watch it progress.
            annealing_steps=20,
            quantization_start_gap=10,
            freeze_bn_at_qat=True,
            track_scale_factors=True,
        ),
        checkpoint=CheckpointConfig(
            monitor_metric="val_acc",
            monitor_mode="max",
            top_k=3,
            save_last=True,
        ),
        # Keep a second best-checkpoint pool by train_loss so the dashboard's
        # reload-best can target either criterion (best_val_acc | best_train_loss).
        secondary_checkpoint_metrics=[("train_loss", "min")],
    )

    trainer = QATTrainerV2(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=nn.CrossEntropyLoss(),
        onnx_dummy_input=torch.zeros(1, 1, 28, 28),
    )

    print(f"\nStart the UI in another terminal:")
    print(f"    python dashboard/serve.py --api http://127.0.0.1:{API_PORT}")
    print(f"then open http://127.0.0.1:8080/")
    print(f"Role histogram:  curl http://127.0.0.1:{API_PORT}/api/v1/quantizers/roles\n")

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
