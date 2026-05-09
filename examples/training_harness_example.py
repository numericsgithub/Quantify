"""
example.py — Full usage example for brevitas_trainer.

Shows how to wire a Brevitas QAT model into the training harness
with all features enabled: checkpointing, logging, plotting,
QAT schedule, calibration, and early stopping.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Try importing Brevitas; fall back to a vanilla model for the demo ──
try:
    import brevitas.nn as qnn
    BREVITAS_AVAILABLE = True
except ImportError:
    BREVITAS_AVAILABLE = False

from training import (
    Trainer,
    TrainerConfig,
    CheckpointConfig,
    LoggingConfig,
    QuantScheduleConfig,
)


# ---------------------------------------------------------------------------
# 1. Build a model
# ---------------------------------------------------------------------------

def make_model(use_brevitas: bool = True) -> nn.Module:
    if use_brevitas and BREVITAS_AVAILABLE:
        return nn.Sequential(
            qnn.QuantConv2d(1, 32, kernel_size=3, padding=1, weight_bit_width=8),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            qnn.QuantConv2d(32, 64, kernel_size=3, padding=1, weight_bit_width=8),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            qnn.QuantLinear(64, 10, bias=True, weight_bit_width=8),
        )
    # Vanilla PyTorch fallback for running the demo without Brevitas
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(64, 10),
    )


# ---------------------------------------------------------------------------
# 2. Fake dataset (replace with your real DataLoader)
# ---------------------------------------------------------------------------

def make_loaders(n_train=512, n_val=128, batch_size=32):
    x = torch.randn(n_train, 1, 28, 28)
    y = torch.randint(0, 10, (n_train,))
    train_ds = TensorDataset(x, y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    xv = torch.randn(n_val, 1, 28, 28)
    yv = torch.randint(0, 10, (n_val,))
    val_ds = TensorDataset(xv, yv)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# 3. Config
# ---------------------------------------------------------------------------

config = TrainerConfig(
    experiment_name = "my_qat_experiment",

    # Training
    epochs         = 20,
    learning_rate  = 1e-3,
    weight_decay   = 1e-4,
    grad_clip_norm = 1.0,

    # Hardware
    device         = "auto",
    mixed_precision = True,

    # Reproducibility
    seed           = 42,

    # Dry-run: flip to True to smoke-test in 2 batches
    dry_run        = False,

    # Root output directory — all checkpoints, logs, and plots go here
    output_dir = "output/my_qat_experiment",

    # Checkpointing
    checkpoint = CheckpointConfig(
        top_k          = 3,
        monitor_metric = "val_loss",
        monitor_mode   = "min",
        save_last      = True,
    ),

    # Logging
    logging = LoggingConfig(
        csv_log         = True,
        use_tensorboard = False,   # flip to True if tensorboard is installed
        use_wandb       = False,   # flip to True + set wandb_project if desired
        save_plots      = True,
    ),

    # QAT schedule
    quant_schedule = QuantScheduleConfig(
        float_warmup_epochs   = 5,     # 5 epochs of float training first
        calibration_batches   = 50,    # then run 50-batch calibration
        freeze_bn_after_epoch = 15,    # freeze BN stats from epoch 15
        track_scale_factors   = True,  # record scale factor evolution
    ),

    # Early stopping
    early_stopping_patience  = 5,
    early_stopping_min_delta = 1e-4,
)


# ---------------------------------------------------------------------------
# 4. Optional: save config to YAML for reproducibility
# ---------------------------------------------------------------------------
# config.to_yaml("my_qat_experiment.yaml")
# config = TrainerConfig.from_yaml("my_qat_experiment.yaml")


# ---------------------------------------------------------------------------
# 5. Optional hooks for custom per-step / per-epoch logic
# ---------------------------------------------------------------------------

def after_step(trainer: Trainer, loss: float, outputs, targets):
    """Called after every optimizer step. Add custom logic here."""
    pass  # e.g. log to a custom dashboard, update a progress bar, etc.


def after_epoch(trainer: Trainer, epoch: int, snap):
    """Called after every epoch. snap is an EpochMetrics object."""
    if snap and epoch % 5 == 0:
        print(f"  [hook] epoch {epoch} summary: {snap}")


# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model        = make_model(use_brevitas=BREVITAS_AVAILABLE)
    train_loader, val_loader = make_loaders()
    optimizer    = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Optional: LR scheduler (step-level cosine warmup)
    from training import WarmupCosineScheduler
    total_steps = config.epochs * len(train_loader)
    scheduler   = WarmupCosineScheduler(
        optimizer,
        warmup_steps = total_steps // 10,
        total_steps  = total_steps,
        eta_min      = 1e-6,
    )

    trainer = Trainer(
        config       = config,
        model        = model,
        optimizer    = optimizer,
        train_loader = train_loader,
        val_loader   = val_loader,
        loss_fn      = nn.CrossEntropyLoss(),
        scheduler    = scheduler,
    )

    # Resume from checkpoint if available:
    # tracker = trainer.fit(resume=True)
    tracker = trainer.fit(
        after_step_hook  = after_step,
        after_epoch_hook = after_epoch,
    )

    print("\nFinal summary:")
    for k, v in tracker.summary().items():
        print(f"  {k}: {v}")