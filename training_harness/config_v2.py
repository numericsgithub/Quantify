"""
config_v2.py — Configuration dataclasses for the V2 QAT training harness.

V2 implements the correct protocol for the project's custom quantizers:
  disable → float warmup → reset calibration → gradual cascade (gating + annealing)

Reuses CheckpointConfig and LoggingConfig from config.py unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from .config import CheckpointConfig, LoggingConfig


@dataclass
class QATScheduleConfigV2:
    """
    Controls the float warmup period and the gradual QAT transition.

    The transition fires when val_loss plateaus for `plateau_patience` epochs.
    At that point:
      1. Calibration buffers are reset (quantizers re-calibrate with converged weights).
      2. Annealing is set: each quantizer ramps from 0 → 1 over `annealing_steps` passes.
      3. Gating staggers activation: quantizer N waits N × `quantization_start_gap` passes.
      4. BN statistics are frozen.
    """

    float_warmup_epochs: int = 10
    """
    Epochs to train in full float before enabling quantization.
    Set to 0 to skip float warmup (e.g. when starting from a pre-trained model).
    """

    plateau_metric: str = "val_loss"
    """
    Metric watched by the plateau detector to decide when to start QAT.
    Must be a DECREASING metric (loss, not accuracy). The detector is
    designed for values that should go down; using an accuracy metric
    here will cause it to fire immediately every patience epochs.
    Falls back to 'train_loss' if the chosen metric is not available.
    """

    plateau_patience: int = 5
    """Epochs of no improvement in plateau_metric before QAT starts."""

    plateau_min_delta: float = 1e-4
    """Minimum decrease in plateau_metric that counts as improvement."""

    annealing_steps: int = 100
    """
    Number of forward passes over which each quantizer ramps annealing_alpha
    from 0.0 (pure float) to 1.0 (fully quantized).
    """

    quantization_start_gap: int = 20
    """
    Forward passes between each successive quantizer activating.
    Quantizer N waits N × gap passes before its gating lifts.
    Gives the model time to adapt to each new source of quantization error.
    """

    freeze_bn_at_qat: bool = True
    """Freeze BatchNorm running statistics when QAT begins."""

    track_scale_factors: bool = True
    """Log per-layer quantization scale factors at the end of each QAT epoch."""


@dataclass
class TrainerConfigV2:
    """
    Master configuration for a V2 QAT training run.

    Usage::

        config = TrainerConfigV2(
            experiment_name="cifar10_fixedpoint",
            epochs=60,
            learning_rate=1e-3,
            qat=QATScheduleConfigV2(
                float_warmup_epochs=10,
                plateau_patience=5,
                annealing_steps=100,
                quantization_start_gap=20,
            ),
        )
    """

    # ---- Identity ----------------------------------------------------------
    experiment_name: str = "experiment_v2"
    run_id: Optional[str] = None
    output_dir: str = "output"

    # ---- Training loop -----------------------------------------------------
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: Optional[float] = 1.0

    # ---- Hardware ----------------------------------------------------------
    device: str = "auto"
    mixed_precision: bool = False
    num_workers: int = 4

    # ---- Reproducibility ---------------------------------------------------
    seed: int = 42
    deterministic: bool = False

    # ---- Dry-run -----------------------------------------------------------
    dry_run: bool = False
    dry_run_batches: int = 2

    # ---- Sub-configs -------------------------------------------------------
    qat: QATScheduleConfigV2 = field(default_factory=QATScheduleConfigV2)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ---- Early stopping ----------------------------------------------------
    early_stopping_patience: Optional[int] = None
    """Only active after QAT has started. Set to None to disable."""

    early_stopping_min_delta: float = 1e-4

    # ---- Helpers -----------------------------------------------------------
    def resolve_device(self) -> str:
        import torch
        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.device

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.output_dir, self.checkpoint.save_dir)

    @property
    def log_dir(self) -> str:
        return os.path.join(self.output_dir, self.logging.log_dir)

    @property
    def plot_dir(self) -> str:
        return os.path.join(self.output_dir, self.logging.plot_dir)

    def make_run_dirs(self) -> None:
        for path in [self.checkpoint_dir, self.log_dir, self.plot_dir]:
            os.makedirs(path, exist_ok=True)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)
