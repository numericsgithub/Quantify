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

    preserve_calibrated_quantizers: bool = False
    """
    If True, quantizers that are already calibrated (search_done=True) when
    QAT activates — e.g. because the model was initialized from a PTQ
    checkpoint produced by examples/find_perfect_lsbs_imagenet_ptq.py — keep
    their existing search_done/search_result_lsb and jump straight to
    annealing_alpha=1.0 instead of being reset to search_done=False and
    annealing_alpha=0.0 like a freshly-built, never-calibrated quantizer.
    Quantizers that are NOT yet calibrated still go through the normal
    reset + gradual annealing ramp. Default False preserves the original
    behavior (every quantizer re-calibrates fresh against converged weights).
    """


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

    # ---- Live monitoring API ------------------------------------------------
    api_port: Optional[int] = None
    """
    If set, expose a read-only HTTP monitoring API on this port while the
    run is in progress (see training_harness/api/). Use 0 to let the OS pick
    a free port. None (default) disables the API entirely.
    """

    api_host: str = "127.0.0.1"
    """Host interface for the monitoring API ("0.0.0.0" for remote access)."""

    # ---- Sub-configs -------------------------------------------------------
    qat: QATScheduleConfigV2 = field(default_factory=QATScheduleConfigV2)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ---- Early stopping ----------------------------------------------------
    early_stopping_patience: Optional[int] = None
    """Only active after QAT has started. Set to None to disable."""

    early_stopping_min_delta: float = 1e-4

    # ---- Reduce LR on plateau ----------------------------------------------
    reduce_lr_on_plateau: bool = False
    """Step a ReduceLROnPlateau scheduler each epoch using val_loss (or train_loss)."""

    reduce_lr_patience: int = 20
    """Epochs of no improvement before LR is reduced."""

    reduce_lr_factor: float = 0.5
    """Factor by which LR is multiplied when a plateau is detected."""

    reduce_lr_min_lr: float = 1e-7
    """Lower bound on the learning rate."""

    reduce_lr_threshold: float = 1e-4
    """Minimum change in monitored metric to count as improvement."""

    reduce_lr_metric: str = "val_loss"
    """Metric to monitor for ReduceLROnPlateau. Use 'val_acc' (mode=max) in QAT
    since val_loss often diverges from accuracy once quantization noise is active."""

    # ---- MixUp / CutMix / Random Erasing ---------------------------------
    mixup: float = 0.0
    """Beta distribution alpha for MixUp. Set > 0 to enable (typical: 0.1)."""

    cutmix: float = 0.0
    """Beta distribution alpha for CutMix. Set > 0 to enable (typical: 1.0)."""

    mixup_prob: float = 1.0
    """Probability of applying mixup or cutmix per batch."""

    mixup_switch_prob: float = 0.5
    """Probability of switching to CutMix when both mixup and cutmix are enabled."""

    smoothing: float = 0.1
    """Label smoothing. Applied via timm Mixup when mixup/cutmix are enabled;
    via LabelSmoothingCrossEntropy otherwise. Set 0 to disable."""

    reprob: float = 0.0
    """Random Erasing probability. Set > 0 to enable (typical: 0.25)."""

    num_classes: int = 1000
    """Number of output classes, used by timm Mixup to build soft-target vectors."""

    # ---- Breakdown detection and recovery ---------------------------------
    breakdown_detection: bool = False
    """Detect catastrophic val_acc drops and recover from the best checkpoint."""

    breakdown_relative_drop: float = 0.7
    """Trigger recovery when val_acc falls below (1 - 0.7) = 30 % of its peak.
    A 70 % relative drop (e.g. 0.67 → 0.20) counts as a breakdown."""

    breakdown_peak_min_factor: float = 10.0
    """Detection is armed only after peak val_acc exceeds this multiple of the
    random-chance accuracy (1/num_classes).  Prevents false triggers at epoch 0."""

    breakdown_max_recoveries: int = 3
    """Maximum number of recovery attempts before giving up."""

    breakdown_lr_factor: float = 0.1
    """LR is multiplied by this factor on each recovery (default: ÷10)."""

    # ---- EMA --------------------------------------------------------------
    ema_decay: float = 0.0
    """EMA decay for shadow model parameters. Set > 0 to enable (typical: 0.9999).
    Validation uses EMA weights via a temporary parameter swap; the EMA state is
    bundled into every checkpoint's 'extra' key for post-training recovery."""

    # ---- BN freezing -------------------------------------------------------
    freeze_bn: bool = False
    """Freeze BatchNorm running stats and affine params throughout training.
    Use when fine-tuning from a converged checkpoint to prevent BN stat drift
    from heavy augmentation causing val_acc regression at low LR.
    Applied at the start of every training epoch (after model.train()), so it
    overrides the recursive model.train() call that would otherwise unfreeze BN."""

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

    @property
    def diagnostics_dir(self) -> str:
        return os.path.join(self.output_dir, "quantizer_diagnostics")

    def make_run_dirs(self) -> None:
        for path in [self.checkpoint_dir, self.log_dir, self.plot_dir]:
            os.makedirs(path, exist_ok=True)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)
