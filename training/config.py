"""
config.py — All configuration dataclasses for the training harness.

A single TrainerConfig object fully describes a training run, making
every run reproducible and self-documenting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Checkpoint sub-config
# ---------------------------------------------------------------------------

@dataclass
class CheckpointConfig:
    """Controls how and when checkpoints are saved."""

    save_dir: str = "checkpoints"
    """
    Subdirectory name for checkpoint files.
    Resolved to <output_dir>/checkpoints by TrainerConfig.
    """

    top_k: int = 3
    """Keep only the K best checkpoints by the watched metric."""

    monitor_metric: str = "val_loss"
    """Metric used to rank checkpoints ('val_loss', 'val_acc', …)."""

    monitor_mode: str = "min"
    """'min' for loss-style metrics, 'max' for accuracy-style metrics."""

    save_last: bool = True
    """Always keep a 'last.pt' checkpoint, regardless of top-k."""

    save_every_n_epochs: Optional[int] = None
    """If set, also save a checkpoint every N epochs unconditionally."""


# ---------------------------------------------------------------------------
# Logging sub-config
# ---------------------------------------------------------------------------

@dataclass
class LoggingConfig:
    """Controls experiment logging behaviour."""

    log_dir: str = "logs"
    """
    Subdirectory name for log files (CSV, TensorBoard events, etc.).
    Resolved to <output_dir>/logs by TrainerConfig.
    """

    use_tensorboard: bool = False
    """Write TensorBoard event files (requires tensorboard package)."""

    use_wandb: bool = False
    """Log to Weights & Biases (requires wandb package)."""

    wandb_project: Optional[str] = None
    """W&B project name (required when use_wandb=True)."""

    wandb_entity: Optional[str] = None
    """W&B entity / user (optional)."""

    csv_log: bool = True
    """Write a simple CSV log file — always available, zero dependencies."""

    log_every_n_steps: int = 10
    """How often (in optimizer steps) to log training loss."""

    plot_dir: str = "plots"
    """
    Subdirectory name for saved plot PNGs.
    Resolved to <output_dir>/plots by TrainerConfig.
    """

    save_plots: bool = True
    """Save plots to disk after training (and optionally at checkpoints)."""


# ---------------------------------------------------------------------------
# Quantization schedule sub-config
# ---------------------------------------------------------------------------

@dataclass
class QuantScheduleConfig:
    """
    Controls when quantization is enabled during training.

    Typical workflow:
      1. Train in full-precision for `float_warmup_epochs` epochs.
      2. Run a calibration pass to set initial quantization ranges.
      3. Continue training with fake-quantization enabled.
    """

    float_warmup_epochs: int = 5
    """Number of epochs to train in full float before enabling fake-quant."""

    calibration_batches: int = 100
    """Number of batches to use for post-warmup calibration."""

    freeze_bn_after_epoch: Optional[int] = None
    """
    If set, freeze BatchNorm statistics after this epoch.
    Useful in late-stage QAT to stabilise ranges.
    """

    track_scale_factors: bool = True
    """
    Record per-layer quantization scale factors at the end of every epoch
    so you can plot how they evolve during training.
    """


# ---------------------------------------------------------------------------
# Top-level trainer config
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """
    Master configuration for a training run.

    Pass a fully-constructed TrainerConfig to Trainer(config=...).
    All fields have sensible defaults so you only need to override what
    differs from a standard QAT experiment.
    """

    # ---- Identity ----------------------------------------------------------
    experiment_name: str = "experiment"
    """Human-readable name, used in filenames and log headers."""

    run_id: Optional[str] = None
    """
    Unique identifier for this specific run.
    Auto-generated from a timestamp if not provided.
    """

    output_dir: str = "output"
    """
    Root directory for ALL output produced by this run.
    Checkpoints, logs, and plots are saved as subdirectories here:

        <output_dir>/
            checkpoints/   <- controlled by checkpoint.save_dir
            logs/          <- controlled by logging.log_dir
            plots/         <- controlled by logging.plot_dir

    Set this to an absolute path to save anywhere on disk, e.g.:
        output_dir = "/data/experiments/my_project"
    """

    # ---- Training loop -----------------------------------------------------
    epochs: int = 50
    """Total number of training epochs."""

    batch_size: int = 64
    """Batch size (informational; the DataLoader is provided externally)."""

    learning_rate: float = 1e-3
    """Initial learning rate for the optimizer."""

    weight_decay: float = 1e-4
    """L2 regularisation weight."""

    grad_clip_norm: Optional[float] = 1.0
    """
    Max gradient norm for clipping.  Set to None to disable.
    Gradient clipping is especially important in QAT to prevent
    scale-factor explosions in early training.
    """

    # ---- Hardware ----------------------------------------------------------
    device: str = "auto"
    """
    'auto'  → use CUDA if available, else CPU.
    'cuda'  → force CUDA (raises if unavailable).
    'cpu'   → force CPU.
    'mps'   → Apple Silicon GPU.
    """

    mixed_precision: bool = False
    """
    Enable torch.autocast + GradScaler for AMP training.
    Automatically disabled on CPU.
    
    Note: Defaults to False for Brevitas QAT compatibility. 
    Mixed precision can interfere with fake-quantization ops and scale learning.
    """

    num_workers: int = 4
    """Number of DataLoader worker processes (informational)."""

    # ---- Reproducibility ---------------------------------------------------
    seed: int = 42
    """Master seed for torch / numpy / random."""

    deterministic: bool = False
    """
    Set torch.backends.cudnn.deterministic = True.
    Slower but fully reproducible on GPU.
    """

    # ---- Dry-run / smoke-test ----------------------------------------------
    dry_run: bool = False
    """
    If True, run only 2 batches per phase (train + val) to verify the
    pipeline end-to-end without a full training loop.
    """

    dry_run_batches: int = 2
    """Number of batches to run when dry_run=True."""

    # ---- Sub-configs -------------------------------------------------------
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    quant_schedule: QuantScheduleConfig = field(default_factory=QuantScheduleConfig)

    # ---- Early stopping ----------------------------------------------------
    early_stopping_patience: Optional[int] = None
    """
    Stop training if the monitored metric does not improve for this many
    epochs. Uses the same metric as CheckpointConfig.monitor_metric.
    Set to None to disable early stopping.
    """

    early_stopping_min_delta: float = 1e-4
    """Minimum change that counts as an improvement for early stopping."""

    # ---- Convenience -------------------------------------------------------
    def resolve_device(self) -> str:
        """Return the actual torch device string after resolving 'auto'."""
        import torch
        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.device

    # ---- Resolved output paths --------------------------------------------

    @property
    def checkpoint_dir(self) -> str:
        """Full path to the checkpoint directory."""
        return os.path.join(self.output_dir, self.checkpoint.save_dir)

    @property
    def log_dir(self) -> str:
        """Full path to the log directory (CSV, TensorBoard, etc.)."""
        return os.path.join(self.output_dir, self.logging.log_dir)

    @property
    def plot_dir(self) -> str:
        """Full path to the plots directory."""
        return os.path.join(self.output_dir, self.logging.plot_dir)

    def make_run_dirs(self) -> None:
        """Create all output directories for this run."""
        for path in [self.checkpoint_dir, self.log_dir, self.plot_dir]:
            os.makedirs(path, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainerConfig":
        """Load a TrainerConfig from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for from_yaml(). Install with: pip install pyyaml")

        with open(path) as f:
            data = yaml.safe_load(f)

        # Unpack nested sub-configs
        cfg = cls()
        for key, value in data.items():
            if key == "checkpoint" and isinstance(value, dict):
                cfg.checkpoint = CheckpointConfig(**value)
            elif key == "logging" and isinstance(value, dict):
                cfg.logging = LoggingConfig(**value)
            elif key == "quant_schedule" and isinstance(value, dict):
                cfg.quant_schedule = QuantScheduleConfig(**value)
            elif hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    def to_yaml(self, path: str) -> None:
        """Save this config to a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for to_yaml(). Install with: pip install pyyaml")

        import dataclasses
        data = dataclasses.asdict(self)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def to_dict(self) -> dict:
        """Return a plain dict representation (useful for W&B / logging)."""
        import dataclasses
        return dataclasses.asdict(self)
