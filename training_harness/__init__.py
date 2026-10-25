"""
training_harness — QAT training harness for Brevitas / Quantify models.

QATTrainer is the trainer: float warmup -> loss plateau -> staggered,
annealed activation of the custom quantizers (see trainer.py).
"""

from .checkpointing import CheckpointManager
from .config import CheckpointConfig, LoggingConfig, QATScheduleConfig, TrainerConfig
from .engine_utils import EarlyStopping, get_hardware_info, set_seed
from .logger import ExperimentLogger
from .lr_finder import LRFindResult, find_lr
from .metrics import MetricsTracker
from .plotting import TrainingPlotter
from .schedulers import WarmupCosineScheduler
from .trainer import QATTrainer

__all__ = [
    "QATTrainer",
    "TrainerConfig",
    "QATScheduleConfig",
    "CheckpointConfig",
    "LoggingConfig",
    "CheckpointManager",
    "MetricsTracker",
    "TrainingPlotter",
    "WarmupCosineScheduler",
    "ExperimentLogger",
    "set_seed",
    "get_hardware_info",
    "EarlyStopping",
    "find_lr",
    "LRFindResult",
]

# Deprecated pre-rename names (the "V2" suffix was dropped): still importable
# from the package, with a DeprecationWarning.
_DEPRECATED_ALIASES = {
    "QATTrainerV2": "QATTrainer",
    "TrainerConfigV2": "TrainerConfig",
    "QATScheduleConfigV2": "QATScheduleConfig",
}


def __getattr__(name):
    if name in _DEPRECATED_ALIASES:
        import warnings

        new = _DEPRECATED_ALIASES[name]
        warnings.warn(f"training_harness.{name} is deprecated; use {new}.", DeprecationWarning, stacklevel=2)
        return globals()[new]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
