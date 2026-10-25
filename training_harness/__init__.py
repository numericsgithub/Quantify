"""
training_harness — QAT training harness for Brevitas / Quantify models.

QATTrainerV2 is the trainer: float warmup -> loss plateau -> staggered,
annealed activation of the custom quantizers (see trainer_v2.py).
"""

from .checkpointing import CheckpointManager
from .config_v2 import CheckpointConfig, LoggingConfig, QATScheduleConfigV2, TrainerConfigV2
from .engine_utils import EarlyStopping, get_hardware_info, set_seed
from .logger import ExperimentLogger
from .lr_finder import LRFindResult, find_lr
from .metrics import MetricsTracker
from .plotting import TrainingPlotter
from .schedulers import WarmupCosineScheduler
from .trainer_v2 import QATTrainerV2

__all__ = [
    "QATTrainerV2",
    "TrainerConfigV2",
    "QATScheduleConfigV2",
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
