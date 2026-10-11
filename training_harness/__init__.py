"""
training_harness — QAT training harness for Brevitas / Quantify models.

QATTrainerV2 is the trainer: float warmup -> loss plateau -> staggered,
annealed activation of the custom quantizers (see trainer_v2.py).
"""

from .trainer_v2 import QATTrainerV2
from .config_v2 import TrainerConfigV2, QATScheduleConfigV2, CheckpointConfig, LoggingConfig
from .checkpointing import CheckpointManager
from .metrics import MetricsTracker
from .plotting import TrainingPlotter
from .schedulers import WarmupCosineScheduler
from .logger import ExperimentLogger
from .engine_utils import set_seed, get_hardware_info, EarlyStopping
from .lr_finder import find_lr, LRFindResult

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
