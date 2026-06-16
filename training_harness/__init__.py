"""
brevitas_trainer — A training_harness harness for Brevitas QAT experiments.
"""

from .trainer import Trainer
from .config import TrainerConfig, QuantScheduleConfig, CheckpointConfig, LoggingConfig
from .checkpointing import CheckpointManager
from .metrics import MetricsTracker
from .plotting import TrainingPlotter
from .schedulers import QATWarmupScheduler, WarmupCosineScheduler
from .calibration import run_calibration, enable_quant, disable_quant
from .logger import ExperimentLogger
from .engine_utils import set_seed, get_hardware_info, EarlyStopping

# V2 — corrected protocol for the project's custom quantizers
from .trainer_v2 import QATTrainerV2
from .config_v2 import TrainerConfigV2, QATScheduleConfigV2

__all__ = [
    # V1
    "Trainer",
    "TrainerConfig",
    "QuantScheduleConfig",
    "CheckpointConfig",
    "LoggingConfig",
    "CheckpointManager",
    "MetricsTracker",
    "TrainingPlotter",
    "QATWarmupScheduler",
    "WarmupCosineScheduler",
    "run_calibration",
    "enable_quant",
    "disable_quant",
    "ExperimentLogger",
    "set_seed",
    "get_hardware_info",
    "EarlyStopping",
    # V2
    "QATTrainerV2",
    "TrainerConfigV2",
    "QATScheduleConfigV2",
]
