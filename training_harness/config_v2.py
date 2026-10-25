"""Deprecated import path: use ``training_harness.config``."""
import warnings

from .config import CheckpointConfig, LoggingConfig, QATScheduleConfig, TrainerConfig

TrainerConfigV2 = TrainerConfig
QATScheduleConfigV2 = QATScheduleConfig

__all__ = [
    "CheckpointConfig",
    "LoggingConfig",
    "QATScheduleConfig",
    "QATScheduleConfigV2",
    "TrainerConfig",
    "TrainerConfigV2",
]

warnings.warn(
    "training_harness.config_v2 is deprecated; import from training_harness.config "
    "(TrainerConfigV2 -> TrainerConfig, QATScheduleConfigV2 -> QATScheduleConfig).",
    DeprecationWarning,
    stacklevel=2,
)
