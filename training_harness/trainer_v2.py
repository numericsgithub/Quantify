"""Deprecated import path: use ``training_harness.trainer`` (``QATTrainer``)."""
import warnings

from .trainer import QATTrainer

QATTrainerV2 = QATTrainer

__all__ = ["QATTrainer", "QATTrainerV2"]

warnings.warn(
    "training_harness.trainer_v2 is deprecated; import from training_harness.trainer "
    "(QATTrainerV2 is now QATTrainer).",
    DeprecationWarning,
    stacklevel=2,
)
