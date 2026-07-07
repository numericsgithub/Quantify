"""
training_harness.api — Read-only HTTP monitoring API for live training runs.

Opt-in via ``TrainerConfig.api_port`` / ``TrainerConfigV2.api_port``.
The server runs in a daemon thread inside the training process and only
*reads* trainer state; it never mutates anything (write/control endpoints
are deliberately out of scope for now, but the /api/v1/ prefix leaves room
to add them later).
"""

from .collector import RunStateCollector, TRAIN_ACC_CAVEAT
from .server import DashboardAPIServer, create_app

__all__ = [
    "RunStateCollector",
    "DashboardAPIServer",
    "create_app",
    "TRAIN_ACC_CAVEAT",
]
