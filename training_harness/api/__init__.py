"""
training_harness.api — Read-only HTTP monitoring API for live training runs.

Opt-in via ``TrainerConfig.api_port`` / ``TrainerConfigV2.api_port``.
The server runs in a daemon thread inside the training process and only
*reads* trainer state; it never mutates anything (write/control endpoints
are deliberately out of scope for now, but the /api/v1/ prefix leaves room
to add them later).
"""

from .collector import RunStateCollector, TRAIN_ACC_CAVEAT
from .control import (
    CallbackRegistry,
    ControlManager,
    ControlValidationError,
)

# server.py pulls in Flask. Expose its symbols lazily (PEP 562) so that
# merely constructing a Trainer — which imports CallbackRegistry from this
# package to build its callback registry — does NOT import Flask unless the
# monitoring API is actually enabled (api_port set).
_LAZY = {"DashboardAPIServer", "create_app"}


def __getattr__(name):
    if name in _LAZY:
        from . import server
        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RunStateCollector",
    "DashboardAPIServer",
    "create_app",
    "TRAIN_ACC_CAVEAT",
    "CallbackRegistry",
    "ControlManager",
    "ControlValidationError",
]
