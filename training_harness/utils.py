"""
utils.py — Utility functions for the training_harness harness.

Covers: reproducibility seeding, hardware diagnostics, early stopping,
progress display, and ETA estimation.
"""

from __future__ import annotations

import random
import time
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set all relevant random seeds for a reproducible run.

    Args:
        seed:          Master seed value.
        deterministic: If True, enable CUDA deterministic algorithms.
                       Slower, but fully reproducible on GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch ≥ 1.8
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass  # Older PyTorch — best effort
    else:
        # benchmark=True can speed up training_harness when input sizes are fixed
        torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Hardware diagnostics
# ---------------------------------------------------------------------------

def get_hardware_info() -> dict:
    """
    Return a dictionary describing the current compute environment.

    Useful for logging at the start of each run so you can later
    reconstruct which machine produced which result.
    """
    info: dict = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": getattr(torch.backends, "mps", None) is not None
                         and torch.backends.mps.is_available(),
        "num_cpus": torch.get_num_threads(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_names"] = [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]
        # Memory in GiB for the default device
        props = torch.cuda.get_device_properties(0)
        info["gpu_memory_gib"] = round(props.total_memory / (1024 ** 3), 2)

    return info


def log_hardware_info(logger=None) -> None:
    """Print (and optionally log) hardware info at run start."""
    info = get_hardware_info()
    lines = ["── Hardware ──────────────────────────────"]
    for k, v in info.items():
        lines.append(f"  {k:<24} {v}")
    lines.append("──────────────────────────────────────────")
    text = "\n".join(lines)
    print(text)
    if logger is not None:
        logger.log_text("hardware_info", text)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Halts training_harness when a monitored metric stops improving.

    Usage::

        stopper = EarlyStopping(patience=10, mode="min")
        for epoch in range(epochs):
            val_loss = validate(...)
            if stopper.step(val_loss):
                print("Early stop!")
                break
        stopper.restore_best_weights(model)
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
        restore_best_weights: bool = True,
    ):
        """
        Args:
            patience:              Number of epochs with no improvement before stopping.
            min_delta:             Minimum change in the metric that counts as improvement.
            mode:                  'min' for loss, 'max' for accuracy.
            restore_best_weights:  If True, save a copy of the best model state dict
                                   and restore it when training_harness stops.
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self._best: float = float("inf") if mode == "min" else float("-inf")
        self._counter: int = 0
        self._best_weights: Optional[dict] = None
        self.stopped_epoch: Optional[int] = None

    # ------------------------------------------------------------------
    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < self._best - self.min_delta
        return value > self._best + self.min_delta

    def step(self, value: float, model: Optional[torch.nn.Module] = None, epoch: int = 0) -> bool:
        """
        Call at the end of each epoch.

        Args:
            value:  Current value of the monitored metric.
            model:  Pass the model here when restore_best_weights=True.
            epoch:  Current epoch number (for bookkeeping).

        Returns:
            True if training_harness should stop, False otherwise.
        """
        if self._is_improvement(value):
            self._best = value
            self._counter = 0
            if self.restore_best_weights and model is not None:
                # Deep-copy the state dict so it's not mutated later
                import copy
                self._best_weights = copy.deepcopy(model.state_dict())
        else:
            self._counter += 1

        if self._counter >= self.patience:
            self.stopped_epoch = epoch
            return True  # Stop!
        return False

    def restore(self, model: torch.nn.Module) -> None:
        """Restore the best weights saved during training_harness."""
        if self._best_weights is not None:
            model.load_state_dict(self._best_weights)
        else:
            import warnings
            warnings.warn(
                "EarlyStopping.restore() called but no best weights were saved. "
                "Pass model= to step() and set restore_best_weights=True."
            )

    @property
    def best(self) -> float:
        return self._best

    @property
    def counter(self) -> int:
        return self._counter


# ---------------------------------------------------------------------------
# Progress / ETA
# ---------------------------------------------------------------------------

class EpochTimer:
    """
    Lightweight per-epoch timer that estimates time-to-completion.

    Usage::

        timer = EpochTimer(total_epochs=50)
        for epoch in range(50):
            timer.start()
            train(...)
            elapsed, eta = timer.stop(epoch)
            print(f"Epoch {epoch}  {elapsed:.1f}s  ETA {eta}")
    """

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self._start: Optional[float] = None
        self._elapsed_history: list[float] = []

    def start(self) -> None:
        self._start = time.time()

    def stop(self, current_epoch: int) -> tuple[float, str]:
        """
        Returns:
            elapsed:  Seconds this epoch took.
            eta_str:  Human-readable ETA string like '3m 42s'.
        """
        assert self._start is not None, "Call start() before stop()"
        elapsed = time.time() - self._start
        self._elapsed_history.append(elapsed)

        remaining = self.total_epochs - (current_epoch + 1)
        avg = sum(self._elapsed_history) / len(self._elapsed_history)
        eta_secs = avg * remaining

        return elapsed, _format_seconds(eta_secs)


def _format_seconds(secs: float) -> str:
    """Convert a duration in seconds to a human-readable string."""
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"
