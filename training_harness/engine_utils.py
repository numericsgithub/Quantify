"""
engine_utils.py — Utility functions for the training_harness harness.

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
# Loss Plateau Detection
# ---------------------------------------------------------------------------

class LossPlateauDetector:
    """
    Detects when a metric (e.g., training loss) stops improving for `patience` steps.
    Useful for triggering phase transitions like switching from float training to QAT.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value: Optional[float] = None
        self.plateau_triggered = False

    def step(self, current_value: float) -> bool:
        """
        Feed the current metric value. Returns True if plateau is detected.
        """
        if self.plateau_triggered:
            return False

        if self.best_value is None:
            self.best_value = current_value
        elif current_value > self.best_value - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.plateau_triggered = True
                return True
        else:
            self.best_value = current_value
            self.counter = 0
        return False

    def reset(self) -> None:
        """Reset to initial state so the detector can be reused after recovery."""
        self.counter = 0
        self.best_value = None
        self.plateau_triggered = False


# ---------------------------------------------------------------------------
# Breakdown detection
# ---------------------------------------------------------------------------

class BreakdownDetector:
    """
    Detects catastrophic training breakdown: val_acc drops dramatically from its peak.

    Breakdown condition (both must hold):
      - The peak val_acc has exceeded peak_min_factor / num_classes, meaning the
        model was genuinely learning before the drop.
      - The current val_acc has fallen below peak * (1 - relative_drop).

    Example (ImageNet, defaults):
      num_classes=1000  → guessing_acc=0.001
      peak_min_factor=10 → peak must exceed 0.01 before detection is armed
      relative_drop=0.7  → breakdown fires when acc < peak * 0.3
      Peak=0.67, current=0.003: 0.003 < 0.67 * 0.3 = 0.20 → breakdown detected ✓
    """

    def __init__(
        self,
        num_classes: int = 1000,
        relative_drop: float = 0.7,
        peak_min_factor: float = 10.0,
    ):
        self._peak_threshold = peak_min_factor / num_classes
        self._keep_fraction = 1.0 - relative_drop
        self._peak_acc: float = 0.0

    def step(self, val_acc: float) -> bool:
        """Update peak tracking and return True if breakdown is detected."""
        self._peak_acc = max(self._peak_acc, val_acc)
        if self._peak_acc < self._peak_threshold:
            return False  # model hasn't trained enough — detection not yet armed
        return val_acc < self._peak_acc * self._keep_fraction

    def reset(self) -> None:
        """Reset peak tracking for a fresh recovery round."""
        self._peak_acc = 0.0

    @property
    def peak_acc(self) -> float:
        return self._peak_acc


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
