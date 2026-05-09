"""
metrics.py — Metrics tracking for the training harness.

Tracks loss, accuracy, and any user-defined metrics per step and per epoch.
Also handles quantization-specific metrics like scale factor evolution.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Running average helper
# ---------------------------------------------------------------------------

class AverageMeter:
    """Computes and stores a running average of a scalar value."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Args:
            val:  Value to add (can be a batch mean).
            n:    Number of samples this value represents.
        """
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


# ---------------------------------------------------------------------------
# Per-epoch metrics snapshot
# ---------------------------------------------------------------------------

class EpochMetrics:
    """Holds all metrics for a single epoch."""

    def __init__(self, epoch: int):
        self.epoch = epoch
        self.metrics: Dict[str, float] = {}
        self.quant_scales: Dict[str, float] = {}  # layer_name → scale value

    def update(self, key: str, value: float) -> None:
        self.metrics[key] = value

    def update_scales(self, scales: Dict[str, float]) -> None:
        self.quant_scales.update(scales)

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            **self.metrics,
        }

    def __repr__(self) -> str:
        parts = [f"Epoch {self.epoch}"]
        for k, v in self.metrics.items():
            parts.append(f"{k}={v:.4f}")
        return "  ".join(parts)


# ---------------------------------------------------------------------------
# Main MetricsTracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    """
    Centralised store for all training metrics.

    Records per-step and per-epoch values, making them available to the
    plotter, logger, and checkpoint manager.

    Usage::

        tracker = MetricsTracker()

        # Inside the training loop
        tracker.update_step("train_loss", loss.item())

        # At the end of an epoch
        tracker.commit_epoch(epoch, phase="train")
        tracker.commit_epoch(epoch, phase="val", extra={"val_loss": 0.42})
    """

    def __init__(self) -> None:
        # Step-level buffers: phase → metric_name → list[float]
        self._step_buffers: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Epoch-level history: list of EpochMetrics
        self.history: List[EpochMetrics] = []

        # Quant-specific: epoch → {layer_name: scale}
        self.scale_history: Dict[int, Dict[str, float]] = {}

        # Running averages for the current epoch
        self._meters: Dict[str, AverageMeter] = {}

    # ------------------------------------------------------------------
    # Step-level API
    # ------------------------------------------------------------------

    def update_step(self, name: str, value: float, phase: str = "train", n: int = 1) -> None:
        """
        Record a single scalar at the current step.

        Args:
            name:   Metric name (e.g. 'loss', 'acc').
            value:  Scalar value.
            phase:  'train' or 'val'.
            n:      Sample count (for weighted averages).
        """
        key = f"{phase}_{name}"
        self._step_buffers[phase][name].append(value)

        if key not in self._meters:
            self._meters[key] = AverageMeter(key)
        self._meters[key].update(value, n)

    def current_avg(self, name: str, phase: str = "train") -> float:
        """Return the running average for a metric in the current epoch."""
        key = f"{phase}_{name}"
        meter = self._meters.get(key)
        return meter.avg if meter is not None else 0.0

    # ------------------------------------------------------------------
    # Epoch-level API
    # ------------------------------------------------------------------

    def commit_epoch(
        self,
        epoch: int,
        phase: str = "train",
        extra: Optional[Dict[str, float]] = None,
    ) -> EpochMetrics:
        """
        Finalise the current epoch, computing averages from step buffers.

        Args:
            epoch:  Epoch index.
            phase:  'train' or 'val'.
            extra:  Any additional metrics to attach (computed outside the loop).

        Returns:
            The resulting EpochMetrics snapshot.
        """
        snap = EpochMetrics(epoch)

        # Average all step buffers for this phase
        for name, values in self._step_buffers[phase].items():
            snap.update(f"{phase}_{name}", float(np.mean(values)))

        # Attach extras
        if extra:
            for k, v in extra.items():
                snap.update(k, v)

        self.history.append(snap)

        # Clear step buffers and meters for the next epoch
        self._step_buffers[phase].clear()
        keys_to_clear = [k for k in self._meters if k.startswith(f"{phase}_")]
        for k in keys_to_clear:
            del self._meters[k]

        return snap

    def record_scale_factors(self, epoch: int, scales: Dict[str, float]) -> None:
        """
        Save per-layer quantization scale factors for a given epoch.

        Args:
            epoch:  Epoch index.
            scales: Dict mapping layer name → scale value.
        """
        self.scale_history[epoch] = scales

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_metric_series(self, metric_name: str) -> tuple[List[int], List[float]]:
        """
        Return (epochs, values) for a named metric across all history.

        Example::

            epochs, losses = tracker.get_metric_series("val_loss")
        """
        epochs, values = [], []
        for snap in self.history:
            if metric_name in snap.metrics:
                epochs.append(snap.epoch)
                values.append(snap.metrics[metric_name])
        return epochs, values

    def best_value(self, metric_name: str, mode: str = "min") -> Optional[float]:
        """Return the best recorded value for a metric."""
        _, values = self.get_metric_series(metric_name)
        if not values:
            return None
        return min(values) if mode == "min" else max(values)

    def latest(self, metric_name: str) -> Optional[float]:
        """Return the most recent value of a metric."""
        for snap in reversed(self.history):
            if metric_name in snap.metrics:
                return snap.metrics[metric_name]
        return None

    def summary(self) -> Dict[str, Any]:
        """Return a flat dict summarising the full training run."""
        result = {"total_epochs": len(self.history)}
        # Collect all metric names that ever appeared
        all_metrics: set = set()
        for snap in self.history:
            all_metrics.update(snap.metrics.keys())
        for name in sorted(all_metrics):
            _, values = self.get_metric_series(name)
            if values:
                result[f"best_{name}"] = min(values) if "loss" in name else max(values)
                result[f"final_{name}"] = values[-1]
        return result

    def all_epoch_dicts(self) -> List[Dict[str, Any]]:
        """Return all epoch snapshots as plain dicts (for CSV export)."""
        # Merge train and val snaps for the same epoch
        by_epoch: Dict[int, dict] = {}
        for snap in self.history:
            d = by_epoch.setdefault(snap.epoch, {"epoch": snap.epoch})
            d.update(snap.metrics)
        return [by_epoch[e] for e in sorted(by_epoch)]
