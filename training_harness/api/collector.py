"""
collector.py — Passive state collection for the read-only monitoring API.

RunStateCollector is the boundary between the training loop and the HTTP
layer: the server only ever talks to the collector, never to the trainer
directly. It works with both Trainer (V1) and QATTrainerV2 (duck-typed).

Data flows in two ways:
  - Push: the collector registers as an ExperimentLogger listener, so the
    training thread hands it step/epoch metrics with a single list append
    (no locks; appends are atomic under the GIL and readers only take
    snapshots of append-only lists).
  - Pull: cheap, request-time reads of trainer attributes (current epoch,
    LR, phase, checkpoint records) when an HTTP client asks for /status.

Every step/epoch event is also appended to ``api_metrics.jsonl`` in the
run's log directory so history survives crashes. Step-level train loss is
otherwise only sent to TensorBoard/W&B — the CSV log is epoch-level only.
"""

from __future__ import annotations

import bisect
import json
import os
import time
from typing import Any, Dict, List, Optional


TRAIN_ACC_CAVEAT = (
    "train_acc is UNRELIABLE on the full-scale pipelines (nonstandard "
    "computation; with mixup/cutmix it is computed against pre-mix hard "
    "labels). Use train_loss and val_acc as the meaningful signals."
)


class RunStateCollector:
    """
    Read-only view of a live training run.

    Args:
        trainer:    A Trainer or QATTrainerV2 instance. Only read, never
                    mutated.
        jsonl_path: Where to append step/epoch events. "auto" resolves to
                    <logger.run_dir>/api_metrics.jsonl; None disables the
                    JSONL file (used by unit tests).
    """

    def __init__(self, trainer: Any, jsonl_path: Optional[str] = "auto"):
        self.trainer = trainer
        self._start_time = time.time()
        self._status: str = "running"

        # Append-only event stores (training thread appends, HTTP threads read)
        self._steps: List[Dict[str, Any]] = []
        self._step_keys: List[int] = []       # parallel list for bisect
        self._epochs: List[Dict[str, Any]] = []
        self._epoch_keys: List[int] = []
        self._last_epoch_end_time: float = self._start_time
        self._global_step_at_epoch_start: int = 0

        self._jsonl_file = None
        if jsonl_path == "auto":
            run_dir = getattr(getattr(trainer, "logger", None), "run_dir", None)
            jsonl_path = os.path.join(run_dir, "api_metrics.jsonl") if run_dir else None
        if jsonl_path:
            try:
                self._jsonl_file = open(jsonl_path, "a")
            except OSError as e:
                print(f"[api] WARNING: cannot open {jsonl_path}: {e}")

    # ------------------------------------------------------------------
    # ExperimentLogger listener interface (called from the training thread)
    # ------------------------------------------------------------------

    def on_step(self, step: int, metrics: Dict[str, float], phase: str = "train") -> None:
        """Record step-level metrics (already throttled by log_every_n_steps)."""
        if phase != "train":
            return
        record: Dict[str, Any] = {"step": step, "t": time.time()}
        record.update(metrics)
        record["lr"] = self._current_lr()
        self._steps.append(record)
        self._step_keys.append(step)
        self._write_jsonl("step", record)

    def on_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Record the merged train+val metrics for a finished epoch."""
        now = time.time()
        record: Dict[str, Any] = {
            "epoch": epoch,
            "t": now,
            "duration_s": round(now - self._last_epoch_end_time, 3),
        }
        record.update(metrics)
        if "lr" not in record:
            lr = self._current_lr()
            if lr is not None:
                record["lr"] = lr
        self._last_epoch_end_time = now
        self._global_step_at_epoch_start = getattr(self.trainer, "_global_step", 0)
        self._epochs.append(record)
        self._epoch_keys.append(epoch)
        self._write_jsonl("epoch", record)

    def mark_finished(self) -> None:
        """Called by the trainer once fit() completes normally."""
        self._status = "finished"
        if self._jsonl_file is not None:
            try:
                self._jsonl_file.close()
            except OSError:
                pass
            self._jsonl_file = None

    # ------------------------------------------------------------------
    # Snapshots for the HTTP layer (called from server threads)
    # ------------------------------------------------------------------

    def status_snapshot(self) -> Dict[str, Any]:
        t = self.trainer
        config = t.config
        total_epochs = getattr(config, "epochs", None)
        global_step = getattr(t, "_global_step", None)

        n_epochs_done = len(self._epochs)
        last_epoch = self._epoch_keys[-1] if self._epoch_keys else None
        current_epoch = (last_epoch + 1) if last_epoch is not None else 0
        if self._status == "finished" and last_epoch is not None:
            current_epoch = last_epoch

        snapshot: Dict[str, Any] = {
            "status": self._status,
            "experiment_name": getattr(config, "experiment_name", None),
            "run_id": getattr(config, "run_id", None),
            "model_class": type(t.model).__name__,
            "trainer_class": type(t).__name__,
            "trainer_version": "v2" if type(t).__name__ == "QATTrainerV2" else "v1",
            "device": str(getattr(t, "device", None)),
            "pid": os.getpid(),
            "uptime_s": round(time.time() - self._start_time, 1),
            "phase": self._phase_snapshot(),
            "epoch": current_epoch,
            "total_epochs": total_epochs,
            "epochs_completed": n_epochs_done,
            "global_step": global_step,
            "epoch_progress": self._epoch_progress(global_step),
            "eta_s": self._eta_seconds(last_epoch, total_epochs),
            "current_lr": self._current_lr(),
            "best_metric": self._best_metric(),
            "last_update": self._last_update(),
        }
        return snapshot

    def config_snapshot(self) -> Dict[str, Any]:
        raw = self.trainer.config.to_dict()
        # Round-trip through JSON so exotic values can never break jsonify
        sanitized = json.loads(json.dumps(raw, default=str))
        sanitized["trainer_class"] = type(self.trainer).__name__
        return sanitized

    def metrics_snapshot(
        self,
        since_step: int = -1,
        since_epoch: int = -1,
    ) -> Dict[str, Any]:
        """Return step/epoch history strictly *after* the given cursors."""
        return {
            "steps": self._after(self._steps, self._step_keys, since_step),
            "epochs": self._after(self._epochs, self._epoch_keys, since_epoch),
            "caveats": {"train_acc": TRAIN_ACC_CAVEAT},
        }

    def latest_snapshot(self) -> Dict[str, Any]:
        return {
            "step": self._steps[-1] if self._steps else None,
            "epoch": self._epochs[-1] if self._epochs else None,
            "status": self._status,
            "caveats": {"train_acc": TRAIN_ACC_CAVEAT},
        }

    def checkpoints_snapshot(self) -> Dict[str, Any]:
        mgr = getattr(self.trainer, "checkpoint_mgr", None)
        config = self.trainer.config
        records = []
        if mgr is not None:
            for rank, rec in enumerate(list(mgr._records)):
                records.append({
                    "rank": rank,
                    "epoch": rec.epoch,
                    "metric_value": rec.metric_value,
                    "path": os.path.abspath(rec.path),
                    "mtime": self._mtime(rec.path),
                })
        last_path = mgr.last_checkpoint_path() if mgr is not None else None
        return {
            "monitor_metric": getattr(config.checkpoint, "monitor_metric", None),
            "monitor_mode": getattr(config.checkpoint, "monitor_mode", None),
            "top_k": getattr(config.checkpoint, "top_k", None),
            "checkpoints": records,
            "last": {
                "path": os.path.abspath(last_path),
                "mtime": self._mtime(last_path),
            } if last_path else None,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _after(records: list, keys: list, cursor: int) -> list:
        """Slice of records whose key is > cursor (lists are append-only)."""
        n = min(len(records), len(keys))  # consistent snapshot length
        if cursor < 0:
            return records[:n]
        idx = bisect.bisect_right(keys, cursor, 0, n)
        return records[idx:n]

    def _phase_snapshot(self) -> Dict[str, Any]:
        t = self.trainer
        qat_active = None
        if hasattr(t, "_qat_active"):                 # QATTrainerV2
            qat_active = bool(t._qat_active)
        elif hasattr(t, "qat_scheduler"):             # V1 Trainer
            qat_active = bool(t.qat_scheduler.in_qat)

        phase = {
            "name": "qat" if qat_active else "float_warmup",
            "qat_active": qat_active,
            "float_warmup_epochs": self._float_warmup_epochs(),
        }
        phase["quantizers"] = self._quantizer_progress()
        return phase

    def _float_warmup_epochs(self) -> Optional[int]:
        config = self.trainer.config
        qs = getattr(config, "quant_schedule", None) or getattr(config, "qat", None)
        return getattr(qs, "float_warmup_epochs", None)

    def _quantizer_progress(self) -> Dict[str, Any]:
        """
        Count calibration/annealing progress across the QuantizerManager
        singleton. There is no discrete "calibration" phase in either trainer:
        quantizers calibrate lazily (search_done) on their first quantized
        forward pass, then anneal alpha 0 -> 1. This is the honest signal.
        """
        try:
            from quantizers.manager import QuantizerManager
            mgr = QuantizerManager()
            # Brevitas injector re-resolution (e.g. during checkpoint loads)
            # registers throwaway quantizer objects that are never wired into
            # the model; they keep inference_sequence_id == -1 because no
            # forward pass ever reaches them (see manager.py). Count only
            # reached quantizers once at least one forward pass has happened.
            quantizers = [
                q for q in mgr.quantizers.values()
                if getattr(q, "inference_sequence_id", -1) != -1
            ] or list(mgr.quantizers.values())
            total = len(quantizers)
            calibrated = 0
            fully_quantized = 0
            for q in quantizers:
                try:
                    if bool(q.search_done):
                        calibrated += 1
                    if float(q.annealing_alpha) >= 1.0:
                        fully_quantized += 1
                except Exception:
                    pass
            return {
                "total": total,
                "calibrated": calibrated,
                "fully_quantized": fully_quantized,
            }
        except Exception:
            return {"total": None, "calibrated": None, "fully_quantized": None}

    def _epoch_progress(self, global_step: Optional[int]) -> Optional[float]:
        """Fraction of the current epoch's train batches already done."""
        try:
            config = self.trainer.config
            if getattr(config, "dry_run", False):
                steps_per_epoch = config.dry_run_batches
            else:
                steps_per_epoch = len(self.trainer.train_loader)
            if not steps_per_epoch or global_step is None:
                return None
            done = global_step - self._global_step_at_epoch_start
            return round(min(max(done / steps_per_epoch, 0.0), 1.0), 4)
        except (TypeError, AttributeError):
            return None

    def _eta_seconds(self, last_epoch: Optional[int], total_epochs: Optional[int]) -> Optional[float]:
        if not self._epochs or last_epoch is None or total_epochs is None:
            return None
        if self._status == "finished":
            return 0.0
        durations = [e["duration_s"] for e in self._epochs if e.get("duration_s")]
        if not durations:
            return None
        remaining = max(total_epochs - (last_epoch + 1), 0)
        return round(sum(durations) / len(durations) * remaining, 1)

    def _current_lr(self) -> Optional[float]:
        try:
            return float(self.trainer.optimizer.param_groups[0]["lr"])
        except (AttributeError, IndexError, KeyError, TypeError):
            return None

    def _best_metric(self) -> Optional[Dict[str, Any]]:
        mgr = getattr(self.trainer, "checkpoint_mgr", None)
        if mgr is None or not mgr._records:
            return None
        best = mgr._records[0]
        return {
            "name": self.trainer.config.checkpoint.monitor_metric,
            "mode": self.trainer.config.checkpoint.monitor_mode,
            "value": best.metric_value,
            "epoch": best.epoch,
        }

    def _last_update(self) -> Optional[float]:
        times = []
        if self._steps:
            times.append(self._steps[-1]["t"])
        if self._epochs:
            times.append(self._epochs[-1]["t"])
        return max(times) if times else None

    @staticmethod
    def _mtime(path: Optional[str]) -> Optional[float]:
        try:
            return os.path.getmtime(path) if path else None
        except OSError:
            return None

    def _write_jsonl(self, kind: str, record: Dict[str, Any]) -> None:
        if self._jsonl_file is None:
            return
        try:
            self._jsonl_file.write(json.dumps({"type": kind, **record}, default=str) + "\n")
            self._jsonl_file.flush()
        except (OSError, ValueError) as e:
            print(f"[api] WARNING: JSONL write failed, disabling: {e}")
            self._jsonl_file = None
