"""
checkpointing.py — Checkpoint management for the training_harness harness.

Handles saving/loading model state, optimizer state, scheduler state,
and training_harness metadata. Maintains a top-K ranking so only the best
checkpoints are kept on disk. Automatically exports to ONNX alongside
each saved checkpoint.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from utils.onnx_export import export_onnx_with_io


# ---------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------------

@dataclass
class CheckpointRecord:
    """Metadata for a single saved checkpoint."""
    epoch: int
    metric_value: float
    path: str

    def to_dict(self) -> dict:
        return {"epoch": self.epoch, "metric_value": self.metric_value, "path": self.path}


# ---------------------------------------------------------------------------
# Checkpoint payload helpers
# -----------------------------------------------------------------------------------

def _build_payload(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    metrics_dict: Dict[str, Any],
    config_dict: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> dict:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics_dict,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if config_dict is not None:
        payload["config"] = config_dict
    if extra:
        payload["extra"] = extra
    return payload


# ---------------------------------------------------------------------------
# CheckpointManager
# -----------------------------------------------------------------------------------

class CheckpointManager:
    """
    Manages saving and loading of training_harness checkpoints.

    Features:
    - Saves model, optimizer, scheduler, and metadata.
    - Keeps only the top-K checkpoints ranked by a monitored metric.
    - Optionally keeps a 'last.pt' checkpoint separate from the top-K.
    - Automatically exports the model to ONNX alongside each saved checkpoint.
    - Provides a simple resume() method to restore a full training_harness state.
    - Gracefully handles Brevitas scale/buffer mismatches during load.

    Usage::

        ckpt = CheckpointManager(save_dir="checkpoints", top_k=3, mode="min")

        # After each epoch:
        ckpt.save(
            epoch=epoch,
            metric_value=val_loss,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        # To resume training_harness:
        start_epoch = ckpt.resume(model, optimizer, scheduler)
    """

    INDEX_FILE = "checkpoint_index.json"

    def __init__(
        self,
        save_dir: str = "checkpoints",
        top_k: int = 3,
        monitor_mode: str = "min",
        save_last: bool = True,
        save_every_n_epochs: Optional[int] = None,
        experiment_name: str = "exp",
    ):
        """
        Args:
            save_dir:             Directory to write checkpoint files.
            top_k:                Maximum number of best checkpoints to keep.
            monitor_mode:         'min' for loss, 'max' for accuracy.
            save_last:            Always keep a 'last.pt' checkpoint.
            save_every_n_epochs:  Unconditionally save every N epochs.
            experiment_name:      Used to prefix checkpoint filenames.
        """
        self.save_dir = save_dir
        self.top_k = top_k
        self.monitor_mode = monitor_mode
        self.save_last = save_last
        self.save_every_n_epochs = save_every_n_epochs
        self.experiment_name = experiment_name

        os.makedirs(save_dir, exist_ok=True)

        # Ranked list of saved checkpoints (best first)
        self._records: List[CheckpointRecord] = []
        self._load_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        epoch: int,
        metric_value: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        metrics_dict: Optional[Dict[str, Any]] = None,
        config_dict: Optional[dict] = None,
        extra: Optional[dict] = None,
        dummy_input: Optional[torch.Tensor] = None,
    ) -> Optional[str]:
        """
        Evaluate whether this epoch should be checkpointed and save if so.

        Args:
            epoch:         Current epoch number.
            metric_value:  Value of the monitored metric for this epoch.
            model:         The model whose state_dict to save.
            optimizer:     Optimizer state to save.
            scheduler:     LR scheduler state to save (optional).
            metrics_dict:  Full metrics snapshot (stored as metadata).
            config_dict:   TrainerConfig as a plain dict (stored for reference).
            extra:         Any extra data to bundle into the checkpoint.
            dummy_input:   Optional tensor for ONNX export. If None, a random
                           tensor of shape (1, 3, 32, 32) is generated.

        Returns:
            Path to the saved checkpoint file, or None if not saved.
        """
        path: Optional[str] = None

        # Always save 'last'
        if self.save_last:
            last_path = os.path.join(self.save_dir, "last.pt")
            payload = _build_payload(
                epoch, model, optimizer, scheduler, metrics_dict or {}, config_dict, extra
            )
            torch.save(payload, last_path)
            self._export_onnx(model, last_path.replace('.pt', '.onnx'), dummy_input)

        # Save every-N if configured
        if self.save_every_n_epochs and (epoch + 1) % self.save_every_n_epochs == 0:
            periodic_path = os.path.join(
                self.save_dir,
                f"{self.experiment_name}_epoch{epoch:04d}.pt"
            )
            payload = _build_payload(
                epoch, model, optimizer, scheduler, metrics_dict or {}, config_dict, extra
            )
            torch.save(payload, periodic_path)

        # Top-K logic
        if self._should_save(metric_value):
            fname = f"{self.experiment_name}_epoch{epoch:04d}_metric{metric_value:.6f}.pt"
            path = os.path.join(self.save_dir, fname)
            payload = _build_payload(
                epoch, model, optimizer, scheduler, metrics_dict or {}, config_dict, extra
            )
            torch.save(payload, path)

            record = CheckpointRecord(epoch=epoch, metric_value=metric_value, path=path)
            self._add_record(record)
            self._prune_to_top_k()
            self._save_index()

            print(
                f"  [ckpt] Saved → {os.path.basename(path)}"
                f"  (top-{self.top_k} pool: {len(self._records)} checkpoints)"
            )

        return path

    def best_checkpoint_path(self) -> Optional[str]:
        """Return the path of the best checkpoint, or None if none saved."""
        if not self._records:
            return None
        return self._records[0].path  # Already sorted best-first

    def last_checkpoint_path(self) -> Optional[str]:
        """Return the path of the 'last.pt' checkpoint."""
        path = os.path.join(self.save_dir, "last.pt")
        return path if os.path.exists(path) else None

    def resume(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        path: Optional[str] = None,
        device: str = "cpu",
        reset_calibration: bool = True,
    ) -> int:
        """
        Load a checkpoint and restore training_harness state.

        Args:
            model:             Model to restore weights into.
            optimizer:         Optimizer to restore state into (optional).
            scheduler:         Scheduler to restore state into (optional).
            path:              Explicit checkpoint path. If None, uses last.pt.
            device:            Device string for torch.load map_location.
            reset_calibration: If True, resets lazy calibration buffers (e.g., `search_done`)
                               so quantizers will re-calibrate on the next forward pass.

        Returns:
            The epoch at which the checkpoint was saved (resume from here + 1).
        """
        if path is None:
            path = self.last_checkpoint_path()
        if path is None:
            print("[ckpt] No checkpoint found to resume from. Starting fresh.")
            return 0
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        print(f"[ckpt] Resuming from {path}")
        payload = torch.load(path, map_location=device)

        # Use strict=False to handle Brevitas scale/buffer mismatches gracefully
        try:
            incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
            if incompatible.missing_keys:
                print(f"[ckpt] Missing keys (expected for Brevitas scales/buffers): {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                print(f"[ckpt] Unexpected keys: {incompatible.unexpected_keys}")
        except Exception as e:
            print(f"[ckpt] Warning during state_dict load: {e}")

        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in payload:
            scheduler.load_state_dict(payload["scheduler_state_dict"])

        if reset_calibration:
            self._reset_calibration_buffers(model)

        epoch = payload.get("epoch", 0)
        metrics = payload.get("metrics", {})
        print(f"[ckpt] Restored to epoch {epoch}. Metrics: {metrics}")
        return epoch + 1  # Next epoch to run

    def load_best(
        self,
        model: nn.Module,
        device: str = "cpu",
        reset_calibration: bool = True,
    ) -> Optional[dict]:
        """
        Load the best checkpoint's weights into the model.

        Args:
            model:             Model to load weights into.
            device:            Device string for torch.load map_location.
            reset_calibration: If True, resets lazy calibration buffers.

        Returns:
            The full payload dict, or None if no checkpoint exists.
        """
        path = self.best_checkpoint_path()
        if path is None or not os.path.exists(path):
            print("[ckpt] No best checkpoint found.")
            return None

        payload = torch.load(path, map_location=device)
        try:
            incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
            if incompatible.missing_keys:
                print(f"[ckpt] Missing keys (expected for Brevitas scales/buffers): {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                print(f"[ckpt] Unexpected keys: {incompatible.unexpected_keys}")
        except Exception as e:
            print(f"[ckpt] Warning during state_dict load: {e}")

        if reset_calibration:
            self._reset_calibration_buffers(model)

        print(f"[ckpt] Loaded best checkpoint from epoch {payload.get('epoch', '?')}")
        return payload

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_save(self, metric_value: float) -> bool:
        """Return True if this metric value should enter the top-K pool."""
        if len(self._records) < self.top_k:
            return True
        # Compare against the worst in the current pool
        worst = self._records[-1].metric_value
        if self.monitor_mode == "min":
            return metric_value < worst
        return metric_value > worst

    def _add_record(self, record: CheckpointRecord) -> None:
        self._records.append(record)
        reverse = (self.monitor_mode == "max")
        self._records.sort(key=lambda r: r.metric_value, reverse=reverse)

    def _prune_to_top_k(self) -> None:
        """Delete checkpoint files that fall outside the top-K pool."""
        while len(self._records) > self.top_k:
            evicted = self._records.pop()  # Worst is at the end
            if os.path.exists(evicted.path):
                os.remove(evicted.path)
                print(f"  [ckpt] Evicted {os.path.basename(evicted.path)}")

    def _index_path(self) -> str:
        return os.path.join(self.save_dir, self.INDEX_FILE)

    def _save_index(self) -> None:
        data = {
            "monitor_mode": self.monitor_mode,
            "top_k": self.top_k,
            "records": [r.to_dict() for r in self._records],
        }
        with open(self._index_path(), "w") as f:
            json.dump(data, f, indent=2)

    def _load_index(self) -> None:
        path = self._index_path()
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self._records = [
            CheckpointRecord(**r)
            for r in data.get("records", [])
            if os.path.exists(r["path"])  # Skip missing files
        ]

    def _reset_calibration_buffers(self, model: nn.Module) -> None:
        """Reset lazy calibration flags (e.g., `search_done`) in Brevitas quantizers."""
        reset_count = 0
        for module in model.modules():
            for name, buffer in module.named_buffers():
                if "search_done" in name or "calibration_done" in name:
                    buffer.fill_(False)
                    reset_count += 1
        if reset_count > 0:
            print(f"  [ckpt] Reset {reset_count} calibration buffer(s) to force re-calibration.")

    def _export_onnx(self, model: nn.Module, onnx_path: str, dummy_input: Optional[torch.Tensor]) -> None:
        """Export model to ONNX format alongside the checkpoint using the centralized exporter."""
        if dummy_input is None:
            # Fallback to a simple random tensor if no dummy input is provided.
            dummy_input = torch.randn(1, 3, 32, 32)
        
        try:
            # export_onnx_with_io handles model.eval() and torch.no_grad() internally
            export_onnx_with_io(
                model=model,
                dummy_input=dummy_input,
                filepath=onnx_path,
                opset_version=13,
                custom_opsets={"Quantify": 1},
                dynamo=False,
            )
            print(f"  [ckpt] Exported ONNX → {os.path.basename(onnx_path)}")
        except Exception as e:
            print(f"  [ckpt] ONNX export skipped: {e}")
