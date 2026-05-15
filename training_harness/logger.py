"""
logger.py — Experiment logging for the training_harness harness.

Provides a unified ExperimentLogger that writes to one or more backends:
  - CSV  (always available, zero extra dependencies)
  - TensorBoard (optional, requires tensorboard package)
  - Weights & Biases (optional, requires wandb package)

The logger is intentionally thin: it records what the MetricsTracker
gives it and routes to whichever backends are enabled.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Any, Dict, Optional


class ExperimentLogger:
    """
    Unified experiment logger.

    Usage::

        logger = ExperimentLogger(
            experiment_name="resnet_qat",
            run_id="2024-01-15_143022",
            log_dir="logs",
            use_tensorboard=True,
            use_wandb=False,
        )

        logger.log_hparams(config.to_dict())

        # During training_harness
        logger.log_step(step=100, metrics={"train_loss": 0.42}, phase="train")
        logger.log_epoch(epoch=5, metrics={"val_loss": 0.38, "val_acc": 0.91})

        logger.close()
    """

    def __init__(
        self,
        experiment_name: str = "experiment",
        run_id: Optional[str] = None,
        log_dir: str = "logs",
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        csv_log: bool = True,
    ):
        self.experiment_name = experiment_name
        self.run_id = run_id or time.strftime("%Y-%m-%d_%H%M%S")
        self.log_dir = log_dir

        # Build the run-specific log directory
        self.run_dir = os.path.join(log_dir, experiment_name, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        # Backends
        self._tb_writer = None
        self._wandb = None
        self._csv_file = None
        self._csv_writer = None
        self._csv_fieldnames: Optional[list] = None

        if use_tensorboard:
            self._init_tensorboard()

        if use_wandb:
            self._init_wandb(wandb_project, wandb_entity)

        if csv_log:
            self._init_csv()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_tensorboard(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(self.run_dir, "tensorboard")
            self._tb_writer = SummaryWriter(log_dir=tb_dir)
            print(f"[logger] TensorBoard: {tb_dir}")
        except ImportError:
            print("[logger] WARNING: tensorboard not installed. Skipping TensorBoard logging.")

    def _init_wandb(self, project: Optional[str], entity: Optional[str]) -> None:
        try:
            import wandb
            wandb.init(
                project=project or self.experiment_name,
                entity=entity,
                name=self.run_id,
                dir=self.run_dir,
            )
            self._wandb = wandb
            print(f"[logger] W&B run: {self.run_id}")
        except ImportError:
            print("[logger] WARNING: wandb not installed. Skipping W&B logging.")

    def _init_csv(self) -> None:
        csv_path = os.path.join(self.run_dir, "metrics.csv")
        self._csv_path = csv_path
        # We open lazily once we know the fieldnames (on first log_epoch call)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_hparams(self, hparams: Dict[str, Any]) -> None:
        """
        Record hyperparameters at the start of a run.

        Writes to all enabled backends.
        """
        # Always write a JSON file
        import json
        hparams_path = os.path.join(self.run_dir, "hparams.json")
        with open(hparams_path, "w") as f:
            json.dump(hparams, f, indent=2, default=str)

        if self._tb_writer is not None:
            try:
                # TensorBoard hparams panel
                self._tb_writer.add_hparams(
                    hparam_dict={k: str(v) for k, v in hparams.items()},
                    metric_dict={},
                )
            except Exception:
                pass  # Some TB versions are picky about types

        if self._wandb is not None:
            self._wandb.config.update(hparams, allow_val_change=True)

    def log_step(
        self,
        step: int,
        metrics: Dict[str, float],
        phase: str = "train",
    ) -> None:
        """
        Log scalar metrics at the step level.

        Args:
            step:    Global training_harness step number.
            metrics: Dict of metric_name → value.
            phase:   'train' or 'val' (used as a prefix in TB).
        """
        if self._tb_writer is not None:
            for name, value in metrics.items():
                self._tb_writer.add_scalar(f"{phase}/{name}", value, step)

        if self._wandb is not None:
            self._wandb.log({f"{phase}/{k}": v for k, v in metrics.items()}, step=step)

    def log_epoch(
        self,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """
        Log scalar metrics at the epoch level.

        Args:
            epoch:   Epoch index.
            metrics: Flat dict of all metrics for this epoch.
        """
        # CSV backend
        if hasattr(self, "_csv_path"):
            self._write_csv_row({"epoch": epoch, **metrics})

        # TensorBoard
        if self._tb_writer is not None:
            for name, value in metrics.items():
                self._tb_writer.add_scalar(f"epoch/{name}", value, epoch)

        # W&B
        if self._wandb is not None:
            self._wandb.log({"epoch": epoch, **metrics})

    def log_scale_factors(
        self,
        epoch: int,
        scales: Dict[str, float],
    ) -> None:
        """
        Log per-layer quantization scale factors.

        These are recorded as scalars with tag 'quant_scales/<layer_name>'.
        """
        if self._tb_writer is not None:
            for layer, scale in scales.items():
                self._tb_writer.add_scalar(f"quant_scales/{layer}", scale, epoch)

        if self._wandb is not None:
            self._wandb.log(
                {f"quant_scales/{k}": v for k, v in scales.items()},
                step=epoch,
            )

        # Also append to a dedicated CSV
        scales_csv = os.path.join(self.run_dir, "scale_factors.csv")
        write_header = not os.path.exists(scales_csv)
        with open(scales_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "layer", "scale"])
            for layer, scale in scales.items():
                writer.writerow([epoch, layer, scale])

    def log_text(self, tag: str, text: str) -> None:
        """Log a block of text (e.g. hardware info, config dump)."""
        # Always write to a file
        path = os.path.join(self.run_dir, f"{tag}.txt")
        with open(path, "w") as f:
            f.write(text)

        if self._tb_writer is not None:
            try:
                self._tb_writer.add_text(tag, text)
            except Exception:
                pass

    def log_image(self, tag: str, image_path: str, epoch: int = 0) -> None:
        """Log a saved image (e.g. a training_harness curve plot) to W&B."""
        if self._wandb is not None:
            try:
                self._wandb.log({tag: self._wandb.Image(image_path)}, step=epoch)
            except Exception:
                pass

    def close(self) -> None:
        """Flush and close all backends."""
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._csv_file is not None:
            self._csv_file.close()
        if self._wandb is not None:
            self._wandb.finish()

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _write_csv_row(self, row: Dict[str, Any]) -> None:
        """Write a row to the CSV log, creating headers on first write."""
        new_file = not os.path.exists(self._csv_path)

        if new_file:
            self._csv_fieldnames = list(row.keys())
            self._csv_file = open(self._csv_path, "a", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=self._csv_fieldnames,
                extrasaction="ignore",
            )
            self._csv_writer.writeheader()
        else:
            if self._csv_file is None:
                # File exists from a previous run (resuming) — open in append mode
                # Infer fieldnames from the existing header row
                with open(self._csv_path) as f:
                    reader = csv.DictReader(f)
                    self._csv_fieldnames = reader.fieldnames or list(row.keys())
                self._csv_file = open(self._csv_path, "a", newline="")
                self._csv_writer = csv.DictWriter(
                    self._csv_file,
                    fieldnames=self._csv_fieldnames,
                    extrasaction="ignore",
                )

        self._csv_writer.writerow(row)
        self._csv_file.flush()  # Ensure writes are visible immediately
