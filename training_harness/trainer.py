"""
trainer.py — The main Trainer class for Brevitas QAT experiments.

Orchestrates: training_harness loop, validation, checkpointing, logging,
metric tracking, QAT schedule, calibration, and plotting.
"""

from __future__ import annotations

import time
import warnings
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpointing import CheckpointManager
from .config import TrainerConfig
from .logger import ExperimentLogger
from .metrics import MetricsTracker
from .plotting import TrainingPlotter
from .schedulers import QATWarmupScheduler, collect_scale_factors
from .engine_utils import EarlyStopping, EpochTimer, log_hardware_info, set_seed, LossPlateauDetector
from quantizers.manager import QuantizerManager


class Trainer:
    """
    End-to-end training_harness harness for Brevitas QAT.

    The Trainer wires together every component of the harness:
      - Config-driven setup (device, seed, AMP)
      - Training + validation loops with gradient clipping
      - QAT warmup: float warmup → calibration → fake-quant
      - Checkpoint management (top-K, last, periodic)
      - Metrics tracking + CSV/TensorBoard/W&B logging
      - Training curve plots saved automatically
      - Early stopping

    Minimal usage::

        config = TrainerConfig(
            experiment_name="resnet18_qat",
            epochs=50,
            learning_rate=1e-3,
        )

        trainer = Trainer(
            config=config,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=nn.CrossEntropyLoss(),
        )

        trainer.fit()

    Advanced usage (custom logic via callbacks/hooks)::

        def after_step(trainer, loss, outputs, targets):
            # Custom per-step logic here
            pass

        trainer.fit(after_step_hook=after_step)
    """

    def __init__(
        self,
        config: TrainerConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
        scheduler=None,
        accuracy_fn: Optional[Callable] = None,
    ):
        """
        Args:
            config:        TrainerConfig describing this run.
            model:         The (Brevitas) model to train.
            optimizer:     PyTorch optimizer.
            train_loader:  Training DataLoader.
            val_loader:    Validation DataLoader (optional but recommended).
            loss_fn:       Loss function (defaults to CrossEntropyLoss).
            scheduler:     LR scheduler (optional; called every step).
            accuracy_fn:   Optional callable (outputs, targets) → float.
                           Defaults to top-1 accuracy for classification.
        """
        self.config   = config
        self.model    = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.loss_fn  = loss_fn or nn.CrossEntropyLoss()
        self.scheduler = scheduler
        self.accuracy_fn = accuracy_fn or _default_accuracy

        # Resolve device
        self.device = torch.device(config.resolve_device())

        # Warn about AMP compatibility with Brevitas
        if config.mixed_precision:
            warnings.warn(
                "Mixed precision (AMP) is enabled. "
                "Brevitas fake-quantization ops may interact poorly with autocast, "
                "potentially causing unstable scale learning. Consider setting mixed_precision=False.",
                UserWarning,
            )

        # ── Components ────────────────────────────────────────────────
        import time as _time
        self.config.run_id = config.run_id or _time.strftime("%Y-%m-%d_%H%M%S")
        config.make_run_dirs()

        self.tracker = MetricsTracker()

        self.checkpoint_mgr = CheckpointManager(
            save_dir       = config.checkpoint_dir,
            top_k          = config.checkpoint.top_k,
            monitor_mode   = config.checkpoint.monitor_mode,
            save_last      = config.checkpoint.save_last,
            save_every_n_epochs = config.checkpoint.save_every_n_epochs,
            experiment_name = config.experiment_name,
        )

        self.logger = ExperimentLogger(
            experiment_name = config.experiment_name,
            run_id          = config.run_id,
            log_dir         = config.log_dir,
            use_tensorboard = config.logging.use_tensorboard,
            use_wandb       = config.logging.use_wandb,
            wandb_project   = config.logging.wandb_project,
            wandb_entity    = config.logging.wandb_entity,
            csv_log         = config.logging.csv_log,
        )

        self.plotter = TrainingPlotter(
            save_dir        = config.plot_dir,
            experiment_name = config.experiment_name,
        )

        self.qat_scheduler = QATWarmupScheduler(
            model               = model,
            float_warmup_epochs = config.quant_schedule.float_warmup_epochs,
            freeze_bn_after_epoch = config.quant_schedule.freeze_bn_after_epoch,
        )

        self.early_stopper: Optional[EarlyStopping] = None
        if config.early_stopping_patience is not None:
            self.early_stopper = EarlyStopping(
                patience   = config.early_stopping_patience,
                min_delta  = config.early_stopping_min_delta,
                mode       = config.checkpoint.monitor_mode,
            )

        # Loss plateau detector for QAT activation
        self.loss_plateau_detector = LossPlateauDetector(patience=5)

        # AMP scaler (disabled on CPU / MPS)
        self._use_amp = config.mixed_precision and str(self.device).startswith("cuda")
        self._scaler  = torch.cuda.amp.GradScaler(enabled=self._use_amp)

        # Global step counter (for step-level logging)
        self._global_step: int = 0

        # LR history (for plotting)
        self._lr_history: List[float] = []

        # Set True by a manual-LR control command so the LR scheduler stops
        # overwriting the override on its next step (see fit()/_run_epoch).
        self._scheduler_suspended: bool = False

        # EpochTimer handle, exposed so the add-epochs control can extend it.
        self._timer = None

        # Names the loop's toggleable behaviors for the dashboard. Always
        # built (default all-enabled = identical behavior); only mutated by
        # control commands when the API is enabled.
        self.callbacks = self._build_callback_registry()

        # Optional monitoring + control API (opt-in via config.api_port)
        self.api_server = None
        self._api_collector = None
        self._control = None
        if config.api_port is not None:
            from .api import DashboardAPIServer, RunStateCollector
            from .api.control import ControlManager
            self._api_collector = RunStateCollector(self)
            self.logger.add_listener(self._api_collector)
            self._control = ControlManager(self, self._api_collector, self.callbacks)
            self.api_server = DashboardAPIServer(
                self._api_collector,
                control=self._control,
                host=config.api_host,
                port=config.api_port,
            )
            self.api_server.start()

    def _build_callback_registry(self):
        """Register the loop behaviors the dashboard can list / toggle."""
        from .api.control import CallbackRegistry
        reg = CallbackRegistry()
        reg.register("checkpointing", "epoch_end",
                     "Save top-K / last / periodic checkpoints", toggleable=True)
        reg.register("scale_factor_tracking", "epoch_end",
                     "Collect per-layer quantization scale factors (QAT only)",
                     toggleable=True)
        reg.register("plateau_qat_activation", "epoch_end",
                     "Auto-activate QAT when training loss plateaus", toggleable=True)
        if self.early_stopper is not None:
            reg.register("early_stopping", "epoch_end",
                         "Stop early once QAT has started and the metric stalls",
                         toggleable=True)
        reg.register("user_after_step", "step_end",
                     "User-provided after_step_hook (if passed to fit)", toggleable=True)
        reg.register("user_after_epoch", "epoch_end",
                     "User-provided after_epoch_hook (if passed to fit)", toggleable=True)
        # Core behaviors — listed for transparency, but never toggleable.
        reg.register("optimizer_step", "step",
                     "The optimizer update; disabling would break training",
                     toggleable=False)
        reg.register("metrics_logging", "step_end",
                     "Metric logging the dashboard depends on", toggleable=False)
        return reg

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def fit(
        self,
        resume: bool = False,
        after_step_hook: Optional[Callable] = None,
        after_epoch_hook: Optional[Callable] = None,
    ) -> MetricsTracker:
        """
        Run the full training_harness loop.

        Args:
            resume:             Resume from the last checkpoint if available.
            after_step_hook:    Called after every optimizer step with
                                (trainer, loss_val, outputs, targets).
            after_epoch_hook:   Called after every epoch with
                                (trainer, epoch, metrics_snapshot).

        Returns:
            The MetricsTracker populated with all metrics for this run.
        """
        # ── Setup ────────────────────────────────────────────────────
        set_seed(self.config.seed, self.config.deterministic)
        self.model.to(self.device)
        log_hardware_info(self.logger)
        self.logger.log_hparams(self.config.to_dict())

        # Log quantization configuration for reproducibility
        quant_config = self._extract_quant_config()
        if quant_config:
            self.logger.log_text("quant_config", str(quant_config))

        print(f"\n{'═'*60}")
        print(f"  Experiment : {self.config.experiment_name}")
        print(f"  Run ID     : {self.config.run_id}")
        print(f"  Device     : {self.device}")
        print(f"  AMP        : {self._use_amp}")
        print(f"  Epochs     : {self.config.epochs}")
        print(f"  Dry run    : {self.config.dry_run}")
        print(f"{'═'*60}\n")

        start_epoch = 0
        if resume:
            start_epoch = self.checkpoint_mgr.resume(
                self.model, self.optimizer, self.scheduler, device=str(self.device)
            )

        timer = EpochTimer(total_epochs=self.config.epochs)
        self._timer = timer

        # ── Main loop ────────────────────────────────────────────────
        # While-loop (not range) so the add-epochs control command can extend
        # self.config.epochs mid-run and have the new budget take effect.
        prev_qat = self.qat_scheduler.in_qat
        epoch = start_epoch
        while epoch < self.config.epochs:
            timer.start()

            # Update QAT state (float → quant transition, BN freeze)
            self.qat_scheduler.step(epoch)
            if self._api_collector is not None and self.qat_scheduler.in_qat != prev_qat:
                self._api_collector.record_event(
                    "phase", f"QAT activated at epoch {epoch}")
                prev_qat = self.qat_scheduler.in_qat

            # Training phase
            train_metrics = self._run_epoch(
                epoch, phase="train", after_step_hook=after_step_hook
            )

            # Validation phase
            val_metrics: Dict[str, float] = {}
            if self.val_loader is not None:
                val_metrics = self._run_epoch(epoch, phase="val")

            # Collect scale factors
            scales: Dict[str, float] = {}
            if (self.callbacks.is_enabled("scale_factor_tracking")
                    and self.config.quant_schedule.track_scale_factors
                    and self.qat_scheduler.in_qat):
                scales = collect_scale_factors(self.model)
                self.tracker.record_scale_factors(epoch, scales)
                self.logger.log_scale_factors(epoch, scales)

            # Log epoch metrics
            all_metrics = {**train_metrics, **val_metrics}
            self.logger.log_epoch(epoch, all_metrics)

            # Checkpoint
            monitor_val = all_metrics.get(
                self.config.checkpoint.monitor_metric,
                train_metrics.get("train_loss", 0.0),
            )
            if self.callbacks.is_enabled("checkpointing"):
                saved_path = self.checkpoint_mgr.save(
                    epoch        = epoch,
                    metric_value = monitor_val,
                    model        = self.model,
                    optimizer    = self.optimizer,
                    scheduler    = self.scheduler,
                    metrics_dict = all_metrics,
                    config_dict  = self.config.to_dict(),
                )
                if saved_path and self._api_collector is not None:
                    self._api_collector.record_event(
                        "checkpoint",
                        f"saved epoch {epoch} "
                        f"({self.config.checkpoint.monitor_metric}={monitor_val:.4f})",
                        {"path": saved_path},
                    )

            # Plateau detection & QAT activation
            if (self.callbacks.is_enabled("plateau_qat_activation")
                    and QuantizerManager().is_not_quantizing_at_all):
                is_plateau = self.loss_plateau_detector.step(monitor_val)
                if is_plateau:
                    print(f"[trainer] Training loss plateaued. Activating QAT...")
                    mgr = QuantizerManager()
                    mgr.set_annealing_for_n_inferences(6)
                    mgr.quantization_start_gap = 20

            # Early stopping: only trigger when QAT has actually started
            stop = False
            if (self.early_stopper is not None
                    and self.callbacks.is_enabled("early_stopping")
                    and not QuantizerManager().is_not_quantizing_at_all):
                stop = self.early_stopper.step(monitor_val, model=self.model, epoch=epoch)

            # Progress line
            elapsed, eta = timer.stop(epoch)
            self._print_epoch_summary(epoch, elapsed, eta, all_metrics)

            # User hook
            if after_epoch_hook is not None and self.callbacks.is_enabled("user_after_epoch"):
                snap = self.tracker.history[-1] if self.tracker.history else None
                after_epoch_hook(self, epoch, snap)

            # Apply queued control commands at the epoch boundary (reload-best,
            # add-epochs, callback toggles). Cheap no-op when nothing queued.
            if self._control is not None:
                self._control.drain("epoch")

            if stop:
                print(f"\n[trainer] Early stopping at epoch {epoch}. "
                      f"Best {self.config.checkpoint.monitor_metric}: "
                      f"{self.early_stopper.best:.4f}")
                if self.early_stopper is not None:
                    self.early_stopper.restore(self.model)
                break

            epoch += 1

        # ── Post-training_harness ─────────────────────────────────────────────
        self._post_training()
        return self.tracker

    # ------------------------------------------------------------------
    # Epoch runner (shared between train and val)
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        epoch: int,
        phase: str,
        after_step_hook: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Run one epoch (training_harness or validation).

        Returns:
            Dict of metric_name → epoch average.
        """
        is_train = (phase == "train")
        loader   = self.train_loader if is_train else self.val_loader

        self.model.train(is_train)
        max_batches = self.config.dry_run_batches if self.config.dry_run else len(loader)

        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            inputs, targets = self._unpack_batch(batch)
            inputs  = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass (with optional AMP)
            with torch.set_grad_enabled(is_train):
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self._use_amp,
                ):
                    outputs = self.model(inputs)
                    loss    = self.loss_fn(outputs, targets)

            if is_train:
                # Backward + optimizer step
                self.optimizer.zero_grad(set_to_none=True)
                self._scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.grad_clip_norm is not None:
                    self._scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip_norm,
                    )

                self._scaler.step(self.optimizer)
                self._scaler.update()

                # Step-level LR scheduling. Skipped while suspended by a manual
                # LR override (a control command), so the override isn't
                # immediately clobbered by the scheduler.
                if self.scheduler is not None and not self._scheduler_suspended:
                    self.scheduler.step()
                    current_lr = self.scheduler.get_last_lr()[0]
                    self._lr_history.append(current_lr)

                self._global_step += 1

                # Apply queued step-boundary control commands (LR / hyperparams)
                # so they take effect within a step, not a whole epoch. Cheap
                # no-op when the queue is empty.
                if self._control is not None:
                    self._control.drain("step")

                # Step-level logging
                if self._global_step % self.config.logging.log_every_n_steps == 0:
                    self.logger.log_step(
                        step=self._global_step,
                        metrics={"loss": loss.item()},
                        phase="train",
                    )

                # User hook
                if after_step_hook is not None and self.callbacks.is_enabled("user_after_step"):
                    after_step_hook(self, loss.item(), outputs.detach(), targets)

            # Accumulate metrics
            batch_size = inputs.size(0)
            self.tracker.update_step(f"{phase}_loss", loss.item(), phase=phase, n=batch_size)

            acc = self.accuracy_fn(outputs.detach(), targets)
            self.tracker.update_step(f"{phase}_acc", acc, phase=phase, n=batch_size)

        snap = self.tracker.commit_epoch(epoch, phase=phase)
        return snap.metrics

    # ------------------------------------------------------------------
    # Post-training_harness
    # ------------------------------------------------------------------

    def _post_training(self) -> None:
        """Save final plots, load best weights, close logger."""
        print("\n[trainer] Training complete. Finalising …")

        if self._api_collector is not None:
            self._api_collector.mark_finished()

        if self.config.logging.save_plots:
            self.plotter.plot_all(self.tracker, lr_history=self._lr_history)

        # Restore best weights
        self.checkpoint_mgr.load_best(self.model, device=str(self.device))

        # Print run summary
        summary = self.tracker.summary()
        print("\n── Run Summary ──────────────────────────────────")
        for k, v in summary.items():
            print(f"  {k:<30} {v:.4f}" if isinstance(v, float) else f"  {k:<30} {v}")
        print("─────────────────────────────────────────────────\n")

        self.logger.close()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_batch(batch):
        """
        Unpack a DataLoader batch into (inputs, targets).

        Handles tuples, lists, and dicts with 'input'/'label' keys.
        """
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]
        if isinstance(batch, dict):
            inputs  = batch.get("input", batch.get("image", batch.get("x")))
            targets = batch.get("label", batch.get("target", batch.get("y")))
            if inputs is None or targets is None:
                raise ValueError(f"Cannot unpack batch dict with keys: {list(batch.keys())}")
            return inputs, targets
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _print_epoch_summary(
        self,
        epoch: int,
        elapsed: float,
        eta: str,
        metrics: Dict[str, float],
    ) -> None:
        parts = [f"Epoch {epoch:4d}/{self.config.epochs - 1}"]
        parts.append(f"  {elapsed:5.1f}s  ETA {eta}")
        for k, v in metrics.items():
            parts.append(f"  {k}: {v:.4f}")
        if self.qat_scheduler.in_float_warmup:
            parts.append("  [float warmup]")
        else:
            parts.append("  [QAT]")
        print("".join(parts))

    # ------------------------------------------------------------------
    # Quantization introspection
    # ------------------------------------------------------------------

    def _extract_quant_config(self) -> dict:
        """
        Walk the model and extract quantizer classes, bit-widths, and current scales.
        Useful for logging full quantization configuration alongside hyperparameters.
        """
        config = {}
        for name, module in self.model.named_modules():
            for attr in ("weight_quant", "input_quant", "output_quant", "act_quant"):
                proxy = getattr(module, attr, None)
                if proxy is None:
                    continue
                info = {"class": type(proxy).__name__}
                if hasattr(proxy, "bit_width"):
                    info["bit_width"] = proxy.bit_width
                if hasattr(proxy, "signed"):
                    info["signed"] = proxy.signed
                try:
                    scale = proxy.scale()
                    if scale is not None:
                        info["scale"] = float(scale.abs().mean().item())
                except Exception:
                    pass
                config[f"{name}.{attr}"] = info
        return config


# ---------------------------------------------------------------------------
# Default accuracy function
# ---------------------------------------------------------------------------

def _default_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy for classification tasks."""
    if outputs.ndim == 1 or outputs.shape[-1] == 1:
        # Binary classification
        preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()
    else:
        preds = outputs.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)
