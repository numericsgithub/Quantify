"""
trainer_v2.py — Corrected QAT training harness for the project's custom quantizers.

V1 bug: disable_quant(model) only toggles the Brevitas-level attribute, which the
custom quantizers don't have. They default to annealing_alpha=1.0, so they calibrate
and fully quantize on the very first forward pass of float warmup. V2 fixes this by
calling QuantizerManager().disable_quantization() (alpha=0, step=0) at the start,
ensuring truly clean float training before the gradual QAT cascade begins.

Correct protocol:
  1. Fully disable all quantization (custom + standard Brevitas layers).
  2. Float warmup — model converges in FP32, monitoring val_loss for plateau.
  3. Plateau detected → reset calibration buffers → set annealing + staggered gating.
  4. Gradual QAT: quantizers activate one-by-one, each annealing 0→1, model adapts.
  5. Continue until epoch budget is spent.
"""

from __future__ import annotations

import time
import warnings
from typing import Callable, Dict, List, Optional

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpointing import CheckpointManager
from .config_v2 import TrainerConfigV2
from .console import TrainingConsole
from .ema import EMAModel
from .logger import ExperimentLogger
from .metrics import MetricsTracker
from .plotting import TrainingPlotter
from .schedulers import collect_scale_factors, freeze_bn, _set_quant_enabled
from .engine_utils import BreakdownDetector, EarlyStopping, EpochTimer, LossPlateauDetector, log_hardware_info, set_seed
from quantizers.manager import QuantizerManager


class QATTrainerV2:
    """
    V2 QAT harness: correct protocol for the project's custom quantizers.

    Minimal usage::

        config = TrainerConfigV2(
            experiment_name="cifar10_qat",
            epochs=60,
            qat=QATScheduleConfigV2(float_warmup_epochs=10, plateau_patience=5),
        )
        trainer = QATTrainerV2(
            config=config,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=nn.CrossEntropyLoss(),
        )
        tracker = trainer.fit()

    Pre-trained model (skip float warmup)::

        config = TrainerConfigV2(
            qat=QATScheduleConfigV2(float_warmup_epochs=0, ...),
        )
    """

    def __init__(
        self,
        config: TrainerConfigV2,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
        scheduler=None,
        accuracy_fn: Optional[Callable] = None,
        onnx_dummy_input: Optional[torch.Tensor] = None,
        extra_checkpoint_fields: Optional[Dict] = None,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.scheduler = scheduler
        self.accuracy_fn = accuracy_fn or _default_accuracy

        self.device = torch.device(config.resolve_device())

        # ── Timm Mixup / CutMix ────────────────────────────────────────────
        # DALILoader already squeezes labels to [B] long, so no squeeze needed.
        self._mixup_fn = None
        if config.mixup > 0 or config.cutmix > 0:
            from timm.data import Mixup as TimmMixup
            self._mixup_fn = TimmMixup(
                mixup_alpha=config.mixup,
                cutmix_alpha=config.cutmix,
                prob=config.mixup_prob,
                switch_prob=config.mixup_switch_prob,
                label_smoothing=config.smoothing,
                num_classes=config.num_classes,
            )

        # ── Random Erasing ──────────────────────────────────────────────────
        # Applied after mixup on GPU-resident, already-normalised images.
        self._erasing_fn = None
        if config.reprob > 0:
            from timm.data.random_erasing import RandomErasing
            self._erasing_fn = RandomErasing(
                probability=config.reprob,
                mode='pixel',
                device='cuda',
            )

        # ── Training loss ───────────────────────────────────────────────────
        # When mixup is active the targets from TimmMixup are soft [B, C]
        # vectors, so we need SoftTargetCrossEntropy.  When only smoothing is
        # requested, fall back to LabelSmoothingCrossEntropy.  Otherwise use
        # the loss_fn passed from outside (plain CE) — this is the disabled
        # path and must behave identically to the pre-augmentation code.
        if self._mixup_fn is not None:
            from timm.loss import SoftTargetCrossEntropy
            self._train_loss_fn = SoftTargetCrossEntropy()
        elif config.smoothing > 0:
            from timm.loss import LabelSmoothingCrossEntropy
            self._train_loss_fn = LabelSmoothingCrossEntropy(smoothing=config.smoothing)
        else:
            self._train_loss_fn = loss_fn or nn.CrossEntropyLoss()

        if config.mixed_precision:
            warnings.warn(
                "Mixed precision (AMP) is enabled. "
                "Brevitas fake-quantization ops may interact poorly with autocast. "
                "Consider mixed_precision=False.",
                UserWarning,
            )

        import time as _time
        self.config.run_id = config.run_id or _time.strftime("%Y-%m-%d_%H%M%S")
        config.make_run_dirs()

        self.tracker = MetricsTracker()

        self.checkpoint_mgr = CheckpointManager(
            save_dir=config.checkpoint_dir,
            top_k=config.checkpoint.top_k,
            monitor_mode=config.checkpoint.monitor_mode,
            save_last=config.checkpoint.save_last,
            save_every_n_epochs=config.checkpoint.save_every_n_epochs,
            experiment_name=config.experiment_name,
        )

        self.logger = ExperimentLogger(
            experiment_name=config.experiment_name,
            run_id=config.run_id,
            log_dir=config.log_dir,
            use_tensorboard=config.logging.use_tensorboard,
            use_wandb=config.logging.use_wandb,
            wandb_project=config.logging.wandb_project,
            wandb_entity=config.logging.wandb_entity,
            csv_log=config.logging.csv_log,
        )

        self.plotter = TrainingPlotter(
            save_dir=config.plot_dir,
            experiment_name=config.experiment_name,
        )

        self.early_stopper: Optional[EarlyStopping] = None
        if config.early_stopping_patience is not None:
            self.early_stopper = EarlyStopping(
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                mode=config.checkpoint.monitor_mode,
            )

        self._plateau_lr_sched = None
        self._plateau_lr_metric = config.reduce_lr_metric
        if config.reduce_lr_on_plateau:
            mode = "max" if config.reduce_lr_metric in ("val_acc", "train_acc") else "min"
            self._plateau_lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                patience=config.reduce_lr_patience,
                factor=config.reduce_lr_factor,
                min_lr=config.reduce_lr_min_lr,
                threshold=config.reduce_lr_threshold,
            )

        self._ema: Optional[EMAModel] = None
        if config.ema_decay > 0:
            self._ema = EMAModel(model, decay=config.ema_decay)

        self._use_amp = config.mixed_precision and str(self.device).startswith("cuda")
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp)
        self._global_step: int = 0
        self._lr_history: List[float] = []
        self._qat_active: bool = False
        self._onnx_dummy_input = onnx_dummy_input
        self._extra_checkpoint_fields: Dict = extra_checkpoint_fields or {}

        # Optional read-only monitoring API (opt-in via config.api_port)
        self.api_server = None
        self._api_collector = None
        if config.api_port is not None:
            from .api import DashboardAPIServer, RunStateCollector
            self._api_collector = RunStateCollector(self)
            self.logger.add_listener(self._api_collector)
            self.api_server = DashboardAPIServer(
                self._api_collector, host=config.api_host, port=config.api_port
            )
            self.api_server.start()

    # ------------------------------------------------------------------
    # Pre-training / standalone evaluation
    # ------------------------------------------------------------------

    def evaluate(self, loader, label: str = "eval") -> Dict[str, float]:
        """
        Evaluate the model in eval mode on the full loader with all quantization
        disabled. Safe to call before fit(); fit() will re-initialise state.

        Returns a dict with ``{label}_loss`` and ``{label}_acc`` keys.
        """
        self.model.to(self.device)
        _reset_and_register(self.model)
        _fully_disable_quantization(self.model)

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            pbar = tqdm(loader, desc=f"  [{label}]", leave=False, dynamic_ncols=True)
            for batch in pbar:
                inputs, targets = _unpack_batch(batch)
                inputs  = inputs.to(self.device)
                targets = targets.to(self.device)
                with torch.autocast(device_type=self.device.type, enabled=self._use_amp):
                    outputs = self.model(inputs)
                    loss    = self.loss_fn(outputs, targets)
                preds = outputs.argmax(dim=-1)
                total_loss    += loss.item() * targets.size(0)
                total_correct += (preds == targets).sum().item()
                total_samples += targets.size(0)
            pbar.close()

        metrics = {
            f"{label}_loss": total_loss / total_samples,
            f"{label}_acc":  total_correct / total_samples,
        }
        print(
            f"  [{label}]  loss={metrics[f'{label}_loss']:.4f}"
            f"  acc={metrics[f'{label}_acc']:.4f}"
            f"  ({total_samples:,} samples)"
        )
        return metrics

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
        Run the full training loop.

        Args:
            resume:            Resume from the last checkpoint if available.
            after_step_hook:   Called after every optimizer step with
                               (trainer, loss_val, outputs, targets).
            after_epoch_hook:  Called after every epoch with
                               (trainer, epoch, metrics_snapshot).

        Returns:
            The MetricsTracker populated with all metrics for this run.
        """
        set_seed(self.config.seed, self.config.deterministic)
        self.model.to(self.device)
        if self._ema is not None:
            self._ema.to(self.device)
        if self._onnx_dummy_input is not None:
            self._onnx_dummy_input = self._onnx_dummy_input.to(self.device)
        log_hardware_info(self.logger)
        self.logger.log_hparams(self.config.to_dict())

        import os as _os
        abs_logs  = _os.path.abspath(self.config.log_dir)
        abs_ckpt  = _os.path.abspath(self.config.checkpoint_dir)
        abs_plots = _os.path.abspath(self.config.plot_dir)
        abs_diag  = _os.path.abspath(self.config.diagnostics_dir)

        print(f"\n{'═'*60}")
        print(f"  Experiment : {self.config.experiment_name}")
        print(f"  Run ID     : {self.config.run_id}")
        print(f"  Device     : {self.device}")
        print(f"  Epochs     : {self.config.epochs}")
        print(f"  Warmup     : {self.config.qat.float_warmup_epochs} epochs")
        print(f"  Gap/Anneal : {self.config.qat.quantization_start_gap} / {self.config.qat.annealing_steps}")
        if self._mixup_fn is not None:
            print(f"  Mixup      : α={self.config.mixup}  CutMix α={self.config.cutmix}"
                  f"  prob={self.config.mixup_prob}  switch={self.config.mixup_switch_prob}"
                  f"  smoothing={self.config.smoothing}")
            print(f"               train_acc is approximate (pre-mixup hard labels)")
        if self._erasing_fn is not None:
            print(f"  RandErase  : prob={self.config.reprob}  mode=pixel")
        if self.config.freeze_bn:
            print(f"  Freeze BN  : True  (running stats locked from checkpoint)")
        print(f"  ── Output paths ──────────────────────────────────────")
        print(f"  Logs       : {abs_logs}")
        print(f"  Checkpoints: {abs_ckpt}")
        print(f"  Plots      : {abs_plots}")
        print(f"  Diagnostics: {abs_diag}")
        print(f"{'═'*60}\n")

        # Reset singleton state from any prior run, then re-register this model's
        # custom quantizers so disable/annealing calls actually reach them.
        _reset_and_register(self.model)

        # Step 1: disable all quantization — critical fix vs V1
        _fully_disable_quantization(self.model)

        # If float_warmup_epochs=0 (pre-trained model), skip warmup and start QAT now
        if self.config.qat.float_warmup_epochs == 0:
            self._activate_qat()

        # Baseline evaluation before any training (useful when starting from pretrained weights)
        if self.val_loader is not None:
            tqdm.write("Evaluating pretrained model on validation set …")
            baseline = self._run_epoch(-1, "val")
            tqdm.write(
                f"  Baseline  val_loss={baseline.get('val_loss', float('nan')):.4f}"
                f"  val_acc={baseline.get('val_acc', float('nan')):.4f}\n"
            )

        plateau_detector = LossPlateauDetector(
            patience=self.config.qat.plateau_patience,
            min_delta=self.config.qat.plateau_min_delta,
        )

        breakdown_detector = (
            BreakdownDetector(
                num_classes=self.config.num_classes,
                relative_drop=self.config.breakdown_relative_drop,
                peak_min_factor=self.config.breakdown_peak_min_factor,
            )
            if self.config.breakdown_detection
            else None
        )

        console = TrainingConsole(self)
        console.start()

        start_epoch = 0
        if resume:
            start_epoch = self.checkpoint_mgr.resume(
                self.model, self.optimizer, self.scheduler, device=str(self.device)
            )

        end_epoch = start_epoch + self.config.epochs
        recovery_count = 0
        all_metrics: Dict[str, float] = {}

        while True:
            timer = EpochTimer(total_epochs=end_epoch - start_epoch)
            breakdown_occurred = False

            for epoch in range(start_epoch, end_epoch):
                timer.start()

                train_metrics = self._run_epoch(epoch, "train", after_step_hook=after_step_hook)

                val_metrics: Dict[str, float] = {}
                if self.val_loader is not None:
                    # Temporarily apply EMA parameters for validation so the
                    # checkpoint monitor metric reflects the averaged weights.
                    # Only parameters are swapped; buffers (BN stats, quant state)
                    # stay as-is so quantization inference remains correct.
                    _ema_stash = self._ema.apply_to(self.model) if self._ema is not None else None
                    val_metrics = self._run_epoch(epoch, "val")
                    if _ema_stash is not None:
                        self._ema.restore(self.model, _ema_stash)

                all_metrics = {**train_metrics, **val_metrics}
                if self._qat_active:
                    fully, total = self._quant_progress()
                    all_metrics["quant_pct"] = fully / total if total > 0 else 0.0

                if self._plateau_lr_sched is not None:
                    plateau_val = all_metrics.get(
                        self._plateau_lr_metric,
                        all_metrics.get("val_loss", all_metrics.get("train_loss", 0.0)),
                    )
                    self._plateau_lr_sched.step(plateau_val)
                    all_metrics["lr"] = self.optimizer.param_groups[0]["lr"]

                self.logger.log_epoch(epoch, all_metrics)

                monitor_val = all_metrics.get(
                    self.config.checkpoint.monitor_metric,
                    train_metrics.get("train_loss", 0.0),
                )

                # Plateau detection: only during float warmup.
                # Uses a dedicated loss metric (plateau_metric) — the detector
                # assumes a decreasing signal; accuracy would fire immediately.
                if not self._qat_active:
                    plateau_val = all_metrics.get(
                        self.config.qat.plateau_metric,
                        train_metrics.get("train_loss", 0.0),
                    )
                    past_warmup = epoch >= self.config.qat.float_warmup_epochs
                    if plateau_detector.step(plateau_val) or past_warmup:
                        self._activate_qat()

                # Scale factor tracking (only once QAT is live)
                if self._qat_active and self.config.qat.track_scale_factors:
                    scales = collect_scale_factors(self.model)
                    self.tracker.record_scale_factors(epoch, scales)
                    self.logger.log_scale_factors(epoch, scales)

                _ckpt_extra = {
                    **({"ema_state_dict": self._ema.state_dict()} if self._ema else {}),
                    **self._extra_checkpoint_fields,
                }
                self.checkpoint_mgr.save(
                    epoch=epoch,
                    metric_value=monitor_val,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics_dict=all_metrics,
                    config_dict=self.config.to_dict(),
                    extra=_ckpt_extra or None,
                    dummy_input=self._onnx_dummy_input,
                )

                # Breakdown detection: check for catastrophic accuracy collapse
                if breakdown_detector is not None and "val_acc" in all_metrics:
                    if breakdown_detector.step(all_metrics["val_acc"]):
                        breakdown_occurred = True

                # Early stopping: only after QAT is active AND every quantizer has
                # finished annealing (alpha=1.0). Stopping during the cascade would
                # cut training before the model has adapted to full quantization.
                stop = False
                if (
                    self.early_stopper is not None
                    and self._qat_active
                    and QuantizerManager().is_quantizing_everything_fully
                ):
                    stop = self.early_stopper.step(monitor_val, model=self.model, epoch=epoch)

                elapsed, eta = timer.stop(epoch)
                self._print_epoch_summary(epoch, elapsed, eta, all_metrics)

                if after_epoch_hook is not None:
                    snap = self.tracker.history[-1] if self.tracker.history else None
                    after_epoch_hook(self, epoch, snap)

                console.drain(epoch)

                if breakdown_occurred:
                    break

                if stop:
                    print(f"\n[trainer_v2] Early stopping at epoch {epoch}. "
                          f"Best {self.config.checkpoint.monitor_metric}: "
                          f"{self.early_stopper.best:.4f}")
                    if self.early_stopper is not None:
                        self.early_stopper.restore(self.model)
                    break

                if console.stop_requested:
                    print(f"\n[console] Stopped at epoch {epoch}.")
                    break

            # ── Outer recovery loop ────────────────────────────────────────
            if not breakdown_occurred:
                break  # training completed (or early-stopped / console-stopped) normally

            if recovery_count >= self.config.breakdown_max_recoveries:
                tqdm.write(
                    f"\n[breakdown] Max recoveries ({self.config.breakdown_max_recoveries}) "
                    f"reached — stopping."
                )
                break

            recovery_count += 1
            best_epoch = self._do_breakdown_recovery(
                breakdown_epoch=epoch,
                recovery_count=recovery_count,
                current_val_acc=all_metrics.get("val_acc", 0.0),
                peak_val_acc=breakdown_detector.peak_acc,
            )
            if best_epoch is None:
                break

            # Fresh epoch range: full budget from the restored checkpoint
            start_epoch = best_epoch + 1
            end_epoch = start_epoch + self.config.epochs

            # Fresh plateau detector (stale state would re-trigger QAT immediately
            # or never fire if it had already saturated)
            plateau_detector = LossPlateauDetector(
                patience=self.config.qat.plateau_patience,
                min_delta=self.config.qat.plateau_min_delta,
            )
            breakdown_detector.reset()

        console.stop()
        self._post_training()
        return self.tracker

    # ------------------------------------------------------------------
    # QAT activation
    # ------------------------------------------------------------------

    def _activate_qat(self) -> None:
        """
        Transition from float warmup to gradual QAT.

        1. Reset search_done so every quantizer re-calibrates against converged weights.
        2. Set annealing (0 → 1 over annealing_steps passes) for all quantizers.
        3. Set staggered gating: quantizer N waits N × gap passes before activating.
        4. Freeze BatchNorm statistics.
        """
        self._qat_active = True
        preserve = self.config.qat.preserve_calibrated_quantizers

        # Reset calibration buffers — forces fresh calibration with converged
        # weights. Skip quantizers that are already calibrated (search_done=True)
        # when preserve_calibrated_quantizers is set, e.g. because the model was
        # initialized from a PTQ checkpoint and its LSBs should be kept as-is.
        for m in self.model.modules():
            for name, buf in m.named_buffers():
                if "search_done" in name or "calibration_done" in name:
                    if preserve and buf.item():
                        continue
                    buf.fill_(False)

        # Re-enable Brevitas proxy layers (disabled during float warmup by
        # _set_quant_enabled). The custom quantizers' own gating/annealing then
        # controls *when* each one actually starts quantizing.
        _set_quant_enabled(self.model, enabled=True)

        mgr = QuantizerManager()
        mgr.set_annealing_for_n_inferences(
            self.config.qat.annealing_steps, skip_calibrated=preserve,
        )
        mgr.quantization_start_gap = self.config.qat.quantization_start_gap
        if preserve:
            mgr.skip_gating_for_calibrated_quantizers()
        mgr.diagnostics_dir = self.config.diagnostics_dir

        if self.config.qat.freeze_bn_at_qat:
            freeze_bn(self.model)

        n_quantizers = len(mgr.quantizers)
        print(
            f"\n[trainer_v2] QAT activated ✓  "
            f"({n_quantizers} quantizers, gap={self.config.qat.quantization_start_gap}, "
            f"anneal={self.config.qat.annealing_steps} passes)\n"
            f"  Quantizer diagnostics → {self.config.diagnostics_dir}"
        )

    # ------------------------------------------------------------------
    # Breakdown recovery
    # ------------------------------------------------------------------

    def _do_breakdown_recovery(
        self,
        breakdown_epoch: int,
        recovery_count: int,
        current_val_acc: float,
        peak_val_acc: float,
    ) -> Optional[int]:
        """
        Respond to a detected training breakdown.

        Steps:
          1. Log the event.
          2. Load the best checkpoint saved so far.
          3. Run a validation pass to confirm the loaded weights are healthy.
          4. Reduce the learning rate by breakdown_lr_factor.
          5. Reinitialize ReduceLROnPlateau (stale state would reduce LR again immediately).
          6. Restore QAT-active state from the checkpoint's metrics.

        Returns:
            The epoch number of the loaded checkpoint (caller sets start_epoch = best + 1),
            or None if recovery is not possible (no best checkpoint, or validation still bad).
        """
        tqdm.write(
            f"\n{'!' * 60}\n"
            f"  [breakdown] TRAINING BREAKDOWN at epoch {breakdown_epoch}\n"
            f"  Peak val_acc: {peak_val_acc:.4f}  →  collapsed to {current_val_acc:.4f}\n"
            f"  Recovery attempt {recovery_count}/{self.config.breakdown_max_recoveries}\n"
            f"{'!' * 60}\n"
        )

        payload = self.checkpoint_mgr.load_best(self.model, device=str(self.device))
        if payload is None:
            tqdm.write("[breakdown] No best checkpoint found — cannot recover.")
            return None

        best_epoch = payload.get("epoch", 0)
        best_metrics = payload.get("metrics", {})
        tqdm.write(
            f"[breakdown] Loaded best checkpoint from epoch {best_epoch}  "
            f"val_acc={best_metrics.get('val_acc', float('nan')):.4f}"
        )

        # Re-register quantizers with fresh runtime counters; alpha → 1.0 for all.
        # This is intentional: we want full quantization (if QAT was active) or
        # full disabling (if it wasn't), rather than replaying the annealing ramp.
        _reset_and_register(self.model)
        was_qat_active = best_metrics.get("quant_pct", 0.0) > 0.0
        if was_qat_active:
            _set_quant_enabled(self.model, enabled=True)
            mgr = QuantizerManager()
            mgr.diagnostics_dir = self.config.diagnostics_dir
            mgr.quantization_start_gap = self.config.qat.quantization_start_gap
            mgr.skip_gating_for_calibrated_quantizers()
            if self.config.qat.freeze_bn_at_qat:
                freeze_bn(self.model)
        else:
            _fully_disable_quantization(self.model)
        self._qat_active = was_qat_active

        # Validate the loaded checkpoint before committing to recovery
        tqdm.write("[breakdown] Validating loaded checkpoint …")
        _ema_stash = self._ema.apply_to(self.model) if self._ema is not None else None
        val_result = self._run_epoch(best_epoch, "val")
        if _ema_stash is not None:
            self._ema.restore(self.model, _ema_stash)
        recovered_acc = val_result.get("val_acc", 0.0)

        min_acceptable = self.config.breakdown_peak_min_factor / self.config.num_classes
        if recovered_acc < min_acceptable:
            tqdm.write(
                f"[breakdown] Best checkpoint acc={recovered_acc:.4f} is still below the "
                f"minimum threshold ({min_acceptable:.4f}) — recovery not possible."
            )
            return None

        tqdm.write(f"[breakdown] Checkpoint validated: val_acc={recovered_acc:.4f} ✓")

        # Reduce learning rate
        old_lr = self.optimizer.param_groups[0]["lr"]
        new_lr = old_lr * self.config.breakdown_lr_factor
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr
        tqdm.write(f"[breakdown] LR: {old_lr:.2e} → {new_lr:.2e}")

        # Reinitialize ReduceLROnPlateau so its stale internal patience counter
        # does not immediately reduce LR again on the first post-recovery epoch.
        if self._plateau_lr_sched is not None:
            mode = "max" if self._plateau_lr_metric in ("val_acc", "train_acc") else "min"
            self._plateau_lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                patience=self.config.reduce_lr_patience,
                factor=self.config.reduce_lr_factor,
                min_lr=self.config.reduce_lr_min_lr,
                threshold=self.config.reduce_lr_threshold,
            )

        tqdm.write(
            f"[breakdown] Recovery complete. "
            f"Restarting from epoch {best_epoch + 1} "
            f"with a fresh {self.config.epochs}-epoch budget.\n"
        )
        return best_epoch

    # ------------------------------------------------------------------
    # Epoch runner
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        epoch: int,
        phase: str,
        after_step_hook: Optional[Callable] = None,
    ) -> Dict[str, float]:
        is_train = (phase == "train")
        loader = self.train_loader if is_train else self.val_loader

        self.model.train(is_train)
        # model.train() is recursive and would re-enable BN stats accumulation.
        # Re-apply freeze after the call so BN uses checkpoint running stats.
        if is_train and (self.config.freeze_bn or
                         (self._qat_active and self.config.qat.freeze_bn_at_qat)):
            freeze_bn(self.model)
        max_batches = self.config.dry_run_batches if self.config.dry_run else len(loader)

        running_loss = 0.0
        running_acc  = 0.0

        pbar = tqdm(
            loader,
            total=max_batches,
            desc=f"Epoch {epoch:4d} [{phase:>5}]",
            leave=False,
            dynamic_ncols=True,
        )

        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= max_batches:
                break

            inputs, targets = _unpack_batch(batch)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Keep hard integer labels for accuracy; mixup will overwrite targets
            hard_targets = targets

            # MixUp / CutMix — training only; produces soft [B, num_classes] targets
            if is_train and self._mixup_fn is not None:
                inputs, targets = self._mixup_fn(inputs, hard_targets)

            # Random Erasing — training only, after mixup, on normalised GPU images
            if is_train and self._erasing_fn is not None:
                inputs = self._erasing_fn(inputs)

            loss_fn = self._train_loss_fn if is_train else self.loss_fn
            with torch.set_grad_enabled(is_train):
                with torch.autocast(device_type=self.device.type, enabled=self._use_amp):
                    outputs = self.model(inputs)
                    loss = loss_fn(outputs, targets)

            if is_train:
                self.optimizer.zero_grad(set_to_none=True)
                self._scaler.scale(loss).backward()

                if self.config.grad_clip_norm is not None:
                    self._scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )

                self._scaler.step(self.optimizer)
                self._scaler.update()

                if self._ema is not None:
                    self._ema.update(self.model)

                if self.scheduler is not None:
                    self.scheduler.step()
                    self._lr_history.append(self.scheduler.get_last_lr()[0])

                self._global_step += 1

                if self._global_step % self.config.logging.log_every_n_steps == 0:
                    self.logger.log_step(
                        step=self._global_step,
                        metrics={"loss": loss.item()},
                        phase="train",
                    )

                if after_step_hook is not None:
                    after_step_hook(self, loss.item(), outputs.detach(), targets)

            batch_size = inputs.size(0)
            self.tracker.update_step(f"{phase}_loss", loss.item(), phase=phase, n=batch_size)
            # Accuracy is always against hard integer labels.  When mixup is
            # active during training, hard_targets are the pre-mix class indices
            # (approximate, since the model saw a blended image).
            acc = self.accuracy_fn(outputs.detach(), hard_targets)
            self.tracker.update_step(f"{phase}_acc", acc, phase=phase, n=batch_size)

            # Running averages in the progress bar postfix
            n = batch_idx + 1
            running_loss = running_loss + (loss.item() - running_loss) / n
            running_acc  = running_acc  + (acc         - running_acc)  / n
            postfix = {"loss": f"{running_loss:.4f}", "acc": f"{running_acc:.3f}"}
            if self._qat_active:
                fully, total = self._quant_progress()
                pct = int(fully / total * 100) if total > 0 else 0
                postfix["quant"] = f"{pct}% ({fully}/{total})"
            pbar.set_postfix(**postfix)

        pbar.close()
        snap = self.tracker.commit_epoch(epoch, phase=phase)
        return snap.metrics

    # ------------------------------------------------------------------
    # Post-training
    # ------------------------------------------------------------------

    def _post_training(self) -> None:
        print("\n[trainer_v2] Training complete. Finalising …")

        if self._api_collector is not None:
            self._api_collector.mark_finished()

        if self.config.logging.save_plots:
            self.plotter.plot_all(self.tracker, lr_history=self._lr_history)

        payload = self.checkpoint_mgr.load_best(self.model, device=str(self.device))

        # If EMA was active, the checkpoint metric was evaluated with EMA params.
        # Restore those params into both the EMA shadow and the main model so the
        # caller gets the same weights that produced the best validation score.
        if self._ema is not None and payload is not None:
            ema_sd = (payload.get("extra") or {}).get("ema_state_dict")
            if ema_sd is not None:
                self._ema.load_state_dict(ema_sd, strict=False)
                for p, sp in zip(self.model.parameters(), self._ema._shadow.parameters()):
                    p.data.copy_(sp.data)
                print("[trainer_v2] EMA parameters applied to model from best checkpoint.")

        summary = self.tracker.summary()
        print("\n── Run Summary ──────────────────────────────────")
        for k, v in summary.items():
            print(f"  {k:<30} {v:.4f}" if isinstance(v, float) else f"  {k:<30} {v}")
        print("─────────────────────────────────────────────────\n")

        self.logger.close()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _quant_progress(self) -> tuple[int, int]:
        """Return (n_fully_quantized, n_total) from the active QuantizerManager."""
        mgr = QuantizerManager()
        total  = len(mgr.quantizers)
        fully  = sum(1 for q in mgr.quantizers.values() if q.annealing_alpha >= 1.0)
        return fully, total

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
            fmt = f"{v:.2e}" if (isinstance(v, float) and 0 < abs(v) < 1e-3) else f"{v:.4f}"
            parts.append(f"  {k}: {fmt}")

        if not self._qat_active:
            parts.append("  [float]")
        else:
            fully, total = self._quant_progress()
            pct = int(fully / total * 100) if total > 0 else 0
            parts.append(f"  [QAT {pct}% ({fully}/{total}) fully quantized]")

        print("".join(parts))


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

# Maps the Brevitas proxy suffix in a named_modules() path to an explicit
# role tag that appears in the final quant_id.  Checked in order so longer
# (more specific) suffixes win.
_PROXY_SUFFIX_TO_ROLE: list[tuple[str, str]] = [
    (".weight_quant.tensor_quant",                              "_weight"),
    (".bias_quant.tensor_quant",                                "_bias"),
    (".input_quant.tensor_quant",                               "_act_in"),
    (".output_quant.tensor_quant",                              "_act_out"),
    (".act_quant.fused_activation_quant_proxy.tensor_quant",    "_act"),
    (".act_quant.tensor_quant",                                 "_act"),
    # proxies without a nested tensor_quant (non-standard direct attachment)
    (".weight_quant",                                           "_weight"),
    (".bias_quant",                                             "_bias"),
    (".input_quant",                                            "_act_in"),
    (".output_quant",                                           "_act_out"),
    (".act_quant",                                              "_act"),
]


def _make_quant_id(path: str) -> str:
    """
    Convert a named_modules() dotted path into a descriptive quant_id.

    Strips the Brevitas proxy/tensor_quant suffix and appends an explicit
    role tag so logs and plot file names are immediately readable:
      features.3.conv.0.weight_quant.tensor_quant  →  features_3_conv_0_weight
      features.3.conv.0.bias_quant.tensor_quant    →  features_3_conv_0_bias
      features.3.conv.0.input_quant.tensor_quant   →  features_3_conv_0_act_in
      features.3.conv.0.output_quant.tensor_quant  →  features_3_conv_0_act_out
    """
    for suffix, role in _PROXY_SUFFIX_TO_ROLE:
        if path.endswith(suffix):
            parent = path[: -len(suffix)]
            base = parent.replace(".", "_") if parent else "root"
            return f"{base}{role}"
    # Fallback: non-standard path — clean dots to underscores
    return path.replace(".", "_")


def _reset_and_register(model: nn.Module) -> None:
    """
    Clear the QuantizerManager singleton state from any prior run, then
    re-register all BaseQuantizer instances found in this model.

    Calling manager.reset() alone is not enough: it wipes the quantizer
    registry, so subsequent disable/annealing calls would iterate an empty
    dict and silently do nothing. Re-registering restores the link between
    the manager and the model's quantizers.
    """
    from quantizers.base_quantizer import BaseQuantizer as _BaseQ
    mgr = QuantizerManager()
    mgr.reset()
    for module in model.modules():
        if isinstance(module, _BaseQ):
            mgr.register_quantizer(module)
            # Reset per-quantizer run state so sequence IDs are fresh
            module.inference_counter = 0
            module.inference_sequence_id = -1
            module.annealing_alpha.data.fill_(1.0)
            module.annealing_alpha_step = 0.1

    # Assign descriptive location-based names.
    # _make_quant_id strips Brevitas proxy noise and appends an explicit role
    # tag (_weight / _bias / _act_in / _act_out / _act).
    seen: dict[str, str] = {}  # qid → original path, for collision diagnostics
    for path, module in model.named_modules():
        if isinstance(module, _BaseQ):
            qid = _make_quant_id(path)
            if qid in seen:
                raise RuntimeError(
                    f"Duplicate quant_id {qid!r} produced from paths "
                    f"{seen[qid]!r} and {path!r}. Please report this as a bug."
                )
            seen[qid] = path
            module.quant_id = qid

    # Keep the manager's registry keys in sync with the descriptive names.
    mgr.quantizers = {q.quant_id: q for q in mgr.quantizers.values()}


def _fully_disable_quantization(model: nn.Module) -> None:
    """
    Disable all quantization for the start of float warmup.

    Covers both the project's custom quantizers (via QuantizerManager) and
    any standard Brevitas layers (via the disable_quant attribute toggle).
    """
    QuantizerManager().disable_quantization()   # custom quantizers: alpha=0, step=0
    _set_quant_enabled(model, enabled=False)     # standard Brevitas layers


def _unpack_batch(batch):
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    if isinstance(batch, dict):
        inputs = batch.get("input", batch.get("image", batch.get("x")))
        targets = batch.get("label", batch.get("target", batch.get("y")))
        if inputs is None or targets is None:
            raise ValueError(f"Cannot unpack batch dict with keys: {list(batch.keys())}")
        return inputs, targets
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def _default_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    if outputs.ndim == 1 or outputs.shape[-1] == 1:
        preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()
    else:
        preds = outputs.argmax(dim=-1)
    return (preds == targets).sum().item() / targets.size(0)
