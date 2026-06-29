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
from .logger import ExperimentLogger
from .metrics import MetricsTracker
from .plotting import TrainingPlotter
from .schedulers import collect_scale_factors, freeze_bn, _set_quant_enabled
from .engine_utils import EarlyStopping, EpochTimer, LossPlateauDetector, log_hardware_info, set_seed
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
        if config.reduce_lr_on_plateau:
            self._plateau_lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=config.reduce_lr_patience,
                factor=config.reduce_lr_factor,
                min_lr=config.reduce_lr_min_lr,
                threshold=config.reduce_lr_threshold,
            )

        self._use_amp = config.mixed_precision and str(self.device).startswith("cuda")
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp)
        self._global_step: int = 0
        self._lr_history: List[float] = []
        self._qat_active: bool = False
        self._onnx_dummy_input = onnx_dummy_input

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

        start_epoch = 0
        if resume:
            start_epoch = self.checkpoint_mgr.resume(
                self.model, self.optimizer, self.scheduler, device=str(self.device)
            )

        timer = EpochTimer(total_epochs=self.config.epochs)

        for epoch in range(start_epoch, self.config.epochs):
            timer.start()

            train_metrics = self._run_epoch(epoch, "train", after_step_hook=after_step_hook)

            val_metrics: Dict[str, float] = {}
            if self.val_loader is not None:
                val_metrics = self._run_epoch(epoch, "val")

            all_metrics = {**train_metrics, **val_metrics}
            if self._qat_active:
                fully, total = self._quant_progress()
                all_metrics["quant_pct"] = fully / total if total > 0 else 0.0

            if self._plateau_lr_sched is not None:
                plateau_loss = all_metrics.get("val_loss", all_metrics.get("train_loss", 0.0))
                self._plateau_lr_sched.step(plateau_loss)
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

            self.checkpoint_mgr.save(
                epoch=epoch,
                metric_value=monitor_val,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                metrics_dict=all_metrics,
                config_dict=self.config.to_dict(),
                dummy_input=self._onnx_dummy_input,
            )

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

            if stop:
                print(f"\n[trainer_v2] Early stopping at epoch {epoch}. "
                      f"Best {self.config.checkpoint.monitor_metric}: "
                      f"{self.early_stopper.best:.4f}")
                if self.early_stopper is not None:
                    self.early_stopper.restore(self.model)
                break

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

            with torch.set_grad_enabled(is_train):
                with torch.autocast(device_type=self.device.type, enabled=self._use_amp):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)

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
            acc = self.accuracy_fn(outputs.detach(), targets)
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

        if self.config.logging.save_plots:
            self.plotter.plot_all(self.tracker, lr_history=self._lr_history)

        self.checkpoint_mgr.load_best(self.model, device=str(self.device))

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
            parts.append(f"  {k}: {v:.4f}")

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
