# Training Harness Pitfalls & Best Practices

## 1. Manually Implementing the Training Loop Instead of Using `QATTrainer`
**When this happens:** You write a custom `for epoch in range(epochs):` loop for a Brevitas QAT model, manually handling batches, metrics, logging, and quantization state transitions.
**The Problem:** The `training_harness.QATTrainer` class is specifically designed to manage QAT-specific workflows that are easy to get wrong manually:
- **Float Warmup & QAT activation:** The harness fully disables the custom quantizers during float warmup (Brevitas' own `disable_quant` toggle does not reach them), then resets calibration and switches them on gradually. Manual loops often start quantization too early, causing poor accuracy.
- **Metric Naming & Tracking:** The harness prefixes metrics with phases (`train_loss`, `val_acc`) and handles epoch commits correctly. Manual implementations often log as `loss` or `acc`, breaking downstream plotting, checkpointing, and early stopping logic.
- **Quantization Gating & Annealing:** The harness coordinates `QuantizerManager` to gradually enable quantization and anneal scales. Manual loops frequently disable this, leading to unstable gradients or incorrect ONNX export states.
**How to Prevent It:**
- Always use `training_harness.QATTrainer` for QAT experiments.
- Configure it with a `TrainerConfig` whose `qat=QATScheduleConfig(...)` sets `float_warmup_epochs`, `annealing_steps` and `quantization_start_gap`.
- Pass your model, optimizer, and data loaders directly to `QATTrainer(config=..., model=..., optimizer=..., train_loader=..., val_loader=...)`.
- Let the harness handle logging, checkpointing, and metric tracking. Only override `QATTrainer` methods if you have highly specific requirements.

## 2. `freeze_bn()` is undone by `model.train()` — must re-apply every epoch

**When this happens:** You call `freeze_bn(model)` once (e.g. in `_activate_qat()`) and expect BN to stay frozen for the entire training run.

**The Problem:** `_run_epoch()` calls `self.model.train(is_train)` at the very start of each epoch. PyTorch's `Module.train()` is **recursive** — it calls `.train()` on every submodule, setting every BN layer back to training mode. The single `freeze_bn()` call is fully undone by the next epoch's `model.train(True)`. This means `freeze_bn_at_qat=True` was silently not working during training epochs.

**Consequence:** BN running statistics are updated throughout training. When fine-tuning at very low LR with heavy augmentation (MixUp, CutMix, RandAugment), the batch statistics are noisier than the clean-data distributions used to build the original running stats. After a full epoch of augmented batches, the running stats drift, and val_acc regresses slightly even though the weights barely changed.

**The Fix:** Re-apply `freeze_bn()` **after** `model.train(True)` at the start of every training epoch. In `_run_epoch()`:
```python
self.model.train(is_train)
if is_train and (self.config.freeze_bn or
                 (self._qat_active and self.config.qat.freeze_bn_at_qat)):
    freeze_bn(self.model)
```

**How to enable for fine-tuning:** Pass `freeze_bn=True` to `TrainerConfig`, or use `--freeze-bn` in `train_imagenet_float.py` (archived on the `old_examples` branch). Always use this when:
- Fine-tuning a converged checkpoint at LR ≤ 1e-5
- Heavy augmentation is active (mixup, cutmix, reprob all on by default in the float script)
- You expect val_acc to improve, not drift downward
