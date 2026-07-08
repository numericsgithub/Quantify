# Training Harness Pitfalls & Best Practices

## 1. Manually Implementing the Training Loop Instead of Using `Trainer`
**When this happens:** You write a custom `for epoch in range(epochs):` loop for a Brevitas QAT model, manually handling batches, metrics, logging, and quantization state transitions.
**The Problem:** The `training_harness.Trainer` class is specifically designed to manage QAT-specific workflows that are easy to get wrong manually:
- **Float Warmup & Calibration:** The harness automatically transitions from float training to calibration, then to full QAT. Manual loops often skip calibration or start quantization too early, causing poor accuracy.
- **Metric Naming & Tracking:** The harness prefixes metrics with phases (`train_loss`, `val_acc`) and handles epoch commits correctly. Manual implementations often log as `loss` or `acc`, breaking downstream plotting, checkpointing, and early stopping logic.
- **Quantization Gating & Annealing:** The harness coordinates `QuantizerManager` to gradually enable quantization and anneal scales. Manual loops frequently disable this, leading to unstable gradients or incorrect ONNX export states.
**How to Prevent It:**
- Always use `training_harness.Trainer` for QAT experiments.
- Configure it with a `TrainerConfig` that specifies `float_warmup_epochs` and `calibration_batches`.
- Pass your model, optimizer, and data loaders directly to `Trainer(config=..., model=..., optimizer=..., train_loader=..., val_loader=...)`.
- Let the harness handle logging, checkpointing, and metric tracking. Only override `Trainer` methods if you have highly specific requirements.

## 2. UnicodeEncodeError on Windows (cp1252) from Box-Drawing Characters
**When this happens:** Running the harness on Windows, especially with stdout redirected to a file (e.g. background runs, CI). Crashes with `UnicodeEncodeError: 'charmap' codec can't encode characters` the moment `log_hardware_info()` or an epoch banner prints.
**The Problem:** The harness prints banners with box-drawing characters (`─`, `═`). Windows defaults to the cp1252 locale encoding for redirected stdout and for `open()` without an explicit `encoding=`, and cp1252 cannot represent those characters. `ExperimentLogger.log_text` writes with `encoding="utf-8"` explicitly, but `print()` output still goes through the console encoding.
**How to Prevent It:** Run Python in UTF-8 mode on Windows: set `PYTHONUTF8=1` (or pass `-X utf8`). When adding new file writes to the harness, always pass `encoding="utf-8"` to `open()`.
