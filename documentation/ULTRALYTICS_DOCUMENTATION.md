# Ultralytics Framework Documentation — Brevitas Integration Guide

> **Scope:** This document covers the **Ultralytics side** of integrating Brevitas-based Quantization-Aware Training (QAT) into the YOLO training infrastructure. Brevitas-side details (quantizer definitions, `ExtendedInjector`, QCDQ export) are assumed known.
> **Verified against:** `ultralytics` source as of late 2025 / early 2026 (YOLO11/YOLO26 era). Some APIs are evolving — verify against the version you're using.

---

## 1. Core Concepts & Architecture

- **`YOLO` high-level API** — Unified wrapper that lazily resolves a task-specific `Trainer`, `Validator`, `Predictor`, and `Model` class via `task_map`. For detection: `DetectionModel`, `DetectionTrainer`, `DetectionValidator`, `DetectionPredictor`.
- **`DetectionModel`** (`ultralytics/nn/tasks.py`) — A `BaseModel`/`nn.Module` whose `self.model` is a `nn.Sequential` produced by `parse_model()` from a YAML config. The forward pass returns either a loss (training) or predictions (inference) depending on whether a batch dict is passed in.
- **`BaseTrainer` → `DetectionTrainer`** (`ultralytics/engine/trainer.py`) — Owns the training loop, validation, EMA, checkpointing, AMP, DDP setup, optimizer/scheduler construction, and callback dispatch. **The primary extension point.**
- **Callback system** (`ultralytics/utils/callbacks/base.py`) — A dict of event-name → list-of-callables, dispatched at fixed lifecycle points. Receives the `Trainer`/`Validator`/`Predictor` instance.
- **Loss** — Detection uses `v8DetectionLoss` (or `E2ELoss` wrapping it for end-to-end models). Computed via `model.loss(batch, preds)` returning `(loss_sum, loss_items_tensor)` where `loss_items_tensor` feeds the progress bar and CSV logger.
- **`parse_model()`** — The function that walks the YAML `backbone` + `head` lists and instantiates each layer. Module name resolution is a **three-tier lookup**: names starting with `nn.` go to `torch.nn`, names with `ops.` go to `torchvision.ops`, everything else is looked up in `parse_model`'s **module-level `globals()`** in `ultralytics.nn.tasks`.

---

## 2. Three Levels of Customization (pick the lightest one that works)

Ultralytics offers a layered customization model. Use the lightest tool for the job:

| Level | When to use | Mechanism |
|---|---|---|
| **Callbacks** | Inject logic at fixed lifecycle points (calibrate, log, save extra artifacts) without touching the loop | `model.add_callback("on_train_epoch_start", fn)` |
| **Custom Trainer subclass** | Replace the model factory, optimizer, validator, or save logic | Subclass `DetectionTrainer`, pass `trainer=MyTrainer` to `model.train()` |
| **Custom YAML + module registration** | Swap `Conv`/`Linear` for `qnn.QuantConv2d`/`qnn.QuantLinear` at architecture-build time | YAML referencing custom module names; modules registered into `ultralytics.nn.tasks` globals |

For Brevitas QAT specifically, you'll typically need **all three**: a custom YAML (or a post-build wrapper) to insert quantized layers, a callback or trainer override to drive Brevitas calibration, and `amp=False` plus possibly `optimizer="SGD"` overrides via trainer args.

---

## 3. Callback Injection

```python
from ultralytics import YOLO

def on_train_start(trainer):
    # e.g., enable Brevitas QAT mode, freeze quantizer ranges, log scales
    apply_brevitas_qat(trainer.model)

def on_train_epoch_end(trainer):
    log_quantizer_scales(trainer.model, epoch=trainer.epoch)

model = YOLO("yolo11n.pt")
model.add_callback("on_train_start", on_train_start)
model.add_callback("on_train_epoch_end", on_train_epoch_end)
model.train(data="coco8.yaml", epochs=10, amp=False)
```

**Full event list** (from `ultralytics/utils/callbacks/base.py`):

- `on_pretrain_routine_start` / `on_pretrain_routine_end` — Before/after model is built and moved to device.
- `on_train_start` — Just before the epoch loop.
- `on_train_epoch_start` / `on_train_epoch_end`
- `on_train_batch_start` / `on_train_batch_end`
- `on_before_zero_grad`, `optimizer_step`
- `on_fit_epoch_end` — **After validation** (use this for metrics that need val results).
- `on_model_save` — Each checkpoint save.
- `on_train_end`, `teardown`
- Validation/predict/export events also exist.

**Timing gotchas:**
- `on_pretrain_routine_end` fires **after** `_setup_train()` — meaning the model has already been wrapped in DDP and EMA has been initialized. If you swap layers here, you must also re-wrap. **Best practice:** insert Brevitas layers in `get_model()` (trainer override) so EMA/DDP wrap the already-quantized model.
- `on_train_epoch_end` fires **before** validation. `on_fit_epoch_end` fires **after**. Use the latter for anything that depends on val metrics.

---

## 4. Custom Trainer

```python
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

class BrevitasDetectionModel(DetectionModel):
    """DetectionModel that swaps Conv/Linear for Brevitas equivalents after build."""
    def __init__(self, cfg, ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        # Walk self.model and replace nn.Conv2d → qnn.QuantConv2d, etc.
        replace_with_brevitas(self.model)

class BrevitasTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = BrevitasDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def build_optimizer(self, model, name="auto", lr=0.01, momentum=0.9, decay=1e-5, iterations=1e5):
        # If you need separate param groups for quantizer scales vs weights, do it here.
        return super().build_optimizer(model, name, lr, momentum, decay, iterations)

YOLO("yolo11n.pt").train(
    trainer=BrevitasTrainer,
    data="coco8.yaml",
    amp=False,           # see §7
    epochs=20,
)
```

**Key overridable methods** (from `BaseTrainer`):

- `get_model(cfg, weights, verbose)` — Build & return the `nn.Module`. **The cleanest insertion point for layer swapping.**
- `build_optimizer(model, name, lr, momentum, decay, iterations)` — Returns the optimizer. Override to put Brevitas scale parameters in a separate group with different LR/no weight decay.
- `get_validator()` — Return a custom validator. Useful if you need to evaluate the *quantized* model (with `eval()` + frozen scales) rather than the QAT-mode model.
- `save_model()` — Override to also dump the dequantized state, ONNX QCDQ, or quantizer config alongside `last.pt`/`best.pt`.
- `preprocess_batch(batch)` — Modify inputs before forward (rarely needed for QAT).

**Pass the class, not an instance.** `model.train(trainer=BrevitasTrainer)` — Ultralytics instantiates it internally so it can attach the right `args`, `data`, and `hub_session`.

---

## 5. Custom Module Registration (the YAML path)

If you want quantization expressed in the architecture YAML rather than as a post-build walk:

### Option A — Editable install + source edit (most robust)

1. `git clone https://github.com/ultralytics/ultralytics && pip install -e .`
2. In `ultralytics/nn/modules/conv.py` (or a new file), define `QuantConv` wrapping `qnn.QuantConv2d`.
3. Re-export it from `ultralytics/nn/modules/__init__.py`.
4. Import it in `ultralytics/nn/tasks.py` so it lands in `parse_model`'s globals.
5. If your module needs special argument handling (e.g., channel inference), add a branch in `parse_model()`'s big `if/elif` chain.
6. Reference it by name in YAML: `[-1, 1, QuantConv, [64, 3, 2, 8]]` (last arg = bit width).

### Option B — Runtime monkey-patch (no source edit)

```python
import ultralytics.nn.tasks as tasks
from my_quant_modules import QuantConv, QuantC2f

# Inject into the module where parse_model() does globals() lookup
tasks.QuantConv = QuantConv
tasks.QuantC2f = QuantC2f

# Now YAML referencing 'QuantConv' will resolve correctly
model = YOLO("yolo11n_quant.yaml")
```

This works because `parse_model()` resolves names via its own module's `globals()` — and Python module dicts are mutable.

### Option C — In-YAML `module: init:` block (newer feature)

Recent Ultralytics versions support inline module definitions inside the YAML itself via a `module: init: |` block that's `exec`'d into globals at parse time. Useful for self-contained configs but messy for complex Brevitas injectors.

---

## 6. Quantization Strategy: PTQ vs QAT in Ultralytics

### Built-in PTQ (export-time, **not** Brevitas)

Ultralytics has built-in INT8 PTQ for **specific export targets**, *not* as a generic ONNX option:

```python
model.export(format="engine", int8=True, data="coco.yaml")     # TensorRT
model.export(format="openvino", int8=True, data="coco.yaml")   # OpenVINO NNCF
model.export(format="tflite", int8=True, data="coco.yaml")     # TFLite
model.export(format="coreml", int8=True)                       # CoreML
```

`int8=True` for plain `format="onnx"` is **not supported** — for ONNX QAT you go through Brevitas + QCDQ.

### QAT with Brevitas (the integration you're building)

There is no first-class Brevitas integration. The community pattern is:

1. **Build** the FP32 model normally via `YOLO(...)` and load pretrained weights.
2. **Swap** `Conv2d`/`Linear`/activations for Brevitas equivalents (either via custom YAML or a `replace_modules()` walk in `get_model()`).
3. **Calibrate** activation ranges with Brevitas' `calibration_mode` over a representative subset *before* QAT begins (use `on_train_start` callback or call manually before `model.train()`).
4. **Train** with `amp=False`, lower LR (typically 0.01–0.1× of FP32 fine-tune LR), shorter schedule (5–20 epochs).
5. **Export** to ONNX QCDQ via Brevitas' `export_onnx_qcdq()`. **Do not** use `model.export(format="onnx")` — it goes through Ultralytics' `torch2onnx` which won't preserve Brevitas-specific graph structure.

---

## 7. AMP & Brevitas

Ultralytics enables AMP (`torch.amp.autocast` + `GradScaler`) by default via `amp=True`. The Ultralytics trainer also runs an "AMP check" at startup that does a forward pass and disables AMP automatically if NaN/inf is detected.

**Recommendation:** Pass `amp=False` for QAT runs. Reasons:
- Brevitas fake-quant ops perform internal arithmetic that can interact badly with `autocast`'s dtype promotion.
- The Ultralytics AMP check itself can fail silently or noisily on a freshly-quantized model with uninitialized scales.
- Quantization noise + AMP gradient scaling has been reported (in PyTorch's native QAT and in Brevitas community threads) to cause unstable scale learning.

If you need AMP for memory reasons: wrap Brevitas `forward` calls in `with torch.cuda.amp.autocast(enabled=False):` and ensure quantizer params stay in `float32`. This is fiddly — `amp=False` is almost always the right call for QAT.

---

## 8. EMA & Quantization

`ModelEMA` (in `ultralytics/utils/torch_utils.py`) maintains a shadow copy of model parameters via `EMA = decay * EMA + (1 - decay) * weights` after every optimizer step. **It copies all `state_dict` floats by default**, including Brevitas-learned scales and zero-points.

**Implications:**
- The EMA model is what gets saved as `best.pt`. If EMA averages quantizer scales over many batches, the scales in `best.pt` may be smoother and more stable than the raw model's — generally a good thing.
- **However:** if you're doing a brief QAT fine-tune (e.g., 5 epochs) starting from already-calibrated scales, EMA may drag scales away from their calibrated values. Consider either:
  - Setting `ModelEMA.enabled = False` early in training (via `on_train_start`), or
  - Excluding quantizer parameters from EMA by patching `ModelEMA.update()` to skip names matching Brevitas scale param patterns.
- The EMA model is also the one used for validation. If your raw model and EMA model diverge in quantization behavior, val mAP becomes hard to interpret. Consider adding a callback that compares both.

---

## 9. DDP & Quantization

Ultralytics supports DDP via `device=[0,1,...]`. Internally, it generates a temporary launcher script and runs `torch.distributed.run`.

**Custom trainers + DDP:** The auto-launcher serializes the trainer class via its import path. **A custom trainer defined in a notebook or `__main__` will not be picklable.** Per Ultralytics docs:

> "If your script contains custom components — such as a custom trainer, validator, dataset, or augmentation pipeline — these objects cannot be automatically serialized and transferred to the DDP subprocesses. In this case, you must launch your script directly with `torch.distributed.run`:"
> ```bash
> python -m torch.distributed.run --nproc_per_node 2 your_training_script.py
> ```

**Quantization-specific DDP concerns:**
- Brevitas fake-quant scales are normal `nn.Parameter`s and are **synced by DDP's gradient all-reduce** like any other learnable parameter. No special handling needed for *learned* scales.
- **Statistics-collection scales** (e.g., a quantizer in calibration mode collecting min/max from activations) are buffers, not parameters — DDP **does not** sync these by default. You'd need `broadcast_buffers=True` (the DDP default) and to ensure all ranks see the same calibration data, or do calibration on rank 0 and broadcast manually.
- Run calibration **before** DDP wrapping. Easiest: do PTQ-style calibration in a single-GPU pass, save the calibrated state, then start DDP QAT from that checkpoint.

---

## 10. Export & Deployment

For **Brevitas QAT models**, prefer Brevitas' native exporters over Ultralytics' `model.export()`:

- **QCDQ ONNX** (`brevitas.export.onnx.standard.qcdq.export_onnx_qcdq`) — Standard `QuantizeLinear`/`DequantizeLinear` nodes. Compatible with ONNX Runtime, TensorRT (post-conversion), and most INT8 inference stacks.
- **QONNX** (`brevitas.export.onnx.qonnx`) — Brevitas' richer format preserving sub-byte and non-uniform quantization. Needed if you're targeting FINN/hardware codegen.

**Bypassing Ultralytics export:**

```python
# Get the underlying nn.Module, NOT the YOLO wrapper
torch_model = trained_yolo.model.eval()
torch_model = torch_model.cpu()  # or wherever your export expects

# Brevitas export
from brevitas.export import export_onnx_qcdq
export_onnx_qcdq(torch_model, args=dummy_input, export_path="model_qcdq.onnx")
```

**If you do want to use Ultralytics' export pipeline** (e.g., for the post-processing graph it adds): note that current versions of `ultralytics/utils/export/engine.py:torch2onnx()` already pass `dynamo=False` internally. So `torch.autograd.Function.symbolic` definitions in Brevitas custom ops *will* work with Ultralytics export — the legacy TorchScript exporter is what's being used. The opset is auto-selected by `best_onnx_opset()` (caps at 20 for torch 2.4–2.8, 23 for torch 2.9+).

**TensorRT INT8** via `model.export(format="engine", int8=True)` is **separate from QAT** — it's a PTQ pipeline run on the FP32 (or QAT-fused FP) model using TensorRT's own calibrator. If you've already done Brevitas QAT, export to QCDQ ONNX first, then convert to TensorRT with `trtexec --onnx=... --int8` or via `polygraphy`.

---

## 11. Common Pitfalls

- **EMA divergence** — `best.pt` is the EMA model; if quantization scales differ between EMA and raw model, your saved checkpoint may behave differently than what you saw in train logs. Validate `best.pt` explicitly after training.
- **AMP check failures** — Even with `amp=False`, the AMP startup check may run briefly. If it fails on a quantized model, set the env var or use `model.train(amp=False)` early.
- **`get_model()` weight-loading** — When you swap layers in `get_model()`, calling `model.load(weights)` afterwards uses `load_state_dict(strict=False)`. Param name mismatches between Brevitas wrappers and original `Conv` layers will silently skip weights. Print missing/unexpected keys to debug.
- **Loss return contract** — Custom loss must return `(loss_scalar, loss_items_tensor)`. The tensor is appended to `tloss` (running mean per epoch) and shown in the progress bar. A mismatched length crashes `progress_string()`.
- **DDP + custom trainer** — Must run as a script with `torch.distributed.run`, not from a notebook.
- **Calibration data** — If your calibration set has different normalization or aug than training, scales will be miscalibrated. Use Ultralytics' actual training dataloader (with aug **disabled**) for the calibration pass.
- **Detection head quantization** — The final `Detect` layer does coordinate decoding and (in end2end models) NMS-equivalent ops. Quantizing it usually hurts accuracy disproportionately. Consider keeping `Detect` in FP32.
- **`fuse()` calls** — Ultralytics auto-calls `model.fuse()` (Conv+BN folding) before validation/export. Brevitas modules don't implement `fuse_conv_and_bn` — override `BaseModel.fuse()` to be a no-op for Brevitas layers, or skip it entirely.

---

## 12. Reference Links (Ultralytics docs only)

**Callbacks**
- https://docs.ultralytics.com/usage/callbacks/
- https://docs.ultralytics.com/reference/utils/callbacks/base/
- https://docs.ultralytics.com/reference/engine/model/

**Custom Trainer / Engine**
- https://docs.ultralytics.com/guides/custom-trainer/
- https://docs.ultralytics.com/usage/engine/
- https://docs.ultralytics.com/reference/engine/trainer/

**Architecture / Module Registration**
- https://docs.ultralytics.com/guides/model-yaml-config/
- https://docs.ultralytics.com/reference/nn/tasks/

**Training & DDP**
- https://docs.ultralytics.com/modes/train/
- https://docs.ultralytics.com/reference/utils/dist/

**Export**
- https://docs.ultralytics.com/modes/export/
- https://docs.ultralytics.com/reference/engine/exporter/
- https://docs.ultralytics.com/reference/utils/export/engine/
- https://docs.ultralytics.com/integrations/onnx/

**Quantization (Ultralytics' own materials, mostly non-Brevitas)**
- https://www.ultralytics.com/glossary/quantization-aware-training-qat
- https://www.ultralytics.com/glossary/model-quantization
- https://docs.ultralytics.com/integrations/

**General guides hub**
- https://docs.ultralytics.com/guides/

---

## 13. Still Open / Worth Verifying On Your Setup

- **Exact `parse_model()` extension shape** for Brevitas modules with bit-width and quantizer-injector arguments. The cleanest pattern probably uses module factory functions (closures) rather than direct class refs in YAML.
- **Whether `BaseModel.fuse()` recurses into Brevitas-wrapped submodules safely.** Test before trusting `model.export()` or any val pass.
- **Detection-head quantization tradeoffs for YOLO11/26.** No public Brevitas+YOLO benchmarks at the time of writing.
- **Compatibility with the new `dynamo=True` ONNX export** that Ultralytics is being asked to adopt (issue #20348). Brevitas' `torch.autograd.Function.symbolic` approach is incompatible with the dynamo path — may need `custom_translation_table` instead.