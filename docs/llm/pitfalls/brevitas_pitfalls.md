# Brevitas Pitfalls & Best Practices

## 1. Hallucinating Non-Existent `Quant*` Layers
**When this happens:** You copy a standard PyTorch architecture and assume every layer has a Brevitas equivalent (e.g., `qnn.QuantGlobalAvgPool2d`, `qnn.QuantLayerNorm`).
**The Problem:** Brevitas only provides quantization wrappers for a specific subset of layers (primarily convolutions, linear layers, batch normalization, and basic activations). Pooling, normalization, and custom ops do not have built-in `Quant*` equivalents.
**How to Prevent It:**
- Verify layer existence in the [Brevitas API docs](https://brevitas.readthedocs.io/) or source before use.
- For unsupported layers, use standard PyTorch implementations (e.g., `nn.AdaptiveAvgPool2d(1)`).
- If you need to quantize activations from unsupported layers, wrap them with `qnn.QuantIdentity` or apply quantization explicitly in `forward()`.

## 2. `QuantLinear` Does Not Auto-Flatten Spatial Dimensions
**When this happens:** You connect a pooling layer (e.g., `AdaptiveAvgPool2d(1)` → shape `(B, C, 1, 1)`) directly to `qnn.QuantLinear`, or you manually miscalculate the flattened feature size.
**The Problem:** `QuantLinear` inherits `nn.Linear`'s strict 2D input requirement `(batch_size, in_features)`. It will not reshape or flatten tensors automatically, causing `RuntimeError: mat1 and mat2 shapes cannot be multiplied`.
**How to Prevent It:**
- Always explicitly flatten before the linear layer: `self.flatten = nn.Flatten()` or `x = x.view(x.size(0), -1)`.
- **Verify flattened dimensions manually or via a dummy pass.** When calculating `in_features` for `QuantLinear` after `Conv2d` + `MaxPool2d`, double-check the spatial reduction formula: `out_size = floor((in_size - kernel + 2*padding) / stride)`. 
- **Pro-tip:** Run a dummy forward pass with `model.eval()` and print `x.shape` right before the linear layer. This catches arithmetic errors (like `16*8*8` vs `16*16*16`) before they cause `RuntimeError` during training or export.
- Remember: quantized layers follow standard PyTorch tensor shape rules; they add no dimension-handling magic.

## 3. Custom ONNX Nodes Require Legacy Exporter (`dynamo=False`)
**When this happens:** You export a model using `torch.onnx.export(model, input, path)` without specifying `dynamo`, or explicitly set `dynamo=True`.
**The Problem:** Your custom quantizer uses `torch.autograd.Function.symbolic` to emit ONNX nodes. This pattern is only supported by the legacy TorchScript-based exporter. Modern `torch.export`-based dynamo tracing does not support `symbolic` methods and will fail with cryptic graph construction errors.
**How to Prevent It:**
- Always explicitly pass `dynamo=False` to `torch.onnx.export()`.
- Pin your PyTorch version if needed (PyTorch 2.9+ deprecates the legacy exporter). If you must use `dynamo=True`, migrate to `torch.export`-compatible custom ops or pre-quantize weights before export.
- **Future-Proofing Note:** The legacy exporter is deprecated. For long-term compatibility, consider migrating to `torch.export`-compatible custom ops or exporting pre-quantized weights as standard ONNX ops.

## 4. Bias Quantization Requires Input Quantization
**When this happens:** You enable `bias_quant=Int8Bias` (or similar) on a `QuantConv2d`/`QuantLinear` without enabling `input_quant`.
**The Problem:** `Int8Bias` assumes bias scale = `input_scale * weight_scale`. Without a quantized input, the layer cannot compute this scale, raising `RuntimeError: QuantLayer is not correctly configured` or `RuntimeError: Input scale required`.
**How to Prevent It:**
- Enable `input_quant` (e.g., `Int8ActPerTensorFloat`) when using `Int8Bias`.
- Alternatively, use bias quantizers with internal scaling like `Int8BiasPerTensorFloatInternalScaling` if you don't want to quantize inputs.

## 5. `load_state_dict` Missing Keys in Quantized Models
**When this happens:** You load a pretrained floating-point `state_dict` into a quantized model using `model.load_state_dict(fp_state_dict)`.
**The Problem:** Quantized layers introduce learned parameters for scales, zero-points, and bit-widths that don't exist in the FP model. PyTorch raises `RuntimeError: Missing key(s) in state_dict`.
**How to Prevent It:**
- Set `config.IGNORE_MISSING_KEYS = True` before loading, or use `strict=False`.
- Brevitas automatically re-initializes scale parameters based on the loaded weights after import.

## 6. `QuantTensor` Validity & Scale Mismatch in Training vs Eval
**When this happens:** You perform element-wise operations (add, cat) on `QuantTensor`s during training and get validity errors or unexpected bit-width expansion.
**The Problem:** During training, activation scales are collected per-batch and differ between tensors. Brevitas allows adding them but marks the result `is_valid=False` and averages scales. In `eval()` mode, scales are fixed (EMA), and mismatched scales will cause errors.
**How to Prevent It:**
- Use `QuantIdentity` to align scales before operations if needed.
- Ensure `model.eval()` before export or inference to stabilize scales.
- Be aware that accumulator bit-widths grow during training (e.g., 8b + 8b → 17b). Output quantization (`output_quant`) is often needed to clamp bit-widths.

## 7. `return_quant_tensor=True` Overhead & Necessity
**When this happens:** You set `return_quant_tensor=True` on every layer unnecessarily.
**The Problem:** It forces Brevitas to maintain and propagate `QuantTensor` metadata through the entire graph, increasing memory and compute overhead. It's only required when downstream layers need quantization metadata (e.g., bias quantization, custom ONNX export, or explicit quantization math).
**How to Prevent It:**
- Keep `return_quant_tensor=False` (default) unless specifically required by the architecture or export target.

## 9. Brevitas `Injector` Subclasses Are Immutable After Definition

**When this happens:** You dynamically create an `Injector` subclass at runtime (e.g., to set `bit_width` or `filepath` from a CLI arg) and then try to set any attribute on the class object itself — including cosmetic ones like `__name__` or `__qualname__`.

**The Problem:** Brevitas injectors inherit from `_dependencies.Injector`, which overrides `__setattr__` to raise `DependencyError: 'Injector' modification is not allowed`. This applies to *any* attribute set on the class after its `class` block is evaluated, not just quantization-specific ones. Example that breaks:

```python
class WeightQuant(FixedPointPerTensorWeightQuant):
    bit_width = bw
WeightQuant.__name__ = "MyQuant"   # ← DependencyError here
```

**How to Prevent It:**
- All injector attributes must be set **inside** the `class` body — never after the `class` statement.
- This includes cosmetic attributes like `__name__`. If you need a descriptive name, embed it directly in the class definition using a string variable before the class block:

```python
# Correct: all attributes set inside the class body
bw = args.weight_bits
class WeightQuant(FixedPointPerTensorWeightQuant):
    bit_width = bw          # ← fine: captured from enclosing scope at class creation time
```

- Never call `setattr()` on an injector class either — it hits the same guard.

## 10. Residual Paths Are Not Quantized Without Explicit `QuantIdentity`

**When this happens:** You build a quantized residual network (ResNet-style) and pass `act_quant` to the blocks, expecting all activations to be quantized.

**The Problem:** Two paths in a residual block produce unquantized tensors unless you explicitly add `QuantIdentity`:

1. **Pre-add main path**: After the last `conv → BN` in each block (before the residual add), the BN output is a plain float. The `QuantReLU` only quantizes the output of the *sum*, not the individual branches.
2. **Downsample skip path**: The `1×1 QuantConv2d → BN` in the downsample branch produces an unquantized tensor. Without a `QuantIdentity` after the BN, the identity branch is float when it reaches the add.

These unquantized tensors are not fake-quantized during QAT, so the simulation does not match the deployed hardware behavior.

**How to Prevent It:**

Add `QuantIdentity` to both paths. In each block, add a `pre_add_quant` module after the last BN:

```python
self.pre_add_quant = qnn.QuantIdentity(act_quant=act_quant) if act_quant else None

# In forward():
out = self.bn2(self.conv2(out))
if self.pre_add_quant is not None:
    out = self.pre_add_quant(out)
```

And append a `QuantIdentity` to the downsample `Sequential` when `act_quant` is set:

```python
ds_modules = [QuantConv2d(...), BatchNorm2d(...)]
if act_quant is not None:
    ds_modules.append(qnn.QuantIdentity(act_quant=act_quant))
downsample = nn.Sequential(*ds_modules)
```

`QuantIdentity` accepts `act_quant` injectors designed for activations. The `FixedPointPerTensorQuantizer` auto-detects signed/unsigned during calibration, so it correctly handles the negative BN outputs on the pre-add path even though `FixedPointPerTensorActivationQuant` defaults to `signed=False`.

Two additional boundary points also require `QuantIdentity`:

- **Network input** (before the first conv): the raw float image must be quantized before entering the stem. Add `self.input_quant = QuantIdentity(act_quant=act_quant)` and call it at the top of `forward()`.
- **`AdaptiveAvgPool2d` output** (before the FC layer): avgpool computes a spatial mean of quantized values, which lands off the quantization grid. Add `self.post_pool_quant = QuantIdentity(act_quant=act_quant)` and call it after avgpool. `MaxPool2d` does *not* need this treatment — selecting a max from quantized values stays on the grid.

The FC output (logits) should intentionally remain unquantized.

## 8. Custom ONNX Nodes Don't Run in ORT
**When this happens:** You export a model with `Quantify::CustomOp` and expect ONNX Runtime to execute it natively.
**The Problem:** ORT only executes standard ONNX ops or registered custom kernels. Unregistered `Quantify::` nodes will cause fallback warnings or runtime errors.
**How to Prevent It:** Use custom nodes for graph inspection/export compatibility only. For ORT deployment, convert to QCDQ (`export_onnx_qcdq`) or implement a custom ORT kernel.

## 9. `model.named_modules()` Order ≠ Forward Execution Order
**When this happens:** You iterate `model.named_modules()` (or `QuantizerManager.quantizers`, which is built from it) assuming the order matches the order layers run in `forward()` — e.g. to greedily search/calibrate quantizers from input to output.
**The Problem:** `named_modules()` walks attribute *declaration* order from `__init__`, not call order from `forward()`. `models/resnet_quant.py`'s `QuantResNet18.__init__` declares `self.input_quant` *after* `self.layer1..layer4`, even though `forward()` calls `input_quant` first. A per-quantizer greedy PTQ search (`examples/find_perfect_lsbs_imagenet_ptq.py`) that iterates in declaration order ends up searching `input_quant` ~29th out of 30 instead of 1st — every quantizer searched before it gets calibrated against an unoptimized input range, then goes stale the moment `input_quant`'s own search later changes that range. The symptom was a huge, suspicious accuracy swing across one quantizer's LSB sweep — not actually a bug in the sweep itself (sweeping a fixed-bit-width quantizer's LSB sweeps its representable *range*; too narrow a range on the stem input clips the image to near-blank, explaining the swing), but a sign the search order was wrong.
**How to Prevent It:** Use `QuantizerManager.quantizers_in_execution_order()` (`quantizers/manager.py`) instead of relying on `named_modules()`/dict iteration order whenever the order quantizers are processed in matters. It raises `RuntimeError` if called before any forward pass has run (order is undefined until then) and drops Brevitas's internal "ghost" quantizer objects (registered but never reached by `forward()`) by default — pass `include_unreached=True` to keep them.

## 10. A BN-Fused Checkpoint Must Be Loaded Into a BN-Fused Model
**When this happens:** `examples/find_perfect_lsbs_imagenet_ptq.py --fuse-bn` folds BatchNorm into the preceding conv/linear (`utils/bn_fusion.py`) before calibrating — the conv gains a bias, the BatchNorm module becomes `nn.Identity()`. The resulting checkpoint's `model_state_dict` reflects that fused structure. A *different* script (`examples/train_imagenet_qat.py`) builds a fresh model with separate, never-fused, randomly-initialized BatchNorm layers and loads that checkpoint via `strict=False`.
**The Problem:** `load_state_dict(strict=False)` doesn't fail — it just silently skips whatever doesn't match: every `bn*.weight/bias/running_mean/running_var` key is "missing" (so those BatchNorm layers stay at their untrained defaults: `weight=1, bias=0, running_mean=0, running_var=1`) and every `conv*.bias` key in the checkpoint is "unexpected" (silently dropped). The model still runs without raising, but inference is essentially random — a 56% PTQ-calibrated accuracy reported moments earlier becomes ~0.1% the instant the checkpoint is loaded somewhere structurally different. `missing_keys`/`unexpected_keys` *are* printed by `_load_ptq_checkpoint`, but a long list of names is easy to skim past and not recognize as catastrophic.
**How to Prevent It:** Whenever a script saves a checkpoint after structurally transforming the model (BN fusion, pruning, layer fusion, etc.), record that fact in the checkpoint's `extra` payload (here: `extra["fuse_bn"]`, set by `find_perfect_lsbs_imagenet_ptq.py` at save time) and have every loader check it and replay the same transform on its own freshly built model *before* `load_state_dict` — see `_load_ptq_checkpoint` in `examples/train_imagenet_qat.py`, which now calls `fuse_bn_into_conv(model)` first when `extra.fuse_bn` is set. Don't rely on the caller remembering to pass a matching flag by hand; detect it from the checkpoint itself. More generally: treat any non-empty `missing_keys`/`unexpected_keys` from a PTQ/pretrained checkpoint load as a signal worth investigating, not just logging — a handful of legitimately-absent keys (e.g. one quantizer role not yet searched) looks very different from dozens of `bn*`/`conv*.bias` keys flipping in lockstep.

## 11. Quantizer Gating and Annealing Are Independent — `preserve_calibrated_quantizers` Only Bypasses One
**When this happens:** Resuming QAT from a PTQ checkpoint via `--init-from-ptq` (`examples/train_imagenet_qat.py`), which sets `preserve_calibrated_quantizers=True` so already-calibrated quantizers skip the annealing ramp (`annealing_alpha` jumps straight to 1.0 instead of ramping 0→1).
**The Problem:** `BaseQuantizer.forward()` (`quantizers/base_quantizer.py`) checks gating *before* it ever looks at `annealing_alpha`/`search_done`:
```python
if self.inference_counter < self.inference_sequence_id * self.quantizer_manager.quantization_start_gap:
    ...
    perform_quantization = False   # float passthrough regardless of alpha
```
`inference_counter` only increments while `self.training`, once per forward call to that specific quantizer (≈ once per training batch), and is never reset or pre-filled by `_activate_qat()`. So a "preserved" quantizer with `alpha=1.0` from forward-pass 1 still silently runs as float passthrough until its *own* `inference_sequence_id * quantization_start_gap` training-mode forward calls have elapsed — deep/late quantizers in the network (high `inference_sequence_id`) can take many epochs to ever actually start quantizing. The visible `quant_pct` progress metric only checks `annealing_alpha >= 1.0` (`training_harness/trainer_v2.py`), not gating, so it reports e.g. "98% quantized" while a meaningful fraction of the forward pass is still running in float — symptom: train/val loss *and* accuracy moving in the same direction for the first several epochs (the network gradually unlocking quantization layer-by-layer, identical to a from-scratch QAT cascade), then everything freezing once the last quantizer's gate clears.
**How to Prevent It:** Gating (`inference_counter` vs. `sequence_id * gap`) and annealing (`annealing_alpha`) are two separate, independently-controlled mechanisms — bypassing one does not bypass the other. Use `QuantizerManager.skip_gating_for_calibrated_quantizers()` (`quantizers/manager.py`) alongside `set_annealing_for_n_inferences(skip_calibrated=True)` whenever resuming from already-calibrated state — it sets `inference_counter = inference_sequence_id * quantization_start_gap` for every `search_done=True` quantizer so gating is immediately satisfied too. `training_harness/trainer_v2.py::_activate_qat()` now calls it when `preserve_calibrated_quantizers=True`.

## 12. `load_state_dict()` Recreates Quantizer Proxies — Stale References & Orphaned Registry Entries

**When this happens:** You call `model.load_state_dict(checkpoint, strict=False)` on any model containing Brevitas-quantized layers (`QuantConv2d`, `QuantLinear`, `QuantIdentity`, etc.), then either (a) keep using a quantizer object reference you captured *before* the load, or (b) iterate `QuantizerManager().quantizers.values()` afterward expecting it to reflect only the model's current, live quantizers.

**The Problem:** Brevitas's `WeightQuantProxyFromInjector`/`BiasQuantProxyFromInjector`/`ActQuantProxyFromInjector` recreate their underlying `tensor_quant` submodule as a brand-new Python object on every `load_state_dict()` call — even when reloading a model's own `state_dict()` onto itself. Verified directly: capturing `id()` of each `BaseQuantizer` instance before and after a single `load_state_dict()` call shows every one is a different object afterward.

This has two consequences:
1. **Stale references**: any reference to a quantizer captured *before* the load (e.g. cached in a variable, a list, a dict built earlier) silently stops being the object the model actually uses. Reading `search_done`/`search_result_lsb` off that stale reference shows pre-load state forever — it will never reflect what was loaded, and no exception is raised.
2. **Orphaned registry entries**: `BaseQuantizer.__init__` registers itself with the singleton `QuantizerManager` and nothing ever removes a superseded object — so each `load_state_dict()` call leaves the previous generation of quantizer objects sitting in `QuantizerManager().quantizers`, unreachable via `model.named_modules()` but still present in the registry. Empirically confirmed: a 3-quantizer model's registry grows from 3 to 6 entries after one `load_state_dict()` call onto itself. Code that iterates `mgr.quantizers.values()` directly after a load (rather than re-deriving from the model's current module tree) risks operating on dead objects with zero effect on the real model, or double-counting in diagnostics/progress output.

**How to Prevent It:**
- Always re-fetch quantizer references from the model *after* calling `load_state_dict()` — never reuse a reference captured before the call.
- After any `load_state_dict()` call whose result will be used with `QuantizerManager` (calibration search, annealing, diagnostics), call `training_harness/trainer_v2.py::_reset_and_register(model)` (or equivalent: `mgr.reset()` then walk `model.modules()` and re-register) to purge orphans before trusting the registry for anything order- or count-sensitive. This is exactly what `QATTrainerV2.fit()` does early in its sequence — before `_activate_qat()` and before any forward pass — making the production training flow safe; it is an implicit property worth keeping explicit via tests rather than relying on call-order alone.
- Never call `load_state_dict()` more than once per model lifecycle with *partial*, complementary checkpoints (e.g. one containing only weight-quantizer keys, another only activation-quantizer keys) expecting the effects to accumulate — each call recreates every quantizer's proxy from scratch, so a second call silently reverts any buffer absent from its own payload back to a freshly-constructed (uncalibrated) default, wiping the first call's effect. Merge state dicts in Python first (`{**a, **b}`) and call `load_state_dict()` exactly once.
- See `tests/test_quantizer_checkpoint_roundtrip.py` for regression tests covering stale references, registry orphaning, the `_reset_and_register` mitigation, and the double-load wipe failure mode.

## 13. Pretrained Model Fidelity — a Model That "Loads" Can Still Be Silently Wrong

**When this happens:** You wire a `timm` pretrained checkpoint into a hand-written quantized model (`utils/weight_mapping.py`) and the load reports success (`Loaded N/N weight tensors`, no skipped keys), but pre-training validation accuracy is far below the checkpoint's published number. `strict=False` and shape-only filtering hide every mismatch that isn't a tensor-shape mismatch. Seen across all four MobileNet loaders; ResNets were unaffected because they load by exact name-match with no architectural reconstruction.

**The Problem:** Four independent, individually-silent ways the reconstructed model can diverge from the checkpoint it loaded — none raises, and weights load "successfully" into all of them:
1. **Phantom randomly-initialised layer.** `QuantInvertedResidual` built the `expand_ratio==1` block *with* an expansion pointwise conv even though torchvision/timm omit it there. The remapper had no weights to fill it, so a Kaiming-random 1×1 conv sat at `features.3.conv.0` scrambling channels → near-random accuracy (~0.001). Fix: omit the expansion conv when `expand_ratio == 1`, matching the reference.
2. **Wrong stride.** The MobileNetV2 `c=32` stage was stride 1 instead of 2 (paper/timm value). Strides don't affect weight *shapes*, so every tensor still loads — the whole network just runs at the wrong spatial resolution and the pretrained BN stats/receptive fields no longer match → ~28% instead of ~72%.
3. **Wrong normalization.** `timm`'s `mobilenetv1_100.ra4_e3600_r224_in1k` was trained with `mean=std=0.5`, not ImageNet stats. The DALI pipeline hard-coded ImageNet mean/std for every model → ~17% instead of ~73%. **Always read the checkpoint's `pretrained_cfg` (HF `config.json`: `mean`, `std`, `crop_pct`, `input_size`) — do not assume ImageNet stats.** Normalization is now per-model via `utils/dali_pipeline.py::norm_for_model()`.
4. **Wrong activation.** Brevitas `QuantReLU` is an *unbounded* ReLU in float mode (and whenever quantization is disabled — i.e. all of float warmup/fine-tuning). MobileNetV1/V2 pretrained weights are trained with **ReLU6**; the missing ceiling cost ~16 pts (V2: 0.56→0.73) and ~9 pts (V1: 0.66→0.75). Use `models/quant_activations.py::QuantReLU6` (clamps input at 6, then `QuantReLU`) — exact ReLU6 in float mode, preserves the *unsigned* activation quantizer for QAT, correct upper-branch gradient for STE.

**How to Prevent It:** After any non-trivial weight remap, **verify the float model numerically against the reference** before trusting it — build the `timm` model and your model, disable quantization (`training_harness/trainer_v2.py::_fully_disable_quantization`), and run both over the val set on the same loader. When the remap and architecture are correct, a quant-disabled model is *numerically identical* to `timm` (both MobileNets now match the reference to 4 decimals: V2 0.7271, V1 0.7514 on full ImageNet val). Anything short of a near-exact match means one of the four failure modes above is present. `Loaded N/N tensors` is necessary but nowhere near sufficient.

## 14. Brevitas Weight Quantizers Don't Support `torch.func` `vmap`/`jacrev` — Even the Built-In Ones

**When this happens:** You try to use `torch.func.vmap`/`torch.func.grad`/`torch.func.jacrev` (e.g. for fast per-sample gradients, as in `importance/engine.py`'s primary path — see `docs/llm/importance_analysis.md`) on *any* model containing a Brevitas `QuantConv2d`/`QuantLinear`/etc., not just ones using Quantify's own custom quantizers.

**The Problem:** Brevitas's weight quantization (including the *default* `Int8WeightPerTensorFloat` injector, with no Quantify code involved at all) goes through a `torch.autograd.Function` internally for the STE rounding step. Any `autograd.Function` used under `torch.func` transforms must implement the `setup_context` staticmethod (see https://pytorch.org/docs/main/notes/extending.func.html); Brevitas's does not. The failure is a `RuntimeError`, not silent wrong output:
```
RuntimeError: In order to use an autograd.Function with functorch transforms
(vmap, grad, jvp, jacrev, ...), it must override the setup_context staticmethod.
```
This means **every** Brevitas-quantized model (Quantify's fixed-point quantizers included) needs a non-`vmap` fallback for any `torch.func`-based analysis — verified empirically on a plain `qnn.QuantConv2d(weight_bit_width=8)` with no custom quantizer at all.

**How to Prevent It:**
- Never assume `vmap`/`jacrev`/`grad` will work on a Brevitas-quantized model without a fallback path; always wrap the attempt in `try/except` and fall back to a plain per-sample loop (regular `backward()` calls work fine — this is standard autograd, exactly what training already exercises).
- If you need per-sample gradients *without* a full per-sample loop and don't need `abs()` of them, remember: summing a batched output column over the batch dimension before calling one plain `.backward()` gives you the exact per-sample gradient in every sample's slot of any *activation* tensor's `.grad` (not a shared parameter's — those only give you the sum) — no `vmap` needed. See `importance/engine.py::_eager_pass`.

## 15. `AdaptiveAvgPool2d(target_size)` Fails ONNX Export Unless `target_size` Evenly Divides the Feature Map

**When this happens:** You use `nn.AdaptiveAvgPool2d(k)` with `k > 1` (e.g. `AdaptiveAvgPool2d(4)`) to pool down to a fixed spatial size before a `Linear` head, then export to ONNX with `utils/onnx_export.py::export_onnx_with_io` (legacy TorchScript exporter, required per pitfall #3).

**The Problem:** PyTorch's legacy ONNX exporter only supports `adaptive_avg_pool2d` when the target output size evenly divides the input feature map size (it lowers to a plain `AveragePool` with a computed stride/kernel, which only works for exact divisors). Training and plain inference work fine either way — the failure only appears at export time, for the specific conv-stride/input-size combination that produces a non-divisible feature map. Concretely: a 28x28 MNIST input through one stride-2 conv gives a 14x14 feature map; `AdaptiveAvgPool2d(4)` (14 is not divisible by 4) raises `SymbolicValueError: Unsupported: ONNX export of operator adaptive_avg_pool2d, output size that are not factor of input size` — while the exact same model trains and runs normally, so this is easy to not notice until export time, potentially several checkpoints into training.

**How to Prevent It:**
- Prefer `nn.AdaptiveAvgPool2d(1)` (global average pool) before the head — it always divides evenly (any size / 1) and is already this repo's documented pattern (CLAUDE.md pitfall #1) for pooling into a quantized `Linear`/`QuantLinear` layer.
- If you need a larger pooled spatial size, pick a target that evenly divides the *actual* feature map size for your specific input resolution and strides, not just a size that looks reasonable — recompute it if you change any conv's stride or the input resolution.
- Since this only fails at export, test `export_onnx_with_io` (or at least trace the model with `torch.onnx.export(..., dynamo=False)`) as part of validating any new architecture, not just a forward pass — see `scripts/train_qat_and_export_onnx.py` for a script that exports every saved training checkpoint, which is exactly when this was caught.

## 16. A Quantizer Mid-Annealing Exports a `Mul`/`Add` Float-Weight Blend Instead of the Quantized Value

**When this happens:** You export to ONNX a checkpoint saved *during* QAT annealing (`BaseQuantizer.annealing_alpha` between 0 and 1 — the ramp `AnnealingBlendFn` uses to blend `(1-alpha)*float_weight + alpha*quantized_weight` as training progresses; see `quantizers/base_quantizer.py`). `Trainer`'s per-epoch checkpoints can easily land here, especially early in the `[QAT]` phase right after `[qat_sched] Epoch N: fake-quantization ENABLED`.

**The Problem:** `BaseQuantizer.forward()` only emits a clean `quantized`-only graph when `annealing_alpha == 1.0`; for any `alpha < 1.0` it routes through `AnnealingBlendFn.apply(x, quantized, alpha)`. That `autograd.Function` has no `symbolic` method (unlike `FixedPointQuantFn`), so the legacy TorchScript exporter traces straight through its `forward()` arithmetic instead of emitting one clean op. The result: right after the `FixedPointQuant` node, the exported graph gets a `Mul` (by the `alpha` constant) then an `Add` (of a `(1-alpha)*original_float_weight` constant) before the value reaches `Conv`/`Gemm` — i.e. the "quantized" weight actually baked into the ONNX graph is a blend that still contains the raw float weight, not a real quantized value. Verified by forcing `annealing_alpha.fill_(0.5)` on a calibrated quantizer and exporting: the graph shows exactly `FixedPointQuant -> Mul -> Add -> Conv` instead of `FixedPointQuant -> Conv`.

**How to Prevent It:** `utils/onnx_export.py::export_onnx_with_io` now handles this automatically (`freeze_annealing=True` by default, added after this was reported): before tracing, it snapshots every `BaseQuantizer`'s `annealing_alpha`, forces each to `1.0`, exports (and runs the reference forward pass for the embedded dummy I/O) with that frozen state, then restores the original values in a `finally` block — even if export raises (e.g. an uncalibrated quantizer elsewhere in the model). Don't call `torch.onnx.export` directly on a Quantify model that might still be mid-anneal without doing this yourself; use `export_onnx_with_io`. See `tests/test_fixedpoint_onnx_export.py::TestAnnealingFrozenForExport` for the regression tests, including one confirming `freeze_annealing=False` reproduces the original leak.
