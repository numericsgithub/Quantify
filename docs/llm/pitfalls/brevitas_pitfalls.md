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

## 8. Custom ONNX Nodes Don't Run in ORT
**When this happens:** You export a model with `Quantify::CustomOp` and expect ONNX Runtime to execute it natively.
**The Problem:** ORT only executes standard ONNX ops or registered custom kernels. Unregistered `Quantify::` nodes will cause fallback warnings or runtime errors.
**How to Prevent It:** Use custom nodes for graph inspection/export compatibility only. For ORT deployment, convert to QCDQ (`export_onnx_qcdq`) or implement a custom ORT kernel.
