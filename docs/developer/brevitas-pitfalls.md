# Brevitas Pitfalls & Best Practices

## 1. Hallucinating Non-Existent `Quant*` Layers
**When this happens:** You copy a standard PyTorch architecture and assume every layer has a Brevitas equivalent.
**The Problem:** Brevitas only provides quantization wrappers for a specific subset of layers. Pooling, normalization, and custom ops do not have built-in `Quant*` equivalents.
**How to Prevent It:** Verify layer existence in the Brevitas API docs. Use standard PyTorch implementations or wrap with `qnn.QuantIdentity`.

## 2. `QuantLinear` Does Not Auto-Flatten Spatial Dimensions
**When this happens:** You connect a pooling layer directly to `qnn.QuantLinear`.
**The Problem:** `QuantLinear` requires strict 2D input `(batch_size, in_features)`.
**How to Prevent It:** Always explicitly flatten before the linear layer: `self.flatten = nn.Flatten()`.

## 3. Custom ONNX Nodes Require Legacy Exporter (`dynamo=False`)
**When this happens:** You export using `torch.onnx.export()` without specifying `dynamo`, or explicitly set `dynamo=True`.
**The Problem:** Custom `torch.autograd.Function.symbolic` methods are only supported by the legacy exporter.
**How to Prevent It:** Always explicitly pass `dynamo=False` to `torch.onnx.export()`.

## 4. Bias Quantization Requires Input Quantization
**When this happens:** You enable `bias_quant=Int8Bias` without enabling `input_quant`.
**The Problem:** `Int8Bias` assumes bias scale = `input_scale * weight_scale`.
**How to Prevent It:** Enable `input_quant` when using `Int8Bias`, or use bias quantizers with internal scaling.

## 5. `load_state_dict` Missing Keys in Quantized Models
**When this happens:** You load a pretrained floating-point `state_dict` into a quantized model.
**The Problem:** Quantized layers introduce learned parameters for scales and zero-points.
**How to Prevent It:** Use `strict=False` or set `config.IGNORE_MISSING_KEYS = True`.

## 6. `QuantTensor` Validity & Scale Mismatch in Training vs Eval
**When this happens:** You perform element-wise operations on `QuantTensor`s during training and get validity errors.
**The Problem:** Activation scales differ per-batch during training. Brevitas marks results `is_valid=False`.
**How to Prevent It:** Use `QuantIdentity` to align scales. Ensure `model.eval()` before export/inference.

## 7. `return_quant_tensor=True` Overhead & Necessity
**When this happens:** You set `return_quant_tensor=True` on every layer unnecessarily.
**The Problem:** Increases memory and compute overhead.
**How to Prevent It:** Keep `return_quant_tensor=False` unless specifically required by the architecture or export target.

## 8. Custom ONNX Nodes Don't Run in ORT
**When this happens:** You export a model with `Quantify::CustomOp` and expect ONNX Runtime to execute it natively.
**The Problem:** ORT only executes standard ONNX ops or registered custom kernels.
**How to Prevent It:** Use custom nodes for graph inspection/export compatibility only. For ORT deployment, convert to QCDQ.
