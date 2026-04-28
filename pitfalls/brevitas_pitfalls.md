# Brevitas Pitfalls & Best Practices

## 1. Hallucinating Non-Existent `Quant*` Layers
**When this happens:** You copy a standard PyTorch architecture and assume every layer has a Brevitas equivalent (e.g., `qnn.QuantGlobalAvgPool2d`, `qnn.QuantLayerNorm`).
**The Problem:** Brevitas only provides quantization wrappers for a specific subset of layers (primarily convolutions, linear layers, batch normalization, and basic activations). Pooling, normalization, and custom ops do not have built-in `Quant*` equivalents.
**How to Prevent It:**
- Verify layer existence in the [Brevitas API docs](https://brevitas.readthedocs.io/) or source before use.
- For unsupported layers, use standard PyTorch implementations (e.g., `nn.AdaptiveAvgPool2d(1)`).
- If you need to quantize activations from unsupported layers, wrap them with `qnn.QuantIdentity` or apply quantization explicitly in `forward()`.

## 2. `QuantLinear` Does Not Auto-Flatten Spatial Dimensions
**When this happens:** You connect a pooling layer (e.g., `AdaptiveAvgPool2d(1)` → shape `(B, C, 1, 1)`) directly to `qnn.QuantLinear`.
**The Problem:** `QuantLinear` inherits `nn.Linear`'s strict 2D input requirement `(batch_size, in_features)`. It will not reshape or flatten tensors automatically, causing `RuntimeError: mat1 and mat2 shapes cannot be multiplied`.
**How to Prevent It:**
- Always explicitly flatten before the linear layer: `self.flatten = nn.Flatten()` or `x = x.view(x.size(0), -1)`.
- Remember: quantized layers follow standard PyTorch tensor shape rules; they add no dimension-handling magic.

## 3. Custom ONNX Nodes Require Legacy Exporter (`dynamo=False`)
**When this happens:** You export a model using `torch.onnx.export(model, input, path)` without specifying `dynamo`, or explicitly set `dynamo=True`.
**The Problem:** Your custom quantizer uses `torch.autograd.Function.symbolic` to emit ONNX nodes. This pattern is only supported by the legacy TorchScript-based exporter. Modern `torch.export`-based dynamo tracing does not support `symbolic` methods and will fail with cryptic graph construction errors.
**How to Prevent It:**
- Always explicitly pass `dynamo=False` to `torch.onnx.export()`.
- Pin your PyTorch version if needed (PyTorch 2.9+ deprecates the legacy exporter). If you must use `dynamo=True`, migrate to `torch.export`-compatible custom ops or pre-quantize weights before export.

## 💡 Quick Checklist Before Export
- [ ] All pooling/normalization layers use standard `nn.*` wrappers.
- [ ] `nn.Flatten()` or `.view()` precedes every `QuantLinear`.
- [ ] `torch.onnx.export(..., dynamo=False)` is explicitly set.
- [ ] Dummy forward pass succeeds in `eval()` mode before export.
