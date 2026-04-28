# Brevitas Pitfalls

## 1. Hallucinating Non-Existent `Quant*` Layers
**The Problem:**
When building models with Brevitas, it's easy to assume that every standard PyTorch layer has a corresponding quantized wrapper (e.g., `qnn.QuantGlobalAvgPool2d`). However, Brevitas only provides quantization wrappers for a specific subset of layers (primarily convolutions, linear layers, batch normalization, and basic activations). Pooling layers, normalization layers beyond BatchNorm, and other custom operations do not have built-in `Quant*` equivalents.

**How to Prevent It:**
- Always verify the existence of a layer in the official [Brevitas API documentation](https://brevitas.readthedocs.io/) or the source code before using it.
- For unsupported layers, use the standard PyTorch implementation (e.g., `nn.AdaptiveAvgPool2d(1)`).
- If you need to quantize the output of an unsupported layer, wrap it with `qnn.QuantIdentity` or apply quantization explicitly in the forward pass.

## 2. `QuantLinear` Does Not Automatically Flatten Spatial Dimensions
**The Problem:**
Brevitas `QuantLinear` (and PyTorch `nn.Linear`) expects a 2D input tensor of shape `(batch_size, in_features)`. When feeding the output of a pooling layer (e.g., `AdaptiveAvgPool2d(1)` which outputs `(batch, channels, 1, 1)`) directly into `QuantLinear`, it will attempt matrix multiplication on the wrong dimensions, resulting in a `RuntimeError: mat1 and mat2 shapes cannot be multiplied`.

**How to Prevent It:**
- Always explicitly flatten the tensor before passing it to a linear layer. Use `nn.Flatten()` or `x.view(x.size(0), -1)` in your model definition.
- Remember that quantized linear layers do not add any special dimension-handling logic compared to standard linear layers; they strictly follow PyTorch's `nn.Linear` input shape requirements.
