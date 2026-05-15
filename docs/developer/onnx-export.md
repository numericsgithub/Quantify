# ONNX Export & Custom Ops

This guide covers exporting PyTorch models with custom quantized layers to ONNX using Brevitas.

## When to Use
- Exporting models with custom `torch.autograd.Function` nodes.
- Using the legacy TorchScript-based exporter (`dynamo=False`).
- Needing custom ONNX nodes for graph inspection or hardware codegen.

## The Pattern
1. Define a `torch.autograd.Function` subclass.
2. Implement `forward`, `backward`, and `symbolic` methods.
3. Use `g.op()` in `symbolic` to emit custom ONNX nodes.
4. Export with `torch.onnx.export(..., dynamo=False)`.

## Pitfalls
- **Legacy Exporter Requirement**: `dynamo=True` does not support `torch.autograd.Function.symbolic`.
- **Attribute Type Suffixes**: ONNX attributes must be suffixed (`_i`, `_f`, `_s`, `_t`).
- **Side Channels**: Use a FIFO deque to pass pre-computed data from `forward` to `symbolic` when sharing `Function` classes across multiple quantizers.
- **ONNX Runtime Compatibility**: Custom nodes (`Quantify::...`) require custom ORT kernels or fallback to standard ops.

For a complete reference implementation, see the codebase examples or `documentation/skills/onnx-brevitas-custom-ops.md`.
