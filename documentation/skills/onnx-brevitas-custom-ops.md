# ONNX Brevitas Custom Ops

## 1. When to use this skill
Use this skill when you need to export a PyTorch model containing custom quantized layers to ONNX while working within a Brevitas workflow. Specifically, apply it when:
- You want to replace standard PyTorch operations (e.g., `Conv2d`, `Linear`) with custom ONNX nodes (e.g., `Quantify::CustomQuantConv`).
- You are explicitly using the legacy TorchScript-based ONNX exporter (`dynamo=False`).
- Your task phrasing involves "export X to ONNX with custom nodes", "make Y compatible with Brevitas QAT export via custom ops", or "override ONNX export behavior for quantized layers".

**Do NOT use this skill when:**
- You are using the modern `torch.export`-based exporter (`dynamo=True`), which does not support `torch.autograd.Function.symbolic`.
- You are exporting models with control flow (if/else, loops).
- You are dealing with standard Brevitas layers that already have built-in ONNX export support.

## 2. The pattern
The pattern wraps a layer's forward pass in a `torch.autograd.Function` to intercept ONNX graph construction. Follow these steps:

1. **Define the Function**: Create a class inheriting from `torch.autograd.Function`.
2. **Implement `forward`**: Define `forward(ctx, inputs...)` to execute the actual PyTorch computation.
   ```python
   return torch.nn.functional.conv2d(x, weight, bias, stride=stride, padding=padding)
   ```
3. **Implement `symbolic`**: Define `symbolic(g, inputs...)` to construct the ONNX graph. The first argument is the graph builder `g`. Emit the custom node and attach attributes:
   ```python
   return g.op("Quantify::CustomOp", x, weight, bias, attr_i=int(val), attr_s=str(val))
   ```
4. **Integrate into Module**: In your `nn.Module.forward`, call `CustomFn.apply(inputs...)` instead of standard layers.
5. **Export**: Call `torch.onnx.export(model, dummy_input, path, dynamo=False)`.

## 3. Pitfalls
- **Legacy Exporter Requirement**: PyTorch 2.9+ deprecates the legacy exporter. The reference example explicitly sets `dynamo=False` because `torch.autograd.Function.symbolic` is only supported by the legacy TorchScript-based exporter. Using `dynamo=True` will fail.
- **Attribute Type Suffixes**: ONNX attributes must be suffixed with a type character (`_i` for int, `_s` for string, `_t` for tensor, `_b` for bool). Using unsuffixed names or incorrect suffixes raises a `ValueError` during export.
- **Signature Divergence**: `symbolic` and `forward` have different first arguments. `symbolic` receives the graph builder `g`, while `forward` receives the context `ctx`. Do not confuse them.
- **Quantization Math is Not Handled**: The reference example uses standard `F.conv2d`/`F.linear` in `forward`. It does not demonstrate embedding quantization math, handling Brevitas `QuantTensor` metadata, or managing scale/zero-point. If you need quantization, you must implement it yourself in `forward` and pass relevant parameters to `symbolic`.
- **Export-Time Calibration Breaks Tracing**: If your quantizer runs search/calibration logic in `forward()`, it will fail during ONNX export. Wrap non-export logic with `if not torch.onnx.is_in_onnx_export():` and register search results as `nn.Module` buffers so they persist through export and `state_dict`.
- **ONNX Runtime Compatibility**: `Quantify::CustomOp` exports fine, but ORT will fall back to standard ops or warn if no custom kernel is registered. Clarify that custom nodes are for export/graph inspection, not necessarily for ORT inference.

## 4. Verification
- **Pre-export Inference**: Run a dummy forward pass before exporting to ensure `forward` executes without errors and produces the expected output shape.
- **Graph Inspection**: Open the exported `.onnx` file in Netron. Verify that custom nodes (e.g., `Quantify::CustomQuantConv`) appear in the graph and that attributes are correctly typed and populated.
- **Round-trip Testing**: The reference example does not demonstrate loading the ONNX model back into PyTorch or comparing outputs against the reference implementation. This step is recommended but not covered by the pattern.
- **Dynamic Axes & Dummy Inputs**: Pass `dynamic_axes` to `torch.onnx.export()` if batch/height/width vary at runtime. Ensure dummy inputs match expected runtime dimensions and data types.

## 5. Reference
- **Example File**: `examples/export_custom_onnx_nodes.py`
- **PyTorch ONNX Export Documentation**: https://docs.pytorch.org/docs/stable/onnx_export.html
- **PyTorch ONNX Control Flow Tutorial**: https://pytorch.org/tutorials/beginner/onnx/export_control_flow_model_to_onnx_tutorial.html
- **Dependencies**: The example imports `brevitas.nn as qnn` and `FixedPointPerTensorWeightQuant`, but relies on standard `torch.nn.functional` for the actual computation. No additional Brevitas-specific export hooks are used.
