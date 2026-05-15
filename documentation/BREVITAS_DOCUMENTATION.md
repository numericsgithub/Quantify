# Brevitas Quantization Framework Documentation

## 1. Core Concepts & Data Structures
- **Fake Quantization**: Brevitas simulates low-precision quantization during training/inference using standard floating-point tensors. Quantized values are wrapped in `QuantTensor` objects.
- **QuantTensor**: A tensor-like structure containing `value` (dequantized float), `scale`, `zero_point`, `bit_width`, `signed`, and `training` flags. Operations preserve `QuantTensor` metadata if invariant to quantization (e.g., `max_pool2d`). Otherwise, they decay to standard `torch.Tensor`.
- **QuantWBIOL**: Base class for `QuantConv2d`, `QuantLinear`, etc. Supports quantization of `weight`, `bias`, `input`, and `output`.

## 2. Quantized Layers & Default Quantizers
- **`QuantConv2d` / `QuantLinear`**: Drop-in replacements for `nn.Conv2d` / `nn.Linear`. Default: `weight_quant=Int8WeightPerTensorFloat`.
- **`QuantReLU`**: Applies ReLU then quantizes. Default: `Uint8ActPerTensorFloat` (unsigned).
- **`QuantIdentity`**: Wraps activations for quantization. Default: `Int8ActPerTensorFloat`.
- **Keyword Arguments**: Override quantizer attributes via layer kwargs (e.g., `weight_bit_width=4`, `weight_scaling_per_output_channel=True`). Prefixes: `weight_`, `input_`, `output_`, `bias_`.
- **Activation Quantization Pairing**: To fully quantize a layer, pair `weight_quant` with `input_quant` and/or `output_quant`. Example: `qnn.QuantConv2d(..., input_quant=Int8ActPerTensorFloat, output_quant=Int8ActPerTensorFloat)`. Ensure `return_quant_tensor=True` if downstream layers require `QuantTensor` metadata.

## 3. Custom Quantizer Development
- **ExtendedInjector**: Brevitas uses auto-wiring dependency injection to assemble quantizers from `brevitas.core` modules.
- **Structure**:
  ```python
  from brevitas.inject import ExtendedInjector
  from brevitas.proxy import WeightQuantProxyFromInjector

  class MyQuantizer(ExtendedInjector):
      proxy_class = WeightQuantProxyFromInjector
      tensor_quant = RescalingIntQuant
      # ... define core modules (int_quant, scaling_impl, etc.) ...
      bit_width = 8
      signed = True
  ```
- **Dynamic Attributes**: Use `@value` decorator to compute attributes at DI time (e.g., `scaling_init` based on `module.weight.abs().max()`).
- **Sharing**: Share quantizer instances across layers by passing `weight_quant=layer1.weight_quant`. Re-initializes automatically when shared.
- **Injector Wiring Clarification**: When using a custom quantizer (e.g., `weight_quant=FixedPointPerTensorWeightQuant`), ensure `proxy_class` and `tensor_quant` align correctly. Class-level attributes like `signed` and `bit_width` act as defaults for the DI system but can be overridden per-layer via kwargs. The `proxy_class` dictates how the quantizer is instantiated and wired into the layer's forward pass.

## 4. Training & Calibration
- **QAT**: Standard PyTorch training loop. Quantization is active by default.
- **PTQ / Calibration**: Use `calibration_mode` to collect statistics without quantization, then `bias_correction_mode`.
  ```python
  from brevitas.graph.calibrate import calibration_mode, bias_correction_mode
  with calibration_mode(model):
      for batch in loader: model(batch)
  with bias_correction_mode(model):
      for batch in loader: model(batch)
  ```
- **State Dict Loading**: Loading FP weights into quantized models raises missing key errors. Enable `config.IGNORE_MISSING_KEYS = True` or `strict=False`.

## 5. ONNX Export & Deployment
- **QCDQ Export**: `export_onnx_qcdq(model, input, path)`. Inserts `QuantizeLinear`/`DeQuantizeLinear`/`Clip` nodes. Supports bit-width < 8 via clipping.
- **QONNX Export**: `export_qonnx(model, input, path)`. Uses custom `Quant` nodes for arbitrary precision/scale.
- **Custom ONNX Nodes**: If using `torch.autograd.Function.symbolic` for custom ops, **must** use legacy exporter: `torch.onnx.export(..., dynamo=False)`. Modern `torch.export` does not support `symbolic`.
- **ONNX Runtime**: QCDQ models run in ORT. QGEMM optimization requires specific conditions (input/weight/output quantized, bias quantized >8bit, output bit-width=8).

## 6. Common Pitfalls & Debugging
- **`QuantLayer is not correctly configured`**: Usually occurs when using bias quantizers like `Int8Bias` without `input_quant` enabled. Bias quantization often requires input scale.
- **Scale Mismatch in Training**: During stats collection, scales differ per batch. Element-wise ops (add, cat) may fail in `eval()` mode if scales don't match. Use `QuantIdentity` to align scales.
- **`return_quant_tensor=True`**: Only necessary if downstream layers require `QuantTensor` metadata (e.g., bias quantization, custom ops). Default is `False` to reduce overhead.
- **Per-Channel Quantization**: Requires `scaling_per_output_channel=True` and `per_channel_broadcastable_shape` for activations. Weights usually handle this automatically.
- **Export Failures**: Ensure `model.eval()` before export. Disable quantization collection if scales are unstable. Check ONNX opset version (>=13 for per-channel).
