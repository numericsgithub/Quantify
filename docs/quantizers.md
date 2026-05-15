# Quantizers & Brevitas Core

This section covers the core quantization concepts and custom quantizers used in the framework.

## Brevitas Core Concepts
- **Fake Quantization**: Simulates low-precision quantization during training/inference using standard floating-point tensors.
- **QuantTensor**: A tensor-like structure containing `value`, `scale`, `zero_point`, `bit_width`, `signed`, and `training` flags.
- **QuantWBIOL**: Base class for `QuantConv2d`, `QuantLinear`, etc. Supports quantization of `weight`, `bias`, `input`, and `output`.

## Custom Quantizers
- **FixedPointPerTensorQuantizer**: Custom fixed-point per-tensor weight/activation quantizer.
- **CoefficientPerTensorWeightQuant**: Coefficient-based per-tensor weight quantizer.
- **SiLUTensorQuant**: Tensor quantizer for SiLU activation.

## Usage
```python
from quantizers import FixedPointPerTensorWeightQuant, FixedPointPerTensorActivationQuant

layer = qnn.QuantConv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    weight_quant=FixedPointPerTensorWeightQuant,
    output_quant=FixedPointPerTensorActivationQuant
)
```

For advanced quantizer development and ONNX export patterns, see [ONNX Export & Custom Ops](developer/onnx-export.md).
