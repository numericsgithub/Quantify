# Quantizers & Brevitas Core

This section covers the core quantization concepts and custom quantizers used in the framework.

## Brevitas Core Concepts
- **Fake Quantization**: Simulates low-precision quantization during training/inference using standard floating-point tensors.
- **QuantTensor**: A tensor-like structure containing `value`, `scale`, `zero_point`, `bit_width`, `signed`, and `training` flags.
- **QuantWBIOL**: Base class for `QuantConv2d`, `QuantLinear`, etc. Supports quantization of `weight`, `bias`, `input`, and `output`.

## Custom Quantizers
- **FixedPointPerTensorQuantizer**: Custom fixed-point per-tensor weight/activation quantizer.
- **CoefficientPerTensorWeightQuant**: Coefficient-based per-tensor weight quantizer.
- **Quantized activations** (`quantizers/activations.py`): `QuantReLU`, `QuantReLU6`, `QuantSigmoid`, `QuantTanh`, `QuantSiLU`, `QuantGELU`, `QuantLeakyReLU`, `QuantSoftmax`, `QuantHardSwish`, `QuantHardSigmoid` — each with an input and an output fixed-point quantizer, exported as a single `Quantify::<Name>` node.
- **SAME-padding layers** (`quantizers/layers.py`): `QuantConv1d`/`QuantConv2d`, `MaxPool1d/2d`, `AvgPool1d/2d` accepting `padding="same"` at any stride (exact TF-SAME pads in the exported graph).

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

For advanced quantizer development and ONNX export patterns, see [ONNX Export & Custom Ops](llm/ONNX_CUSTOM_NODE_EXPORT.md).
