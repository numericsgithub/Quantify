# Quantizers & Brevitas Core

This section covers the core quantization concepts and custom quantizers used in the framework.

## Brevitas Core Concepts
- **Fake Quantization**: Simulates low-precision quantization during training/inference using standard floating-point tensors.
- **QuantTensor**: A tensor-like structure containing `value`, `scale`, `zero_point`, `bit_width`, `signed`, and `training` flags.
- **QuantWBIOL**: Base class for `QuantConv2d`, `QuantLinear`, etc. Supports quantization of `weight`, `bias`, `input`, and `output`.

## Custom Quantizers
- **FixedPointPerTensorQuantizer**: Custom fixed-point per-tensor weight/activation quantizer.
- **CoefficientPerTensorWeightQuant**: Coefficient-based per-tensor weight quantizer: snaps each weight to the nearest value of a coefficient set (one set per line of a text file) times a power of two. Calibration picks the (set, shift) with the lowest SAD among the 13 shifts around the one that covers `max|w|`, rejecting candidates that clip more than `max_clip_pct` (15%) of the weights. Supports `clipped_ste`. Has no bit width; the `Quantify::CoefficientQuant` node holds the coefficient set as an attribute and carries `num_coefficients`, `min_value` and `max_value` instead.
- **Quantized activations** (`quantizers/activations.py`): `QuantReLU`, `QuantReLU6`, `QuantSigmoid`, `QuantTanh`, `QuantSiLU`, `QuantGELU`, `QuantLeakyReLU`, `QuantSoftmax`, `QuantHardSwish`, `QuantHardSigmoid` — each with an input and an output fixed-point quantizer, exported as a single `Quantify::<Name>` node.
- **SAME-padding layers** (`quantizers/layers.py`): `QuantConv1d`/`QuantConv2d`, `MaxPool1d/2d` accepting `padding="same"`; average pooling is `QuantAvgPool1d/2d` / `QuantGlobalAvgPool1d/2d` (frozen depthwise conv, 12-bit weights, 10-bit output) at any stride (exact TF-SAME pads in the exported graph).

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
