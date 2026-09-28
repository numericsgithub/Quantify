"""Shared quantization-aware activations for the model zoo.

``QuantReLU6`` is re-exported from ``quantizers.activations`` -- it applies a
true ReLU6 (single `Quantify::Relu6` ONNX node on export) followed by a
separate output quantizer, replacing the old clamp-then-``QuantReLU``
two-node implementation. Kept importable from here for backward
compatibility with ``models/mobilenetv1_quant.py`` / ``mobilenetv2_quant.py``.
"""

from quantizers.activations import QuantReLU6

__all__ = ["QuantReLU6"]
