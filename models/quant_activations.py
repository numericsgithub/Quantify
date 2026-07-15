"""Shared quantization-aware activations for the model zoo."""

import torch
import torch.nn as nn
import brevitas.nn as qnn


class QuantReLU6(nn.Module):
    """ReLU6 activation with Brevitas activation quantization.

    Brevitas ``QuantReLU`` implements an *unbounded* ReLU in float mode (and
    when quantization is disabled). MobileNetV1/V2 pretrained weights are
    trained with the ReLU6 ceiling, so an unbounded ReLU corrupts the
    activations and drops top-1 accuracy substantially.

    This clamps the input at 6 before the ReLU, which yields exact ReLU6 in
    float mode while preserving the *unsigned* activation quantizer that
    ``QuantReLU`` provides for QAT (all post-ReLU6 values are in ``[0, 6]``).
    ``torch.clamp(max=6)`` also gives the correct ReLU6 upper-branch gradient
    (0 above 6), so QAT/STE training is unaffected.

    All kwargs (e.g. ``act_quant``, ``bit_width``) are forwarded to
    ``qnn.QuantReLU``.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.relu = qnn.QuantReLU(**kwargs)

    def forward(self, x):
        return self.relu(torch.clamp(x, max=6.0))
