import torch
import torch.nn as nn
from brevitas.quant.base import *
from brevitas.inject.enum import ScalingImplType, BitWidthImplType, FloatToIntImplType, QuantType, ScalingPerOutputType, RestrictValueType
from brevitas.proxy import WeightQuantProxyFromInjector

# Quantizer 1: Fixed-point per-tensor weight quantizer
# Following Brevitas pattern by composing from base classes
class Quantizer1(
    NarrowIntQuant,  # 8-bit narrow signed int
    ParameterScaling,  # Parameter-based scaling
    PerTensorFloatScaling8bit):  # 8-bit per-tensor float scaling
    """
    Quantizer 1 implementation as a Brevitas QuantType.
    This is a fixed-point per-tensor weight quantizer.
    """
    pass
