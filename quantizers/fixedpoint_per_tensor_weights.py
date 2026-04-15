import torch
import torch.nn as nn
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.inject import value
from brevitas.inject.enum import ScalingImplType, BitWidthImplType, FloatToIntImplType, QuantType

# Quantizer 1: Fixed-point per-tensor weight quantizer
# Using existing Brevitas quantizer as base
class Quantizer1(Int8WeightPerTensorFloat):
    """
    Quantizer 1 implementation as a Brevitas QuantType.
    This is a fixed-point per-tensor weight quantizer.
    """
    
    # Override to use fixed-point instead of floating-point
    # This creates a fixed-point quantizer with 8-bit precision
    bit_width = 8
    scaling_impl_type = ScalingImplType.PARAMETER
    signed = True
    narrow_range = True
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.ROUND
