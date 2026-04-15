import torch
import torch.nn as nn
from brevitas.quant.base import *
from brevitas.inject import ExtendedInjector
from brevitas.inject.enum import ScalingImplType, BitWidthImplType, FloatToIntImplType, QuantType, ScalingPerOutputType, RestrictValueType

# Quantizer 1: Fixed-point per-tensor weight quantizer
# Using Brevitas ExtendedInjector pattern
class Quantizer1(ExtendedInjector):
    """
    Quantizer 1 implementation as a Brevitas QuantType.
    This is a fixed-point per-tensor weight quantizer.
    """
    
    # Define the quantizer properties using Brevitas injector pattern
    quant_type = QuantType.INT
    bit_width = 8
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.ROUND
    narrow_range = True
    signed = True
    scaling_impl_type = ScalingImplType.PARAMETER
    scaling_per_output_type = ScalingPerOutputType.TENSOR
    restrict_scaling_type = RestrictValueType.FP
    tensor_clamp_impl = TensorClamp
    zero_point_impl = ZeroZeroPoint
    # Use a fixed-point scaling implementation
    scaling_impl = ParameterScaling
    scaling_shape = SCALAR_SHAPE
    scaling_min_val = 1e-10
