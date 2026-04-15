import torch
import torch.nn as nn
from brevitas.quant.base import *
from brevitas.core.scaling import ParameterFromStatsFromParameterScaling
from brevitas.inject import ExtendedInjector
from brevitas.inject.enum import ScalingImplType, BitWidthImplType, FloatToIntImplType, QuantType, ScalingPerOutputType, RestrictValueType
from brevitas.core.function_wrapper import TensorClamp, Identity
from brevitas.core.scaling import SCALAR_SHAPE
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.proxy import WeightQuantProxyFromInjector

# Quantizer 1: Fixed-point per-tensor weight quantizer
# Following Brevitas pattern by inheriting from a proper base class
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
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    scaling_per_output_type = ScalingPerOutputType.TENSOR
    restrict_scaling_type = RestrictValueType.FP
    tensor_clamp_impl = TensorClamp
    zero_point_impl = ZeroZeroPoint
    # Use a fixed-point scaling implementation
    scaling_impl = ParameterFromStatsFromParameterScaling
    scaling_shape = SCALAR_SHAPE
    scaling_min_val = 1e-10
    # Required for Brevitas injector pattern
    proxy_class = WeightQuantProxyFromInjector
    # Required for proper initialization
    input_view_impl = Identity
    # Required for proper initialization
    stats_reduce_dim = None
    
    # This is the key - we need to provide a tensor_quant that will be properly resolved
    # by the Brevitas dependency injection system
    @value
    def tensor_quant():
        # Import here to avoid circular imports
        from brevitas.core.quant import IntQuant
        return IntQuant
