import torch
import torch.nn as nn
from brevitas.quant.scaled_int import IntQuant
from brevitas.inject import value
from brevitas.inject.enum import ScalingImplType, BitWidthImplType, FloatToIntImplType, QuantType

# Quantizer 1: Fixed-point per-tensor weight quantizer
class Quantizer1(IntQuant):
    """
    Quantizer 1 implementation as a Brevitas QuantType.
    This is a fixed-point per-tensor weight quantizer.
    """
    
    # Define the quantization type as weight quantization
    quant_type = QuantType.INT
    
    # Set the bit width for the quantizer
    bit_width = value(8)
    
    # Set the scaling factor to be learned during training
    scaling_impl_type = ScalingImplType.PARAMETER
    
    # Set the quantization to be per-tensor
    scaling_per_output_type = ScalingImplType.PARAMETER
    
    # Set the quantization to be symmetric
    signed = True
    
    # Set the quantization to be fixed-point
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.ROUND
