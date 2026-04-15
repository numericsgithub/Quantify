import torch
import torch.nn as nn
from brevitas.quant import QuantType
from brevitas.quant.scaled_int import IntQuant
from brevitas.quant.fixed_point import FixedPointQuant
from brevitas.inject import value
from brevitas.inject import param as param_injector
from brevitas.inject import indicator as indicator_injector

# Quantizer 1: Fixed-point per-tensor weight quantizer
class Quantizer1(FixedPointQuant):
    """
    Quantizer 1 implementation as a Brevitas QuantType.
    This is a fixed-point per-tensor weight quantizer.
    """
    
    # Define the quantization type as weight quantization
    quant_type = QuantType.WEIGHT
    
    # Set the bit width for the quantizer
    bit_width = value(8)
    
    # Set the scaling factor to be learned during training
    scaling_per_tensor = True
    
    # Set the zero point to be learned during training
    zero_point_per_tensor = True
    
    # Set the quantization mode to per-tensor
    per_tensor = True
    
    # Set the quantization to be symmetric
    symmetric = True
    
    # Set the quantization to be signed
    signed = True
    
    # Set the scaling factor to be learned
    scaling_impl = param_injector.LearnedPerTensorScaling
    
    # Set the zero point to be learned
    zero_point_impl = param_injector.LearnedPerTensorZeroPoint
    
    # Set the quantization to be fixed-point
    quant_impl = indicator_injector.FixedPointQuant
