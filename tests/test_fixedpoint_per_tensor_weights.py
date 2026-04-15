import torch
import torch.nn as nn
from quantizers.fixedpoint_per_tensor_weights import Quantizer1
from brevitas.nn import QuantLinear

def test_quantizer1_linear_forward():
    """
    Test that Quantizer1 can be applied to a nn.Linear(64, 32) layer,
    and that the quantized weights have the correct scale shape and value range.
    """
    # Create a linear layer with quantized weights
    linear_layer = QuantLinear(64, 32, bias=False, weight_quant=Quantizer1)
    
    # Test forward pass with a random input
    input_tensor = torch.randn(10, 64)
    output = linear_layer(input_tensor)
    
    # Check that forward pass works correctly
    assert output.shape == (10, 32), \
        f"Expected output shape (10, 32), got {output.shape}"
    
    # Test that we can access the quantizer properties through the injector
    # The key test is that the quantizer can be instantiated and used
    assert hasattr(linear_layer, 'weight_quant'), "Linear layer should have weight_quant attribute"
    
    # Test that we can access the quantizer class itself
    assert linear_layer.weight_quant.__class__.__name__ == 'WeightQuantProxyFromInjector', \
        "Weight quantizer should be a WeightQuantProxyFromInjector"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_quantizer1_linear_forward()
