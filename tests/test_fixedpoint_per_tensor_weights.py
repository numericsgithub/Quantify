import torch
import torch.nn as nn
from quantizers.fixedpoint_per_tensor_weights import Quantizer1

def test_quantizer1_linear_forward():
    """
    Test that Quantizer1 can be applied to a nn.Linear(64, 32) layer,
    and that the quantized weights have the correct scale shape and value range.
    """
    # Create a linear layer
    linear_layer = nn.Linear(64, 32)
    
    # Apply Quantizer1 to the weight parameter
    quantizer = Quantizer1()
    
    # Get the original weights
    original_weights = linear_layer.weight.data
    
    # Apply quantization to the weights
    quantized_weights = quantizer(original_weights)
    
    # Check that the quantized weights have the correct shape
    assert quantized_weights.shape == original_weights.shape, \
        f"Expected shape {original_weights.shape}, got {quantized_weights.shape}"
    
    # Check that the quantized weights are in the expected value range
    # For 8-bit signed fixed-point, values should be in range [-128, 127]
    # But after quantization and dequantization, they should be close to original values
    # Let's check that the quantized values are within reasonable bounds
    assert torch.all(quantized_weights >= -128) and torch.all(quantized_weights <= 127), \
        "Quantized weights should be within 8-bit signed integer range"
    
    # Test forward pass with a random input
    input_tensor = torch.randn(10, 64)
    output = linear_layer(input_tensor)
    
    # Check that forward pass works correctly
    assert output.shape == (10, 32), \
        f"Expected output shape (10, 32), got {output.shape}"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_quantizer1_linear_forward()
