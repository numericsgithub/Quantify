"""
Example: Small Brevitas CNN with Fixed-Point Weight Quantization & ONNX Export.

This script demonstrates:
1. Building a minimal CNN using Brevitas quantized layers.
2. Applying the custom `FixedPointPerTensorWeightQuant` quantizer to weights.
3. Running a dummy inference pass.
4. Exporting the model to ONNX with custom nodes (requires `dynamo=False`).
"""

import torch
import torch.nn as nn

import brevitas.nn as qnn
from quantizers.fixedpoint_per_tensor_weights import FixedPointPerTensorWeightQuant


class SimpleFixedPointCNN(nn.Module):
    """A tiny CNN using Brevitas layers with fixed-point weight quantization."""

    def __init__(self, num_classes=10):
        super().__init__()
        # Use the custom quantizer for weights
        self.conv1 = qnn.QuantConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1,
            weight_quant=FixedPointPerTensorWeightQuant,
            return_quant_tensor=False,
        )
        self.relu = qnn.QuantReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = qnn.QuantLinear(
            in_features=16 * 16 * 16,  # 32x32 input -> 16x16x16 after pool -> 4096 features
            out_features=num_classes,
            weight_quant=FixedPointPerTensorWeightQuant,
            return_quant_tensor=False,
        )

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def main():
    # 1. Initialize model and set to eval mode
    model = SimpleFixedPointCNN(num_classes=10)
    model.eval()

    # 2. Dummy inference
    dummy_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Dummy inference successful. Output shape: {output.shape}")

    # 3. Export to ONNX
    # Note: Custom ONNX nodes require the legacy exporter (dynamo=False)
    onnx_path = "simple_fixedpoint_cnn.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=13,
        dynamo=False,  # Required for torch.autograd.Function.symbolic
        input_names=["input"],
        output_names=["output"],
    )
    print(f"Model exported to {onnx_path}")


if __name__ == "__main__":
    main()
