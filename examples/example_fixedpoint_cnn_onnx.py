"""
Example: Small Brevitas CNN with Fixed-Point Weight Quantization & ONNX Export

This script demonstrates:
1. Building a small CNN using Brevitas quantized layers.
2. Applying the custom FixedPointPerTensorWeightQuant quantizer to weights.
3. Running a dummy forward pass.
4. Exporting the model to ONNX using the legacy exporter (dynamo=False), 
   which is required to support the custom autograd.Function symbolic shim.
"""

import torch
import torch.nn as nn
import brevitas.nn as qnn
from quantizers.fixedpoint_per_tensor_weights import FixedPointPerTensorWeightQuant


class SmallFixedPointCNN(nn.Module):
    """A minimal CNN for demonstration purposes."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = qnn.QuantConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_quant=FixedPointPerTensorWeightQuant,
        )
        self.relu = qnn.QuantReLU()
        # Brevitas does not provide a QuantGlobalAvgPool2d wrapper.
        # Use standard PyTorch adaptive average pooling instead.
        self.pool = nn.AdaptiveAvgPool2d(1)
        # QuantLinear expects a 2D input (batch, in_features).
        # We must explicitly flatten spatial dimensions before the linear layer.
        self.flatten = nn.Flatten()
        self.fc = qnn.QuantLinear(
            in_features=16,
            out_features=num_classes,
            weight_quant=FixedPointPerTensorWeightQuant,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def main():
    # 1. Initialize model and set to evaluation mode
    model = SmallFixedPointCNN(num_classes=10)
    model.eval()

    # 2. Create dummy input matching expected shape (batch, channels, height, width)
    dummy_input = torch.randn(1, 3, 32, 32)

    # 3. Perform dummy inference
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Dummy inference successful. Output shape: {output.shape}")

    # 4. Export to ONNX
    # Note: dynamo=False is explicitly required because the quantizer uses
    # torch.autograd.Function.symbolic to emit custom ONNX nodes.
    # The modern dynamo exporter does not support this pattern.
    onnx_path = "small_fixedpoint_cnn.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=11,
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"Model successfully exported to {onnx_path}")


if __name__ == "__main__":
    main()
