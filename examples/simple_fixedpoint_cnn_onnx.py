"""
Example: Simple Fixed-Point CNN with Brevitas & ONNX Export

This script demonstrates how to:
1. Build a small CNN using Brevitas layers and the custom FixedPointPerTensorWeightQuant quantizer.
2. Run a dummy inference pass.
3. Export the model to ONNX using the legacy exporter (dynamo=False), which is required
   for custom torch.autograd.Function symbolic methods.
"""

import torch
import torch.nn as nn
import brevitas.nn as qnn
from quantizers.fixedpoint_per_tensor_weights import FixedPointPerTensorWeightQuant


class SimpleFixedPointCNN(nn.Module):
    """A minimal CNN for CIFAR-10 using fixed-point weight quantization."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            qnn.QuantConv2d(3, 16, kernel_size=3, padding=1, weight_quant=FixedPointPerTensorWeightQuant),
            qnn.QuantReLU(),
            nn.MaxPool2d(2),
            qnn.QuantConv2d(16, 32, kernel_size=3, padding=1, weight_quant=FixedPointPerTensorWeightQuant),
            qnn.QuantReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            qnn.QuantLinear(32 * 8 * 8, 64, weight_quant=FixedPointPerTensorWeightQuant),
            qnn.QuantReLU(),
            qnn.QuantLinear(64, num_classes, weight_quant=FixedPointPerTensorWeightQuant),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    device = torch.device("cpu")
    model = SimpleFixedPointCNN(num_classes=10).to(device)
    model.eval()

    # Create a dummy input matching expected runtime dimensions
    dummy_input = torch.randn(1, 3, 32, 32).to(device)

    # 1. Dummy Inference
    print("Running dummy inference...")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Dummy inference output shape: {output.shape}")

    # 2. Export to ONNX
    onnx_path = "simple_fixedpoint_cnn.onnx"
    print(f"Exporting to ONNX: {onnx_path}...")
    
    # NOTE: dynamo=False is strictly required because FixedPointQuantFn uses
    # torch.autograd.Function.symbolic, which is only supported by the legacy exporter.
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        dynamo=False,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"Successfully exported model to {onnx_path}")


if __name__ == "__main__":
    main()
