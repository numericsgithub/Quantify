import torch
import torch.nn as nn
import brevitas.nn as qnn
from quantizers.fixedpoint_per_tensor_weights import FixedPointPerTensorWeightQuant

class SimpleFixedPointCNN(nn.Module):
    """
    A small CNN demonstrating the FixedPointPerTensorWeightQuant quantizer.
    Uses Brevitas QuantConv2d, QuantReLU, and QuantLinear layers.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            qnn.QuantConv2d(3, 16, kernel_size=3, padding=1, weight_quant=FixedPointPerTensorWeightQuant),
            qnn.QuantReLU(),
            qnn.QuantConv2d(16, 32, kernel_size=3, padding=1, weight_quant=FixedPointPerTensorWeightQuant),
            qnn.QuantReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = qnn.QuantLinear(32, num_classes, weight_quant=FixedPointPerTensorWeightQuant)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def main():
    # 1. Initialize model and set to evaluation mode
    model = SimpleFixedPointCNN(num_classes=10)
    model.eval()
    
    # 2. Perform dummy inference
    dummy_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"✅ Dummy inference successful. Output shape: {output.shape}")
    
    # 3. Export to ONNX
    # NOTE: dynamo=False is strictly required because FixedPointQuantFn uses 
    # torch.autograd.Function.symbolic, which is only supported by the legacy exporter.
    onnx_path = "simple_fixedpoint_cnn.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        dynamo=False,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"✅ Model successfully exported to {onnx_path}")


if __name__ == "__main__":
    main()
