"""
Example: Exporting Brevitas Quantized Models with Custom ONNX Nodes.

This script demonstrates how to override the default ONNX export behavior
of Brevitas layers to emit custom nodes (e.g., mydomain::CustomQuantConv)
with additional attributes (strings, tensors, scalars, booleans).

It shows the simplest, most robust pattern for replacing standard PyTorch/Brevitas
ops with custom ONNX nodes while keeping the forward pass compatible with Brevitas.
"""

import torch
import torch.nn as nn
import torch.onnx
import brevitas.nn as qnn
from quantizers.fixedpoint_per_tensor_weights import FixedPointPerTensorWeightQuant


# -----------------------------------------------------------------------------
# 1. Custom torch.autograd.Function with symbolic method for ONNX export
# -----------------------------------------------------------------------------
class CustomQuantConvFn(torch.autograd.Function):
    """
    Wraps a convolution forward pass but emits a custom ONNX node
    instead of the standard PyTorch/Brevitas ops.
    """
    @staticmethod
    def symbolic(g, x, weight, bias, stride, padding, bit_width, rounding_mode, is_signed):
        # Emit custom node with various attribute types as requested
        return g.op(
            "mydomain::CustomQuantConv",
            x, weight, bias,
            stride_i=int(stride),
            padding_i=int(padding),
            bit_width_i=int(bit_width),
            rounding_mode_s=str(rounding_mode),
            is_signed_i=int(is_signed),
            dummy_tensor_t=torch.tensor([1.0, 2.0, 3.0]),
        )

    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, bit_width, rounding_mode, is_signed):
        # Perform actual forward pass. In a real Brevitas workflow, you would
        # apply the quantizer here before or after the convolution.
        # For this export demo, we use standard conv to keep tracing simple
        # and avoid Brevitas' internal export hooks interfering.
        return torch.nn.functional.conv2d(x, weight, bias, stride=stride, padding=padding)


class CustomQuantLinearFn(torch.autograd.Function):
    """
    Wraps a linear forward pass but emits a custom ONNX node.
    """
    @staticmethod
    def symbolic(g, x, weight, bias, bit_width, rounding_mode, is_signed):
        return g.op(
            "mydomain::CustomQuantLinear",
            x, weight, bias,
            bit_width_i=int(bit_width),
            rounding_mode_s=str(rounding_mode),
            is_signed_i=int(is_signed),
            dummy_tensor_t=torch.tensor([4.0, 5.0]),
        )

    @staticmethod
    def forward(ctx, x, weight, bias, bit_width, rounding_mode, is_signed):
        # Standard linear layer forward pass
        return torch.nn.functional.linear(x, weight, bias)


# -----------------------------------------------------------------------------
# 2. Simple Quantized Model using the custom functions
# -----------------------------------------------------------------------------
class SimpleQuantizedModel(nn.Module):
    def __init__(self, in_ch=3, out_ch=10, bit_width=8, rounding_mode="round"):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.bit_width = bit_width
        self.rounding_mode = rounding_mode
        self.is_signed = True

        # Initialize weights and biases
        self.conv_weight = nn.Parameter(torch.randn(16, 3, 3, 3))
        self.conv_bias = nn.Parameter(torch.zeros(16))
        self.fc_weight = nn.Parameter(torch.randn(10, 16))
        self.fc_bias = nn.Parameter(torch.zeros(10))

    def forward(self, x):
        x = CustomQuantConvFn.apply(
            x, self.conv_weight, self.conv_bias,
            1, 1, self.bit_width, self.rounding_mode, self.is_signed
        )
        x = torch.relu(x)
        # Pool spatial dimensions to match the fully connected layer's input size
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = CustomQuantLinearFn.apply(
            x, self.fc_weight, self.fc_bias,
            self.bit_width, self.rounding_mode, self.is_signed
        )
        return x


# -----------------------------------------------------------------------------
# 3. Dummy Inference & ONNX Export
# -----------------------------------------------------------------------------
def main():
    device = torch.device("cpu")
    model = SimpleQuantizedModel().to(device)
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, 32, 32, device=device)

    # Dummy inference
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Dummy inference successful. Output shape: {output.shape}")

    # Export to ONNX
    onnx_path = "custom_quantized_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,  # Force legacy exporter to support torch.autograd.Function.symbolic
    )
    print(f"Model exported to {onnx_path}")
    print("Open the file in Netron to verify the custom 'mydomain::CustomQuantConv' and 'mydomain::CustomQuantLinear' nodes.")


if __name__ == "__main__":
    main()
