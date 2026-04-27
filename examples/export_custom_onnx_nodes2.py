import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx

# 1. Define the Function
class CustomQuantConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding):
        # 2. Implement forward: execute the actual PyTorch computation
        return F.conv2d(x, weight, bias, stride=stride, padding=padding)

    @staticmethod
    def symbolic(g, x, weight, bias, stride, padding):
        # 3. Implement symbolic: construct the ONNX graph
        # Emit the custom node and attach attributes with correct type suffixes (_i for int)
        return g.op(
            "mydomain::CustomQuantConv",
            x,
            weight,
            bias,
            stride_i=stride,
            padding_i=padding
        )

# 4. Integrate into Module
class BrevitasCustomModel(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_ch))

    def forward(self, x):
        # Call CustomFn.apply instead of standard layers
        return CustomQuantConvFn.apply(
            x, self.weight, self.bias, self.stride, self.padding
        )

def main():
    # Setup model and dummy input
    model = BrevitasCustomModel(in_ch=3, out_ch=16, kernel_size=3, stride=1, padding=1)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # Pre-export Inference: Run a dummy forward pass to ensure forward executes without errors
    print("Running dummy inference...")
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Dummy inference output shape: {output.shape}")
        
    # Export to ONNX
    onnx_path = "custom_quant_conv.onnx"
    print(f"Exporting to ONNX with dynamo=False...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=14,
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
        verbose=False
    )
    print(f"Model successfully exported to {onnx_path}")
    print("Open the file in Netron to verify the custom node 'mydomain::CustomQuantConv'.")

if __name__ == "__main__":
    main()
