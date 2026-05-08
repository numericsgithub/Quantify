"""
Example: Simple Fixed-Point CNN with Brevitas & ONNX Export

This script demonstrates how to:
1. Build a small CNN using Brevitas layers and the custom FixedPointPerTensorWeightQuant quantizer.
2. Run a dummy inference pass.
3. Export the model to ONNX using the legacy exporter (dynamo=False), which is required
   for custom torch.autograd.Function symbolic methods.
4. Load the exported ONNX model and verify that the custom quantizer node is present.
5. Inspect the quantizer attributes embedded in the ONNX graph.
"""

import torch
import torch.nn as nn
import brevitas.nn as qnn
import onnx
import onnxruntime as ort
import numpy as np
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


def verify_onnx_model(onnx_path: str) -> bool:
    """Load ONNX model and check for the custom quantizer node."""
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    
    custom_node_found = False
    for node in model.graph.node:
        if node.op_type == "FixedPointQuant" and node.domain == "mydomain":
            custom_node_found = True
            print(f"  Found custom node: {node.op_type} (domain: {node.domain})")
            print(f"    Attributes: {[(a.name, a.i if a.i else a.f if a.f else a.s) for a in node.attribute]}")
            break
            
    if not custom_node_found:
        print("  WARNING: Custom 'mydomain::FixedPointQuant' node not found in ONNX graph!")
        return False
    return True


def compare_outputs(pytorch_model, onnx_path, dummy_input, atol=1e-4):
    """Run inference on PyTorch and ONNX Runtime, then compare outputs."""
    try:
        pytorch_model.eval()
        with torch.no_grad():
            pt_output = pytorch_model(dummy_input).numpy()
            
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_inputs = {sess.get_inputs()[0].name: dummy_input.numpy()}
        ort_output = sess.run(None, ort_inputs)[0]
        
        if np.allclose(pt_output, ort_output, atol=atol):
            print(f"  Outputs match! Max diff: {np.max(np.abs(pt_output - ort_output)):.2e}")
            return True
        else:
            print(f"  Outputs mismatch! Max diff: {np.max(np.abs(pt_output - ort_output)):.2e}")
            return False
    except Exception as e:
        print(f"  ONNX Runtime inference skipped or failed (custom op may not be registered): {e}")
        return None


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

    # 3. Verify ONNX Model
    print("\nVerifying ONNX model...")
    if verify_onnx_model(onnx_path):
        print("  ONNX model structure verified successfully.")
    else:
        print("  ONNX model verification failed.")
        return

    # 4. Compare Outputs (Optional, requires onnxruntime)
    print("\nComparing PyTorch and ONNX Runtime outputs...")
    compare_outputs(model, onnx_path, dummy_input)


if __name__ == "__main__":
    main()
