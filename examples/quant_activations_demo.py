"""
Dummy model exercising every quantized activation in `quantizers/activations.py`
(ReLU, ReLU6, Sigmoid, Tanh, SiLU, GELU, LeakyReLU, Softmax), each with its
default fixed-point output quantizer, then exports it to ONNX.

Run:
    python examples/quant_activations_demo.py

Produces `runs/quant_activations_demo/model.onnx` -- open it in Netron to see
each activation's single ONNX node feeding into its own
`Quantify::FixedPointQuant` quantizer node.
"""

from pathlib import Path

import torch
import torch.nn as nn
import brevitas.nn as qnn

from quantizers import FixedPointPerTensorWeightQuant
from quantizers.activations import (
    QuantReLU,
    QuantReLU6,
    QuantSigmoid,
    QuantTanh,
    QuantSiLU,
    QuantGELU,
    QuantLeakyReLU,
    QuantSoftmax,
)
from utils.onnx_export import export_onnx_with_io


class ActivationZoo(nn.Module):
    """A tiny conv/linear stack where every stage is followed by a different
    quantized activation -- just enough structure (varying channel/feature
    counts) to give each activation a nontrivial input distribution to
    calibrate against.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = qnn.QuantConv2d(
            3, 8, 3, padding=1, bias=False, weight_quant=FixedPointPerTensorWeightQuant
        )
        self.act1 = QuantReLU(bit_width=8)

        self.conv2 = qnn.QuantConv2d(
            8, 8, 3, padding=1, bias=False, weight_quant=FixedPointPerTensorWeightQuant
        )
        self.act2 = QuantReLU6(bit_width=8)

        self.conv3 = qnn.QuantConv2d(
            8, 8, 3, padding=1, bias=False, weight_quant=FixedPointPerTensorWeightQuant
        )
        self.act3 = QuantSigmoid(bit_width=8)

        self.conv4 = qnn.QuantConv2d(
            8, 8, 3, padding=1, bias=False, weight_quant=FixedPointPerTensorWeightQuant
        )
        self.act4 = QuantTanh(bit_width=8)

        self.conv5 = qnn.QuantConv2d(
            8, 8, 3, padding=1, bias=False, weight_quant=FixedPointPerTensorWeightQuant
        )
        self.act5 = QuantSiLU(bit_width=8)

        self.conv6 = qnn.QuantConv2d(
            8, 8, 3, padding=1, bias=False, weight_quant=FixedPointPerTensorWeightQuant
        )
        self.act6 = QuantGELU(bit_width=8)

        self.conv7 = qnn.QuantConv2d(
            8, 8, 3, padding=1, bias=False, weight_quant=FixedPointPerTensorWeightQuant
        )
        self.act7 = QuantLeakyReLU(bit_width=8)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = qnn.QuantLinear(
            8, 10, bias=True, weight_quant=FixedPointPerTensorWeightQuant
        )
        self.act8 = QuantSoftmax(dim=-1, bit_width=8)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.act5(self.conv5(x))
        x = self.act6(self.conv6(x))
        x = self.act7(self.conv7(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.act8(x)
        return x


def main():
    torch.manual_seed(0)
    model = ActivationZoo()

    dummy_input = torch.randn(2, 3, 16, 16)

    # Calibrate: one forward pass in training mode lets every quantizer
    # (weight and activation) run its calibration search.
    model.train()
    model(dummy_input)

    model.eval()
    out_dir = Path(__file__).resolve().parent.parent / "runs" / "quant_activations_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"

    export_onnx_with_io(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
    )

    print(f"Exported ONNX model to: {onnx_path.resolve()}")

    import onnx

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("\nONNX graph nodes:")
    for node in onnx_model.graph.node:
        domain = node.domain or "(default)"
        print(f"  {node.op_type:20s} domain={domain}")


if __name__ == "__main__":
    main()
