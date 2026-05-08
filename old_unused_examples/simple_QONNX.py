import torch
import torch.nn as nn

from brevitas.nn import QuantLinear
from brevitas.nn import QuantReLU
from brevitas.export import export_qonnx

class QuantMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = QuantLinear(
            16, 32,
            weight_bit_width=4,
            bias=True
        )

        self.act1 = QuantReLU(bit_width=4)

        self.fc2 = QuantLinear(
            32, 10,
            weight_bit_width=4,
            bias=True
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

model = QuantMLP()
model.eval()

dummy_input = torch.randn(1, 16)

torch.onnx.export(
    model,
    dummy_input,
    "quant_model.onnx",
    dynamo=False
)