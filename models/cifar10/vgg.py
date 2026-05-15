import torch.nn as nn
import brevitas.nn as qnn


class QuantVGG(nn.Module):
    """
    Small VGG-style quantized CNN for CIFAR-10.

    Brevitas cheat sheet
    --------------------
    * ``qnn.QuantIdentity``  – fake-quantizes the (float) network input.
    * ``qnn.QuantConv2d`` /
      ``qnn.QuantLinear``    – standard conv/linear with fake-quant
                               applied to weights during forward.
    * ``qnn.QuantReLU``      – ReLU whose output is fake-quantized to
                               an unsigned integer range.
    * ``nn.BatchNorm*``      – kept in float, can be folded into the
                               preceding conv/linear later.
    """

    def __init__(self,
                 num_classes: int = 10,
                 weight_bit_width: int = 8,
                 act_bit_width: int = 8):
        super().__init__()

        self.quant_inp = nn.Identity()

        self.features = nn.Sequential(
            *self._conv_block(3,   64,  weight_bit_width, act_bit_width),
            *self._conv_block(64,  64,  weight_bit_width, act_bit_width),
            nn.MaxPool2d(2),                        # 32 -> 16
            *self._conv_block(64,  128, weight_bit_width, act_bit_width),
            *self._conv_block(128, 128, weight_bit_width, act_bit_width),
            nn.MaxPool2d(2),                        # 16 -> 8
            *self._conv_block(128, 256, weight_bit_width, act_bit_width),
            *self._conv_block(256, 256, weight_bit_width, act_bit_width),
            nn.AdaptiveAvgPool2d(1),                #  8 -> 1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            qnn.QuantLinear(256, 256, bias=False,
                            weight_bit_width=weight_bit_width),
            nn.BatchNorm1d(256),
            qnn.QuantReLU(bit_width=act_bit_width),
            qnn.QuantLinear(256, num_classes, bias=True,
                            weight_bit_width=weight_bit_width),
        )

    @staticmethod
    def _conv_block(in_ch, out_ch, w_bits, a_bits):
        return [
            qnn.QuantConv2d(in_ch, out_ch, kernel_size=3, padding=1,
                            bias=False, weight_bit_width=w_bits),
            nn.BatchNorm2d(out_ch),
            qnn.QuantReLU(bit_width=a_bits),
        ]

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.features(x)
        x = self.classifier(x)
        return x
