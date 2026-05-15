import torch.nn as nn
import brevitas.nn as qnn

from .blocks import DepthwiseSeparableBlock


class QuantMobileNetCIFAR(nn.Module):
    """Small MobileNet-style quantized CNN for CIFAR-10.

    Stem (3x3 conv, stride 1) keeps 32x32 resolution, then six DS
    blocks with two stride-2 downsamples bring us to 8x8 x 256.
    Global average pooling and a single QuantLinear produce the logits.
    """

    # (out_channels, stride) for each depthwise-separable block.
    BLOCK_CFG = [
        (16, 1),
        (64, 1),
        (32, 2),# 32 -> 16
        (64, 2), # 16 -> 8
        (128, 2),  # 8 ->  4
        (128, 2),  # 4 ->  2
        (64, 2),  # 4 ->  1
    ]

    def __init__(self,
                 num_classes: int = 10,
                 weight_bit_width: int = 8,
                 act_bit_width: int = 8):
        super().__init__()

        self.quant_inp = nn.Identity()

        # Standard 3x3 stem conv — first layer is usually kept dense
        # (no depthwise) because it has only 3 input channels.
        self.stem = nn.Sequential(
            qnn.QuantConv2d(3, 32, kernel_size=3, padding=1, bias=True,
                            weight_bit_width=weight_bit_width),
            nn.BatchNorm2d(32),
            qnn.QuantReLU(bit_width=act_bit_width),
        )

        blocks = []
        in_ch = 32
        for out_ch, stride in self.BLOCK_CFG:
            blocks.append(DepthwiseSeparableBlock(
                in_ch, out_ch, stride, weight_bit_width, act_bit_width))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
           nn.AdaptiveAvgPool2d(1),
           nn.Flatten(),
           qnn.QuantLinear(in_ch, num_classes, bias=True,
                           weight_bit_width=weight_bit_width),
        )

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
