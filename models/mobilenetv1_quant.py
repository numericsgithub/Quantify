"""
Quantized MobileNetV1 for ImageNet.

Implements the original Howard et al. (2017) MobileNet with depthwise-separable
convolutions. No torchvision pretrained weights are available for V1, so this
model must be trained from scratch or initialised from a custom checkpoint.

Architecture:
    Stem (3→32 conv, s=2) + 13 depthwise-separable blocks + GlobalAvgPool + FC
    Total 28 convolutional layers.
"""

import torch.nn as nn
import brevitas.nn as qnn


def _relu(act_quant):
    if act_quant is not None:
        return qnn.QuantReLU(act_quant=act_quant)
    return nn.ReLU(inplace=True)


class QuantDWSepBlock(nn.Module):
    """Depthwise-separable block: DW 3×3 → BN → ReLU → PW 1×1 → BN → ReLU."""

    def __init__(self, in_ch, out_ch, stride, weight_quant=None, act_quant=None):
        super().__init__()
        self.dw = qnn.QuantConv2d(
            in_ch, in_ch, 3, stride=stride, padding=1,
            groups=in_ch, bias=False, weight_quant=weight_quant,
        )
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.relu_dw = _relu(act_quant)

        self.pw = qnn.QuantConv2d(in_ch, out_ch, 1, bias=False, weight_quant=weight_quant)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        self.relu_pw = _relu(act_quant)

    def forward(self, x):
        x = self.relu_dw(self.bn_dw(self.dw(x)))
        return self.relu_pw(self.bn_pw(self.pw(x)))


class QuantMobileNetV1(nn.Module):
    """
    Quantized MobileNetV1 for ImageNet (224×224 input).

    Args:
        num_classes:  Output logits. Default 1000.
        weight_quant: Brevitas injector class for weight quantization.
        act_quant:    Brevitas injector class for activation quantization.
        bias_quant:   Brevitas injector class for bias quantization (fc only).
    """

    # (out_channels, stride) for the 13 depthwise-separable blocks
    _CFG = [
        (64,   1),
        (128,  2), (128,  1),
        (256,  2), (256,  1),
        (512,  2),
        (512,  1), (512,  1), (512,  1), (512,  1), (512,  1),
        (1024, 2), (1024, 1),
    ]

    def __init__(self, num_classes=1000, weight_quant=None, act_quant=None, bias_quant=None):
        super().__init__()

        self.stem = nn.Sequential(
            qnn.QuantConv2d(3, 32, 3, stride=2, padding=1, bias=False, weight_quant=weight_quant),
            nn.BatchNorm2d(32),
            _relu(act_quant),
        )

        blocks = []
        in_ch = 32
        for out_ch, stride in self._CFG:
            blocks.append(QuantDWSepBlock(in_ch, out_ch, stride, weight_quant, act_quant))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        fc_kw = {"bias_quant": bias_quant} if bias_quant is not None else {}
        self.fc = qnn.QuantLinear(1024, num_classes, bias=True,
                                   weight_quant=weight_quant, output_quant=None, **fc_kw)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)
