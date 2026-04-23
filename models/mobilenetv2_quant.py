import torch
import torch.nn as nn
import brevitas.nn as qnn
from quantizers.fixedpoint_per_tensor_weights import FixedPointPerTensorWeightQuant

class QuantInvertedResidual(nn.Module):
    """Quantized Inverted Residual Block for MobileNetV2."""
    def __init__(self, inp, oup, stride, expand_ratio, weight_bit_width, act_bit_width, weight_quant):
        super().__init__()
        self.stride = stride
        self.use_res_connect = stride == 1 and inp == oup

        hidden_dim = int(round(inp * expand_ratio))
        
        # Expansion
        self.conv = nn.Sequential(
            # pw
            qnn.QuantConv2d(inp, hidden_dim, 1, 1, 0, bias=False, 
                            weight_bit_width=weight_bit_width, weight_quant=weight_quant),
            nn.BatchNorm2d(hidden_dim),
            qnn.QuantReLU(bit_width=act_bit_width),
            # dw
            qnn.QuantConv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False, 
                            weight_bit_width=weight_bit_width, weight_quant=weight_quant),
            nn.BatchNorm2d(hidden_dim),
            qnn.QuantReLU(bit_width=act_bit_width),
            # pw-linear
            qnn.QuantConv2d(hidden_dim, oup, 1, 1, 0, bias=False, 
                            weight_bit_width=weight_bit_width, weight_quant=weight_quant),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class QuantMobileNetV2(nn.Module):
    """
    Quantized MobileNetV2 for ImageNet.
    Mirrors the structure of torchvision.models.mobilenet_v2.MobileNet_V2
    to allow easy weight mapping.
    """
    def __init__(self, num_classes=1000, weight_bit_width=8, act_bit_width=8):
        super().__init__()

        # Dynamic injector subclass to set bit_width
        class FixedPointWeightQuant(FixedPointPerTensorWeightQuant):
            bit_width = weight_bit_width

        # MobileNetV2 Config: (expand_ratio, channels, stride)
        self.config = [
            # t:.relu, type:Conv, stride:1
            (1, 16, 1),
            # t:.relu, type:InvertedResidual, stride:1, expand_ratio:1
            (1, 24, 2),
            (2, 32, 1),
            (3, 64, 2),
            (4, 96, 1),
            (5, 160, 2),
            (6, 160, 1),
            (6, 160, 1),
            (6, 160, 2),
            (6, 160, 1),
            (6, 160, 1),
            (6, 160, 2),
            (6, 160, 1),
            (6, 160, 1),
            (6, 160, 2),
            (6, 160, 1),
            (6, 160, 1),
        ]

        # Stem
        self.features = [
            qnn.QuantConv2d(3, 32, 3, 2, 1, bias=False, 
                            weight_bit_width=weight_bit_width, weight_quant=FixedPointWeightQuant),
            nn.BatchNorm2d(32),
            qnn.QuantReLU(bit_width=act_bit_width),
        ]

        # Inverted Residual Blocks
        in_channels = 32
        for exp, out_channels, stride in self.config:
            self.features.append(
                QuantInvertedResidual(in_channels, out_channels, stride, exp, 
                                      weight_bit_width, act_bit_width, FixedPointWeightQuant)
            )
            in_channels = out_channels

        # Final Conv layer
        self.features.append(
            qnn.QuantConv2d(in_channels, 1280, 1, 1, 0, bias=False, 
                            weight_bit_width=weight_bit_width, weight_quant=FixedPointWeightQuant)
        )
        self.features.append(nn.BatchNorm2d(1280))
        self.features.append(qnn.QuantReLU(bit_width=act_bit_width))

        self.features = nn.Sequential(*self.features)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            qnn.QuantLinear(1280, num_classes, bias=True, 
                            weight_bit_width=weight_bit_width, weight_quant=FixedPointWeightQuant)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
