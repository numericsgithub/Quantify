import torch
import torch.nn as nn
import brevitas.nn as qnn
from quantizers.fixedpoint_per_tensor import FixedPointPerTensorWeightQuant

class QuantInvertedResidual(nn.Module):
    """Quantized Inverted Residual Block for MobileNetV2.
    
    Mirrors torchvision.models.mobilenetv2._InvertedResidual.
    """
    def __init__(self, inp, oup, stride, expand_ratio, weight_bit_width, act_bit_width, weight_quant, act_quant=None):
        super().__init__()
        self.stride = stride
        self.use_res_connect = stride == 1 and inp == oup

        hidden_dim = int(round(inp * expand_ratio))
        
        # The 'conv' attribute in torchvision is a Sequential containing:
        # 0: Conv2d (pw)
        # 1: BatchNorm2d
        # 2: ReLU6
        # 3: Conv2d (dw)
        # 4: BatchNorm2d
        # 5: ReLU6
        # 6: Conv2d (pw-linear)
        # 7: BatchNorm2d
        self.conv = nn.Sequential(
            # pw
            qnn.QuantConv2d(inp, hidden_dim, 1, 1, 0, bias=False, 
                            weight_bit_width=weight_bit_width, weight_quant=weight_quant),
            nn.BatchNorm2d(hidden_dim),
            qnn.QuantReLU(bit_width=act_bit_width, act_quant=act_quant),
            # dw
            qnn.QuantConv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False, 
                            weight_bit_width=weight_bit_width, weight_quant=weight_quant),
            nn.BatchNorm2d(hidden_dim),
            qnn.QuantReLU(bit_width=act_bit_width, act_quant=act_quant),
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

    Args:
        num_classes:     Output logits. Default 1000.
        weight_bit_width: Bit width for fixed-point weight quant (ignored when
                          weight_quant is provided explicitly).
        act_bit_width:   Bit width passed to QuantReLU layers.
        weight_quant:    Brevitas injector class for weight quantization. When
                         None (default), a FixedPointPerTensorWeightQuant
                         subclass with weight_bit_width is created automatically.
        act_quant:       Brevitas injector class for activation quantization.
        bias_quant:      Brevitas injector class for bias quantization (fc only).
    """
    def __init__(self, num_classes=1000, weight_bit_width=8, act_bit_width=8,
                 weight_quant=None, act_quant=None, bias_quant=None):
        super().__init__()

        if weight_quant is None:
            class weight_quant(FixedPointPerTensorWeightQuant):
                bit_width = weight_bit_width

        # Official MobileNetV2 Config: (expand_ratio, channels, num_blocks, stride)
        self.config = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 1],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 160, 1, 1],
        ]

        # Stem
        self.features = []
        self.features.append(
            qnn.QuantConv2d(3, 32, 3, 2, 1, bias=False,
                            weight_bit_width=weight_bit_width, weight_quant=weight_quant)
        )
        self.features.append(nn.BatchNorm2d(32))
        self.features.append(qnn.QuantReLU(bit_width=act_bit_width, act_quant=act_quant))

        # Inverted Residual Blocks
        in_channels = 32
        for t, c, n, s in self.config:
            for i in range(n):
                # Only the first block of each group uses the specified stride
                stride = s if i == 0 else 1
                self.features.append(
                    QuantInvertedResidual(in_channels, c, stride, t,
                                         weight_bit_width, act_bit_width, weight_quant, act_quant)
                )
                in_channels = c

        # Final Conv layer
        self.features.append(
            qnn.QuantConv2d(in_channels, 1280, 1, 1, 0, bias=False,
                            weight_bit_width=weight_bit_width, weight_quant=weight_quant)
        )
        self.features.append(nn.BatchNorm2d(1280))
        self.features.append(qnn.QuantReLU(bit_width=act_bit_width, act_quant=act_quant))

        self.features = nn.Sequential(*self.features)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        fc_kw = {"bias_quant": bias_quant} if bias_quant is not None else {}
        self.classifier = nn.Sequential(
            nn.Flatten(),
            qnn.QuantLinear(1280, num_classes, bias=True,
                            weight_bit_width=weight_bit_width, weight_quant=weight_quant,
                            output_quant=None, **fc_kw)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
