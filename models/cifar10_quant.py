import torch.nn as nn
import brevitas.nn as qnn
import torch.nn as nn


class DepthwiseSeparableBlockFloat(nn.Module):
    """Depthwise 3x3 (BN, ReLU) followed by pointwise 1x1 (BN, ReLU).

    Stride is applied on the depthwise conv, which is also where
    spatial downsampling happens.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int,
                  weight_bit_width: int, act_bit_width: int):
        super().__init__()

        # Depthwise: one 3x3 filter per input channel (groups == in_ch).
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1,
            groups=in_ch, bias=True)
        #self.bn_dw = nn.BatchNorm2d(in_ch)
        self.relu_dw = nn.ReLU()

        # Pointwise: 1x1 conv mixing the channels.
        self.pw = nn.Conv2d(
            in_ch, out_ch, kernel_size=1, bias=True)
        #self.bn_pw = nn.BatchNorm2d(out_ch)
        self.relu_pw = nn.ReLU()

    def forward(self, x):
        x = self.relu_dw(self.bn_dw(self.dw(x)))
        x = self.relu_pw(self.bn_pw(self.pw(x)))
        return x

class DepthwiseSeparableBlock(nn.Module):
    """Depthwise 3x3 (BN, ReLU) followed by pointwise 1x1 (BN, ReLU).

    Stride is applied on the depthwise conv, which is also where
    spatial downsampling happens.

    Note on quantization
    --------------------
    Depthwise convs in particular often benefit noticeably from
    per-channel weight quantization (each depthwise kernel is
    independent), whereas the default here is per-tensor. For an
    8-bit run the difference is small; at 4 bits it becomes visible.
    Swap ``weight_quant=Int8WeightPerChannelFloat`` into the depthwise
    ``QuantConv2d`` if you want to try it.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int,
                 weight_bit_width: int, act_bit_width: int):
        super().__init__()

        # Depthwise: one 3x3 filter per input channel (groups == in_ch).
        self.dw = qnn.QuantConv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1,
            groups=in_ch, bias=True,
            weight_bit_width=weight_bit_width)
        #self.bn_dw = nn.BatchNorm2d(in_ch)
        # self.relu_dw = qnn.QuantReLU(bit_width=act_bit_width)
        self.relu_dw = nn.ReLU()



        # Pointwise: 1x1 conv mixing the channels.
        self.pw = qnn.QuantConv2d(
            in_ch, out_ch, kernel_size=1, bias=True,
            weight_bit_width=weight_bit_width)
        #self.bn_pw = nn.BatchNorm2d(out_ch)
        # self.relu_pw = qnn.QuantReLU(bit_width=act_bit_width)
        self.relu_pw = nn.ReLU()

    def forward(self, x):
        # x = self.relu_dw(self.bn_dw(self.dw(x)))
        # x = self.relu_pw(self.bn_pw(self.pw(x)))
        x = self.relu_dw(self.dw(x))
        x = self.relu_pw(self.pw(x))
        return x

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
            #nn.BatchNorm2d(32),
            qnn.QuantReLU(bit_width=act_bit_width),
        )

        blocks = []
        in_ch = 32
        for out_ch, stride in self.BLOCK_CFG:
            blocks.append(DepthwiseSeparableBlock(
                in_ch, out_ch, stride, weight_bit_width, act_bit_width))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        #self.head = nn.Sequential(
        #    nn.AdaptiveAvgPool2d(1),
        #    nn.Flatten(),
        #    qnn.QuantLinear(in_ch, num_classes, bias=True,
        #                    weight_bit_width=weight_bit_width),
        #)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, num_classes, bias=True),
        )

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

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
            #nn.BatchNorm2d(out_ch),
            qnn.QuantReLU(bit_width=a_bits),
        ]

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.features(x)
        x = self.classifier(x)
        return x
