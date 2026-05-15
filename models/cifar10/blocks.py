import torch.nn as nn
import brevitas.nn as qnn


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
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.relu_dw = qnn.QuantReLU(bit_width=act_bit_width)

        # Pointwise: 1x1 conv mixing the channels.
        self.pw = qnn.QuantConv2d(
            in_ch, out_ch, kernel_size=1, bias=True,
            weight_bit_width=weight_bit_width)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        self.relu_pw = qnn.QuantReLU(bit_width=act_bit_width)

    def forward(self, x):
        x = self.relu_dw(self.bn_dw(self.dw(x)))
        x = self.relu_pw(self.bn_pw(self.pw(x)))
        return x
