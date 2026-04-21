import torch.nn as nn

class DepthwiseSeparableBlock(nn.Module):
    """Depthwise 3x3 (BN, ReLU) followed by pointwise 1x1 (BN, ReLU).

    Stride is applied on the depthwise conv, which is also where
    spatial downsampling happens.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()

        # Depthwise: one 3x3 filter per input channel (groups == in_ch).
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1,
            groups=in_ch, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.relu_dw = nn.ReLU()

        # Pointwise: 1x1 conv mixing the channels.
        self.pw = nn.Conv2d(
            in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        self.relu_pw = nn.ReLU()

    def forward(self, x):
        x = self.relu_dw(self.bn_dw(self.dw(x)))
        x = self.relu_pw(self.bn_pw(self.pw(x)))
        return x

class MobileNetCIFAR(nn.Module):
    """Small MobileNet-style CNN for CIFAR-10 (Floating Point).

    Stem (3x3 conv, stride 1) keeps 32x32 resolution, then six DS
    blocks with two stride-2 downsamples bring us to 8x8 x 256.
    Global average pooling and a single Linear produce the logits.
    """

    # (out_channels, stride) for each depthwise-separable block.
    BLOCK_CFG = [
        (16, 1),
        (32, 2), # 32 -> 16
        (32, 1),
        (64, 2), # 16 -> 8
        (128, 2),  # 8 ->  4
        (128, 2),  # 4 ->  2
        (64, 2),  # 4 ->  1
    ]

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.inp = nn.Identity()

        # Standard 3x3 stem conv — first layer is usually kept dense
        # (no depthwise) because it has only 3 input channels.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        blocks = []
        in_ch = 32
        for out_ch, stride in self.BLOCK_CFG:
            blocks.append(DepthwiseSeparableBlock(
                in_ch, out_ch, stride))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, num_classes, bias=True),
        )

    def forward(self, x):
        x = self.inp(x)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

class VGG(nn.Module):
    """
    Small VGG-style CNN for CIFAR-10 (Floating Point).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.inp = nn.Identity()

        self.features = nn.Sequential(
            *self._conv_block(3,   64),
            *self._conv_block(64,  64),
            nn.MaxPool2d(2),                        # 32 -> 16
            *self._conv_block(64,  128),
            *self._conv_block(128, 128),
            nn.MaxPool2d(2),                        # 16 -> 8
            *self._conv_block(128, 256),
            *self._conv_block(256, 256),
            nn.AdaptiveAvgPool2d(1),                #  8 -> 1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes, bias=True),
        )

    @staticmethod
    def _conv_block(in_ch, out_ch):
        return [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        ]

    def forward(self, x):
        x = self.inp(x)
        x = self.features(x)
        x = self.classifier(x)
        return x
