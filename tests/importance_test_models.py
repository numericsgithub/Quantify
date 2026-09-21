"""Small model zoo shared by the importance test suite (not a test file itself)."""
import torch
import torch.nn as nn
import brevitas.nn as qnn

from quantizers import FixedPointPerTensorWeightQuant


class Small2DCNN(nn.Module):
    def __init__(self, in_ch=3, n_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 4, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4, n_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.flatten(self.pool(x))
        return self.fc(x)


class Small1DCNN(nn.Module):
    def __init__(self, in_ch=2, n_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 4, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4, n_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.flatten(self.pool(x))
        return self.fc(x)


class SmallMLP(nn.Module):
    def __init__(self, in_features=10, n_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, n_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class ConvBNModel(nn.Module):
    """Conv immediately followed by BatchNorm -- the scale-invariance case
    that makes weight-level importance misleading (see
    docs/llm/importance_analysis.md)."""

    def __init__(self, in_ch=3, n_classes=4, out_ch=6):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_ch, n_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.flatten(self.pool(x))
        return self.fc(x)


class DepthwiseGroupedModel(nn.Module):
    def __init__(self, ch=4, groups=4, n_classes=3):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=groups)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(ch, n_classes)

    def forward(self, x):
        x = self.relu(self.dw(x))
        return self.fc(self.flatten(self.pool(x)))


class BrevitasQuantModel(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.conv = qnn.QuantConv2d(3, 4, 3, padding=1, weight_bit_width=8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = qnn.QuantLinear(4, n_classes, weight_bit_width=8)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(self.flatten(self.pool(x)))


class QuantifyFixedPointModel(nn.Module):
    """Uses Quantify's own custom-autograd-Function-based quantizer."""

    def __init__(self, n_classes=3):
        super().__init__()
        self.conv = qnn.QuantConv2d(3, 4, 3, padding=1, weight_quant=FixedPointPerTensorWeightQuant)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4, n_classes)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(self.flatten(self.pool(x)))

    def calibrate(self, sample_input):
        self.train()
        with torch.no_grad():
            self(sample_input)
        self.eval()


class TupleOutputModel(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4, n_classes)

    def forward(self, x):
        feat = self.flatten(self.pool(self.conv(x)))
        return self.fc(feat), feat


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class ResidualModel(nn.Module):
    def __init__(self, ch=4, n_classes=3):
        super().__init__()
        self.stem = nn.Conv2d(3, ch, 3, padding=1)
        self.block = ResBlock(ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(ch, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block(x)
        return self.fc(self.flatten(self.pool(x)))


def make_image_loader(n_batches=3, batch_size=4, in_ch=3, hw=8, n_classes=None, as_dict=False):
    batches = []
    for _ in range(n_batches):
        x = torch.randn(batch_size, in_ch, hw, hw)
        if as_dict:
            item = {"image": x}
            if n_classes is not None:
                item["label"] = torch.randint(0, n_classes, (batch_size,))
            batches.append(item)
        else:
            if n_classes is not None:
                batches.append((x, torch.randint(0, n_classes, (batch_size,))))
            else:
                batches.append((x,))
    return batches


def make_1d_loader(n_batches=3, batch_size=4, in_ch=2, length=16):
    return [(torch.randn(batch_size, in_ch, length),) for _ in range(n_batches)]


def make_flat_loader(n_batches=3, batch_size=4, in_features=10):
    return [(torch.randn(batch_size, in_features),) for _ in range(n_batches)]
