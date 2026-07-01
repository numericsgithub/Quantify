"""
Quantized ResNet-18 and ResNet-50 for ImageNet.

Architecture mirrors torchvision.models.resnet exactly (module names, layer
structure, expansion factors) so that load_pretrained_weights() can transfer
float checkpoint weights without any key renaming.

ReLU layers use different names than torchvision (relu1/relu2 vs relu) because
torchvision's single relu module is called twice in forward. We use distinct
modules to give each quantizer its own calibration state. Since ReLU has no
learnable parameters, this difference is invisible to weight loading.
"""

import torch.nn as nn
import brevitas.nn as qnn


def _relu(act_quant):
    if act_quant is not None:
        return qnn.QuantReLU(act_quant=act_quant)
    return nn.ReLU(inplace=True)


def _quant_identity(act_quant):
    """Return a QuantIdentity for quantizing pre-add / skip-path activations, or None."""
    if act_quant is not None:
        return qnn.QuantIdentity(act_quant=act_quant)
    return None


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class QuantBasicBlock(nn.Module):
    """Quantized BasicBlock for ResNet-18 / ResNet-34."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,
                 weight_quant=None, act_quant=None, downsample=None):
        super().__init__()
        self.conv1 = qnn.QuantConv2d(
            in_planes, planes, 3, stride=stride, padding=1, bias=False,
            weight_quant=weight_quant,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = _relu(act_quant)
        self.conv2 = qnn.QuantConv2d(
            planes, planes, 3, stride=1, padding=1, bias=False,
            weight_quant=weight_quant,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.pre_add_quant = _quant_identity(act_quant)
        self.relu2 = _relu(act_quant)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.pre_add_quant is not None:
            out = self.pre_add_quant(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu2(out + identity)


class QuantBottleneck(nn.Module):
    """Quantized Bottleneck for ResNet-50 / ResNet-101 / ResNet-152."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1,
                 weight_quant=None, act_quant=None, downsample=None):
        super().__init__()
        self.conv1 = qnn.QuantConv2d(in_planes, planes, 1, bias=False, weight_quant=weight_quant)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = _relu(act_quant)
        self.conv2 = qnn.QuantConv2d(
            planes, planes, 3, stride=stride, padding=1, bias=False,
            weight_quant=weight_quant,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = _relu(act_quant)
        self.conv3 = qnn.QuantConv2d(
            planes, planes * self.expansion, 1, bias=False, weight_quant=weight_quant,
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.pre_add_quant = _quant_identity(act_quant)
        self.relu3 = _relu(act_quant)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.pre_add_quant is not None:
            out = self.pre_add_quant(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu3(out + identity)


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------

class QuantResNet(nn.Module):
    """
    Generic quantized ResNet. Use QuantResNet18 / QuantResNet50 factory
    functions rather than instantiating this directly.

    Args:
        block:        QuantBasicBlock or QuantBottleneck.
        layers:       Number of blocks per stage, e.g. [2, 2, 2, 2].
        num_classes:  Output logits. Default 1000 for ImageNet.
        weight_quant: Brevitas injector class for weight quantization.
        act_quant:    Brevitas injector class for activation quantization.
        bias_quant:   Brevitas injector class for bias quantization (fc only;
                      conv layers use bias=False because BN follows).
    """

    def __init__(self, block, layers, num_classes=1000,
                 weight_quant=None, act_quant=None, bias_quant=None):
        super().__init__()
        self._in_planes = 64
        self._weight_quant = weight_quant
        self._act_quant = act_quant

        # Stem — names match torchvision exactly for weight mapping
        self.conv1 = qnn.QuantConv2d(
            3, 64, 7, stride=2, padding=3, bias=False, weight_quant=weight_quant,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = _relu(act_quant)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Residual stages — names layer1..layer4 match torchvision
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.input_quant = _quant_identity(act_quant)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.post_pool_quant = _quant_identity(act_quant)
        self.flatten = nn.Flatten()
        fc_kw = {"bias_quant": bias_quant} if bias_quant is not None else {}
        self.fc = qnn.QuantLinear(
            512 * block.expansion, num_classes, bias=True,
            weight_quant=weight_quant, output_quant=None, **fc_kw,
        )

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self._in_planes != planes * block.expansion:
            ds_modules = [
                qnn.QuantConv2d(
                    self._in_planes, planes * block.expansion, 1,
                    stride=stride, bias=False, weight_quant=self._weight_quant,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            ]
            if self._act_quant is not None:
                ds_modules.append(qnn.QuantIdentity(act_quant=self._act_quant))
            downsample = nn.Sequential(*ds_modules)

        layers = [block(
            self._in_planes, planes, stride=stride,
            weight_quant=self._weight_quant, act_quant=self._act_quant,
            downsample=downsample,
        )]
        self._in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(
                self._in_planes, planes,
                weight_quant=self._weight_quant, act_quant=self._act_quant,
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.input_quant is not None:
            x = self.input_quant(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        if self.post_pool_quant is not None:
            x = self.post_pool_quant(x)
        x = self.flatten(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def QuantResNet18(num_classes=1000, weight_quant=None, act_quant=None, bias_quant=None):
    """Quantized ResNet-18. Mirrors torchvision.models.resnet18 for weight loading."""
    return QuantResNet(QuantBasicBlock, [2, 2, 2, 2], num_classes, weight_quant, act_quant, bias_quant)


def QuantResNet50(num_classes=1000, weight_quant=None, act_quant=None, bias_quant=None):
    """Quantized ResNet-50. Mirrors torchvision.models.resnet50 for weight loading."""
    return QuantResNet(QuantBottleneck, [3, 4, 6, 3], num_classes, weight_quant, act_quant, bias_quant)
