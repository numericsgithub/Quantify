"""Train (briefly) a small Quantify-quantized MNIST CNN, run importance
analysis over it, save the result, and open the interactive viewer.

    python examples/basics/importance_mnist.py

Falls back to synthetic random data if MNIST can't be downloaded (offline
sandboxes), so the importance.analyze()/save()/view() flow is always
runnable end to end.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import brevitas.nn as qnn

from importance import analyze, view
from quantizers import FixedPointPerTensorWeightQuant


class ImportanceMNISTNet(nn.Module):
    """Conv+BN+ReLU stem (Quantify fixed-point weight quant -- this is what
    triggers importance's vmap-fallback path, see docs/llm/importance_analysis.md)
    followed by a plain Linear head."""

    def __init__(self):
        super().__init__()
        self.conv1 = qnn.QuantConv2d(
            1, 8, kernel_size=3, padding=1, bias=False,
            weight_quant=FixedPointPerTensorWeightQuant,
        )
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.conv2 = qnn.QuantConv2d(
            8, 16, kernel_size=3, padding=1, stride=2, bias=False,
            weight_quant=FixedPointPerTensorWeightQuant,
        )
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        # AdaptiveAvgPool2d(1) (global average pool), not a larger target size:
        # ONNX's adaptive_avg_pool2d exporter only supports output sizes that
        # evenly divide the input size, which a fixed target like (4, 4) can
        # silently violate once conv strides change the feature map size
        # (e.g. 14x14 here) -- see docs/llm/pitfalls/brevitas_pitfalls.md.
        # 1x1 always divides evenly and is this repo's documented pattern for
        # pooling into a quantized Linear head (CLAUDE.md pitfall #1).
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.flatten(self.pool(x))
        return self.fc(x)


def _load_mnist_or_synthetic(train: bool, n_synthetic: int = 512):
    try:
        from torchvision import datasets, transforms

        ds = datasets.MNIST(
            root="/tmp/mnist_data", train=train, download=True,
            transform=transforms.ToTensor(),
        )
        return ds
    except Exception as e:  # noqa: BLE001 - offline sandboxes, etc.
        print(f"Could not download MNIST ({e}); using synthetic random data instead.")
        x = torch.randn(n_synthetic, 1, 28, 28)
        y = torch.randint(0, 10, (n_synthetic,))
        return TensorDataset(x, y)


def main():
    torch.manual_seed(0)
    model = ImportanceMNISTNet()

    train_ds = _load_mnist_or_synthetic(train=True)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    # A short training loop -- just enough for BN stats and weights to settle
    # into something non-random; this example is about importance.analyze(),
    # not about achieving high accuracy.
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for step, (x, y) in enumerate(train_loader):
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        if step >= 100:
            break
    model.eval()

    result = analyze(
        model, train_loader,
        loss_fn=nn.CrossEntropyLoss(),
        max_samples=1000,
        store_samples=16,
    )
    print(f"analyzed {len(result.layer_names())} layers: {result.layer_names()}")
    print(f"path used: {result.manifest['settings']['path_used']}")

    out_dir = "runs/imp_mnist_example"
    result.save(out_dir)
    print(f"saved importance result to {out_dir}")

    view(out_dir)


if __name__ == "__main__":
    main()
