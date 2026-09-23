"""Train a plain (non-quantized) PyTorch MNIST CNN with a standard training
loop, save it exactly the normal PyTorch way (`torch.save(model.state_dict(),
path)`), and compare the resulting .pt file's *structure* against the
`*_state_dict.pt` companion files that `training_harness.checkpointing`
auto-exports next to its own (QAT) checkpoints.

"Structure" here means the file format / container -- both should be a bare
`OrderedDict[str, Tensor]` saved with plain `torch.save`, loadable via
`model.load_state_dict(torch.load(path))` with zero knowledge of this repo.
The actual *keys* will differ (the QAT model has extra quantizer buffers
like `annealing_alpha`/`search_done`) -- that's an expected architecture
difference, not a structural one.

    python scripts/train_plain_mnist_and_compare.py
"""
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from examples.basics.importance_mnist import _load_mnist_or_synthetic


class PlainMNISTNet(nn.Module):
    """Same shape story as ImportanceMNISTNet (examples/basics/importance_mnist.py)
    -- Conv+BN+ReLU x2, global average pool, Linear head -- but with plain
    nn.Conv2d instead of Brevitas QuantConv2d/Quantify weight quantizers, so
    there is no QAT involved at all."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.flatten(self.pool(x))
        return self.fc(x)


def train_plain_model() -> PlainMNISTNet:
    torch.manual_seed(0)
    model = PlainMNISTNet()
    train_ds = _load_mnist_or_synthetic(train=True)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for step, (x, y) in enumerate(train_loader):
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        if step % 25 == 0:
            acc = (out.argmax(1) == y).float().mean().item()
            print(f"step {step:3d}  loss {loss.item():.3f}  batch_acc {acc:.3f}")
        if step >= 150:
            break
    model.eval()
    return model


def describe(path: str, label: str) -> dict:
    obj = torch.load(path, map_location="cpu")
    print(f"\n--- {label} ({path}) ---")
    print(f"  python type:        {type(obj)}")
    print(f"  is OrderedDict:     {isinstance(obj, collections.OrderedDict)}")
    print(f"  is plain dict:      {isinstance(obj, dict)}")
    print(f"  has wrapper keys:   {[k for k in obj.keys() if k in ('model_state_dict', 'epoch', 'optimizer_state_dict')] if isinstance(obj, dict) else 'n/a'}")
    all_tensors = isinstance(obj, dict) and all(torch.is_tensor(v) for v in obj.values())
    print(f"  every value tensor: {all_tensors}")
    print(f"  num entries:        {len(obj) if isinstance(obj, dict) else 'n/a'}")
    print(f"  first 5 keys:       {list(obj.keys())[:5] if isinstance(obj, dict) else 'n/a'}")
    return obj


def main():
    print("=== 1. Train a plain (non-quantized) PyTorch model ===")
    model = train_plain_model()

    plain_path = "runs/plain_mnist/model_state_dict.pt"
    import os
    os.makedirs("runs/plain_mnist", exist_ok=True)
    torch.save(model.state_dict(), plain_path)
    print(f"\nSaved the normal-PyTorch way: torch.save(model.state_dict(), {plain_path!r})")

    # Sanity: loadable by a fresh instance with zero knowledge of how it was trained
    fresh = PlainMNISTNet()
    fresh.load_state_dict(torch.load(plain_path))
    print("Loaded back into a fresh PlainMNISTNet with model.load_state_dict(torch.load(path)) -- OK")

    qat_path = "runs/qat_mnist/checkpoints/last_state_dict.pt"
    if not os.path.exists(qat_path):
        print(f"\n(no {qat_path} found -- run scripts/train_qat_and_export_onnx.py first to compare against it)")
        return

    print("\n=== 2. Compare file *structure* against the QAT export's plain companion ===")
    plain_obj = describe(plain_path, "plain PyTorch model (this script)")
    qat_obj = describe(qat_path, "QAT model's auto-exported *_state_dict.pt companion")

    print("\n=== 3. Verdict ===")
    same_container_type = type(plain_obj) is type(qat_obj)
    both_plain_tensor_dicts = (
        isinstance(plain_obj, dict) and isinstance(qat_obj, dict)
        and all(torch.is_tensor(v) for v in plain_obj.values())
        and all(torch.is_tensor(v) for v in qat_obj.values())
    )
    neither_has_wrapper = all(
        k not in obj for obj in (plain_obj, qat_obj) for k in ("model_state_dict", "epoch", "optimizer_state_dict")
    )
    print(f"  same container type ({type(plain_obj).__name__}):           {same_container_type}")
    print(f"  both are plain {{str: Tensor}} dicts (no wrapper):    {both_plain_tensor_dicts}")
    print(f"  neither has the harness's wrapper keys:            {neither_has_wrapper}")
    print(
        f"\n  => STRUCTURALLY THE SAME: {same_container_type and both_plain_tensor_dicts and neither_has_wrapper} "
        f"(both are exactly what plain `torch.save(model.state_dict(), path)` produces)"
    )

    print("\n  Key sets differ, as expected -- the QAT model has extra Brevitas/Quantify")
    print("  quantizer buffers (annealing_alpha, search_done, ...) that a plain float")
    print("  model has no equivalent of; that's an architecture difference, not a")
    print("  file-format one:")
    plain_keys = set(plain_obj.keys())
    qat_keys = set(qat_obj.keys())
    print(f"  plain-only keys (sample): {sorted(plain_keys - qat_keys)[:5]}")
    print(f"  qat-only keys (sample):   {sorted(qat_keys - plain_keys)[:5]}")


if __name__ == "__main__":
    main()
