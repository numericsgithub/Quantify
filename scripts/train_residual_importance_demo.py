"""Train a small residual MNIST CNN and run importance analysis over it --
a showcase for the graph viewer (residual adds) and its zero marking.

    python scripts/train_residual_importance_demo.py [--out runs/imp_residual_mnist] [--epochs 2]
    python -m importance.serve runs/imp_residual_mnist --host 0.0.0.0

The model mixes Brevitas-quantized convs (8- and 4-bit weights -> exact
zeros) with plain float convs (zeros via the float tolerance rule). Optional
sparsity pressure: --l1 (per-weight), --wd (AdamW weight decay -- shrinks
BN-followed layers uniformly, so it does NOT zero kernels on its own), and
--kernel-lasso/--filter-lasso (proximal group lasso -> exact zero
kernels/filters, e.g. --kernel-lasso 2 --filter-lasso 1).
Plain Brevitas quantizers train fine in an ordinary loop; this script does
not use Quantify's QuantizerManager-driven quantizers (those need the
training harness, see docs/llm/pitfalls/training_harness_pitfalls.md).
"""
import argparse

import brevitas.nn as qnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from importance import analyze


class ResBlock(nn.Module):
    def __init__(self, ch, quant_bits=None):
        super().__init__()
        if quant_bits:
            self.conv1 = qnn.QuantConv2d(ch, ch, 3, padding=1, bias=False, weight_bit_width=quant_bits)
            self.conv2 = qnn.QuantConv2d(ch, ch, 3, padding=1, bias=False, weight_bit_width=quant_bits)
        else:
            self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.bn2 = nn.BatchNorm2d(ch)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu2(out + x)


class ResidualMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = qnn.QuantConv2d(1, 16, 3, padding=1, bias=False, weight_bit_width=8)
        self.stem_bn = nn.BatchNorm2d(16)
        self.stem_relu = nn.ReLU()
        self.block1 = ResBlock(16)                       # float
        self.down = qnn.QuantConv2d(16, 32, 3, stride=2, padding=1, bias=False, weight_bit_width=8)
        self.down_bn = nn.BatchNorm2d(32)
        self.down_relu = nn.ReLU()
        self.block2 = ResBlock(32, quant_bits=4)         # 4-bit weights
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.stem_relu(self.stem_bn(self.stem(x)))
        x = self.block1(x)
        x = self.down_relu(self.down_bn(self.down(x)))
        x = self.block2(x)
        return self.fc(self.flatten(self.pool(x)))


@torch.no_grad()
def group_lasso_prox(weights, kernel_thresh, filter_thresh):
    """Proximal step of a group-lasso penalty: shrink every group's L2 norm
    by `thresh` and set the group exactly to 0 once its norm drops below it.
    Unlike weight decay this survives BatchNorm (it's not a uniform rescale:
    weak groups die, strong ones barely move), giving real structured zeros."""
    for w in weights:
        if kernel_thresh:
            # conv: one group per kernel [F, C]; Linear: one group per input column
            g = w.flatten(2).norm(dim=2) if w.dim() > 2 else w.norm(dim=0, keepdim=True)
            factor = (1 - kernel_thresh / g.clamp_min(1e-12)).clamp_min(0)
            w.mul_(factor.view(*factor.shape, *([1] * (w.dim() - factor.dim()))))
        if filter_thresh:
            g = w.flatten(1).norm(dim=1)
            factor = (1 - filter_thresh / g.clamp_min(1e-12)).clamp_min(0)
            w.mul_(factor.view(-1, *([1] * (w.dim() - 1))))


def mnist(train):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    return datasets.MNIST("data", train=train, download=False, transform=tf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/imp_residual_mnist")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--l1", type=float, default=2e-5, help="L1 penalty on conv/linear weights (induces zeros)")
    ap.add_argument("--wd", type=float, default=0.0,
                    help="decoupled (AdamW) weight decay on conv/linear weights. Note: before a BatchNorm this "
                         "only shrinks the whole layer (BN undoes the scale), it does not zero kernels")
    ap.add_argument("--kernel-lasso", type=float, default=0.0,
                    help="proximal group-lasso strength per kernel (Linear: per input column); zeroes whole kernels")
    ap.add_argument("--filter-lasso", type=float, default=0.0,
                    help="proximal group-lasso strength per filter/neuron; zeroes whole filters")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--analyze-samples", type=int, default=1000)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResidualMNIST().to(device)
    train_loader = DataLoader(mnist(True), batch_size=128, shuffle=True, num_workers=4)
    test_ds = mnist(False)
    test_loader = DataLoader(test_ds, batch_size=512)

    weights = [m.weight for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    # decay only the conv/linear weights -- not BN affine params or biases
    wd_ids = {id(w) for w in weights}
    opt = torch.optim.AdamW([
        {"params": weights, "weight_decay": args.wd},
        {"params": [p for p in model.parameters() if id(p) not in wd_ids], "weight_decay": 0.0},
    ], lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(train_loader))
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y) + args.l1 * sum(w.abs().sum() for w in weights)
            opt.zero_grad()
            loss.backward()
            lr_now = sched.get_last_lr()[0]
            opt.step()
            sched.step()
            if args.kernel_lasso or args.filter_lasso:
                group_lasso_prox(weights, lr_now * args.kernel_lasso, lr_now * args.filter_lasso)
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        print(f"epoch {epoch + 1}: loss {loss.item():.4f}  test acc {100 * correct / len(test_ds):.2f}%")

    analysis_loader = DataLoader(Subset(test_ds, range(args.analyze_samples)), batch_size=50)
    result = analyze(model, analysis_loader, device=device, max_samples=args.analyze_samples,
                     store_samples=16, loss_fn=nn.CrossEntropyLoss())
    result.save(args.out)
    for lid in result.layer_names():
        i = result.sparsity(lid)["info"]
        print(f"{lid:14s} {i['method']:9s} zero filters {i['zero_filters']}/{i['n_filters']}  "
              f"zero kernels {i['zero_kernels']}/{i['n_kernels']}  zero weights {i['zero_weights']}/{i['n_weights']}")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
