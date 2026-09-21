"""Quick demo: train a small quantized MNIST CNN briefly, then run
importance.analyze() over it and save the result for the viewer.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from importance import analyze

from examples.basics.importance_mnist import ImportanceMNISTNet, _load_mnist_or_synthetic


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImportanceMNISTNet().to(device)

    train_ds = _load_mnist_or_synthetic(train=True)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    n_steps = 150
    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        if step % 25 == 0:
            acc = (out.argmax(1) == y).float().mean().item()
            print(f"step {step:3d}  loss {loss.item():.3f}  batch_acc {acc:.3f}")
        if step >= n_steps:
            break
    model.eval()

    result = analyze(
        model, train_loader,
        loss_fn=nn.CrossEntropyLoss(),
        device=device,
        max_samples=1000,
        store_samples=16,
    )
    print(f"analyzed layers: {result.layer_names()}")
    print(f"path used: {result.manifest['settings']['path_used']}")

    out_dir = "runs/imp_demo"
    result.save(out_dir)
    print(f"saved importance result to {out_dir}")


if __name__ == "__main__":
    main()
