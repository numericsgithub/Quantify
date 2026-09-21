#!/usr/bin/env python
"""Validate importance.analyze()'s Taylor filter scores against real ablation.

For a chosen layer/output, zeroes one filter at a time, measures the change
in that output feature averaged over the dataset, and reports the Spearman
rank correlation between the ablation-measured importance and the Taylor
`mean_abs_s` filter score from `importance.analyze()`. This is only an
approximation of "true" importance (first-order Taylor vs. a finite,
single-filter perturbation), so a sensible threshold -- not near-1.0
agreement -- is what to expect.

Usage:
    python scripts/importance_ablation_check.py
"""
from __future__ import annotations

import argparse
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from importance import analyze
from importance.discovery import detect_batch


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def _mean_output(model: nn.Module, dataloader, output_index: int, output_spec, batch_fn, device) -> float:
    total = 0.0
    n = 0
    for batch in dataloader:
        x, _ = detect_batch(batch, batch_fn)
        x = x.to(device)
        out = output_spec.transform(model(x))
        total += out[:, output_index].sum().item()
        n += out.shape[0]
    return total / n


def ablation_filter_scores(
    model: nn.Module, dataloader, layer_name: str, output_index: int,
    output_spec, batch_fn=None, device: str = "cpu",
) -> np.ndarray:
    """|change in mean output_index| when each filter of `layer_name` is
    zeroed, one at a time (weights and bias).

    `output_index` follows importance.Result's convention (0 == the
    "__all__" mean row, 1..K == the real output features); output_spec's raw
    [B, K] transform has no __all__ row, so we shift by one here.
    """
    module = dict(model.named_modules())[layer_name]
    raw_index = output_index - 1
    baseline = _mean_output(model, dataloader, raw_index, output_spec, batch_fn, device)

    F = module.weight.shape[0]
    orig_w = module.weight.data.clone()
    orig_b = module.bias.data.clone() if module.bias is not None else None
    scores = np.zeros(F)
    try:
        for f in range(F):
            module.weight.data[f] = 0
            if module.bias is not None:
                module.bias.data[f] = 0
            ablated = _mean_output(model, dataloader, raw_index, output_spec, batch_fn, device)
            scores[f] = abs(ablated - baseline)
            module.weight.data.copy_(orig_w)
            if module.bias is not None:
                module.bias.data.copy_(orig_b)
    finally:
        module.weight.data.copy_(orig_w)
        if module.bias is not None:
            module.bias.data.copy_(orig_b)
    return scores


def taylor_vs_ablation(
    model: nn.Module, dataloader, layer_name: str, output_index: int = 1,
    metric: str = "mean_abs_s", device: str = "cpu", max_samples: int = 256,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Returns (taylor_scores, ablation_scores, spearman_correlation)."""
    result = analyze(model, dataloader, device=device, max_samples=max_samples, store_samples=0)
    taylor_scores = result.filter(layer_name, output=output_index, metric=metric)

    model_device = torch.device(device)
    model = model.to(model_device)
    output_spec = _rebuild_output_spec(result)
    ablation_scores = ablation_filter_scores(
        model, dataloader, layer_name, output_index, output_spec, device=model_device,
    )
    corr = _spearman(taylor_scores, ablation_scores)
    return taylor_scores, ablation_scores, corr


def _rebuild_output_spec(result):
    """Rebuild a minimal OutputSpec compatible with the one analyze() used,
    from the manifest (good enough for a plain tensor / channel-reduced
    output; use output_fn directly for anything more exotic)."""
    from importance.discovery import OutputSpec

    of = result.manifest["output_features"]
    channel_reduce = of["reduce"] in ("channel",)
    return OutputSpec(k=of["count"] - 1, names=of["names"][1:], source_shape=tuple(of["source_shape"]),
                       reduce_kind=of["reduce"], channel_reduce=channel_reduce)


def _demo_model_and_data():
    class DemoCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            return self.fc(self.flatten(self.pool(x)))

    torch.manual_seed(0)
    model = DemoCNN().eval()
    data = [(torch.randn(8, 3, 12, 12),) for _ in range(6)]
    return model, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", default="conv1")
    parser.add_argument("--output-index", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, data = _demo_model_and_data()
    taylor, ablation, corr = taylor_vs_ablation(
        model, data, args.layer, output_index=args.output_index, device=args.device,
    )
    print(f"Taylor mean_abs_s scores:  {taylor}")
    print(f"Ablation |delta| scores:   {ablation}")
    print(f"Spearman rank correlation: {corr:.3f}")


if __name__ == "__main__":
    main()
