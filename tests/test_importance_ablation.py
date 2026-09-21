"""Validate Taylor filter scores against real single-filter ablation.

This is necessarily an approximation (first-order Taylor attribution vs. a
finite single-filter perturbation) -- see scripts/importance_ablation_check.py.
"""
import torch

from scripts.importance_ablation_check import taylor_vs_ablation
from tests.importance_test_models import Small2DCNN, make_image_loader


def test_taylor_scores_correlate_with_ablation():
    torch.manual_seed(7)
    model = Small2DCNN(in_ch=3, n_classes=4).eval()
    # wider filters + more samples than the default smoke models so ablation
    # deltas are not dominated by noise
    model.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
    model.fc = torch.nn.Linear(8, 4)
    data = make_image_loader(n_batches=6, batch_size=8, hw=12)

    taylor, ablation, corr = taylor_vs_ablation(model, data, "conv1", output_index=1, max_samples=48)

    assert taylor.shape == ablation.shape == (8,)
    assert corr > 0.6, (
        f"expected a reasonably strong rank correlation between Taylor scores and "
        f"real ablation (got {corr:.3f}); taylor={taylor}, ablation={ablation}"
    )
