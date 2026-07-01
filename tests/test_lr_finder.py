"""
Unit tests for training_harness/lr_finder.py::_suggest_lr.

Regression for a real bug: steep_lr used to be computed from a raw
single-step difference in EMA-smoothed loss vs. log10(LR). In an actual LR
sweep, a near-flat noisy region early in the sweep had a single adjacent-step
jitter that registered a steeper slope than anywhere in the genuine,
sustained descent region later in the sweep -- so steep_lr locked onto noise
(picking an LR so small training was effectively frozen) instead of the real
descent. min_loss_lr (a global argmin over the whole curve) was unaffected,
which is why it remained a sane recommendation throughout.

The fix computes the gradient over a multi-step window instead of adjacent
single steps, so an isolated single-step jitter can no longer out-rank a
slope that's sustained across many steps.
"""

import math

from training_harness.lr_finder import _suggest_lr


def _make_lrs(n=100, start=1e-8, end=1e-2):
    log_start, log_end = math.log10(start), math.log10(end)
    return [10 ** (log_start + (log_end - log_start) * i / (n - 1)) for i in range(n)]


def _make_noisy_then_descending_losses(n=100, noisy_end=30, descend_end=70):
    """Near-flat noisy region [0, noisy_end), then a clear sustained descent
    [noisy_end, descend_end), then a rise [descend_end, n) -- the same shape
    as the real failing sweep."""
    losses = []
    for i in range(n):
        if i < noisy_end:
            losses.append(5.40 + (0.02 if i % 2 == 0 else -0.02))
        elif i < descend_end:
            frac = (i - noisy_end) / (descend_end - noisy_end)
            losses.append(5.40 - 0.80 * frac)
        else:
            frac = (i - descend_end) / (n - descend_end)
            losses.append(4.60 + 1.50 * frac)
    return losses


class TestSuggestLR:

    def test_steep_lr_ignores_single_step_noise_in_flat_region(self):
        lrs = _make_lrs()
        losses = _make_noisy_then_descending_losses()
        steep_lr, _ = _suggest_lr(lrs, losses)

        noisy_end_lr = lrs[30]
        assert steep_lr >= noisy_end_lr, (
            f"steep_lr={steep_lr} fell inside the noisy flat region "
            f"(ends at {noisy_end_lr}) instead of the real sustained descent"
        )

    def test_steep_lr_matches_old_single_step_behavior_when_window_is_1(self):
        """window=1 reproduces the original (buggy) single-step behavior --
        confirms windowing is the only change, not a different selection rule."""
        lrs = _make_lrs()
        losses = _make_noisy_then_descending_losses()
        steep_lr_w1, _ = _suggest_lr(lrs, losses, window=1)

        noisy_end_lr = lrs[30]
        assert steep_lr_w1 < noisy_end_lr, (
            "expected window=1 to pick a noisy point in the flat region, "
            "reproducing the bug this fix addresses"
        )

    def test_min_loss_lr_unaffected_by_window(self):
        lrs = _make_lrs()
        losses = _make_noisy_then_descending_losses()
        _, min_loss_lr_w1 = _suggest_lr(lrs, losses, window=1)
        _, min_loss_lr_w5 = _suggest_lr(lrs, losses, window=5)
        assert min_loss_lr_w1 == min_loss_lr_w5

    def test_suggest_lr_handles_short_input(self):
        assert _suggest_lr([], []) == (1e-4, 1e-4)

        lr, lr2 = _suggest_lr([3e-5], [1.0])
        assert lr == lr2 == 3e-5

        # Fewer points than the default window must not raise.
        lrs = [1e-6, 2e-6, 3e-6]
        losses = [5.0, 4.5, 4.8]
        steep_lr, min_loss_lr = _suggest_lr(lrs, losses)
        assert steep_lr in lrs
        assert min_loss_lr in lrs
