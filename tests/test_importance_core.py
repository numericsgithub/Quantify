"""Core correctness tests for importance/: model coverage, aggregation,
BN scale-invariance handling, save/load roundtrip.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from importance import analyze, load
from importance.discovery import BatchDetectionError

from tests.importance_test_models import (
    Small2DCNN, Small1DCNN, SmallMLP, ConvBNModel, DepthwiseGroupedModel,
    BrevitasQuantModel, QuantifyFixedPointModel, TupleOutputModel, ResidualModel,
    make_image_loader, make_1d_loader, make_flat_loader,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_small_2d_cnn():
    model = Small2DCNN().eval()
    data = make_image_loader()
    result = analyze(model, data, max_samples=12, store_samples=2, device="cpu")
    assert result.layer_names() == ["conv1", "fc"]
    filt = result.filter("conv1")
    assert filt.shape == (1 + 5, 4)  # __all__ + 5 classes, 4 filters
    assert np.all(np.isfinite(filt))
    assert result.manifest["dataset_info"]["n_samples"] == 12


def test_small_1d_cnn():
    model = Small1DCNN().eval()
    data = make_1d_loader()
    result = analyze(model, data, max_samples=12, store_samples=2, device="cpu")
    kernel = result.kernel("conv1")
    weight = result.weight("conv1")
    assert weight.ndim == 4  # [n_out, F, C, k]
    assert kernel.shape == weight.shape[:3]


def test_mlp():
    model = SmallMLP().eval()
    data = make_flat_loader()
    result = analyze(model, data, max_samples=12, store_samples=2, device="cpu")
    assert result.layer_names() == ["fc1", "fc2"]
    # Linear layers have no spatial kernel dims: kernel level == weight level shape
    assert result.kernel("fc1").shape == result.weight("fc1").shape


def test_all_outputs_row_is_mean_of_real_outputs():
    model = Small2DCNN(n_classes=3).eval()
    data = make_image_loader()
    result = analyze(model, data, max_samples=8, store_samples=1, device="cpu")
    filt = result.filter("conv1", metric="mean_s")
    all_row = filt[0]
    real_rows_mean = filt[1:].mean(axis=0)
    assert np.allclose(all_row, real_rows_mean, atol=1e-4)


def test_conv_bn_weight_score_fails_but_act_filter_does_not():
    """The known BN scale-invariance failure mode (see new_feature.md and
    docs/llm/importance_analysis.md): output is exactly invariant to
    rescaling a single filter's conv weights by any constant c, because BN
    normalizes by the (batch) standard deviation of that same filter's
    output, which scales by exactly c too. By Euler's homogeneous-function
    theorem this makes w . grad_w(output) == 0 for that filter's weights
    *exactly* -- so the signed per-weight scores of a BN-following filter
    cancel almost perfectly when summed (positive and negative contributions
    wash out), while the same filter's act_filter score (based on actual
    activation magnitudes, not a weight-rescale direction) does not collapse
    the same way. We use BatchNorm2d(track_running_stats=False) so the
    invariance holds exactly even with the model in eval() (required by
    analyze()) -- it always normalizes by the current batch's statistics
    rather than fixed running stats, which would only make the model
    *approximately* invariant. We compare against an architecturally
    identical conv-only (no BN) control to isolate the effect of BN itself.
    """
    torch.manual_seed(3)

    class ConvBNNoTrack(nn.Module):
        """Like ConvBNModel but BN always uses batch statistics (even in
        eval()), which is what makes the weight-rescale invariance exact."""

        def __init__(self, out_ch=4, n_classes=3):
            super().__init__()
            self.conv1 = nn.Conv2d(3, out_ch, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_ch, track_running_stats=False)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(out_ch, n_classes)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            return self.fc(self.flatten(self.pool(x)))

    bn_model = ConvBNNoTrack(out_ch=4, n_classes=3).eval()
    plain_model = Small2DCNN(n_classes=3).eval()
    with torch.no_grad():
        plain_model.conv1.weight.copy_(bn_model.conv1.weight)

    data = make_image_loader(hw=8, n_batches=6)
    result_bn = analyze(bn_model, data, max_samples=24, store_samples=2, device="cpu")
    result_plain = analyze(plain_model, data, max_samples=24, store_samples=2, device="cpu")

    def cancellation_ratio(result, layer, output):
        # per-weight signed scores [F, C, kh, kw] -> per-filter |sum| / sum(|.|)
        signed = result.weight(layer, output=output, metric="mean_s")
        F = signed.shape[0]
        flat = signed.reshape(F, -1)
        num = np.abs(flat.sum(axis=1))
        den = np.abs(flat).sum(axis=1) + 1e-12
        return num / den

    ratio_bn = cancellation_ratio(result_bn, "conv1", output=1)
    ratio_plain = cancellation_ratio(result_plain, "conv1", output=1)

    # BN's scale-invariance drives the signed weight score toward exact cancellation
    assert ratio_bn.mean() < 0.01, f"expected near-exact cancellation with BN, got {ratio_bn}"
    # the architecturally identical no-BN control shows essentially no cancellation
    assert ratio_plain.mean() > 0.9, f"expected no cancellation without BN, got {ratio_plain}"

    # act_filter does not suffer the same collapse: it stays a comparable,
    # non-trivial fraction of its own scale for the BN model's filters too.
    act_bn = result_bn.filter("conv1", output=1, metric="act_filter")
    assert act_bn.max() > 0
    normalized_act = act_bn / act_bn.max()
    assert normalized_act.min() > 0.01, (
        "act_filter should not collapse toward zero for BN-following filters "
        f"the way the signed weight score does: {act_bn}"
    )


def test_depthwise_grouped_conv_shapes():
    model = DepthwiseGroupedModel(ch=4, groups=4).eval()
    data = make_image_loader(in_ch=4)
    result = analyze(model, data, max_samples=8, store_samples=1, device="cpu")
    weight = result.weight("dw")
    # groups=4, in_ch=4 -> C/groups == 1
    assert weight.shape[2] == 1
    entry = [e for e in result.manifest["layers"] if e["id"] == "dw"][0]
    assert entry["groups"] == 4


def test_brevitas_quant_conv2d_model():
    model = BrevitasQuantModel().eval()
    data = make_image_loader()
    result = analyze(model, data, max_samples=8, store_samples=1, device="cpu")
    assert set(result.layer_names()) == {"conv", "fc"}
    assert np.all(np.isfinite(result.filter("conv")))


def test_quantify_fixedpoint_model_triggers_fallback():
    model = QuantifyFixedPointModel().eval()
    model.calibrate(torch.randn(4, 3, 8, 8))
    data = make_image_loader()
    result = analyze(model, data, max_samples=8, store_samples=1, device="cpu")
    assert result.manifest["settings"]["path_used"] == "fallback"
    assert np.all(np.isfinite(result.filter("conv")))


def test_tuple_output_model_warns_and_uses_first_tensor():
    model = TupleOutputModel().eval()
    data = make_image_loader()
    with pytest.warns(UserWarning, match="tuple"):
        result = analyze(model, data, max_samples=8, store_samples=1, device="cpu")
    assert np.all(np.isfinite(result.filter("conv")))


def test_residual_connection_model():
    model = ResidualModel().eval()
    data = make_image_loader()
    result = analyze(model, data, max_samples=8, store_samples=1, device="cpu")
    assert result.layer_names() == ["stem", "block.conv1", "block.conv2", "fc"]


def test_dict_dataloader_auto_detection():
    model = Small2DCNN(n_classes=3).eval()
    data = make_image_loader(as_dict=True)
    result = analyze(model, data, max_samples=8, store_samples=1, device="cpu")
    assert np.all(np.isfinite(result.filter("conv1")))


def test_dict_dataloader_without_known_key_raises_helpful_error():
    model = Small2DCNN(n_classes=3).eval()
    data = [{"weird_key": torch.randn(4, 3, 8, 8)}]
    with pytest.raises(BatchDetectionError, match="batch_fn"):
        analyze(model, data, max_samples=4, store_samples=0, device="cpu")


def test_batch_fn_override():
    model = Small2DCNN(n_classes=3).eval()
    data = [{"weird_key": torch.randn(4, 3, 8, 8)}]
    result = analyze(
        model, data, max_samples=4, store_samples=0, device="cpu",
        batch_fn=lambda b: (b["weird_key"], None),
    )
    assert np.all(np.isfinite(result.filter("conv1")))


def test_output_fn_override():
    model = TupleOutputModel(n_classes=3).eval()
    data = make_image_loader()
    result = analyze(
        model, data, max_samples=8, store_samples=1, device="cpu",
        output_fn=lambda out: out[0],
    )
    assert np.all(np.isfinite(result.filter("conv")))


def test_loss_row():
    model = Small2DCNN(n_classes=4).eval()
    data = make_image_loader(n_classes=4)
    loss_fn = nn.CrossEntropyLoss()
    result = analyze(model, data, max_samples=8, store_samples=1, device="cpu", loss_fn=loss_fn)
    assert result.manifest["output_features"]["names"][-1] == "loss"
    assert np.all(np.isfinite(result.filter("conv1", output="loss")))


def test_save_load_roundtrip(tmp_path):
    model = Small2DCNN(n_classes=3).eval()
    data = make_image_loader()
    result = analyze(model, data, max_samples=8, store_samples=3, device="cpu")
    out_dir = tmp_path / "result"
    result.save(str(out_dir))
    assert (out_dir / "manifest.json").exists()

    reloaded = load(str(out_dir))
    np.testing.assert_allclose(reloaded.filter("conv1"), result.filter("conv1"), rtol=0.05, atol=1e-3)
    np.testing.assert_allclose(reloaded.layer_score("fc"), result.layer_score("fc"), rtol=0.05, atol=1e-3)
    assert reloaded.samples["meta"]["count"] == 3


def test_raw_weight_values_are_actual_parameters(tmp_path):
    """raw_weight()/raw_bias() expose the literal float parameter values
    (not an importance score, not output-indexed) -- what the viewer's
    weight-level heatmap now shows alongside the importance scores."""
    model = Small2DCNN(n_classes=3).eval()
    data = make_image_loader()
    result = analyze(model, data, max_samples=8, store_samples=1, device="cpu")

    raw_w = result.raw_weight("conv1")
    np.testing.assert_allclose(raw_w, model.conv1.weight.detach().numpy(), atol=1e-5)
    assert raw_w.shape == tuple(model.conv1.weight.shape)
    np.testing.assert_allclose(result.raw_bias("conv1"), model.conv1.bias.detach().numpy(), atol=1e-5)

    raw_fc_bias = result.raw_bias("fc")
    np.testing.assert_allclose(raw_fc_bias, model.fc.bias.detach().numpy(), atol=1e-5)

    out_dir = tmp_path / "raw_result"
    result.save(str(out_dir))
    reloaded = load(str(out_dir))
    np.testing.assert_allclose(reloaded.raw_weight("conv1"), raw_w, atol=1e-3)
    np.testing.assert_allclose(reloaded.raw_bias("fc"), raw_fc_bias, atol=1e-3)

    # a layer with bias=False should report no raw bias at all
    bn_model = ConvBNModel(n_classes=3).eval()
    bn_result = analyze(bn_model, make_image_loader(), max_samples=8, store_samples=0, device="cpu")
    assert bn_result.raw_bias("conv1") is None


def test_to_dataframe():
    model = SmallMLP().eval()
    data = make_flat_loader()
    result = analyze(model, data, max_samples=8, store_samples=1, device="cpu")
    df = result.to_dataframe(level="filter")
    assert set(["layer", "output_index", "output_name", "position", "value"]).issubset(df.columns)
    assert (df["layer"] == "fc1").sum() > 0


def test_levels_config_limits_finest_computed_level():
    model = Small2DCNN().eval()
    data = make_image_loader()
    result = analyze(model, data, max_samples=8, store_samples=1, device="cpu", levels=("filter",))
    entry = [e for e in result.manifest["layers"] if e["id"] == "conv1"][0]
    assert "weight" not in entry["levels"]
    assert "kernel" not in entry["levels"]
    assert "filter" in entry["levels"] and "layer" in entry["levels"]


def test_max_outputs_never_silently_explodes():
    class BigOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1)
            self.fc = nn.Conv2d(4, 4, 1)  # keeps spatial [B,4,8,8] -> too many "flat" features if mishandled

        def forward(self, x):
            return self.fc(self.conv(x))  # [B, 4, 8, 8]

    model = BigOutputModel().eval()
    data = make_image_loader()
    result = analyze(model, data, max_samples=4, store_samples=0, device="cpu", max_outputs=16)
    # spatial output reduces to channel count (4), well under max_outputs
    assert result.manifest["output_features"]["count"] <= 17
