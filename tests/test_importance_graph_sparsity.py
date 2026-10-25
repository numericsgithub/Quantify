"""Graph tracing (importance/graph.py), zero detection (importance/sparsity.py),
the float16 storage scale, and the viewer endpoints built on them."""
import json
import threading
import urllib.request

import numpy as np
import pytest
import torch

from importance import analyze, load
from importance.graph import trace_graph
from importance.serve import make_server
from importance.sparsity import float_zero_threshold, layer_sparsity
from tests.importance_test_models import (
    BrevitasQuantModel,
    ResidualModel,
    Small2DCNN,
    TupleOutputModel,
    make_image_loader,
)


# ---------------------------------------------------------------------------
# graph tracing
# ---------------------------------------------------------------------------
def _by_name(graph):
    return {n["name"]: n for n in graph["nodes"]}


def _preds(graph, node_id):
    return {e["source"] for e in graph["edges"] if e["target"] == node_id}


def test_residual_add_has_both_branches_as_inputs():
    g = trace_graph(ResidualModel(ch=4).eval(), torch.randn(1, 3, 8, 8), analyzed_ids={"stem"})
    nodes = _by_name(g)
    add = nodes["block.add"]
    assert add["kind"] == "op"
    assert _preds(g, add["id"]) == {nodes["block.bn2"]["id"], nodes["stem"]["id"]}
    assert nodes["stem"]["analyzed"] and not nodes["block.conv1"]["analyzed"]
    assert nodes["stem"]["params"][0] == {"name": "weight", "shape": [4, 3, 3, 3], "dtype": "float32"}
    assert nodes["stem"]["output_shapes"] == [[1, 4, 8, 8]]


def test_reused_module_gets_one_node_per_call():
    # ResBlock calls self.relu twice
    g = trace_graph(ResidualModel(ch=4).eval(), torch.randn(1, 3, 8, 8))
    names = [n["name"] for n in g["nodes"]]
    assert "block.relu" in names and "block.relu#2" in names


def test_dtype_moves_between_modules_keep_the_edge():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(4, 4)
            self.b = torch.nn.Linear(4, 2)

        def forward(self, x):
            return self.b(self.a(x).double().float().contiguous())

    g = trace_graph(M().eval(), torch.randn(1, 4))
    nodes = _by_name(g)
    assert _preds(g, nodes["b"]["id"]) == {nodes["a"]["id"]}
    assert [n["kind"] for n in g["nodes"]] == ["input", "module", "module", "output"]


def test_tuple_output_gives_one_output_node_per_tensor():
    g = trace_graph(TupleOutputModel().eval(), torch.randn(1, 3, 8, 8))
    outs = [n for n in g["nodes"] if n["kind"] == "output"]
    assert len(outs) == 2
    nodes = _by_name(g)
    assert _preds(g, outs[1]["id"]) == {nodes["flatten"]["id"]}


def test_brevitas_layer_is_one_node_with_quant_info():
    g = trace_graph(BrevitasQuantModel().eval(), torch.randn(1, 3, 8, 8))
    nodes = _by_name(g)
    assert [n["kind"] for n in g["nodes"]] == ["input", "module", "module", "module", "module", "output"]
    q = nodes["conv"]["quantization"]["weight_quant"]
    assert q["enabled"] and q["bit_width"] == 8.0


# ---------------------------------------------------------------------------
# zero rule
# ---------------------------------------------------------------------------
def test_float_threshold_example_from_spec():
    th = float_zero_threshold(np.array([3000.0, 0.1, 5.0]))
    assert th["rel_bound"] == pytest.approx(3.0)
    assert th["threshold"] == pytest.approx(0.1)  # 0.1 is zero, 5 is not


def test_mass_budget_keeps_many_small_weights_nonzero():
    # a million 0.1s next to one 3000: the 1/1000 rule alone would zero them
    # all, but together they hold ~97% of the L1 mass
    w = np.concatenate([[3000.0], np.full(1_000_000, 0.1)])
    assert float_zero_threshold(w)["threshold"] < 0.1


def test_layer_sparsity_masks_and_grouped_inputs():
    w = np.random.RandomState(0).randn(8, 4, 3, 3)
    w[2] = 0.0                 # zero filter
    w[5, 1] = 1e-9             # zero kernel (tiny float values)
    sp = layer_sparsity(w, bias=np.ones(8))
    assert sp["mask_filter"].tolist() == [i == 2 for i in range(8)]
    assert sp["mask_kernel"][5, 1] and sp["info"]["zero_kernels"] == 5
    assert sp["info"]["zero_filters_with_bias"] == 1

    dw = np.random.RandomState(1).randn(6, 1, 3, 3)
    dw[3] = 0.0
    sp = layer_sparsity(dw, groups=6, in_channels=6)
    assert sp["mask_input"].tolist() == [False, False, False, True, False, False]


def test_quantized_weights_use_exact_zeros():
    w = np.array([[0.01, 0.5], [0.3, -0.2]])
    q = np.array([[0.0, 0.5], [0.25, -0.25]])
    sp = layer_sparsity(w, qweight=q)
    assert sp["info"]["method"] == "quantized"
    assert sp["mask_weight"].tolist() == [[True, False], [False, False]]


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------
def test_tiny_scores_survive_float16_storage(tmp_path):
    torch.manual_seed(0)
    model = Small2DCNN(n_classes=3).eval()
    result = analyze(model, make_image_loader(n_batches=2), max_samples=8, store_samples=0, device="cpu",
                     progress=False)
    # shrink to well below float16's smallest subnormal (6e-8)
    result.arrays["conv1"]["filter"]["mean_abs_s"] = result.arrays["conv1"]["filter"]["mean_abs_s"] * 1e-9
    expected = np.asarray(result.arrays["conv1"]["filter"]["mean_abs_s"])
    result.save(str(tmp_path / "r"))
    loaded = load(str(tmp_path / "r"))
    got = loaded.filter("conv1")
    assert np.abs(got).max() > 0
    np.testing.assert_allclose(got, expected, rtol=2e-3, atol=0)


def test_brevitas_result_stores_quantized_weight_and_graph(tmp_path):
    torch.manual_seed(0)
    result = analyze(BrevitasQuantModel().eval(), make_image_loader(n_batches=2), max_samples=4,
                     store_samples=0, device="cpu", progress=False)
    result.save(str(tmp_path / "r"))
    loaded = load(str(tmp_path / "r"))
    assert loaded.raw_qweight("conv") is not None
    assert loaded.sparsity("conv")["info"]["method"] == "quantized"
    assert not loaded.graph.get("fallback")
    assert any(n.get("module") == "fc" for n in loaded.graph["nodes"])


def test_result_without_graph_falls_back_to_chain(tmp_path):
    result = analyze(Small2DCNN(n_classes=3).eval(), make_image_loader(n_batches=1), max_samples=4,
                     store_samples=0, device="cpu", progress=False)
    result.save(str(tmp_path / "r"))
    (tmp_path / "r" / "graph.json").unlink()
    g = load(str(tmp_path / "r")).graph
    assert g["fallback"]
    assert [n["kind"] for n in g["nodes"]] == ["input", "module", "module", "output"]


# ---------------------------------------------------------------------------
# viewer endpoints
# ---------------------------------------------------------------------------
def _get(url):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())


def test_viewer_graph_sparsity_overview_unit_endpoints(tmp_path):
    torch.manual_seed(0)
    model = ResidualModel(ch=4, n_classes=3).eval()
    with torch.no_grad():
        model.stem.weight[1] = 0.0
    result = analyze(model, make_image_loader(n_batches=2), max_samples=8, store_samples=2, device="cpu",
                     progress=False)
    result.save(str(tmp_path / "r"))
    httpd = make_server(str(tmp_path / "r"), port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        graph = _get(f"{base}/api/graph")
        assert any(n["name"] == "block.add" for n in graph["nodes"])

        sp = _get(f"{base}/api/sparsity")
        assert sp["stem"]["mask_filter"] == [0, 1, 0, 0]
        assert sp["stem"]["info"]["zero_filters"] == 1

        ov = _get(f"{base}/api/overview?output=1&metric=auto")
        assert ov["layers"]["block.conv1"]["metric"] == "act_filter"  # followed by BN
        assert ov["layers"]["stem"]["metric"] == "mean_abs_s"
        assert len(ov["layers"]["stem"]["filter"]) == 4

        unit = _get(f"{base}/api/layer/stem/unit/1?output=0&metric=auto")
        assert unit["is_zero_filter"] and unit["shape"] == [3, 3, 3]
        assert unit["zero_kernels"] == [1, 1, 1]
        assert len(unit["kernel_importance"]) == 3
        assert np.asarray(unit["weight_importance"]).shape == (3, 3, 3)

        manifest = _get(f"{base}/api/manifest")
        assert manifest["samples"]["count"] == 2
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_input_importance_sums_kernels_per_real_channel():
    from importance.serve import input_importance
    from tests.importance_test_models import DepthwiseGroupedModel

    torch.manual_seed(0)
    model = DepthwiseGroupedModel(ch=4, groups=4).eval()
    result = analyze(model, make_image_loader(n_batches=1, in_ch=4), max_samples=4, store_samples=0,
                     device="cpu", progress=False)
    entry = result._layer_entry("dw")
    metric, per_in = input_importance(entry, result.arrays["dw"], 0, "auto")
    kernel = np.asarray(result.arrays["dw"]["kernel"]["mean_abs_s"][0])  # [4, 1]: filter f reads channel f
    assert metric == "mean_abs_s"
    np.testing.assert_allclose(per_in, kernel[:, 0], rtol=1e-6)

    _, fc_in = input_importance(result._layer_entry("fc"), result.arrays["fc"], 0, "mean_abs_s")
    np.testing.assert_allclose(fc_in, np.asarray(result.arrays["fc"]["kernel"]["mean_abs_s"][0]).sum(0), rtol=1e-6)


def test_viewer_input_endpoint(tmp_path):
    torch.manual_seed(0)
    model = Small2DCNN(n_classes=3).eval()
    with torch.no_grad():
        model.fc.weight[:, 2] = 0.0  # input feature 2 of fc unused
    result = analyze(model, make_image_loader(n_batches=1), max_samples=4, store_samples=0, device="cpu",
                     progress=False)
    result.save(str(tmp_path / "r"))
    httpd = make_server(str(tmp_path / "r"), port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        ov = _get(f"{base}/api/overview?output=0&metric=auto")
        assert len(ov["layers"]["conv1"]["inputs"]) == 3 and len(ov["layers"]["fc"]["inputs"]) == 4
        assert ov["layers"]["fc"]["inputs"][2] == 0.0
        d = _get(f"{base}/api/layer/fc/input/2?output=0&metric=auto")
        assert d["unused"] and d["filters"] == [0, 1, 2] and d["zero_kernels"] == [1, 1, 1]
        d = _get(f"{base}/api/layer/conv1/input/1?output=0")
        assert np.asarray(d["weight"]).shape == (4, 3, 3) and len(d["kernel_importance"]) == 4
    finally:
        httpd.shutdown()
        httpd.server_close()
