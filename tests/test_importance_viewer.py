"""Smoke test for the offline viewer backend: start the server on a freshly
generated result, fetch the manifest and one array slice.
"""
import json
import urllib.request

import torch

from importance import analyze
from importance.serve import make_server

from tests.importance_test_models import Small2DCNN, make_image_loader


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, json.loads(resp.read())


def test_viewer_serves_manifest_and_array_slice(tmp_path):
    torch.manual_seed(0)
    model = Small2DCNN(n_classes=3).eval()
    data = make_image_loader(n_batches=3)
    result = analyze(model, data, max_samples=12, store_samples=4, device="cpu")
    out_dir = tmp_path / "viewer_result"
    result.save(str(out_dir))

    httpd = make_server(str(out_dir), port=0)
    import threading
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"

    try:
        status, manifest = _get(f"{base}/api/manifest")
        assert status == 200
        assert manifest["tool_version"]
        assert manifest["layers"][0]["id"] == "conv1"

        status, slice_data = _get(f"{base}/api/layer/conv1/filter/mean_abs_s?output=0")
        assert status == 200
        assert slice_data["shape"] == [4]
        assert len(slice_data["data"]) == 4

        status, sample_out = _get(f"{base}/api/samples/outputs?index=0")
        assert status == 200
        assert len(sample_out["data"]) == manifest["output_features"]["count"]

        status, page = None, None
        with urllib.request.urlopen(f"{base}/", timeout=5) as resp:
            status = resp.status
            body = resp.read().decode()
        assert status == 200
        assert "Importance Viewer" in body

        with urllib.request.urlopen(f"{base}/static/app.js", timeout=5) as resp:
            assert resp.status == 200
            assert resp.getheader("Content-Type").startswith("application/javascript") or \
                resp.getheader("Content-Type").startswith("text/javascript")
    finally:
        httpd.shutdown()
        httpd.server_close()
