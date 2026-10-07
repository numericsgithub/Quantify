"""Local, offline web viewer backend: stdlib http.server serving the static
app plus JSON endpoints that stream manifest.json and array slices on
demand -- a full weight-level array is never sent to the browser in one
response, only a single output-row slice at a time.

No new dependency: FastAPI/uvicorn are not already project dependencies (see
CONVENTIONS.md), so this uses only the standard library.
"""
from __future__ import annotations

import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs

import numpy as np

from importance.storage import Result

STATIC_DIR = Path(__file__).parent / "static"


def _json_bytes(obj) -> bytes:
    return json.dumps(obj).encode("utf-8")


def _clean(arr) -> list:
    """ndarray -> JSON-safe nested list (NaN/inf would make JSON.parse throw)."""
    a = np.asarray(arr, dtype=np.float64)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).tolist()


def resolve_metric(layer_entry: dict, metric: str, level: str = "filter") -> str:
    """'auto' -> act_filter after BN (see docs/llm/importance_analysis.md),
    else mean_abs_s; falls back to mean_abs_s at levels that don't store the
    requested metric (act_filter only exists at filter level)."""
    if metric == "auto":
        metric = "act_filter" if layer_entry.get("has_bn_after") else "mean_abs_s"
    available = layer_entry.get("levels", {}).get(level, {})
    if metric in available:
        return metric
    if metric in ("mean_s", "act_filter_signed") and "mean_s" in available:
        return "mean_s"
    return "mean_abs_s" if "mean_abs_s" in available else next(iter(available), metric)


def input_importance(layer_entry: dict, arrays: dict, output: int, metric: str):
    """Importance per *real* input channel (in_features for Linear): the
    kernel-level scores summed over every filter that reads the channel.
    Returns (metric, [in_channels]) or None without kernel-level data.
    act_filter has no per-input breakdown, so `auto` uses mean_abs_s here."""
    if "kernel" not in arrays:
        return None
    m = resolve_metric(layer_entry, "mean_abs_s" if metric in ("auto", "act_filter") else metric, "kernel")
    k = np.asarray(arrays["kernel"][m][output], dtype=np.float64)  # [F, C/groups]
    if k.ndim != 2:
        return None
    F, Cg = k.shape
    groups = layer_entry.get("groups") or 1
    per = max(1, F // groups)
    in_ch = layer_entry.get("in_channels") if layer_entry.get("kind") == "conv" else None
    out = np.zeros(in_ch or Cg * groups)
    for g in range(groups):
        out[g * Cg:(g + 1) * Cg] += k[g * per:(g + 1) * per].sum(axis=0)[: max(0, out.size - g * Cg)]
    return m, out


class ViewerHandler(BaseHTTPRequestHandler):
    result: Result = None  # set per-instance via factory in serve()
    server_version = "ImportanceViewer/0.2"

    def log_message(self, fmt, *args):  # quieter default logging
        pass

    def _send(self, code: int, body: bytes, content_type: str = "application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code: int = 200):
        self._send(code, _json_bytes(obj), "application/json")

    def do_GET(self):  # noqa: N802 - stdlib method name
        parsed = urlparse(self.path)
        path = parsed.path
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        try:
            if path == "/" or path == "":
                self._serve_static("index.html")
            elif path == "/api/manifest":
                self._send_json({**self.result.manifest, "samples": self.result.samples.get("meta", {})})
            elif path == "/api/graph":
                self._send_json(self.result.graph)
            elif path == "/api/sparsity":
                self._serve_sparsity()
            elif path == "/api/overview":
                self._serve_overview(query)
            elif path.startswith("/api/layer/") and "/unit/" in path:
                self._serve_unit(path, query)
            elif path.startswith("/api/layer/") and "/input/" in path:
                self._serve_input(path, query)
            elif path.startswith("/api/layer/"):
                self._serve_layer_slice(path, query)
            elif path.startswith("/api/samples/"):
                self._serve_sample(path, query)
            elif path.startswith("/static/"):
                self._serve_static(path[len("/static/"):])
            else:
                self._send(404, b"not found", "text/plain")
        except KeyError as e:
            self._send_json({"error": str(e)}, code=404)
        except Exception as e:  # noqa: BLE001
            self._send_json({"error": f"{type(e).__name__}: {e}"}, code=500)

    def _serve_static(self, rel_path: str):
        fpath = (STATIC_DIR / rel_path).resolve()
        if STATIC_DIR.resolve() not in fpath.parents and fpath != STATIC_DIR.resolve():
            self._send(403, b"forbidden", "text/plain")
            return
        if not fpath.exists():
            self._send(404, b"not found", "text/plain")
            return
        ctype = mimetypes.guess_type(str(fpath))[0] or "application/octet-stream"
        self._send(200, fpath.read_bytes(), ctype)

    def _serve_sparsity(self):
        out = {}
        for entry in self.result.manifest["layers"]:
            sp = self.result.sparsity(entry["id"])
            out[entry["id"]] = {
                "info": sp["info"],
                "mask_filter": sp["mask_filter"].astype(int).tolist(),
                "mask_input": sp["mask_input"].astype(int).tolist(),
                "bias_zero": sp["bias_zero"].astype(int).tolist() if sp["bias_zero"] is not None else None,
                "zero_kernels_per_filter": sp["zero_kernels_per_filter"].tolist(),
                "zero_weights_per_filter": sp["zero_weights_per_filter"].tolist(),
            }
        self._send_json(out)

    def _serve_overview(self, query: dict):
        """Filter-level scores of every layer for one output row -- everything
        the graph needs to draw its per-node histograms, in one request."""
        output = int(query.get("output", 0))
        metric = query.get("metric", "auto")
        out = {}
        for entry in self.result.manifest["layers"]:
            arrays = self.result.arrays[entry["id"]]
            if "filter" not in arrays:
                continue
            m = resolve_metric(entry, metric, "filter")
            filt = np.asarray(arrays["filter"][m][output])
            layer_score = float(np.abs(filt).sum())
            out[entry["id"]] = {"metric": m, "filter": _clean(filt), "layer": layer_score}
            per_input = input_importance(entry, arrays, output, metric)
            if per_input is not None:
                out[entry["id"]]["input_metric"] = per_input[0]
                out[entry["id"]]["inputs"] = _clean(per_input[1])
        self._send_json({"output": output, "layers": out})

    def _serve_input(self, path: str, query: dict):
        """/api/layer/<id>/input/<channel>?output=&metric= -- one *real* input
        channel across every filter that reads it (only its group's filters
        for grouped/depthwise convs): kernel importance, weights, zero masks."""
        rest = path[len("/api/layer/"):]
        layer_id, _, c_str = rest.partition("/input/")
        c = int(c_str)
        entry = self.result._layer_entry(layer_id)
        arrays = self.result.arrays[layer_id]
        output = int(query.get("output", 0))
        metric = query.get("metric", "auto")
        raw_w = arrays["raw"]["weight"]
        F, Cg = raw_w.shape[0], raw_w.shape[1]
        groups = entry.get("groups") or 1
        per = max(1, F // groups)
        g, cl = c // Cg, c % Cg
        filters = list(range(g * per, (g + 1) * per))
        sp = self.result.sparsity(layer_id)
        body = {
            "layer": layer_id, "channel": c, "local_channel": cl, "filters": filters,
            "weight": _clean(np.asarray(raw_w)[filters, cl]),
            "zero_mask": sp["mask_weight"][filters, cl].astype(int).tolist(),
            "zero_kernels": sp["mask_kernel"][filters, cl].astype(int).tolist(),
            "unused": bool(sp["mask_input"][c]),
        }
        if "qweight" in arrays["raw"]:
            body["qweight"] = _clean(np.asarray(arrays["raw"]["qweight"])[filters, cl])
        if "kernel" in arrays:
            km = resolve_metric(entry, metric, "kernel")
            body["kernel_metric"] = km
            body["kernel_importance"] = _clean(np.asarray(arrays["kernel"][km][output])[filters, cl])
        self._send_json(body)

    def _serve_unit(self, path: str, query: dict):
        """/api/layer/<id>/unit/<filter>?output=&metric= -- everything the
        detail panel shows for one filter/neuron, sliced server-side so a
        512x512x3x3 layer never ships whole."""
        rest = path[len("/api/layer/"):]
        layer_id, _, f_str = rest.partition("/unit/")
        f = int(f_str)
        entry = self.result._layer_entry(layer_id)
        arrays = self.result.arrays[layer_id]
        output = int(query.get("output", 0))
        metric = query.get("metric", "auto")
        sp = self.result.sparsity(layer_id)
        raw_w = np.asarray(arrays["raw"]["weight"][f])
        C = raw_w.shape[0] if raw_w.ndim else 1
        groups = entry.get("groups") or 1
        F = entry["weight_shape"][0]
        group_idx = f // max(1, F // groups)
        body = {
            "layer": layer_id, "filter": f, "shape": list(raw_w.shape),
            "channel_offset": group_idx * C,
            "weight": _clean(raw_w),
            "zero_mask": sp["mask_weight"][f].astype(int).tolist(),
            "zero_kernels": sp["mask_kernel"][f].astype(int).tolist(),
            "is_zero_filter": bool(sp["mask_filter"][f]),
        }
        if "qweight" in arrays["raw"]:
            body["qweight"] = _clean(arrays["raw"]["qweight"][f])
        if "bias" in arrays["raw"]:
            body["bias"] = float(arrays["raw"]["bias"][f])
            body["bias_zero"] = bool(sp["bias_zero"][f])
        filt_levels = arrays.get("filter", {})
        for key in ("bias_mean_abs_s", "bias_mean_s"):
            if key in filt_levels:
                body[key] = float(np.asarray(filt_levels[key][output])[f])
        if "kernel" in arrays:
            km = resolve_metric(entry, metric, "kernel")
            body["kernel_metric"] = km
            body["kernel_importance"] = _clean(np.asarray(arrays["kernel"][km][output][f]))
        if "weight" in arrays:
            wm = resolve_metric(entry, metric, "weight")
            body["weight_metric"] = wm
            body["weight_importance"] = _clean(np.asarray(arrays["weight"][wm][output][f]))
            if "mean_s" in arrays["weight"]:
                body["weight_signed"] = _clean(np.asarray(arrays["weight"]["mean_s"][output][f]))
        self._send_json(body)

    def _serve_layer_slice(self, path: str, query: dict):
        # /api/layer/<layer_id>/<level>/<metric>?output=<idx>
        rest = path[len("/api/layer/"):]
        parts = rest.split("/")
        if len(parts) < 3:
            self._send(400, b"expected /api/layer/<layer_id>/<level>/<metric>", "text/plain")
            return
        layer_id, level, metric = parts[0], parts[1], parts[2]
        arr = self.result.arrays[layer_id][level][metric]
        if "output" in query:
            idx = int(query["output"])
            row = np.asarray(arr[idx])
            self._send_json({"shape": list(row.shape), "data": _clean(row)})
        else:
            arr_np = np.asarray(arr)
            self._send_json({"shape": list(arr_np.shape), "data": _clean(arr_np)})

    def _serve_sample(self, path: str, query: dict):
        # /api/samples/inputs?index=0   /api/samples/outputs
        name = path[len("/api/samples/"):]
        arrays = self.result.samples.get("arrays", {})
        if name not in arrays:
            self._send_json({"error": f"no sample array {name!r}"}, code=404)
            return
        arr = arrays[name]
        if "index" in query:
            idx = int(query["index"])
            row = np.asarray(arr[idx])
            self._send_json({"shape": list(row.shape), "data": row.tolist()})
        else:
            arr_np = np.asarray(arr)
            self._send_json({"shape": list(arr_np.shape), "data": arr_np.tolist()})


def make_server(result_dir: str, port: int = 8000, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    result = Result.load(result_dir)

    class BoundHandler(ViewerHandler):
        pass

    BoundHandler.result = result
    return ThreadingHTTPServer((host, port), BoundHandler)


def _lan_ip() -> Optional[str]:
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return None


def serve(result_dir: str, port: int = 8000, open_browser: bool = True, block: bool = True,
          host: str = "127.0.0.1") -> ThreadingHTTPServer:
    httpd = make_server(result_dir, port=port, host=host)
    actual_port = httpd.server_address[1]
    url = f"http://127.0.0.1:{actual_port}/"
    print(f"importance viewer serving {result_dir!r} at {url}")
    if host == "0.0.0.0":
        lan_ip = _lan_ip()
        if lan_ip:
            print(f"also reachable on the LAN at http://{lan_ip}:{actual_port}/")
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    if block:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()
        return httpd
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Serve the importance viewer for a result directory.")
    parser.add_argument("result_dir")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to expose on the LAN.")
    args = parser.parse_args()
    serve(args.result_dir, port=args.port, open_browser=not args.no_browser, host=args.host)


if __name__ == "__main__":
    main()
