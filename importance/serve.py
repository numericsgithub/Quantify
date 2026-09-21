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


class ViewerHandler(BaseHTTPRequestHandler):
    result: Result = None  # set per-instance via factory in serve()
    server_version = "ImportanceViewer/0.1"

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
                self._send_json(self.result.manifest)
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
            self._send_json({"shape": list(row.shape), "data": row.tolist()})
        else:
            arr_np = np.asarray(arr)
            self._send_json({"shape": list(arr_np.shape), "data": arr_np.tolist()})

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
