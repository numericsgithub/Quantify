"""
serve.py — Tiny static server for the training dashboard UI.

The dashboard is a single self-contained HTML page that polls the training
process's monitoring API (see training_harness/api/) over HTTP with CORS,
so this server only needs to serve static files — zero dependencies.

Usage:
    python dashboard/serve.py                          # http://127.0.0.1:8080
    python dashboard/serve.py --port 9000
    python dashboard/serve.py --api http://127.0.0.1:8765   # pre-fill API base

Then open the printed URL. The API base URL can also be changed at any time
in the page header (recent bases are remembered per browser), so one UI
instance can switch between multiple concurrent runs on different ports.
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Quantify training dashboard UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    parser.add_argument("--api", default=None,
                        help="Training API base URL to pre-fill, e.g. http://127.0.0.1:8765")
    args = parser.parse_args()

    directory = os.path.dirname(os.path.abspath(__file__))
    handler = partial(SimpleHTTPRequestHandler, directory=directory)

    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    if args.api:
        url += f"?api={args.api}"
    print(f"[dashboard] Serving UI at {url}")
    print("[dashboard] Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
