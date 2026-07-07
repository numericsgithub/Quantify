"""
server.py — Flask HTTP server for the read-only monitoring API.

Runs werkzeug's threaded WSGI server in a daemon thread inside the training
process. All endpoints are GET-only and versioned under /api/v1/ so that
control endpoints (pause, LR changes, ...) can be added later without
redesign. Flask was chosen over FastAPI because it is a single small WSGI
dependency that runs happily in a background thread — no asyncio event
loop, uvicorn, or pydantic needed for a low-traffic read-only JSON API.

CORS is wide open (GET-only, read-only data) because the dashboard UI is
served from a different port than the API.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from flask import Flask, jsonify, request

from .collector import RunStateCollector


def create_app(collector: RunStateCollector) -> Flask:
    """Build the Flask app serving the given collector's state."""
    app = Flask("quantify_dashboard_api")

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        return response

    @app.get("/api/v1/health")
    def health():
        return jsonify({"ok": True})

    @app.get("/api/v1/status")
    def status():
        return jsonify(collector.status_snapshot())

    @app.get("/api/v1/config")
    def config():
        return jsonify(collector.config_snapshot())

    @app.get("/api/v1/metrics")
    def metrics():
        since_step = request.args.get("since_step", default=-1, type=int)
        since_epoch = request.args.get("since_epoch", default=-1, type=int)
        return jsonify(collector.metrics_snapshot(since_step, since_epoch))

    @app.get("/api/v1/metrics/latest")
    def metrics_latest():
        return jsonify(collector.latest_snapshot())

    @app.get("/api/v1/checkpoints")
    def checkpoints():
        return jsonify(collector.checkpoints_snapshot())

    return app


class DashboardAPIServer:
    """
    Threaded WSGI server wrapper.

    Usage::

        server = DashboardAPIServer(collector, host="127.0.0.1", port=8765)
        server.start()
        print(server.port)   # actual port (useful with port=0)
        ...
        server.shutdown()    # optional; the daemon thread dies with the process
    """

    def __init__(self, collector: RunStateCollector, host: str = "127.0.0.1", port: int = 8765):
        self.collector = collector
        self.host = host
        self._requested_port = port
        self._server = None
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> Optional[int]:
        return self._server.server_port if self._server is not None else None

    def start(self) -> bool:
        """Bind and serve in a daemon thread. Returns False if binding fails."""
        from werkzeug.serving import make_server

        # Silence werkzeug's per-request log lines — they would interleave
        # with tqdm progress bars in the training console.
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

        app = create_app(self.collector)
        try:
            self._server = make_server(self.host, self._requested_port, app, threaded=True)
        except OSError as e:
            print(f"[api] WARNING: could not bind {self.host}:{self._requested_port} — "
                  f"monitoring API disabled: {e}")
            return False

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="dashboard-api",
            daemon=True,
        )
        self._thread.start()
        print(f"[api] Monitoring API listening on http://{self.host}:{self.port}/api/v1/")
        return True

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
