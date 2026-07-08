"""
server.py — Flask HTTP server for the dashboard monitoring + control API.

Runs werkzeug's threaded WSGI server in a daemon thread inside the training
process. Read endpoints are GET; control endpoints are POST and versioned
under /api/v1/. Flask was chosen over FastAPI because it is a single small
WSGI dependency that runs happily in a background thread — no asyncio event
loop, uvicorn, or pydantic needed for a low-traffic JSON API.

Control endpoints never mutate the trainer directly: they validate and
enqueue a command (see control.py), returning 202 Accepted. The training
loop applies it at a safe boundary. CORS is wide open because the dashboard
UI is served from a different port than the API.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from flask import Flask, jsonify, request

from .collector import RunStateCollector
from .control import ControlManager, ControlValidationError


def create_app(collector: RunStateCollector,
               control: Optional[ControlManager] = None) -> Flask:
    """Build the Flask app serving the given collector's state.

    If ``control`` is None the write endpoints return 503 (read-only mode).
    """
    app = Flask("quantify_dashboard_api")

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    def _require_control():
        if control is None:
            return jsonify({"error": "control API not enabled for this run"}), 503
        return None

    def _submit(ctype, params):
        gate = _require_control()
        if gate is not None:
            return gate
        try:
            cmd = control.submit(ctype, params)
        except ControlValidationError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify(cmd.to_dict()), 202

    # ── Read endpoints ────────────────────────────────────────────────

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

    @app.get("/api/v1/events")
    def events():
        since_id = request.args.get("since_id", default=-1, type=int)
        return jsonify(collector.events_snapshot(since_id))

    @app.get("/api/v1/callbacks")
    def callbacks():
        if control is None:
            return jsonify({"callbacks": []})
        return jsonify({"callbacks": control.callbacks.list()})

    @app.get("/api/v1/commands")
    def commands():
        if control is None:
            return jsonify({"commands": []})
        return jsonify({"commands": control.list_commands()})

    @app.get("/api/v1/commands/<cid>")
    def command(cid):
        if control is None:
            return jsonify({"error": "control API not enabled for this run"}), 503
        cmd = control.get_command(cid)
        if cmd is None:
            return jsonify({"error": f"no such command: {cid}"}), 404
        return jsonify(cmd)

    # ── Control (write) endpoints ─────────────────────────────────────

    @app.post("/api/v1/control/hyperparams")
    def control_hyperparams():
        return _submit("set_hyperparams", request.get_json(silent=True) or {})

    @app.post("/api/v1/control/callbacks/<name>")
    def control_toggle_callback(name):
        body = request.get_json(silent=True) or {}
        return _submit("toggle_callback", {"name": name, "enabled": body.get("enabled")})

    @app.post("/api/v1/control/reload-best")
    def control_reload_best():
        return _submit("reload_best", request.get_json(silent=True) or {})

    @app.post("/api/v1/control/add-epochs")
    def control_add_epochs():
        return _submit("add_epochs", request.get_json(silent=True) or {})

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

    def __init__(self, collector: RunStateCollector,
                 control: Optional[ControlManager] = None,
                 host: str = "127.0.0.1", port: int = 8765):
        self.collector = collector
        self.control = control
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

        app = create_app(self.collector, self.control)
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
