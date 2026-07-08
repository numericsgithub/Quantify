"""
control.py — Live control layer for the training dashboard (read/write).

This is the write half of the dashboard. It never mutates the trainer's
shared objects (optimizer, scheduler, model, callbacks) from the API
thread. Instead it uses a command-queue:

  1. An HTTP handler on the API thread calls ``ControlManager.submit()``,
     which VALIDATES the command immediately (bad input -> raise, the
     route turns that into HTTP 400) and, if valid, records it and pushes
     it onto a thread-safe queue. The command starts life as "pending".
  2. The training loop (main thread) calls ``ControlManager.drain(boundary)``
     at a SAFE boundary — "step" (end of a training step) for fast, cheap
     mutations like LR, "epoch" (between epochs) for structural ones like
     reloading weights or extending the epoch budget. drain() applies the
     command on the training thread and marks it "applied" or "failed".

So the actual mutation always happens on the thread that owns the object,
at a point where nothing is mid-flight. The API thread only ever touches
the queue and the (lock-guarded) command records.

Callbacks are not a real framework in this codebase — see CallbackRegistry
for how existing hardcoded loop behaviors are surfaced as named,
toggleable entries without moving any logic out of the loop.
"""

from __future__ import annotations

import math
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# Hard bounds for validation. LR above ~1 is already 100-1000x typical QAT
# values; anything larger is almost certainly a fat-finger and would waste
# hours of compute, so we reject it rather than trust it.
MAX_LR = 10.0
MAX_WEIGHT_DECAY = 1.0
MAX_ADD_EPOCHS = 100_000


class ControlValidationError(ValueError):
    """Raised when a submitted command is invalid (-> HTTP 400)."""


# ---------------------------------------------------------------------------
# Callback registry
# ---------------------------------------------------------------------------

@dataclass
class CallbackInfo:
    """One named, introspectable loop behavior."""
    name: str
    event: str            # "step_end" | "epoch_end" | ...
    description: str
    toggleable: bool
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "event": self.event,
            "description": self.description,
            "toggleable": self.toggleable,
            "enabled": self.enabled,
        }


class CallbackRegistry:
    """
    Names the loop behaviors that already exist so they can be listed and
    (where safe) toggled. This is NOT a callback framework — the logic
    stays inline in the trainer; each behavior just gains a one-line
    ``if registry.is_enabled(name)`` guard.

    Core behaviors that would corrupt the run or blind the dashboard
    (the optimizer step, metrics logging) are registered as
    ``toggleable=False`` so they show up for transparency but reject any
    toggle attempt.
    """

    def __init__(self) -> None:
        self._items: Dict[str, CallbackInfo] = {}
        self._lock = threading.Lock()

    def register(self, name: str, event: str, description: str,
                 toggleable: bool = True, enabled: bool = True) -> None:
        self._items[name] = CallbackInfo(name, event, description, toggleable, enabled)

    def is_enabled(self, name: str) -> bool:
        # Fail-open: an unregistered name reads as enabled so a typo in a
        # guard can never silently switch a core behavior off.
        info = self._items.get(name)
        return info.enabled if info is not None else True

    def validate_toggle(self, name: str) -> None:
        """Submit-time check (API thread). Raises on unknown/non-toggleable."""
        info = self._items.get(name)
        if info is None:
            raise ControlValidationError(f"unknown callback: {name!r}")
        if not info.toggleable:
            raise ControlValidationError(
                f"callback {name!r} is core and cannot be toggled")

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Apply-time mutation (training thread)."""
        with self._lock:
            info = self._items.get(name)
            if info is None:
                raise ControlValidationError(f"unknown callback: {name!r}")
            if not info.toggleable:
                raise ControlValidationError(
                    f"callback {name!r} is core and cannot be toggled")
            info.enabled = bool(enabled)

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [info.to_dict() for info in self._items.values()]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@dataclass
class ControlCommand:
    id: str
    type: str
    params: Dict[str, Any]
    apply_at: str                       # "step" | "epoch"
    status: str = "pending"             # pending | applied | failed
    result: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    applied_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "params": self.params,
            "apply_at": self.apply_at,
            "status": self.status,
            "result": self.result,
            "created_at": self.created_at,
            "applied_at": self.applied_at,
        }


class ControlManager:
    """
    Validates, queues, and applies control commands for one live run.

    Bound to a single V1 ``Trainer``. The trainer calls ``drain("step")``
    after each training step and ``drain("epoch")`` between epochs.
    """

    # Which boundary each command type is applied at.
    _APPLY_AT = {
        "set_hyperparams": "step",
        "toggle_callback": "epoch",
        "reload_best": "epoch",
        "add_epochs": "epoch",
    }

    def __init__(self, trainer: Any, collector: Any, callbacks: CallbackRegistry):
        self.trainer = trainer
        self.collector = collector
        self.callbacks = callbacks
        self._queues: Dict[str, "queue.Queue[ControlCommand]"] = {
            "step": queue.Queue(),
            "epoch": queue.Queue(),
        }
        self._records: Dict[str, ControlCommand] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Submit (API thread)
    # ------------------------------------------------------------------

    def submit(self, ctype: str, params: Optional[Dict[str, Any]]) -> ControlCommand:
        """Validate + enqueue a command. Raises ControlValidationError on bad input."""
        if ctype not in self._APPLY_AT:
            raise ControlValidationError(f"unknown command type: {ctype!r}")
        clean = self._validate(ctype, params or {})
        cmd = ControlCommand(
            id=uuid.uuid4().hex[:12],
            type=ctype,
            params=clean,
            apply_at=self._APPLY_AT[ctype],
        )
        with self._lock:
            self._records[cmd.id] = cmd
        self._queues[cmd.apply_at].put(cmd)
        self._event("command_submitted", f"{ctype} submitted", {"id": cmd.id, "params": clean})
        return cmd

    # ------------------------------------------------------------------
    # Drain + apply (training thread, at a safe boundary)
    # ------------------------------------------------------------------

    def drain(self, boundary: str) -> None:
        """Apply every queued command for this boundary. Cheap when empty."""
        q = self._queues.get(boundary)
        if q is None:
            return
        while True:
            try:
                cmd = q.get_nowait()
            except queue.Empty:
                break
            self._apply_one(cmd)

    def _apply_one(self, cmd: ControlCommand) -> None:
        try:
            result = self._apply(cmd)
            with self._lock:
                cmd.status = "applied"
                cmd.result = result
                cmd.applied_at = time.time()
            self._event("command_applied", f"{cmd.type}: {result}", {"id": cmd.id})
        except Exception as e:  # apply-time failure (e.g. no checkpoint yet)
            with self._lock:
                cmd.status = "failed"
                cmd.result = str(e)
                cmd.applied_at = time.time()
            self._event("command_failed", f"{cmd.type} failed: {e}", {"id": cmd.id})

    # ------------------------------------------------------------------
    # History (API thread)
    # ------------------------------------------------------------------

    def list_commands(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [c.to_dict() for c in self._records.values()]

    def get_command(self, cid: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cmd = self._records.get(cid)
            return cmd.to_dict() if cmd is not None else None

    # ------------------------------------------------------------------
    # Validation (runs on the API thread; raises -> HTTP 400)
    # ------------------------------------------------------------------

    def _validate(self, ctype: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(params, dict):
            raise ControlValidationError("request body must be a JSON object")
        method: Callable = getattr(self, f"_validate_{ctype}")
        return method(params)

    def _validate_set_hyperparams(self, params: Dict[str, Any]) -> Dict[str, Any]:
        clean: Dict[str, Any] = {}
        if "lr" in params and params["lr"] is not None:
            lr = self._as_float(params["lr"], "lr")
            if not (0.0 < lr <= MAX_LR):
                raise ControlValidationError(
                    f"lr must be in (0, {MAX_LR}]; got {lr}")
            clean["lr"] = lr
        if "weight_decay" in params and params["weight_decay"] is not None:
            wd = self._as_float(params["weight_decay"], "weight_decay")
            if not (0.0 <= wd <= MAX_WEIGHT_DECAY):
                raise ControlValidationError(
                    f"weight_decay must be in [0, {MAX_WEIGHT_DECAY}]; got {wd}")
            clean["weight_decay"] = wd
        if "suspend_scheduler" in params and params["suspend_scheduler"] is not None:
            if not isinstance(params["suspend_scheduler"], bool):
                raise ControlValidationError("suspend_scheduler must be a boolean")
            clean["suspend_scheduler"] = params["suspend_scheduler"]
        if not clean:
            raise ControlValidationError(
                "provide at least one of: lr, weight_decay, suspend_scheduler")
        return clean

    def _validate_toggle_callback(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise ControlValidationError("callback name is required")
        if "enabled" not in params or not isinstance(params["enabled"], bool):
            raise ControlValidationError("enabled (boolean) is required")
        self.callbacks.validate_toggle(name)  # unknown / non-toggleable -> 400
        return {"name": name, "enabled": params["enabled"]}

    def _validate_reload_best(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if params.get("confirm") is not True:
            raise ControlValidationError(
                "reload-best is destructive to in-flight progress; "
                "resend with {\"confirm\": true}")
        return {"confirm": True}

    def _validate_add_epochs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        count = params.get("count")
        if not isinstance(count, int) or isinstance(count, bool):
            raise ControlValidationError("count must be an integer")
        if not (0 < count <= MAX_ADD_EPOCHS):
            raise ControlValidationError(
                f"count must be in (0, {MAX_ADD_EPOCHS}]; got {count}")
        return {"count": count}

    @staticmethod
    def _as_float(value: Any, field_name: str) -> float:
        try:
            f = float(value)
        except (TypeError, ValueError):
            raise ControlValidationError(f"{field_name} must be a number")
        if not math.isfinite(f):
            raise ControlValidationError(f"{field_name} must be finite")
        return f

    # ------------------------------------------------------------------
    # Application (runs on the training thread, at a safe boundary)
    # ------------------------------------------------------------------

    def _apply(self, cmd: ControlCommand) -> str:
        return getattr(self, f"_apply_{cmd.type}")(cmd.params)

    def _apply_set_hyperparams(self, params: Dict[str, Any]) -> str:
        t = self.trainer
        changes: List[str] = []
        if "lr" in params:
            for pg in t.optimizer.param_groups:
                pg["lr"] = params["lr"]
            changes.append(f"lr={params['lr']:g}")
        if "weight_decay" in params:
            for pg in t.optimizer.param_groups:
                pg["weight_decay"] = params["weight_decay"]
            changes.append(f"weight_decay={params['weight_decay']:g}")
        # Scheduler suspension. Only meaningful when a scheduler exists (it
        # would otherwise overwrite the LR on its next step). Default: an LR
        # change suspends it; suspend_scheduler=false explicitly resumes.
        if t.scheduler is not None:
            suspend = params.get("suspend_scheduler", "lr" in params)
            if "suspend_scheduler" in params or "lr" in params:
                t._scheduler_suspended = bool(suspend)
                changes.append("scheduler " + ("suspended" if suspend else "resumed"))
        return ", ".join(changes) if changes else "no-op"

    def _apply_toggle_callback(self, params: Dict[str, Any]) -> str:
        self.callbacks.set_enabled(params["name"], params["enabled"])
        state = "enabled" if params["enabled"] else "disabled"
        return f"callback {params['name']} {state}"

    def _apply_reload_best(self, params: Dict[str, Any]) -> str:
        t = self.trainer
        mgr = getattr(t, "checkpoint_mgr", None)
        best = mgr.best_checkpoint_path() if mgr is not None else None
        if not best:
            raise RuntimeError("no best checkpoint available to reload")
        # Weights only, no optimizer state, calibration preserved — mirrors
        # the interactive console's load-best (console.py).
        mgr.resume(t.model, path=best, device=str(t.device), reset_calibration=False)
        return f"reloaded weights from {best}"

    def _apply_add_epochs(self, params: Dict[str, Any]) -> str:
        t = self.trainer
        old = t.config.epochs
        t.config.epochs = old + params["count"]
        timer = getattr(t, "_timer", None)
        if timer is not None:
            timer.total_epochs = t.config.epochs
        return f"epoch budget {old} -> {t.config.epochs}"

    # ------------------------------------------------------------------

    def _event(self, kind: str, message: str, detail: Optional[dict] = None) -> None:
        if self.collector is not None:
            try:
                self.collector.record_event(kind, message, detail)
            except Exception:
                pass
