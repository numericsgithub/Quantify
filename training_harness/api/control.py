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
import os
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

# Valid quantizer groups for the Phase-4 group-targeted QAT controls.
_QUANT_GROUPS = ("weights", "biases", "activations", "all")


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
        "end_epoch_early": "step",
        "halt": "epoch",
        "set_scheduler_params": "epoch",
        # QAT group controls (Phase 4)
        "set_annealing": "step",
        "set_lsb": "step",
        "recalibrate": "epoch",
        "disable_quant": "epoch",
        # NB: pause/resume are NOT queued — see pause()/resume() below.
    }

    def __init__(self, trainer: Any, collector: Any,
                 callbacks: Optional[CallbackRegistry] = None):
        self.trainer = trainer
        self.collector = collector
        # May be None: the V2 trainer has no CallbackRegistry, so callback
        # toggling is rejected there (see _validate_toggle_callback).
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
        if self.callbacks is None:
            raise ControlValidationError(
                "callback toggling is not supported on this trainer "
                "(V2 has no callback registry - a known regression from V1, "
                "pending the callback-system work)")
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
        # Which checkpoint pool to reload from: "best" (primary monitor) or
        # "best_<metric>" (a secondary pool, e.g. best_train_loss). Resolved at
        # apply time against the trainer's pools; an unknown one fails there.
        criterion = params.get("criterion", "best")
        if not isinstance(criterion, str) or not criterion:
            raise ControlValidationError("criterion must be a non-empty string")
        # weights_only=True (default, safer) restores only model weights; False
        # also restores optimizer + scheduler state.
        weights_only = params.get("weights_only", True)
        if not isinstance(weights_only, bool):
            raise ControlValidationError("weights_only must be a boolean")
        return {"confirm": True, "criterion": criterion, "weights_only": weights_only}

    def _validate_end_epoch_early(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def _validate_halt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if params.get("confirm") is not True:
            raise ControlValidationError(
                "halt ends the run and is NOT resumable; "
                "resend with {\"confirm\": true}")
        return {"confirm": True}

    def _validate_set_scheduler_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Edit the live ReduceLROnPlateau (patience / factor / min_lr)."""
        clean: Dict[str, Any] = {}
        if params.get("patience") is not None:
            p = params["patience"]
            if not isinstance(p, int) or isinstance(p, bool) or p < 0:
                raise ControlValidationError("patience must be a non-negative integer")
            clean["patience"] = p
        if params.get("factor") is not None:
            f = self._as_float(params["factor"], "factor")
            if not (0.0 < f < 1.0):
                raise ControlValidationError("factor must be in (0, 1)")
            clean["factor"] = f
        if params.get("min_lr") is not None:
            m = self._as_float(params["min_lr"], "min_lr")
            if m < 0.0:
                raise ControlValidationError("min_lr must be >= 0")
            clean["min_lr"] = m
        if not clean:
            raise ControlValidationError(
                "provide at least one of: patience, factor, min_lr")
        return clean

    # ---- QAT group controls (Phase 4) ---------------------------------

    def _require_group(self, params: Dict[str, Any]) -> str:
        group = params.get("group")
        if group not in _QUANT_GROUPS:
            raise ControlValidationError(
                f"group must be one of {list(_QUANT_GROUPS)}; got {group!r}")
        return group

    def _validate_set_annealing(self, params: Dict[str, Any]) -> Dict[str, Any]:
        group = self._require_group(params)
        mode = params.get("mode")
        # ramp vs absolute vs step are kept distinct on purpose — see the
        # semantics trap in the Phase-4 notes.
        if mode not in ("ramp", "absolute", "step"):
            raise ControlValidationError(
                "mode must be one of: 'ramp' (alpha 0->1 over n passes), "
                "'absolute' (set alpha=X now), 'step' (set per-forward increment)")
        clean: Dict[str, Any] = {"group": group, "mode": mode}
        if mode == "ramp":
            n = params.get("n")
            if not isinstance(n, int) or isinstance(n, bool) or n < 1:
                raise ControlValidationError("ramp mode requires integer n >= 1")
            clean["n"] = n
        elif mode == "absolute":
            a = self._as_float(params.get("alpha"), "alpha")
            if not (0.0 <= a <= 1.0):
                raise ControlValidationError("alpha must be in [0, 1]")
            clean["alpha"] = a
        else:  # step
            s = self._as_float(params.get("step"), "step")
            if not (0.0 <= s <= 1.0):
                raise ControlValidationError("step must be in [0, 1]")
            clean["step"] = s
        return clean

    def _validate_set_lsb(self, params: Dict[str, Any]) -> Dict[str, Any]:
        qid = params.get("quant_id")
        if not isinstance(qid, str) or not qid:
            raise ControlValidationError("quant_id (string) is required")
        lsb = params.get("lsb")
        if not isinstance(lsb, int) or isinstance(lsb, bool):
            raise ControlValidationError("lsb must be an integer")
        if not (-32 <= lsb <= 32):
            raise ControlValidationError("lsb must be in [-32, 32]")
        return {"quant_id": qid, "lsb": lsb}

    def _validate_recalibrate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        group = self._require_group(params)
        if params.get("confirm") is not True:
            raise ControlValidationError(
                "recalibration re-runs the LSB search on the next forward "
                "(against whatever batch comes next); resend with "
                "{\"confirm\": true}")
        return {"group": group, "confirm": True}

    def _validate_disable_quant(self, params: Dict[str, Any]) -> Dict[str, Any]:
        group = self._require_group(params)
        if params.get("confirm") is not True:
            raise ControlValidationError(
                "disabling quantization changes the model mid-run; "
                "resend with {\"confirm\": true}")
        return {"group": group, "confirm": True}

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
        # V2 can run TWO LR-writing schedulers (a per-step self.scheduler AND an
        # epoch-stepped ReduceLROnPlateau); the flag suspends BOTH (see
        # trainer_v2 where both step() calls are guarded by _scheduler_suspended).
        has_scheduler = (t.scheduler is not None
                         or getattr(t, "_plateau_lr_sched", None) is not None)
        if has_scheduler:
            suspend = params.get("suspend_scheduler", "lr" in params)
            if "suspend_scheduler" in params or "lr" in params:
                t._scheduler_suspended = bool(suspend)
                changes.append("scheduler(s) " + ("suspended" if suspend else "resumed"))
        return ", ".join(changes) if changes else "no-op"

    def _apply_toggle_callback(self, params: Dict[str, Any]) -> str:
        self.callbacks.set_enabled(params["name"], params["enabled"])
        state = "enabled" if params["enabled"] else "disabled"
        return f"callback {params['name']} {state}"

    def _apply_reload_best(self, params: Dict[str, Any]) -> str:
        t = self.trainer
        criterion = params.get("criterion", "best")
        weights_only = params.get("weights_only", True)

        # Resolve which pool: "best" -> primary; "best_<metric>"/"<metric>" ->
        # a secondary pool. Falls back to the single checkpoint_mgr on trainers
        # without the pool registry (V1).
        pools = getattr(t, "_checkpoint_pools", None)
        if pools and criterion not in ("best", ""):
            metric = criterion[len("best_"):] if criterion.startswith("best_") else criterion
            mgr = pools.get(metric)
            if mgr is None:
                raise RuntimeError(
                    f"no checkpoint pool tracks {metric!r}; available criteria: "
                    f"'best' + {sorted('best_' + m for m in pools)}")
        else:
            mgr = getattr(t, "checkpoint_mgr", None)

        rec = mgr.best_checkpoint_record() if mgr is not None else None
        if rec is None:
            raise RuntimeError("no best checkpoint available to reload")

        # weights_only (default) mirrors the safe console load-best: model
        # weights only, calibration preserved, optimizer/scheduler untouched.
        if weights_only:
            mgr.resume(t.model, path=rec.path, device=str(t.device),
                       reset_calibration=False)
        else:
            mgr.resume(t.model, optimizer=getattr(t, "optimizer", None),
                       scheduler=getattr(t, "scheduler", None), path=rec.path,
                       device=str(t.device), reset_calibration=False)

        scope = "weights" if weights_only else "weights+optimizer"
        return (f"restored epoch {rec.epoch} ({rec.metric_value:.4f}) "
                f"[{criterion}, {scope}] from {os.path.basename(rec.path)}")

    def _apply_end_epoch_early(self, params: Dict[str, Any]) -> str:
        t = self.trainer
        if not hasattr(t, "_end_epoch_early"):
            raise RuntimeError("end-epoch-early is not supported by this trainer")
        t._end_epoch_early = True
        return "current epoch will end after this step, then validation runs"

    def _apply_halt(self, params: Dict[str, Any]) -> str:
        t = self.trainer
        if not hasattr(t, "_halt_requested"):
            raise RuntimeError("halt is not supported by this trainer")
        t._halt_requested = True
        return "run will halt after the current epoch (not resumable)"

    def _apply_set_scheduler_params(self, params: Dict[str, Any]) -> str:
        sched = getattr(self.trainer, "_plateau_lr_sched", None)
        if sched is None:
            raise RuntimeError(
                "no ReduceLROnPlateau scheduler is active on this run "
                "(set reduce_lr_on_plateau=True in the config)")
        changes: List[str] = []
        if "patience" in params:
            sched.patience = params["patience"]
            changes.append(f"patience={params['patience']}")
        if "factor" in params:
            sched.factor = params["factor"]
            changes.append(f"factor={params['factor']:g}")
        if "min_lr" in params:
            # ReduceLROnPlateau keeps one min_lr per param group.
            sched.min_lrs = [params["min_lr"]] * len(sched.min_lrs)
            changes.append(f"min_lr={params['min_lr']:g}")
        return "plateau scheduler: " + ", ".join(changes)

    # ---- QAT group controls (Phase 4) ---------------------------------

    def _apply_set_annealing(self, params: Dict[str, Any]) -> str:
        from quantizers.manager import QuantizerManager
        mgr = QuantizerManager()
        group, mode = params["group"], params["mode"]
        if mode == "ramp":
            res = mgr.set_group_annealing_ramp(group, params["n"])
            what = f"ramp alpha 0->1 over {params['n']} passes"
        elif mode == "absolute":
            res = mgr.set_group_annealing_alpha(group, params["alpha"])
            what = f"alpha={params['alpha']:g}"
        else:
            res = mgr.set_group_annealing_step(group, params["step"])
            what = f"alpha_step={params['step']:g}"
        return self._group_result(what, res)

    def _apply_set_lsb(self, params: Dict[str, Any]) -> str:
        from quantizers.manager import QuantizerManager
        res = QuantizerManager().set_lsb(params["quant_id"], params["lsb"])
        return f"lsb={res['lsb']} set on {res['quant_id']} ({res['role']})"

    def _apply_recalibrate(self, params: Dict[str, Any]) -> str:
        from quantizers.manager import QuantizerManager
        res = QuantizerManager().recalibrate_group(params["group"])
        return self._group_result(
            "recalibration armed (runs on each quantizer's next forward)", res)

    def _apply_disable_quant(self, params: Dict[str, Any]) -> str:
        from quantizers.manager import QuantizerManager
        res = QuantizerManager().disable_group(params["group"])
        return self._group_result("quantization disabled (alpha=0, step=0)", res)

    @staticmethod
    def _group_result(what: str, res: Dict[str, Any]) -> str:
        msg = f"{what}: {res['count']} {res['group']} quantizer(s)"
        if res.get("unknown_role"):
            msg += (f" — WARNING: {res['unknown_role']} quantizer(s) have unknown "
                    f"role and were NOT affected")
        return msg

    # ------------------------------------------------------------------
    # Pause / resume — direct, off-queue, by design
    # ------------------------------------------------------------------
    #
    # These are the sanctioned exceptions to "never mutate off the queue".
    # Rationale: only the *blocking* must happen at a safe boundary, and it
    # does — the loop's pause GATE (a threading.Event.wait() at the step
    # boundary) runs on the training thread. Setting/clearing that Event is a
    # cross-thread signal, exactly what threading.Event is for.
    #
    # Pause must be direct too (not queued): if it were queued, a resume could
    # arrive and run BEFORE the still-queued pause applied, and the stale pause
    # would then fire and re-pause the run. Direct set/clear has no such race.

    def pause(self) -> Dict[str, Any]:
        """Request a pause. The loop blocks at its next step-boundary gate."""
        t = self.trainer
        ev = getattr(t, "_pause_event", None)
        if ev is None:
            return {"paused": False, "reason": "pause not supported by this trainer"}
        already = not ev.is_set()
        t._paused = True
        ev.clear()
        self._event("paused",
                    "pause requested (already paused)" if already
                    else "pause requested — blocks at next step boundary")
        return {"paused": True, "was_paused": already}

    def resume(self) -> Dict[str, Any]:
        """Release a paused run by setting the gate Event."""
        t = self.trainer
        ev = getattr(t, "_pause_event", None)
        if ev is None:
            return {"resumed": False, "reason": "pause not supported by this trainer"}
        was_paused = not ev.is_set()
        t._paused = False
        ev.set()
        self._event("resumed",
                    "training resumed" if was_paused else "resume (was not paused)")
        return {"resumed": True, "was_paused": was_paused}

    def _apply_add_epochs(self, params: Dict[str, Any]) -> str:
        t = self.trainer
        old = t.config.epochs
        t.config.epochs = old + params["count"]
        timer = getattr(t, "_timer", None)          # V1 exposes its EpochTimer
        if timer is not None:
            timer.total_epochs = t.config.epochs
        # V2 runs `while epoch < self._end_epoch`; extend that re-read bound so
        # the extra epochs actually run this session (V1 relies on config.epochs).
        if getattr(t, "_end_epoch", None) is not None:
            t._end_epoch += params["count"]
        return f"epoch budget {old} -> {t.config.epochs}"

    # ------------------------------------------------------------------

    def _event(self, kind: str, message: str, detail: Optional[dict] = None) -> None:
        if self.collector is not None:
            try:
                self.collector.record_event(kind, message, detail)
            except Exception:
                pass
