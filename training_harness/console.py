"""
Interactive training console.

A background thread reads lines from stdin. The main training loop drains
the command queue between epochs and dispatches each command. Only activates
when stdin is a TTY; silently does nothing when output is redirected.
"""

from __future__ import annotations

import queue
import sys
import threading
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from .trainer_v2 import QATTrainerV2


_HELP = """
  lr <value>      Set learning rate now,  e.g.  lr 3e-5
  stop            Stop after the current epoch finishes
  load-best       Restore best checkpoint weights into the running model
  load-last       Restore last checkpoint weights
  status          Print epoch, LR, and best metric so far
  patience <n>    Change ReduceLROnPlateau patience
  factor <f>      Change ReduceLROnPlateau reduction factor
  help            Show this message
""".rstrip()


class TrainingConsole:
    """
    Attach to a QATTrainerV2 to get a live command prompt during training.

    Commands are collected in a background thread and executed between epochs
    so they never interrupt a batch mid-flight.

    Usage inside QATTrainerV2.fit()::

        console = TrainingConsole(self)
        console.start()
        for epoch in ...:
            ...
            console.drain(epoch)
            if console.stop_requested:
                break
        console.stop()
    """

    def __init__(self, trainer: "QATTrainerV2") -> None:
        self.trainer = trainer
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stop_requested: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """
        Start the background stdin reader.

        Returns True if the console is active, False when stdin is not a TTY
        (e.g. output redirected to a file) in which case nothing is started.
        """
        if not sys.stdin.isatty():
            return False
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="training-console",
            daemon=True,
        )
        self._thread.start()
        tqdm.write(
            "\n[console] Interactive console active — "
            "type 'help' for commands (applied between epochs).\n"
        )
        return True

    def stop(self) -> None:
        """Signal the reader thread to exit."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Per-epoch drain (call from main thread between epochs)
    # ------------------------------------------------------------------

    def drain(self, epoch: int) -> None:
        """Process every command that arrived since the last drain."""
        while True:
            try:
                raw = self._queue.get_nowait()
            except queue.Empty:
                break
            self._dispatch(raw, epoch)

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    # ------------------------------------------------------------------
    # Background reader
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                line = sys.stdin.readline()
            except (EOFError, OSError):
                break
            if not line:        # EOF
                break
            cmd = line.strip()
            if cmd:
                self._queue.put(cmd)

    # ------------------------------------------------------------------
    # Command dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, raw: str, epoch: int) -> None:
        parts = raw.split()
        if not parts:
            return
        verb, args = parts[0].lower(), parts[1:]
        t = self.trainer

        if verb == "help":
            tqdm.write("[console] Available commands:" + _HELP)

        elif verb == "status":
            lr = t.optimizer.param_groups[0]["lr"]
            if t.checkpoint_mgr._records:
                best_val = t.checkpoint_mgr._records[0].metric_value
                best_str = f"{best_val:.4f}"
            else:
                best_str = "n/a"
            phase = "QAT" if t._qat_active else "float-warmup"
            tqdm.write(
                f"[console] epoch={epoch}/{t.config.epochs - 1}  "
                f"phase={phase}  lr={lr:.3e}  "
                f"best {t.config.checkpoint.monitor_metric}={best_str}"
            )

        elif verb == "lr":
            if not args:
                tqdm.write("[console] Usage:  lr <value>   e.g.  lr 3e-5")
                return
            try:
                new_lr = float(args[0])
            except ValueError:
                tqdm.write(f"[console] Bad LR value: {args[0]!r}")
                return
            for pg in t.optimizer.param_groups:
                pg["lr"] = new_lr
            tqdm.write(f"[console] Learning rate → {new_lr:.3e}")

        elif verb == "stop":
            self._stop_requested = True
            tqdm.write("[console] Stop requested — will halt after this epoch.")

        elif verb in ("load-best", "load_best"):
            path = t.checkpoint_mgr.best_checkpoint_path()
            if path is None:
                tqdm.write("[console] No best checkpoint saved yet.")
                return
            t.checkpoint_mgr.resume(
                t.model, path=path, device=str(t.device), reset_calibration=False,
            )
            tqdm.write(f"[console] Loaded best checkpoint: {path}")

        elif verb in ("load-last", "load_last"):
            path = t.checkpoint_mgr.last_checkpoint_path()
            if path is None:
                tqdm.write("[console] No last checkpoint saved yet.")
                return
            t.checkpoint_mgr.resume(
                t.model, path=path, device=str(t.device), reset_calibration=False,
            )
            tqdm.write(f"[console] Loaded last checkpoint: {path}")

        elif verb == "patience":
            if t._plateau_lr_sched is None:
                tqdm.write("[console] ReduceLROnPlateau is not active.")
                return
            if not args:
                tqdm.write("[console] Usage:  patience <n>")
                return
            try:
                t._plateau_lr_sched.patience = int(args[0])
                tqdm.write(f"[console] Plateau patience → {args[0]}")
            except ValueError:
                tqdm.write(f"[console] Bad patience value: {args[0]!r}")

        elif verb == "factor":
            if t._plateau_lr_sched is None:
                tqdm.write("[console] ReduceLROnPlateau is not active.")
                return
            if not args:
                tqdm.write("[console] Usage:  factor <f>")
                return
            try:
                t._plateau_lr_sched.factor = float(args[0])
                tqdm.write(f"[console] Plateau factor → {args[0]}")
            except ValueError:
                tqdm.write(f"[console] Bad factor value: {args[0]!r}")

        else:
            tqdm.write(
                f"[console] Unknown command: {raw!r}  — type 'help' for available commands."
            )
