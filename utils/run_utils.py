"""
run_utils.py — Shared utilities for training script entry points.

- next_run_dir: auto-increment output directories so runs don't clobber each other
- env_default:  read argparse defaults from environment variables
- setup_output_tee: duplicate stdout/stderr into a log file in the output dir
"""

import os
import sys
from typing import Optional


def next_run_dir(base_dir: str) -> str:
    """
    Return base_dir if it doesn't already exist; otherwise return
    base_dir_1, base_dir_2, ... until a free name is found.

    Useful for training scripts so successive runs don't overwrite each other.
    """
    if not os.path.exists(base_dir):
        return base_dir
    i = 1
    while True:
        candidate = f"{base_dir}_{i}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


def env_default(var: str, fallback: Optional[str] = None) -> Optional[str]:
    """
    Return os.environ.get(var, fallback).

    Use as the default= value in argparse add_argument() calls so that
    environment variables act as user-level defaults that CLI flags can still
    override:

        parser.add_argument("--data-dir", default=env_default("IMAGENET_DALI_PATH"))
    """
    return os.environ.get(var, fallback)


class _Tee:
    """Writes to both the original stream and a log file simultaneously."""

    def __init__(self, original, logfile):
        self._orig = original
        self._log = logfile

    def write(self, data):
        self._orig.write(data)
        self._log.write(data)

    def flush(self):
        self._orig.flush()
        self._log.flush()

    def isatty(self):
        return self._orig.isatty()

    def fileno(self):
        return self._orig.fileno()

    def __getattr__(self, name):
        return getattr(self._orig, name)


def setup_output_tee(output_dir: str, filename: str = "run.log"):
    """
    Create output_dir if needed, open a log file there, and redirect
    both sys.stdout and sys.stderr through a Tee so every print/warning
    also lands in the log file.

    Call this early in main(), after output_dir is finalised.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, filename)
    log_file = open(log_path, "a", buffering=1)  # line-buffered
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    print(f"[run] Logging to {log_path}")
