"""Lightweight CSV logging helper for standalone scripts.

For full experiment tracking (TensorBoard, W&B, hparams, epoch/step routing),
use `training_harness.logger.ExperimentLogger` instead.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence


class CSVLogger:
    """Minimal line-buffered CSV logger.

    Use as a context manager so the file is flushed and closed on exit::

        with CSVLogger(path, ["epoch", "loss", "acc"]) as log:
            log.log(epoch=1, loss=0.5, acc=80.0)

    The file is flushed after every ``log()`` call so partial runs
    are still inspectable from another shell while training.
    """

    def __init__(self, path: Path | str, fieldnames: Sequence[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)

        self._file = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()
        self._file.flush()

    def log(self, **row: Any) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
