"""
Where to save and load data?
This workspace enviroment describes and manages where to save/load datasets, models, and other data created or needed at runtime.

Every example follows the same layout::

    <root>/
    ├── data/          # dataset downloads
    ├── checkpoints/   # model weights
    └── logs/          # CSV logs, tensorboard, etc.


The root is controlled by (in priority order):

1. The ``--workdir`` CLI flag.
2. The ``$QATLAB_WORKDIR`` environment variable, combined with the
   example's ``name`` (e.g. ``$QATLAB_WORKDIR/cifar10_vgg``).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_WORKDIR_ENV = "QUANT_WORKDIR"
DEFAULT_ROOT = None


@dataclass(frozen=True)
class Workspace:
    """Bundle of directories produced by a single training run.

    ``root`` is the only real attribute; the subdirectories are
    derived from it so there is a single source of truth.
    """

    root_folder_path: Path
    root_sub_folder_path: Path

    @property
    def root(self) -> Path:
        return self.root_sub_folder_path

    @property
    def data(self) -> Path:
        return self.root_sub_folder_path / "data"

    @property
    def datasets(self) -> Path:
        return self.root_folder_path / "cached_datasets"

    @property
    def checkpoints(self) -> Path:
        return self.root_sub_folder_path / "checkpoints"

    @property
    def logs(self) -> Path:
        return self.root_sub_folder_path / "logs"

    def ensure(self) -> "Workspace":
        """Create every subdirectory if it doesn't already exist."""
        for d in (self.root_folder_path, self.root, self.data, self.checkpoints, self.logs, self.datasets):
            d.mkdir(parents=True, exist_ok=True)
        return self

    @classmethod
    def at(cls, root: os.PathLike | str, sub_folder_name: os.PathLike | str) -> "Workspace":
        """Build a Workspace rooted at ``root`` (expanded & resolved),
        and create its subdirectories."""
        return cls(root_folder_path=Path(root / sub_folder_name).expanduser().resolve(), root_sub_folder_path=Path(root / sub_folder_name).expanduser().resolve()).ensure()


def add_workspace_args(
    parser: argparse.ArgumentParser,
    *,
    name: str,
    env_var: str = DEFAULT_WORKDIR_ENV,
) -> None:
    """Register a ``--workdir`` flag on ``parser``.

    Parameters
    ----------
    parser
        Your example's ``argparse.ArgumentParser``.
    name
        Short identifier for this example (e.g. ``"cifar10_vgg"``).
        Used to build a per-example subdirectory when the env var
        fallback is active.
    env_var
        Environment variable consulted for the base directory when
        ``--workdir`` is not given on the command line.
    """

    parser.add_argument(
        "--workdir",
        type=Path,
        default=name,
        help=(
            f"Base directory for data, checkpoints and logs. "
        ),
    )


def workspace_from_args(args: argparse.Namespace) -> Workspace:
    """Create and prepare a :class:`Workspace` from parsed CLI args."""
    env_root = os.environ.get(DEFAULT_WORKDIR_ENV)
    if env_root:
        main_root = Path(env_root)
    else:
        if DEFAULT_ROOT is None:
            raise Exception("Set enviroment variable " + env_root)
        main_root = DEFAULT_ROOT

    if "/" in str(args.workdir).strip() or "\\" in str(args.workdir).strip():
        raise Exception("The workdir is not a simple folder name")

    return Workspace.at(main_root, args.workdir)
