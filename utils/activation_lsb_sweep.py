"""Deprecated import path: moved to ``quantizers.activation_lsb_sweep``.

``python -m utils.activation_lsb_sweep`` still works; prefer
``python -m quantizers.activation_lsb_sweep``.
"""
import warnings

from quantizers.activation_lsb_sweep import *  # noqa: F401,F403
from quantizers.activation_lsb_sweep import print_all_tables

warnings.warn(
    "utils.activation_lsb_sweep moved to quantizers.activation_lsb_sweep.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    print_all_tables()
