"""Deprecated import path: moved to ``quantizers.diagnostics``."""
import warnings

from quantizers.diagnostics import *  # noqa: F401,F403
from quantizers.diagnostics import compute_metrics, plot_quantizer_metrics, plot_quantizer_metrics_grid, run_diagnostics  # noqa: F401

warnings.warn(
    "utils.quantizer_diagnostics moved to quantizers.diagnostics.",
    DeprecationWarning,
    stacklevel=2,
)
