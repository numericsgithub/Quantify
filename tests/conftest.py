"""Shared pytest fixtures for the test suite."""
import pytest
import torch


@pytest.fixture
def device():
    """Return the default device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dummy_input(device):
    """Return a standard dummy input tensor for image models."""
    return torch.randn(2, 3, 64, 64, device=device)


@pytest.fixture
def dummy_classification_input(device):
    """Return a dummy input for classification models."""
    return torch.randn(4, 3, 32, 32, device=device)


@pytest.fixture
def dummy_linear_input(device):
    """Return a dummy input for linear/fully-connected layers."""
    return torch.randn(8, 64, device=device)


@pytest.fixture(autouse=True)
def _reset_quantizer_manager_switches():
    """QuantizerManager is a process-wide singleton, so a test that disables
    quantization or sets a start gap / forced recalibration would otherwise
    leak that into whatever runs next on the same pytest-xdist worker (seen
    as order-dependent failures in test_quant_activations.py). Reset just the
    global switches -- not the quantizer registry, which module-scoped
    fixtures may rely on."""
    from quantizers.manager import QuantizerManager

    mgr = QuantizerManager()
    mgr.quantization_globally_disabled = False
    mgr.quantization_start_gap = 0
    mgr.force_recalibration = False
    yield
