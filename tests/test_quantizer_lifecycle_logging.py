"""Lifecycle-event logging in BaseQuantizer.forward(): gate-open, calibration,
annealing-started, annealing-complete. These fire once per quantizer object
via the standard `logging` module (logger name "quantizers"), so a user can
explain from the logs alone when each quantizer actually started quantizing
-- not just the coarse, model-wide "[qat_sched] fake-quantization ENABLED"
message the training harness prints.
"""
import logging

import pytest
import torch
import torch.nn as nn
import brevitas.nn as qnn

from quantizers import FixedPointPerTensorWeightQuant, FixedPointPerTensorQuantizer
from quantizers.manager import QuantizerManager


@pytest.fixture(autouse=True)
def _isolated_manager():
    QuantizerManager().reset()
    yield
    QuantizerManager().reset()


def _quantized_conv():
    return qnn.QuantConv2d(3, 4, 3, padding=1, weight_quant=FixedPointPerTensorWeightQuant)


def _messages(caplog):
    return [r.message for r in caplog.records if r.name == "quantizers"]


def test_gate_open_is_logged_once_when_gap_delays_it(caplog):
    """quantization_start_gap staggers which forward call a quantizer's gate
    opens on; confirm the event is logged exactly once, on that call."""
    mgr = QuantizerManager()
    conv = _quantized_conv()
    mgr.quantization_start_gap = 3  # this quantizer's gate opens after gap*sequence_id passes

    conv.train()
    with caplog.at_level(logging.INFO, logger="quantizers"):
        for _ in range(5):
            with torch.no_grad():
                conv(torch.randn(2, 3, 8, 8))

    gate_msgs = [m for m in _messages(caplog) if "gate opened" in m]
    assert len(gate_msgs) == 1, gate_msgs


def test_gate_opens_immediately_with_zero_gap(caplog):
    mgr = QuantizerManager()
    conv = _quantized_conv()
    mgr.quantization_start_gap = 0

    conv.train()
    with caplog.at_level(logging.INFO, logger="quantizers"):
        with torch.no_grad():
            conv(torch.randn(2, 3, 8, 8))

    gate_msgs = [m for m in _messages(caplog) if "gate opened" in m]
    assert len(gate_msgs) == 1


def test_calibration_logged_once_then_recalibration_logged_separately(caplog):
    mgr = QuantizerManager()
    conv = _quantized_conv()
    conv.train()

    with caplog.at_level(logging.INFO, logger="quantizers"):
        with torch.no_grad():
            conv(torch.randn(2, 3, 8, 8))  # triggers first calibration

    cal_msgs = [m for m in _messages(caplog) if "calibration" in m]
    assert len(cal_msgs) == 1
    assert "completed" in cal_msgs[0]

    caplog.clear()
    mgr.force_recalibration = True
    with caplog.at_level(logging.INFO, logger="quantizers"):
        with torch.no_grad():
            conv(torch.randn(2, 3, 8, 8))  # forces a second calibration

    cal_msgs = [m for m in _messages(caplog) if "calibration" in m]
    assert len(cal_msgs) == 1
    assert "re-run" in cal_msgs[0]


def test_annealing_started_and_complete_are_logged_once_each(caplog):
    mgr = QuantizerManager()
    conv = _quantized_conv()
    conv.train()
    mgr.set_annealing_for_n_inferences(4)  # ramps annealing_alpha 0 -> 1 over 4 training forwards

    with caplog.at_level(logging.INFO, logger="quantizers"):
        for _ in range(6):
            with torch.no_grad():
                conv(torch.randn(2, 3, 8, 8))

    msgs = _messages(caplog)
    started = [m for m in msgs if "annealing started" in m]
    complete = [m for m in msgs if "annealing complete" in m]
    assert len(started) == 1, started
    assert len(complete) == 1, complete
    # started must be logged before complete
    assert msgs.index(started[0]) < msgs.index(complete[0])


def test_no_annealing_ramp_logs_immediate_complete_but_no_started(caplog):
    """Default annealing_alpha=1.0 (no set_annealing_for_n_inferences call):
    the quantizer is fully quantized from its very first active forward --
    "annealing complete" should still fire (useful to know it's fully
    quantized), but "annealing started" (mid-blend) never should."""
    conv = _quantized_conv()
    conv.train()

    with caplog.at_level(logging.INFO, logger="quantizers"):
        with torch.no_grad():
            conv(torch.randn(2, 3, 8, 8))

    msgs = _messages(caplog)
    assert not [m for m in msgs if "annealing started" in m]
    assert [m for m in msgs if "annealing complete" in m]


def test_quant_id_appears_in_log_messages_when_set(caplog):
    """The quantizer's identity (quant_id if set, else repr(id(self))) is
    embedded in every lifecycle log message so a specific layer can be
    picked out of a run with many quantizers."""
    q = FixedPointPerTensorQuantizer(bit_width=8)
    q.quant_id = "my_special_conv.weight"
    q.train()

    with caplog.at_level(logging.INFO, logger="quantizers"):
        q(torch.randn(4, 4))

    msgs = _messages(caplog)
    assert any("my_special_conv.weight" in m for m in msgs), msgs
