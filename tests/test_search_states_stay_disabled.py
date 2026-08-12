"""
Regression tests for _set_search_states: quantizers it disables must STAY
disabled for the whole role search.

The bug this guards against
---------------------------
_set_search_states disabled a quantizer by setting annealing_alpha=0, but left
annealing_alpha_step at its default (0.1). BaseQuantizer.forward anneals
alpha += alpha_step on every *training* forward, and the LSB search runs one
training (calibration) forward per quantizer. So the "disabled" quantizers
silently annealed back to alpha=1.0 after just 10 calibration steps, and from
weight quantizer #10 onward the whole activation stack was fake-quantizing with
LSBs that had never been searched.

The consequence was silent and severe: every subsequent LSB was picked by
"min val_loss" against a model that was already destroyed by garbage activation
quantization. In the MobileNetV2 ImageNet search this dragged val_acc from
72.4% down to 5.3% across the weight role, and the accuracy snapped back to
57.3% the moment the next role called _set_search_states again (which reset
alpha to 0) -- the tell-tale that exposed it.
"""

import torch
import torch.nn as nn
from brevitas.nn import QuantConv2d, QuantIdentity

from quantizers import (
    FixedPointPerTensorActivationQuant,
    FixedPointPerTensorWeightQuant,
)
from quantizers.manager import QuantizerManager
from examples.find_perfect_lsbs_imagenet_ptq import (
    _assign_descriptive_ids,
    _set_search_states,
)


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = QuantConv2d(3, 4, 3, padding=1,
                              weight_quant=FixedPointPerTensorWeightQuant, bias=False)
        self.a1 = QuantIdentity(act_quant=FixedPointPerTensorActivationQuant)
        self.c2 = QuantConv2d(4, 4, 3, padding=1,
                              weight_quant=FixedPointPerTensorWeightQuant, bias=False)
        self.a2 = QuantIdentity(act_quant=FixedPointPerTensorActivationQuant)

    def forward(self, x):
        return self.a2(self.c2(self.a1(self.c1(x))))


def _calibrated_net():
    """A net whose quantizers have all been calibrated, with ids assigned."""
    torch.manual_seed(0)
    model = _Net()
    x = torch.randn(2, 3, 8, 8)
    model.train()
    for _ in range(15):          # let every quantizer calibrate + finish annealing
        model(x)
    model.eval()
    model(x)                     # establishes inference_sequence_id / execution order
    _assign_descriptive_ids(model)
    return model, x


def _roles(mgr, role):
    return [q for q in mgr.quantizers.values() if q.quantizer_role == role]


def test_disabled_quantizers_do_not_anneal_back_on():
    """The core bug: 10 calibration forwards must not revive disabled quantizers."""
    model, x = _calibrated_net()
    mgr = QuantizerManager()

    _set_search_states(mgr, target_role="weight", active_roles=set())
    acts = _roles(mgr, "activation")
    assert acts, "test model must have activation quantizers"
    assert all(q.annealing_alpha.item() == 0.0 for q in acts)

    # Exactly what search_role_lsbs does: one training forward per quantizer.
    model.train()
    for _ in range(12):
        model(x)

    for q in acts:
        assert q.annealing_alpha.item() == 0.0, (
            f"{q.display_name} annealed back to alpha={q.annealing_alpha.item()} "
            "during the weight search -- it is fake-quantizing with an unsearched LSB"
        )


def test_unsearched_target_quantizers_stay_disabled():
    """Target-role quantizers awaiting their turn must not self-activate either.

    During the activation search, activation quantizer #1 is calibrated while
    #2..#N are still unsearched. They must stay passthrough, or #1's LSB gets
    chosen against garbage downstream.
    """
    model, x = _calibrated_net()
    mgr = QuantizerManager()

    _set_search_states(mgr, target_role="activation", active_roles={"weight"})
    targets = _roles(mgr, "activation")

    model.train()
    for _ in range(12):
        model(x)

    for q in targets:
        assert q.annealing_alpha.item() == 0.0, (
            f"unsearched target {q.display_name} self-activated to "
            f"alpha={q.annealing_alpha.item()} before its own search ran"
        )


def test_active_roles_stay_fully_quantized():
    """Roles marked active must sit at alpha=1 and never drift back down."""
    model, x = _calibrated_net()
    mgr = QuantizerManager()

    _set_search_states(mgr, target_role="activation", active_roles={"weight"})
    weights = _roles(mgr, "weight")

    model.train()
    for _ in range(12):
        model(x)

    for q in weights:
        assert q.annealing_alpha.item() == 1.0, (
            f"active-role {q.display_name} drifted to alpha={q.annealing_alpha.item()}"
        )


def test_disabled_model_output_is_stable_across_calibration_steps():
    """End-to-end: with only weights active, eval output must not change as the
    search grinds through calibration forwards.

    This is the property the search actually depends on -- if the model drifts
    under it, every val_loss it compares is measured against a different model.
    """
    model, x = _calibrated_net()
    mgr = QuantizerManager()
    _set_search_states(mgr, target_role="weight", active_roles=set())

    # Make the weight quantizers active, as they become once searched.
    for q in _roles(mgr, "weight"):
        q.annealing_alpha.data.fill_(1.0)
        q.annealing_alpha_step = 0.0

    model.eval()
    before = model(x).detach().clone()

    model.train()
    for _ in range(12):
        model(x)

    model.eval()
    after = model(x).detach()

    assert torch.allclose(before, after, atol=1e-6), (
        "model output drifted during calibration forwards -- the LSB sweep is "
        "comparing val_loss across a moving model"
    )
