"""Old `utils.*` import paths for the modules moved into `quantizers/` keep working,
and `iter_quantizers` finds exactly the BaseQuantizer submodules."""
import importlib
import sys

import pytest
import torch.nn as nn

from quantizers.activations import QuantReLU
from quantizers.base_quantizer import BaseQuantizer, iter_quantizers


@pytest.mark.parametrize("old,new,attr", [
    ("utils.quantizer_diagnostics", "quantizers.diagnostics", "compute_metrics"),
    ("utils.activation_lsb_sweep", "quantizers.activation_lsb_sweep", "build_lsb_table"),
])
def test_old_utils_paths_reexport_with_warning(old, new, attr):
    sys.modules.pop(old, None)
    with pytest.warns(DeprecationWarning, match="moved to"):
        old_mod = importlib.import_module(old)
    assert getattr(old_mod, attr) is getattr(importlib.import_module(new), attr)


class TestIterQuantizers:
    def test_yields_named_base_quantizers_only(self):
        model = nn.Sequential(nn.Linear(4, 4), QuantReLU(), nn.ReLU())
        found = list(iter_quantizers(model))
        assert found, "QuantReLU should contain quantizers"
        assert all(isinstance(q, BaseQuantizer) for _, q in found)
        # names are the dotted paths named_modules() reports
        named = dict(model.named_modules())
        for name, q in found:
            assert named[name] is q

    def test_plain_model_has_none(self):
        assert list(iter_quantizers(nn.Sequential(nn.Linear(2, 2), nn.ReLU()))) == []

    def test_order_matches_named_modules(self):
        model = nn.Sequential(QuantReLU(), nn.Linear(4, 4), QuantReLU())
        found = [n for n, _ in iter_quantizers(model)]
        expected = [n for n, m in model.named_modules() if isinstance(m, BaseQuantizer)]
        assert found == expected
