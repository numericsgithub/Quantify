"""The V2-suffixed names were renamed; old import paths keep working with a warning."""
import importlib
import sys

import pytest

import training_harness as th


@pytest.mark.parametrize("old,new", [
    ("QATTrainerV2", "QATTrainer"),
    ("TrainerConfigV2", "TrainerConfig"),
    ("QATScheduleConfigV2", "QATScheduleConfig"),
])
def test_package_level_alias_warns_and_resolves(old, new):
    with pytest.warns(DeprecationWarning, match=old):
        obj = getattr(th, old)
    assert obj is getattr(th, new)


def test_unknown_attribute_still_raises():
    with pytest.raises(AttributeError):
        th.DoesNotExist


@pytest.mark.parametrize("module,old,new", [
    ("training_harness.trainer_v2", "QATTrainerV2", "QATTrainer"),
    ("training_harness.config_v2", "TrainerConfigV2", "TrainerConfig"),
    ("training_harness.config_v2", "QATScheduleConfigV2", "QATScheduleConfig"),
])
def test_old_module_paths_reexport_with_warning(module, old, new):
    sys.modules.pop(module, None)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        mod = importlib.import_module(module)
    assert getattr(mod, old) is getattr(th, new)
