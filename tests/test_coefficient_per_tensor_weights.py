import os
import unittest

import torch
from brevitas.nn import QuantLinear

from quantizers.coefficient_per_tensor_weights import (
    CoefficientPerTensorWeightQuant,
    CoefficientPerTensorWeightQuantizer,
)


class TestCoefficientQuantizer(unittest.TestCase):
    def setUp(self):
        self.coeff_file = "tests/dummy_coeffs.txt"
        # Verify the dummy file exists before proceeding
        if not os.path.exists(self.coeff_file):
            self.fail(f"Required coefficient file {self.coeff_file} not found. Please ensure it is created.")

    def test_init_and_loading(self):
        """Test that coefficients are correctly loaded from the text file."""
        quantizer = CoefficientPerTensorWeightQuantizer(self.coeff_file)
        self.assertEqual(len(quantizer.coefficient_sets), 2)
        # Set 0: [-1.0, 0.0, 1.0]
        self.assertTrue(torch.allclose(quantizer.coefficient_sets[0], torch.tensor([-1.0, 0.0, 1.0])))
        # Set 1: [-0.5, -0.25, 0.0, 0.25, 0.5]
        self.assertTrue(torch.allclose(quantizer.coefficient_sets[1], torch.tensor([-0.5, -0.25, 0.0, 0.25, 0.5])))

    def test_optimal_search_simple(self):
        """Test search for the best set and scale when weights clearly fit a specific set."""
        quantizer = CoefficientPerTensorWeightQuantizer(self.coeff_file)
        # Weights clearly fitting set 0 with scale 1.0 (bit_shift_scale=0); none clipped
        weights = torch.tensor([-0.9, 0.1, 0.9])
        q, scale, zp, bw = quantizer(weights)
        
        self.assertEqual(quantizer.best_set_idx.item(), 0)
        self.assertEqual(quantizer.best_bit_shift_scale.item(), 0)
        self.assertTrue(torch.allclose(q, torch.tensor([-1.0, 0.0, 1.0])))
        self.assertEqual(scale.item(), 1.0)
        self.assertEqual(bw.item(), 3.0)

    def test_optimal_search_scaling(self):
        """Test that the quantizer finds the optimal power-of-two scale."""
        quantizer = CoefficientPerTensorWeightQuantizer(self.coeff_file)
        # Weights fitting set 0 with scale 2.0 (bit_shift_scale=1) -> [-2.0, 0.0, 2.0]
        weights = torch.tensor([-1.9, 0.1, 1.9])
        q, scale, zp, bw = quantizer(weights)
        
        self.assertEqual(quantizer.best_bit_shift_scale.item(), 1)
        self.assertTrue(torch.allclose(q, torch.tensor([-2.0, 0.0, 2.0])))
        self.assertEqual(scale.item(), 2.0)

    def test_optimal_search_set_selection(self):
        """Test that the quantizer selects the set that minimizes SAD."""
        quantizer = CoefficientPerTensorWeightQuantizer(self.coeff_file)
        # Weights fitting set 1 with scale 1.0 (bit_shift_scale=0) -> [-0.5, -0.25, 0.0, 0.25, 0.5]
        weights = torch.tensor([-0.48, -0.26, 0.01, 0.24, 0.51])
        q, scale, zp, bw = quantizer(weights)
        
        self.assertEqual(quantizer.best_set_idx.item(), 1)
        self.assertEqual(bw.item(), 5.0)

    def test_serialization(self):
        """Test that search results are preserved in the state_dict."""
        quantizer = CoefficientPerTensorWeightQuantizer(self.coeff_file)
        weights = torch.tensor([-0.9, 0.1, 0.9])
        quantizer(weights) # Trigger search
        
        # Simulate saving and loading
        state = quantizer.state_dict()
        new_quantizer = CoefficientPerTensorWeightQuantizer(self.coeff_file)
        new_quantizer.load_state_dict(state)
        
        self.assertTrue(new_quantizer.search_done.item())
        self.assertEqual(new_quantizer.best_set_idx.item(), 0)
        self.assertEqual(new_quantizer.best_bit_shift_scale.item(), 0)

    def test_brevitas_integration(self):
        """Test integration with Brevitas QuantLinear via the Injector."""
        # Create a subclass to specify the filepath for the test
        class TestCoeffQuant(CoefficientPerTensorWeightQuant):
            filepath = "tests/dummy_coeffs.txt"

        layer = QuantLinear(
            in_features=10,
            out_features=5,
            weight_quant=TestCoeffQuant
        )
        
        # Forward pass should trigger the search in the internal quantizer
        _ = layer(torch.randn(1, 10))
        
        # In Brevitas, the weight quantizer is instantiated as a proxy.
        # The actual quantizer module is stored in the tensor_quant attribute.
        self.assertTrue(layer.weight_quant.tensor_quant.search_done.item())

    def test_all_zeros(self):
        """Test behavior with all-zero weights."""
        quantizer = CoefficientPerTensorWeightQuantizer(self.coeff_file)
        weights = torch.zeros(10)
        q, scale, zp, bw = quantizer(weights)
        self.assertTrue(torch.all(q == 0))

    def test_missing_file(self):
        """Test that a missing file raises the appropriate error."""
        with self.assertRaises(FileNotFoundError):
            CoefficientPerTensorWeightQuantizer("non_existent_file_12345.txt")




# ---------------------------------------------------------------------------
# Calibration: covering shift, search window, clipping ceiling
# ---------------------------------------------------------------------------

import pytest

from quantizers.coefficient_per_tensor_weights import SHIFT_SEARCH_RADIUS


def _make(tmp_path, lines, **kwargs):
    path = tmp_path / "coeffs.txt"
    path.write_text("\n".join(lines) + "\n")
    return CoefficientPerTensorWeightQuantizer(str(path), **kwargs)


def _calibrate(q, x):
    q.train()
    with torch.no_grad():
        q(x)
    assert q.search_done_value
    return q._set_idx_value, q._bit_shift_scale_value


class TestCalibrationSearch:
    @pytest.mark.parametrize("exp", [-30, -14, 0, 14, 30])
    def test_finds_shift_far_outside_the_old_fixed_window(self, tmp_path, exp):
        """The old search only tried shifts in [-12, 12]; the window now follows
        the weights' range, so any magnitude is found."""
        q = _make(tmp_path, ["-1 0 1"])
        x = torch.tensor([-0.9, 0.1, 0.9]) * 2.0 ** exp
        _, shift = _calibrate(q, x)
        assert shift == exp
        assert torch.allclose(q(x)[0], torch.tensor([-1.0, 0.0, 1.0]) * 2.0 ** exp)

    def test_covering_shift(self):
        f = CoefficientPerTensorWeightQuantizer._covering_shift
        c = torch.tensor([-1.0, 0.0, 1.0])
        assert f(1.0, c) == 0
        assert f(1.01, c) == 1       # needs 2.0 to cover
        assert f(0.4, c) == -1       # 0.5 covers
        assert f(0.0, c) == 0        # all-zero weights
        assert f(5.0, torch.zeros(3)) == 0  # degenerate set

    def test_asymmetric_set_uses_largest_magnitude(self):
        c = torch.tensor([-0.25, 0.0, 1.0])
        assert CoefficientPerTensorWeightQuantizer._covering_shift(3.9, c) == 2

    def test_search_tries_exactly_2r_plus_1_shifts_per_set(self, tmp_path, monkeypatch):
        import quantizers.coefficient_per_tensor_weights as m
        seen = []
        real = m.apply_non_uniform_quantization

        def spy(w, c, shift):
            seen.append((c.numel(), shift))
            return real(w, c, shift)

        monkeypatch.setattr(m, "apply_non_uniform_quantization", spy)
        q = _make(tmp_path, ["-1 0 1", "-0.5 -0.25 0 0.25 0.5"])
        x = torch.tensor([-0.9, 0.1, 0.9]) * 2.0 ** 3
        q.train()
        with torch.no_grad():
            q._calibrate(x)
        assert SHIFT_SEARCH_RADIUS == 6
        for n_coeffs, center in ((3, 3), (5, 4)):  # set 1's max is 0.5 -> one more shift
            shifts = sorted(s for n, s in seen if n == n_coeffs)
            assert shifts == list(range(center - 6, center + 7))
        assert len(seen) == 26

    def test_each_set_gets_its_own_window(self, tmp_path):
        """A set whose largest coefficient is 8x smaller is centered 3 shifts higher."""
        q = _make(tmp_path, ["-8 0 8", "-1 0 1"])
        x = torch.tensor([-0.9, 0.1, 0.9])
        idx, shift = _calibrate(q, x)
        # set 1 at shift 0 reproduces +-1 exactly-ish; set 0 would need shift -3
        assert (idx, shift) in {(0, -3), (1, 0)}

    def test_clip_ceiling_rejects_the_sad_optimum(self, tmp_path):
        """Clipping the single huge outlier lowers SAD, but it is 1/10 = 10%..."""
        x = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 3.0])
        # ceiling 15% -> 10% clipping is allowed, so the finer grid wins (shift 0)
        q = _make(tmp_path, ["-1 -0.5 0 0.5 1"], max_clip_pct=15.0)
        assert _calibrate(q, x)[1] == 0
        # ceiling 5% -> clipping the outlier is not allowed, shift 2 is the covering one
        q = _make(tmp_path, ["-1 -0.5 0 0.5 1"], max_clip_pct=5.0)
        assert _calibrate(q, x)[1] == 2

    def test_default_ceiling_matches_fixedpoint_weights(self, tmp_path):
        from quantizers.fixedpoint_per_tensor import _MAX_CLIP_PCT_BY_ROLE
        assert _make(tmp_path, ["-1 0 1"]).max_clip_pct == _MAX_CLIP_PCT_BY_ROLE["weight"]

    def test_falls_back_to_least_clipping_when_nothing_qualifies(self, tmp_path):
        """No candidate can satisfy a negative ceiling, so the least-clipping
        one is taken -- the covering shift (0% clipped) -- even though the
        SAD optimum (which clips the outlier) is a finer grid."""
        x = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 3.0])
        q = _make(tmp_path, ["-1 -0.5 0 0.5 1"], max_clip_pct=-1.0)
        assert _calibrate(q, x)[1] == 2

    def test_all_zero_weights_do_not_break_the_search(self, tmp_path):
        q = _make(tmp_path, ["-1 0 1"])
        _calibrate(q, torch.zeros(8))
        assert torch.all(q(torch.zeros(8))[0] == 0)


# ---------------------------------------------------------------------------
# No bit_width field
# ---------------------------------------------------------------------------

class TestNoBitWidthField:
    def test_constructor_rejects_bit_width(self, tmp_path):
        path = tmp_path / "c.txt"
        path.write_text("-1 0 1\n")
        with pytest.raises(TypeError):
            CoefficientPerTensorWeightQuantizer(str(path), bit_width=3)

    def test_returned_width_is_number_of_coefficients(self, tmp_path):
        q = _make(tmp_path, ["-1 0 1", "-0.5 -0.25 0 0.25 0.5"])
        _, _, _, bw = q(torch.tensor([-0.48, -0.26, 0.01, 0.24, 0.51]))
        assert bw.item() == 5.0


# ---------------------------------------------------------------------------
# Clipped STE through the Brevitas injector
# ---------------------------------------------------------------------------

def _single_weight_layer(tmp_path, mode):
    path = tmp_path / "c.txt"
    path.write_text("0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5\n")

    class Q(CoefficientPerTensorWeightQuant):
        filepath = str(path)
        clipped_ste = mode

    layer = QuantLinear(1, 1, bias=False, weight_quant=Q)
    q = layer.weight_quant.tensor_quant
    with torch.no_grad():
        layer.weight.fill_(2.0)
    layer.train()
    layer(torch.ones(1, 1))  # calibrate
    assert q.search_done_value
    return layer, q


class TestClippedSTEViaInjector:
    @pytest.mark.parametrize("mode", ["not_clip", "clip", "single_direction_clip"])
    def test_mode_reaches_quantizer(self, tmp_path, mode):
        _, q = _single_weight_layer(tmp_path, mode)
        assert q.ste_mode == mode

    @pytest.mark.parametrize("mode", ["not_clip", "clip", "single_direction_clip"])
    @pytest.mark.parametrize("g", [+1.0, -1.0])
    def test_gradient_of_clipped_weight(self, tmp_path, mode, g):
        """A weight far above the coefficient span: forward saturates at the top coefficient."""
        layer, _ = _single_weight_layer(tmp_path, mode)
        with torch.no_grad():
            layer.weight.fill_(100.0)
        (layer(torch.ones(1, 1)) * g).sum().backward()
        # loss = g * q(w); SGD steps w -= lr * grad, so grad < 0 pushes the already
        # saturated weight further out.
        w_grad = layer.weight.grad.item()
        if mode == "not_clip":
            assert w_grad == pytest.approx(g)
        elif mode == "clip":
            assert w_grad == 0.0
        else:
            # Only the inward direction (grad > 0) passes.
            assert w_grad == (g if g > 0 else 0.0)

    def test_in_range_weight_keeps_gradient_in_every_mode(self, tmp_path):
        for mode in ("not_clip", "clip", "single_direction_clip"):
            layer, _ = _single_weight_layer(tmp_path, mode)
            layer.weight.grad = None
            layer(torch.ones(1, 1)).sum().backward()
            assert layer.weight.grad.item() == pytest.approx(1.0), mode


# ---------------------------------------------------------------------------
# ONNX node attributes
# ---------------------------------------------------------------------------

class TestOnnxNodeAttributes:
    def _export(self, tmp_path, lines, weight):
        import onnx

        from utils.onnx_export import export_onnx_with_io

        path = tmp_path / "c.txt"
        path.write_text("\n".join(lines) + "\n")

        class Q(CoefficientPerTensorWeightQuant):
            filepath = str(path)

        layer = QuantLinear(3, 2, bias=False, weight_quant=Q)
        with torch.no_grad():
            layer.weight.copy_(weight)
        x = torch.randn(1, 3)
        layer.train(); layer(x); layer.eval()
        out = tmp_path / "m.onnx"
        export_onnx_with_io(layer, x, str(out), dynamo=False, canonical_checks="off")
        model = onnx.load(str(out))
        nodes = [n for n in model.graph.node if n.op_type == "CoefficientQuant"]
        assert len(nodes) == 1
        return {a.name: a for a in nodes[0].attribute}

    def test_count_and_range_attributes(self, tmp_path):
        w = torch.tensor([[-3.9, 0.1, 3.9], [1.0, -1.0, 2.0]])
        attrs = self._export(tmp_path, ["-1 -0.5 0 0.5 1"], w)
        shift = attrs["bit_shift_scale"].i
        assert shift == 2  # covers 3.9
        assert attrs["num_coefficients"].i == 5
        assert attrs["min_value"].f == pytest.approx(-1.0 * 2.0 ** shift)
        assert attrs["max_value"].f == pytest.approx(1.0 * 2.0 ** shift)

    def test_range_of_asymmetric_set(self, tmp_path):
        w = torch.tensor([[0.0, 0.1, 0.9], [0.5, 0.25, 0.75]])
        attrs = self._export(tmp_path, ["0 0.25 0.5 0.75 1"], w)
        shift = attrs["bit_shift_scale"].i
        assert attrs["num_coefficients"].i == 5
        assert attrs["min_value"].f == pytest.approx(0.0)
        assert attrs["max_value"].f == pytest.approx(2.0 ** shift)

    def test_no_bit_width_attribute(self, tmp_path):
        attrs = self._export(tmp_path, ["-1 0 1"], torch.tensor([[-0.9, 0.1, 0.9], [0.5, 0.2, 0.3]]))
        assert not any("bit_width" in name for name in attrs)


class TestCoefficientsAreNodeAttribute:
    """The coefficient set is part of the CoefficientQuant node (an attribute,
    like chosen_indices), not a graph input or a stray Constant node."""

    SETS = ["-1 -0.5 0 0.5 1", "-2 -1 0 1 2"]

    def _export(self, tmp_path):
        import onnx

        from utils.onnx_export import export_onnx_with_io

        paths = []
        for i, line in enumerate(self.SETS):
            path = tmp_path / f"c{i}.txt"
            path.write_text(line + "\n")
            paths.append(str(path))

        def quant(path):
            class Q(CoefficientPerTensorWeightQuant):
                filepath = path
            return Q

        model = torch.nn.Sequential(
            QuantLinear(3, 4, bias=False, weight_quant=quant(paths[0])),
            QuantLinear(4, 2, bias=False, weight_quant=quant(paths[1])),
        )
        x = torch.randn(1, 3)
        model.train(); model(x); model.eval()
        out = tmp_path / "m.onnx"
        export_onnx_with_io(model, x, str(out), dynamo=False, canonical_checks="off",
                            input_names=["x"], output_names=["y"])
        return onnx.load(str(out)), model, x

    def test_no_constant_nodes_and_single_input(self, tmp_path):
        model, _, _ = self._export(tmp_path)
        assert not [n for n in model.graph.node if n.op_type == "Constant"]
        quants = [n for n in model.graph.node if n.op_type == "CoefficientQuant"]
        assert len(quants) == 2
        for n in quants:
            assert len(n.input) == 1  # just the weight
        # the sets (5 values each) are not graph initializers; what is left is
        # the two weights and the zero biases the Dense lowering adds
        assert not [i.name for i in model.graph.initializer if list(i.dims) == [5]]

    def test_attribute_holds_the_calibrated_set(self, tmp_path):
        import numpy as np
        from onnx import numpy_helper
        model, torch_model, _ = self._export(tmp_path)
        quants = [n for n in model.graph.node if n.op_type == "CoefficientQuant"]
        for n, line in zip(quants, self.SETS):
            attrs = {a.name: a for a in n.attribute}
            coeffs = numpy_helper.to_array(attrs["coefficients"].t)
            assert np.allclose(coeffs, [float(v) for v in line.split()])
            assert attrs["num_coefficients"].i == len(coeffs)
            assert attrs["min_value"].f == pytest.approx(coeffs.min() * attrs["scale"].f)
            assert attrs["max_value"].f == pytest.approx(coeffs.max() * attrs["scale"].f)

    def test_function_takes_only_x_and_references_the_attribute(self, tmp_path):
        model, _, _ = self._export(tmp_path)
        f = next(f for f in model.functions if f.name == "CoefficientQuant")
        assert list(f.input) == ["x"]
        assert set(f.attribute) == {"scale", "coefficients"}

    def test_exported_model_still_matches_eager(self, tmp_path):
        import numpy as np

        from tests.onnx_run import run_onnx
        model, torch_model, x = self._export(tmp_path)
        with torch.no_grad():
            eager = torch_model(x).detach().numpy()
        out = run_onnx(model, {"x": x.numpy().astype(np.float64)})[0]
        assert np.allclose(out, eager, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
