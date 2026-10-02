# STE clipping modes (`clipped_ste`)

Every `BaseQuantizer` takes a `clipped_ste` setting that controls the backward
pass for inputs the forward clamp saturated (values outside the quantizer's
representable range). Forward values are identical in all modes, and the
ONNX-export path ignores the setting.

| `clipped_ste`                      | gradient outside the range                         | effect on a clipped weight |
|------------------------------------|----------------------------------------------------|----------------------------|
| `"not_clip"` / `False` (default)   | passed (slope 1 everywhere, plain STE)             | drifts arbitrarily far out without changing the output; lags on the way back |
| `"clip"` / `True`                  | zeroed (`ClippedSTEFn`)                            | gets **stuck** for good: no gradient ever again unless the range changes |
| `"single_direction_clip"`          | zeroed if it pushes further out, passed if it points back in (`SingleDirectionClippedSTEFn`) | no drift, and recovers the moment the loss wants it back inside |

"Points back in" is judged on the sign of the gradient w.r.t. the quantized
output, assuming gradient descent (`w -= lr * grad`): above the range a
positive gradient passes, below it a negative one. With momentum/Adam an
already-built-up velocity can still carry a value outward for a few steps.

## Setting it

```python
# On a Brevitas injector (the usual way)
class MyWeightQuant(FixedPointPerTensorWeightQuant):
    clipped_ste = "single_direction_clip"

# Directly / after construction
q = FixedPointPerTensorQuantizer(clipped_ste="single_direction_clip")
q.clipped_ste = "clip"      # q.ste_mode == "clip"; q.clipped_ste is True
```

`q.ste_mode` holds the normalized mode string; `q.clipped_ste` reads back as a
bool (True for either clipping mode) for backward compatibility. Invalid
values raise `ValueError`.

## Why "clip" is a trap for weights here

Clipped STE is the textbook choice because in the usual setups nothing gets
stuck: activations are recomputed every step, and PyTorch/TF weight ranges
come from a min/max observer of the weights themselves (or are learned, as in
LSQ/PACT). Quantify's ranges are fixed by calibration and weights may be
deliberately clipped (up to 15% per layer), so a weight that crosses the limit
under `"clip"` never comes back. `examples/ste_clipping_demo.py` shows all
three modes step by step on a single 10-bit weight.

## Adding a new quantizer

Override `_in_range_mask(x, params)` (for `"clip"`) and `_clip_side(x, params)`
(for `"single_direction_clip"`; return -1/0/+1 per element, boundaries count
as in range). Returning `None` from either degrades that mode to plain STE.
