"""
The three STE clipping modes on a weight that is pushed past the quantizer's range.

Model: a single QuantLinear(1, 1): y = q(w) * x + b, with a 10-bit signed
fixed-point weight quantizer pinned to lsb = -7, i.e. a representable range
of [-4.0, 3.9921875] in steps of 1/128. The bias is left unquantized.

Training (plain SGD, no momentum, no weight decay, x = 1):
  phase "up":   MSE towards target +20 -> pushes w up, past q_max
  phase "down": MSE towards target -20 -> pushes w back down

Run once per STE mode (the quantizer's `clipped_ste` setting), logging every step:
  "not_clip"               plain STE: the clipped float weight keeps drifting
                           outward, then lags on the way back
  "clip"                   clipped STE: the clipped weight gets stuck for good
  "single_direction_clip"  stops the outward drift, but lets the gradient
                           through as soon as it points back into the range

Usage:
    python examples/ste_clipping_demo.py [--steps-up 30] [--steps-down 60] [--lr 0.02]
                                         [--modes not_clip clip single_direction_clip]
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")  # Brevitas named-tensor / diagnostics noise

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import brevitas.nn as qnn

from quantizers import FixedPointPerTensorWeightQuant
from quantizers.base_quantizer import BaseQuantizer, STE_MODES
from quantizers.manager import QuantizerManager

BIT_WIDTH = 10
LSB = -7  # step 2**-7 = 0.0078125 -> range [-512, 511] * step = [-4.0, 3.9921875]


class Weight10Bit(FixedPointPerTensorWeightQuant):
    bit_width = BIT_WIDTH


def build_model(mode: str, w0: float, b0: float):
    QuantizerManager().reset()
    # Set the mode the way a user would: as an attribute of the Brevitas injector.
    weight_quant = type(f"Weight10Bit_{mode}", (Weight10Bit,), {"clipped_ste": mode})
    model = qnn.QuantLinear(1, 1, bias=True, weight_quant=weight_quant, return_quant_tensor=False)
    with torch.no_grad():
        model.weight.fill_(w0)
        model.bias.fill_(b0)
    (quant,) = [m for m in model.modules() if isinstance(m, BaseQuantizer)]
    assert quant.ste_mode == mode
    # Pin the grid. A single-element tensor has only one unique value, so the
    # quantizer would never set search_done and would re-calibrate on every
    # forward -- its range would just follow the weight and never clip.
    quant.search_result_lsb.fill_(LSB)
    quant.search_result_is_signed.fill_(True)
    quant.signed = True
    quant.set_search_done(True)
    return model, quant


def run(mode: str, steps_up: int, steps_down: int, lr: float, w0: float, b0: float):
    model, quant = build_model(mode, w0, b0)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    x = torch.ones(1, 1)
    step = 2.0 ** LSB
    q_min, q_max = -(2 ** (BIT_WIDTH - 1)) * step, (2 ** (BIT_WIDTH - 1) - 1) * step

    print("=" * 132)
    print(f"clipped_ste={mode!r}   bit_width={BIT_WIDTH}  lsb={LSB}  step={step}  "
          f"range=[{q_min}, {q_max}]   lr={lr}  x=1  w0={w0}  b0={b0}")
    print("dL/dq = gradient arriving at the quantizer output;  w.grad = what the STE passes on to the float weight;")
    print("dir = which way dL/dq pushes w under gradient descent (out/in = away from/towards the range, for a clipped w);")
    print("outside by = distance of the float w beyond the nearest range limit (0 inside the range)")
    print("=" * 132)
    header = (f"{'phase':>5} {'step':>4} | {'w (float)':>10} {'q(w)':>10} {'clipped':>8} {'dir':>4} {'outside by':>10} | "
              f"{'b':>9} {'y':>9} {'target':>6} {'loss':>10} | {'dL/dq':>9} {'w.grad':>9} {'b.grad':>9} | "
              f"{'dw':>9} {'db':>9}")
    print(header)
    print("-" * len(header))

    # Capture q(w) (the quantizer's output) so we can read its gradient too.
    captured = {}

    def hook(mod, inp, out):
        out[0].retain_grad()
        captured["q"] = out[0]

    handle = quant.register_forward_hook(hook)
    global_step = 0
    for phase, target_value, n in (("up", 20.0, steps_up), ("down", -20.0, steps_down)):
        target = torch.full((1, 1), target_value)
        for i in range(n):
            w_before, b_before = model.weight.item(), model.bias.item()
            opt.zero_grad()
            y = model(x)
            loss = ((y - target) ** 2).mean()
            loss.backward()
            qw = captured["q"].item()
            dq = captured["q"].grad.item()
            gw, gb = model.weight.grad.item(), model.bias.grad.item()
            opt.step()
            w_after, b_after = model.weight.item(), model.bias.item()
            clipped = not (q_min <= w_before <= q_max)
            # Descent moves w by -lr*dq: outward if that increases |distance to the range|.
            if not clipped:
                direction = "-"
            elif (w_before > q_max) == (dq < 0):
                direction = "out"
            else:
                direction = "in"
            print(f"{phase:>5} {global_step:>4} | {w_before:>10.5f} {qw:>10.5f} {str(clipped):>8} {direction:>4} "
                  f"{max(w_before - q_max, q_min - w_before, 0.0):>10.5f} | {b_before:>9.4f} {y.item():>9.4f} {target_value:>6.1f} "
                  f"{loss.item():>10.4f} | {dq:>+9.4f} {gw:>+9.4f} {gb:>+9.4f} | "
                  f"{w_after - w_before:>+9.5f} {b_after - b_before:>+9.4f}")
            global_step += 1
        print("-" * len(header))
    handle.remove()

    w, b = model.weight.item(), model.bias.item()
    print(f"FINAL clipped_ste={mode!r}: w (float) = {w:.5f}, q(w) = {quant(model.weight)[0].item():.5f}, "
          f"b = {b:.4f}, annealing_alpha = {quant.annealing_alpha_value}, lsb = {int(quant.search_result_lsb)}")
    print()
    return w, b


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps-up", type=int, default=30)
    parser.add_argument("--steps-down", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--w0", type=float, default=3.0)
    parser.add_argument("--b0", type=float, default=0.0)
    parser.add_argument("--modes", nargs="+", choices=STE_MODES, default=None,
                        help="STE modes to run (default: all three)")
    args = parser.parse_args()
    for mode in args.modes or STE_MODES:
        run(mode, args.steps_up, args.steps_down, args.lr, args.w0, args.b0)


if __name__ == "__main__":
    main()
