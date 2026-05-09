
from quantizers import FixedPointPerTensorQuantizer
import torch

grid = [k * 0.5 + 0.1 for k in range(-2,11)]  # 0.0 .. 3.5
# # grid_floaty = [k * 0.5 + 0.1 for k in range(8)]  # 0.0 .. 3.5
# weights = torch.tensor(grid, dtype=torch.float32)
# q = quantize_fixed_point(weights, lsb=2, bit_width=3, signed=True, rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN)
#
# print(grid)
# print(q)


quantizer = FixedPointPerTensorQuantizer(bit_width=3)
weights = torch.tensor([0.1, -0.5, 1.0, 2.0, 19.0])

for i in range(15):
    result, _, _, _ = quantizer(weights)

    print(result)

    result = result.numpy()
    weights_np = weights.numpy()

    for i in range(len(weights_np)):

        print(str(result[i]).ljust(7), str(abs(result[i]-weights_np[i])).ljust(7), weights_np[i])