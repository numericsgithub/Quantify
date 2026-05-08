

from quantizers.fixedpoint_per_tensor_weights import *

grid = [k * 0.5 + 0.1 for k in range(-2,11)]  # 0.0 .. 3.5
# # grid_floaty = [k * 0.5 + 0.1 for k in range(8)]  # 0.0 .. 3.5
# weights = torch.tensor(grid, dtype=torch.float32)
# q = quantize_fixed_point(weights, lsb=2, bit_width=3, signed=True, rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN)
#
# print(grid)
# print(q)


quantizer = FixedPointPerTensorWeightQuantizer(bit_width=3)
weights = torch.tensor([0.1, -0.5, 1.0, 2.0, 19.0])
result, _, _, _ = quantizer(weights)

print(result)

result = result.numpy()
weights = weights.numpy()

for i in range(len(weights)):

    print(str(result[i]).ljust(7), str(abs(result[i]-weights[i])).ljust(7), weights[i])