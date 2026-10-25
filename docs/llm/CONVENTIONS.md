# Dependencies
When you need a new package, add it to `pyproject.toml` (`dependencies`, or an optional extra if imported lazily) before importing it.

# Finding new skills
Think about the task you were given. Do you see a pattern in your solution that could describe a skill specifically bound to developing this quantization framework?
One example is the already existing skill, creating custom ONNX nodes when exporting brevitas models.
If you realize this, create a new .md file in the skills folder, like skills/<pattern-name>.md, but keep it empty for now. 
The user will prompt you to fill it with the description later with something like "Fill the description for skills/<filename>.md" so the empty file serves as a context placeholder. 
This is done so you do not get confused and lose track of your current task. 
Only create a new skill file if the pattern is reusable across multiple tasks or tightly coupled to this quantization framework.

# Documenting Pitfalls
When you encounter a common error, unexpected behavior, or debugging tip while working on this project, add it to `pitfalls/brevitas_pitfalls.md`. 
Follow the established format: explain when it happens, what the problem is, and how to prevent or fix it. 
This keeps a centralized knowledge base for the framework and helps avoid repeating mistakes in future tasks.

# Naming
- `*Quant` (e.g. `FixedPointPerTensorWeightQuant`): the Brevitas **injector** that plugs a quantizer into a layer.
- `*Quantizer` (e.g. `FixedPointPerTensorQuantizer`): the `BaseQuantizer` **module** doing the work.
- `*QuantFn` / `*Fn`: a `torch.autograd.Function` (forward/backward, and `symbolic()` when exported).
- `Quant<Layer>` (e.g. `QuantConv2d`, `QuantReLU`): a layer/activation wrapper.
- No version suffixes (`V2`) on new names; rename and leave a deprecated alias instead.
- Iterate a model's quantizers with `quantizers.iter_quantizers(model)` instead of re-writing the `isinstance(..., BaseQuantizer)` walk.
