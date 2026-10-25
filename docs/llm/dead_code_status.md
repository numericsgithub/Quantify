# Dead-code audit status

Replaces the old root-level `POSSIBLY_DEAD_QUANTIZER_CODE.md`. Each former candidate was re-checked by grepping src, tests, scripts and docs.

## Removed
- `BaseQuantizer.backward` (`quantizers/base_quantizer.py`): a leftover `nn.Module` method with a debug `print`. Nothing called it.
- `QuantizerManager.stop_quantization_for_n_inferences` (`quantizers/manager.py`): no callers anywhere.
- Debug `print("grad_quantizedgrad_quantized", ...)` in `FixedPointQuantFn.backward`. The straight-through backward itself is live and stays. The print fired on every backward pass.

## Already gone before this audit
- `print("find_optimal_lsb was called!")`.
- The duplicate `return` in `CoefficientPerTensorWeightQuantizer._quantize`.

## Kept on purpose
- `quantize_fixed_point`: called by `find_optimal_lsb`, `FixedPointPerTensorQuantizer._quantize` and many tests.
- `FixedPointQuantFnTestingThings`: still applied inside `quantize_fixed_point`. Replacing it with a direct `quantize_fixed_point_with_integers` call is a hot-path change (calibration search loop), so it is a separate, test-gated refactor.
- `detect_signed`, `QuantizerManager.enable_quantization`: no src callers, but public helpers exercised by many tests.
- `RoundingMode.ROUND` vs `ROUND_TO_NEAREST_EVEN`: both used by tests and by `utils/onnx_self_contained.py`. Merging them would change numerics.
- Unused but public: `EpochMetrics.update_scales` and `MetricsTracker.current_avg`/`latest`/`all_epoch_dicts` in `training_harness/metrics.py`, `ExperimentLogger.log_image`, `QATTrainer.evaluate`, scheduler `get_lr`. Cheap, plausible user-facing API; remove only with a deliberate API decision.
