# Training Harness

The `training_harness` provides an end-to-end QAT pipeline.

## Pipeline Phases
1. **Float Warmup**: Train in full precision to learn robust weights.
2. **Calibration**: Run a PTQ-style pass to initialize quantization ranges.
3. **QAT**: Enable fake-quantization and fine-tune with learned scales.
4. **Checkpointing & Logging**: Automatic top-K checkpointing, CSV/TensorBoard/W&B logging, and training curve plotting.

## Configuration
Use `TrainerConfig` to define experiment parameters:
```python
config = TrainerConfig(
    experiment_name="my_qat_run",
    epochs=50,
    learning_rate=1e-3,
    quant_schedule=QuantScheduleConfig(float_warmup_epochs=5, calibration_batches=100),
)
```

## Key Classes
- `Trainer`: Orchestrates the training loop, validation, checkpointing, and logging.
- `CheckpointManager`: Handles saving/loading model state and ONNX export.
- `ExperimentLogger`: Routes metrics to CSV, TensorBoard, and W&B.
- `QATWarmupScheduler`: Manages phase transitions (float → calibration → QAT).

For usage examples, see the [README](../README.md) or `examples/` directory.
