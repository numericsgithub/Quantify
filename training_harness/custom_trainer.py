"""
custom_trainer.py — Custom Ultralytics Trainer for YOLOv8nPANOnly with Brevitas QAT.
"""

import copy

import numpy as np
import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import RANK

from models.yolov8PanOnly import YOLOv8nPANOnly
import quantizers as q
from quantizers.manager import QuantizerManager


class CustomYOLOv8nTrainer(DetectionTrainer):
    """
    DetectionTrainer subclass that builds our clean YOLOv8nPANOnly nn.Module
    instead of parsing yolov8n.yaml.

    Only get_model() is overridden. Everything else — loss, optimizer,
    scheduler, augmentation, logging, checkpointing — is Ultralytics stock.

    Compatibility notes
    -------------------
    set_model_attributes() sets model.nc, model.names, model.args.
    Our YOLOv8nPANOnly doesn't define those, but Python allows setting arbitrary
    attributes on nn.Module instances, so this works without any change.

    The DFL freeze ("always_freeze_names = ['.dfl']") matches our
    detect.dfl submodule name, so DFL weights are correctly frozen
    during training_harness (they are fixed by construction anyway).

    The loss function (v8DetectionLoss) reads model.model[-1] to get the
    Detect head's stride, nc, and reg_max. We attach a .model attribute
    that exposes this so the loss can find it.
    """
    def __init__(self, *args, checkpoint: str = None, **kwargs):
        # Store checkpoint path before super().__init__ validates overrides
        self._checkpoint = checkpoint
        super().__init__(*args, **kwargs)
        # Disable EMA to prevent it from averaging/corrupting quantizer scales.
        # `ema` is not a valid override argument, so we nullify it after init.
        self.ema = None

    def save_model(self):
        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_args": vars(self.args),
        }
        torch.save(ckpt, self.last)

        export_model = copy.deepcopy(self.model).float().cpu().eval()

        dummy = torch.zeros(1, 3, 640, 640)
        torch.onnx.export(
            export_model, dummy, str(self.last) + ".onnx",
            dynamo=False,
            opset_version=13,
            custom_opsets={"Quantify": 1},
            do_constant_folding=False,  # keep the custom node visible
            input_names=["input"],
            output_names=["output"],
        )
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
            torch.onnx.export(
                export_model, dummy, str(self.best) + ".onnx",
                dynamo=False,
                opset_version=13,
                custom_opsets={"Quantify": 1},
                do_constant_folding=False,  # keep the custom node visible
                input_names=["input"],
                output_names=["output"],
            )
        del export_model
        return True

    def final_eval(self):
        export_model = copy.deepcopy(self.model).float().cpu().eval()
        self.metrics = self.validator(model=export_model)
        self.metrics.pop("fitness", None)
        self.run_callbacks("on_fit_epoch_end")

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build our custom YOLOv8nPANOnly, optionally loading a saved state dict."""
        nc = self.data["nc"]
        
        # Explicitly instantiate a local QuantizerManager to avoid global state leakage.
        # This manager coordinates inference gating, annealing, and recalibration
        # specifically for this training run.
        quantizer_mgr = QuantizerManager()
        quantizer_mgr.quantization_start_gap = 100
        quantizer_mgr.set_annealing_for_n_inferences(20)
        quantizer_mgr.stop_quantization_for_n_inferences(917*25)
        
        # Note: Brevitas DI instantiates quantizer classes internally.
        # If your quantizer classes inherit from BaseQuantizer, you can pass the manager
        # via a subclass or wrapper. For now, the manager is configured and ready
        # to be attached to quantizer proxies post-instantiation if needed.
        model = YOLOv8nPANOnly(nc=nc, weight_quant=q.FixedPointPerTensorWeightQuant, act_quant=q.FixedPointPerTensorActivationQuant)
        model = model.to(self.device)

        # Load a previously saved state dict if provided via --checkpoint.
        # self.args.checkpoint is set from overrides in main().
        checkpoint = self._checkpoint
        torch.serialization.add_safe_globals([
            np.core.multiarray.scalar,
            np.dtype,
            np.dtypes.Float64DType,  # may also appear
            np.int64,
            np.float64,
            np.ndarray,
        ])
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
            # Support both raw state dicts and Ultralytics-style ckpt dicts
            if isinstance(ckpt, dict) and "model" in ckpt:
                state_dict = ckpt["model"]
            elif isinstance(ckpt, dict) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
                state_dict = ckpt  # already a state dict
            else:
                state_dict = ckpt.state_dict()
            missing, unexpected = model.load_state_dict(state_dict, strict=True)
            if verbose and RANK in {-1, 0}:
                print(f"  Loaded checkpoint: {checkpoint}")
                if missing:
                    print(f"  ⚠️  Missing keys: {len(missing)}")
                if unexpected:
                    print(f"  ⚠️  Unexpected keys: {len(unexpected)}")

        model.detect.nc = nc
        model.detect.stride = model.stride
        model.end2end = False

        if verbose and RANK in {-1, 0}:
            n_params = sum(p.numel() for p in model.parameters())
            mode = "fine-tuning" if checkpoint else "scratch"
            print(f"Custom YOLOv8nPANOnly ({mode}): nc={nc}, strides={model.stride.tolist()}, {n_params:,} parameters")

        return model
