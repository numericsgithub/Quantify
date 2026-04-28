"""
train_custom_yolo.py
--------------------
Train our custom YOLOv8nPANOnly reimplementation from scratch on COCO using the
Ultralytics training infrastructure (optimizer, scheduler, augmentation,
loss, logging — all unchanged).

The key mechanism: DetectionTrainer.setup_model() checks whether
self.model is already an nn.Module before calling get_model(). If it is,
it skips get_model() entirely. We subclass DetectionTrainer and override
get_model() to inject our custom model instead of building from YAML,
which is cleaner than relying on that implicit skip behaviour.

Run
---
    # Full COCO training (300 epochs, Ultralytics defaults):
    python train_custom_yolo.py \\
        --data /path/to/coco_yolo_fmt/coco_train.yaml \\
        --workdir ./runs/custom_yolo_scratch

    # Quick smoke-test (coco8, 3 epochs):
    python train_custom_yolo.py \\
        --data coco8.yaml --epochs 3 --smoke-test

    # With explicit device and batch size:
    python train_custom_yolo.py \\
        --data coco.yaml --device cuda --batch 64 --epochs 300
"""

import argparse
from pathlib import Path

import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import RANK

# from models.yolov8n_model import YOLOv8nPANOnly
from models.yolov8PanOnly import YOLOv8nPANOnly
import dill
import quantizers as q
from brevitas.export import export_onnx_qcdq
from quantizers import FixedPointPerTensorWeightQuantizer

from contextlib import contextmanager

# Example usage python -m examples.train_custom_yolo --data /home/th/tmp/quanttests/cached_datasets/coco_yolo_fmt/coco.yaml --device cuda --batch 128 --epochs 1

@contextmanager
def fixed_point_export_mode(model):
    targets = [m for m in model.modules()
               if isinstance(m, FixedPointPerTensorWeightQuantizer)]
    for m in targets:
        m.export_mode = True
    try:
        yield
    finally:
        pass
        for m in targets:
            m.export_mode = False

MAX_BATCHES = 5
counter = {"n": 0}

def stop_early(trainer):
    counter["n"] += 1
    if counter["n"] >= MAX_BATCHES:
        trainer.stop = True

# ---------------------------------------------------------------------------
# Custom trainer
# ---------------------------------------------------------------------------

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
    during training (they are fixed by construction anyway).

    The loss function (v8DetectionLoss) reads model.model[-1] to get the
    Detect head's stride, nc, and reg_max. We attach a .model attribute
    that exposes this so the loss can find it.
    """
    def __init__(self, *args, checkpoint: str = None, **kwargs):
        # Store checkpoint path before super().__init__ validates overrides
        self._checkpoint = checkpoint
        super().__init__(*args, **kwargs)

    def save_model(self):
        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_args": vars(self.args),
        }
        torch.save(ckpt, self.last)

        export_model = self.model.float().cpu().eval()
        dummy = torch.zeros(1, 3, 640, 640)
        torch.onnx.export(
            export_model, dummy, str(self.last) + ".onnx",
            dynamo=False,
            opset_version=13,
            custom_opsets={"mydomain": 1},
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
                custom_opsets={"mydomain": 1},
                do_constant_folding=False,  # keep the custom node visible
                input_names=["input"],
                output_names=["output"],
            )
        return True

    def final_eval(self):
        pass


    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build our custom YOLOv8nPANOnly, optionally loading a saved state dict."""
        nc = self.data["nc"]
        model = YOLOv8nPANOnly(nc=nc, weight_quant=q.FixedPointPerTensorWeightQuant, act_quant=None)# q.FixedPointPerTensorActivationQuant

        # Load a previously saved state dict if provided via --checkpoint.
        # self.args.checkpoint is set from overrides in main().
        checkpoint = self._checkpoint
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
            # Support both raw state dicts and Ultralytics-style ckpt dicts
            if isinstance(ckpt, dict) and "model" in ckpt:
                state_dict = ckpt["model"].state_dict()
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
        model.model = [model.detect]

        if verbose and RANK in {-1, 0}:
            n_params = sum(p.numel() for p in model.parameters())
            mode = "fine-tuning" if checkpoint else "scratch"
            print(f"Custom YOLOv8nPANOnly ({mode}): nc={nc}, strides={model.stride.tolist()}, {n_params:,} parameters")

        return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train custom YOLOv8nPANOnly from scratch on COCO")
    p.add_argument("--data", required=True, help="Dataset YAML path")
    p.add_argument("--workdir", default="./runs/custom_yolo_scratch",
                   help="Output directory")
    p.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    p.add_argument("--batch", type=int, default=256, help="Batch size")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size")
    p.add_argument("--device", default="cuda", help="Device (cuda / cpu)")
    p.add_argument("--workers", type=int, default=16, help="Dataloader workers")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a previously saved state dict (.pt) to resume from")
    p.add_argument("--smoke-test", action="store_true",
                   help="Override to coco8.yaml + 3 epochs for a quick sanity check")
    return p.parse_args()


def main():
    args = parse_args()

    if args.smoke_test:
        args.data = "coco8.yaml"
        args.epochs = 3
        args.batch = 4
        print("Smoke-test mode: coco8.yaml, 3 epochs, batch=4")

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  Custom YOLOv8nPANOnly — {'Fine-tuning' if args.checkpoint else 'Training from Scratch'}")
    print("=" * 60)
    print(f"  Data    : {args.data}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  Imgsz   : {args.imgsz}")
    print(f"  Device  : {args.device}")
    print(f"  Workdir : {workdir}")
    if args.checkpoint:
        print(f"  Checkpoint : {args.checkpoint}")

    trainer = CustomYOLOv8nTrainer(
        checkpoint=args.checkpoint,
        overrides=dict(
            model="yolov8n.yaml",  # placeholder — get_model() ignores this
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            project=str(workdir),
            name="train",
            # Ultralytics defaults for all other hyperparameters:
            # lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
            # warmup_epochs=3, warmup_momentum=0.8,
            # box=7.5, cls=0.5, dfl=1.5,
            # mosaic=1.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, ...
            # (see ultralytics/cfg/default.yaml for the full list)
            pretrained=False,  # scratch training — no weight loading
            verbose=True,
        )
    )

    trainer.add_callback("on_train_batch_end", stop_early)
    trainer.train()

    # trainer.save_dir is the actual run directory (e.g. .../train-10/weights/)
    # — always correct regardless of the auto-incremented suffix.
    best_ckpt = Path(trainer.save_dir) / "weights" / "best.pt"
    out_path = Path(trainer.save_dir) / "weights" / "best_custom_statedict.pt"

    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        model = ckpt.get("model") or ckpt
        if hasattr(model, "state_dict"):
            torch.save(model.state_dict(), out_path)
            print(f"\nClean state dict saved to: {out_path}")
        # try:
        #     import onnx
        #     from onnxsim import simplify
        #
        #     model = onnx.load(str(Path(trainer.save_dir) / "weights" / "best2.onnx"))
        #     model_simp, check = simplify(model)
        #
        #     assert check
        #     onnx.save(model_simp, str(Path(trainer.save_dir) / "weights" / "best6.onnx"))
        # except Exception as e:
        #     print(f"⚠️  ONNX SIMPLIFY export failed: {e}")
    else:
        print(f"\n⚠️  best.pt not found at {best_ckpt}")

    print(f"\nTraining complete. Results in: {trainer.save_dir}")


if __name__ == "__main__":
    main()
