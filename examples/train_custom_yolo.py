"""
train_custom_yolo.py
--------------------
Train our custom YOLOv8nPANOnly reimplementation from scratch on COCO using the
Ultralytics training_harness infrastructure (optimizer, scheduler, augmentation,
loss, logging — all unchanged).

The key mechanism: DetectionTrainer.setup_model() checks whether
self.model is already an nn.Module before calling get_model(). If it is,
it skips get_model() entirely. We subclass DetectionTrainer and override
get_model() to inject our custom model instead of building from YAML,
which is cleaner than relying on that implicit skip behaviour.

Run
---
    # Full COCO training_harness (300 epochs, Ultralytics defaults):
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
from ultralytics.utils import RANK

from training_harness.custom_trainer import CustomYOLOv8nTrainer


# Example usage python -m examples.train_custom_yolo --data /home/th/tmp/quanttests/cached_datasets/coco_yolo_fmt/coco.yaml --device cuda --batch 128 --epochs 1

MAX_BATCHES = 5
counter = {"n": 0}

def stop_early(trainer):
    counter["n"] += 1
    if counter["n"] >= MAX_BATCHES:
        trainer.stop = True


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
            amp=False,      # Prevents CPU master weight drift & dtype mismatch with Brevitas
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
            pretrained=False,  # scratch training_harness — no weight loading
            verbose=True,
        )
    )

    # trainer.add_callback("on_train_batch_end", stop_early)
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
