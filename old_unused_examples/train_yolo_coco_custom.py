"""
train_yolo_coco.py  (updated)
------------------------------
COCO Training for YOLOv8 — now supports swapping in our custom nn.Module.

New flag:
    --use-custom-model   If set, uses our YOLOv8n reimplementation instead
                         of loading the architecture from the .pt file.
                         The pretrained weights are still loaded.

Everything else (dataset, YAML, training_harness loop) is unchanged.
"""

import argparse
import torch
from ultralytics import YOLO
from datasets import load_dataset
from pathlib import Path
from PIL import Image
import os

from utils import add_workspace_args, workspace_from_args
from tqdm import tqdm


def save_hf_dataset_for_yolo(dataset, output_dir: Path, split_name: str):
    img_dir = output_dir / "images" / split_name
    lbl_dir = output_dir / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {split_name} split to YOLO format...")
    for i, item in tqdm(enumerate(dataset)):
        image: Image.Image = item["image"]
        w, h = image.size

        img_path = img_dir / f"{i:06d}.jpg"
        image.save(img_path)

        label_lines = []
        for box, cat in zip(item["objects"]["bbox"], item["objects"]["category"]):
            x1, y1, x2, y2 = box
            nw = (x2 - x1) / w
            nh = (y2 - y1) / h
            cx = x1 / w + nw / 2
            cy = y1 / h + nh / 2
            label_lines.append(f"{cat} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        lbl_path = lbl_dir / f"{i:06d}.txt"
        lbl_path.write_text("\n".join(label_lines))


def create_yolo_yaml(output_dir: Path, yaml_path: Path):
    yaml_content = (
        f"path: {output_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 80\n"
        f"names: { [f'class_{i}' for i in range(80)] }\n"
    )
    yaml_path.write_text(yaml_content)


def parse_args():
    p = argparse.ArgumentParser(description="Train pretrained YOLOv8 on COCO via Hugging Face")
    add_workspace_args(p, name="yolo_coco_train")
    p.add_argument("--batch-size",        type=int,  default=256)
    p.add_argument("--epochs",            type=int,  default=1)
    p.add_argument("--imgsz",             type=int,  default=640)
    p.add_argument("--model-variant",     type=str,  default="yolov8n.pt",
                   help="YOLOv8 variant (e.g., yolov8n.pt, yolov8s.pt)")
    p.add_argument("--use-custom-model",  action="store_true",
                   help="Use our custom YOLOv8n nn.Module instead of the Ultralytics default")
    return p.parse_args()


def main(args):
    ws     = workspace_from_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Workspace: {ws.root}")
    print(f"Device:    {device}")

    # ── Model ──────────────────────────────────────────────────────────────
    if args.use_custom_model:
        print(f"Loading CUSTOM YOLOv8n model (our reimplementation)...")
        from yolov8n_adapter import build_yolo_with_custom_model
        model = build_yolo_with_custom_model(nc=80, pretrained=args.model_variant)
    else:
        print(f"Loading pretrained YOLOv8 model: {args.model_variant}...")
        model = YOLO(args.model_variant)

    # ── Data ───────────────────────────────────────────────────────────────
    print("Loading COCO dataset from Hugging Face...")
    dataset_dict  = load_dataset("detection-datasets/coco", trust_remote_code=True)
    train_dataset = dataset_dict["train"]
    val_dataset   = dataset_dict["val"]

    output_dir = Path(ws.datasets / "coco_yolo_fmt")

    # Uncomment to reconvert:
    save_hf_dataset_for_yolo(train_dataset, output_dir, "train")
    save_hf_dataset_for_yolo(val_dataset,   output_dir, "val")

    yaml_path = output_dir / "coco.yaml"
    create_yolo_yaml(output_dir, yaml_path)
    print(f"Dataset YAML created at: {yaml_path}")

    # ── Training ───────────────────────────────────────────────────────────
    print("Starting YOLOv8 training_harness...")
    results = model.train(
        data    = str(yaml_path),
        epochs  = args.epochs,
        imgsz   = args.imgsz,
        batch   = args.batch_size,
        device  = device,
        project = str(ws.root),
        name    = "yolo_train_run",
    )

    torch.save(model.model.state_dict(), f"{ws.root}/yolo_train_run/trained_model.pt")

    print("\nTraining complete.")
    print(f"Results saved to: {ws.root}/yolo_train_run")


if __name__ == "__main__":
    main(parse_args())