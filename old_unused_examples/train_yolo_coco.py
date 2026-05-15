"""
COCO Training for YOLOv8.

This script loads a pretrained YOLOv8 model from Ultralytics, 
downloads the COCO dataset from Hugging Face, converts it to 
YOLO format, and starts the training_harness process.

Run
---
    python examples/train_yolo_coco.py --workdir ./runs/yolo_coco_train
"""

import argparse
import torch
from ultralytics import YOLO
from datasets import load_dataset
from pathlib import Path
from PIL import Image
import os

from utils import add_workspace_args, workspace_from_args

def save_hf_dataset_for_yolo(dataset, output_dir: Path, split_name: str):
    """
    Save HuggingFace COCO dataset to YOLOv8-compatible format for a specific split.
    
    Args:
        dataset: The HF dataset split.
        output_dir: Root directory for the YOLO dataset.
        split_name: 'train' or 'val'.
    """
    img_dir = output_dir / "images" / split_name
    lbl_dir = output_dir / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {split_name} split to YOLO format...")
    for i, item in enumerate(dataset):
        image: Image.Image = item["image"]
        w, h = image.size

        # Save image
        img_path = img_dir / f"{i:06d}.jpg"
        image.save(img_path)

        # Convert COCO [x, y, w, h] -> YOLO [cx, cy, w, h] normalized
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
    """Create the data.yaml file required by Ultralytics YOLO."""
    yaml_content = (
        f"path: {output_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 80\n"
        f"names: { [f'class_{i}' for i in range(80)] }\n" # Simplified names
    )
    yaml_path.write_text(yaml_content)

def parse_args():
    p = argparse.ArgumentParser(description="Train pretrained YOLOv8 on COCO via Hugging Face")
    add_workspace_args(p, name="yolo_coco_train")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--model-variant", type=str, default="yolov8n.pt", 
                   help="YOLOv8 variant (e.g., yolov8n.pt, yolov8s.pt)")
    return p.parse_args()

def main(args):
    ws = workspace_from_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Workspace: {ws.root}")
    print(f"Device:    {device}")

    # ---------------- model ----------------
    print(f"Loading pretrained YOLOv8 model: {args.model_variant}...")
    model = YOLO(args.model_variant).to(device)

    # ---------------- data ----------------
    print("Loading COCO dataset from Hugging Face...")
    # Load both train and val splits
    dataset_dict = load_dataset("detection-datasets/coco", trust_remote_code=True)
    train_dataset = dataset_dict["train"]
    val_dataset = dataset_dict["validation"]

    output_dir = Path(ws.datasets / "coco_yolo_fmt")
    
    # Convert splits to YOLO format
    save_hf_dataset_for_yolo(train_dataset, output_dir, "train")
    save_hf_dataset_for_yolo(val_dataset, output_dir, "val")

    # Create the YAML config
    yaml_path = output_dir / "coco.yaml"
    create_yolo_yaml(output_dir, yaml_path)
    print(f"Dataset YAML created at: {yaml_path}")

    # ---------------- training_harness loop ----------------
    print("Starting YOLOv8 training_harness...")
    # model.train() handles the training_harness loop, optimizer, and logging internally
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=device,
        project=str(ws.root),
        name="yolo_train_run"
    )

    print("\nTraining complete.")
    print(f"Results saved to: {ws.root}/yolo_train_run")

if __name__ == "__main__":
    main(parse_args())
