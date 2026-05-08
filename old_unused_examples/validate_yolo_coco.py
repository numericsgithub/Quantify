"""
COCO Validation for pretrained YOLOv8.

This script loads a pretrained YOLOv8 model from Ultralytics and validates it 
on the COCO validation dataset loaded from Hugging Face.

Run
---
    python examples/validate_yolo_coco.py --workdir ./runs/yolo_coco_val
"""

import argparse
import torch
from ultralytics import YOLO
from datasets import load_dataset
from typing import List, Tuple

from utils import add_workspace_args, workspace_from_args
from utils.logging import CSVLogger


def calculate_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    Boxes are expected in (x1, y1, x2, y2) format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def parse_args():
    p = argparse.ArgumentParser(description="Validate pretrained YOLOv8 on COCO via Hugging Face")
    add_workspace_args(p, name="yolo_coco_val")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--model-variant", type=str, default="yolov8n.pt", 
                   help="YOLOv8 variant (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt)")
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
    print("Loading COCO validation set from Hugging Face...")
    # Note: 'coco' dataset on HF contains 'image' and 'objects' (bboxes and categories)
    dataset = load_dataset("detection-datasets/coco", split="validation", trust_remote_code=True)

    # ---------------- validation loop ----------------
    log_path = ws.logs / "yolo_coco_val_log.csv"
    
    # We track a simple metric: "Hit Rate" 
    # (Percentage of images where at least one ground truth box is matched by a prediction with IoU > 0.5)
    hits = 0
    total_images = len(dataset)

    print(f"Starting validation on {total_images} images...")
    
    with CSVLogger(log_path, fieldnames=["image_idx", "has_hit", "num_preds", "num_gt"]) as log:
        for i in range(total_images):
            item = dataset[i]
            image = item["image"]
            gt_objects = item["objects"] # contains 'bbox' and 'category'
            
            # Run inference
            # YOLOv8 model handles the image preprocessing internally
            results = model.predict(image, conf=0.25, verbose=False)[0]
            
            # Extract predictions: boxes are in [x1, y1, x2, y2]
            pred_boxes = results.boxes.xyxy.cpu().numpy()
            gt_boxes = gt_objects["bbox"] # COCO HF format is usually [x, y, w, h]
            
            # Convert GT from [x, y, w, h] to [x1, y1, x2, y2]
            gt_boxes_converted = []
            for box in gt_boxes:
                x, y, w, h = box
                gt_boxes_converted.append((x, y, x + w, y + h))

            # Check for hits (IoU > 0.5)
            image_hit = False
            for gt in gt_boxes_converted:
                for pred in pred_boxes:
                    if calculate_iou(gt, pred) > 0.5:
                        image_hit = True
                        break
                if image_hit:
                    break
            
            if image_hit:
                hits += 1
            
            log.log(
                image_idx=i, 
                has_hit=int(image_hit), 
                num_preds=len(pred_boxes), 
                num_gt=len(gt_boxes)
            )

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{total_images} images... Current Hit Rate: {100.0 * hits / (i + 1):.2f}%")

    final_hit_rate = 100.0 * hits / total_images
    print(f"\nValidation Complete.")
    print(f"Final Hit Rate (IoU > 0.5): {final_hit_rate:.2f}%")
    print(f"Results logged to: {log_path}")


if __name__ == "__main__":
    main(parse_args())
