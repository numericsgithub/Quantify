"""
Dump 10 sample images from the DALI training pipeline as PNG files.

Usage:
    python scripts/dump_dali_augmentation_samples.py \
        --data-dir /home/th/tmp/datasets/imagenet \
        --output-dir /tmp/dali_samples

Images are denormalized back to RGB [0, 255] before saving so you can
visually verify the crop, flip, brightness, and saturation augmentations.
"""

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True,
                   help="ImageFolder root with train/ and val/ subdirs")
    p.add_argument("--output-dir", default="/tmp/dali_samples",
                   help="Directory to write PNG files (default: /tmp/dali_samples)")
    p.add_argument("--num-images", type=int, default=10)
    p.add_argument("--num-threads", type=int, default=4)
    return p.parse_args()


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet mean/std normalization and return a uint8 CHW tensor."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import here so the script fails fast if DALI is not installed.
    from utils.dali_pipeline import build_dali_loaders

    print(f"Building DALI train pipeline from {args.data_dir} …")
    train_loader, _ = build_dali_loaders(
        data_dir=args.data_dir,
        batch_size=args.num_images,
        num_threads=args.num_threads,
    )

    print("Fetching one train batch …")
    images, labels = next(iter(train_loader))  # [N, 3, 224, 224] float GPU

    print("\n--- Train samples ---")
    saved = 0
    for i in range(min(args.num_images, images.shape[0])):
        img_raw = images[i]
        img = denormalize(img_raw)            # float [0, 1] CHW on GPU
        img_pil = TF.to_pil_image(img.cpu()) # converts to PIL RGB
        out_path = out_dir / f"train_sample_{i:02d}_label{labels[i].item()}.png"
        img_pil.save(out_path)
        saved += 1
        print(f"  [{i:02d}] min={img_raw.min():.4f}  max={img_raw.max():.4f}  saved {out_path}")

    print(f"\nDone — {saved} train images written to {out_dir}")

    print(f"\nBuilding DALI val pipeline from {args.data_dir} …")
    _, val_loader = build_dali_loaders(
        data_dir=args.data_dir,
        batch_size=args.num_images,
        num_threads=args.num_threads,
    )

    print("Fetching one val batch …")
    val_images, val_labels = next(iter(val_loader))

    print("\n--- Val samples ---")
    val_saved = 0
    for i in range(min(args.num_images, val_images.shape[0])):
        img_raw = val_images[i]
        img = denormalize(img_raw)
        img_pil = TF.to_pil_image(img.cpu())
        out_path = out_dir / f"val_sample_{i:02d}_label{val_labels[i].item()}.png"
        img_pil.save(out_path)
        val_saved += 1
        print(f"  [{i:02d}] min={img_raw.min():.4f}  max={img_raw.max():.4f}  saved {out_path}")

    print(f"\nDone — {val_saved} val images written to {out_dir}")


if __name__ == "__main__":
    # Allow running as `python scripts/dump_dali_augmentation_samples.py` from repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
