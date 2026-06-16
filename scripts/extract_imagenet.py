"""
extract_imagenet.py — dump ILSVRC/imagenet-1k from HuggingFace to ImageFolder on disk.

Run once. Output structure:
    <output_dir>/train/<label_id:04d>/<index:08d>.jpg
    <output_dir>/val/<label_id:04d>/<index:08d>.jpg

After extraction, pass --data-dir <output_dir> to train_imagenet_qat.py to use
the DALI pipeline instead of the HuggingFace dataloader.

Usage:
    python scripts/extract_imagenet.py --output-dir /data/imagenet_jpeg
    python scripts/extract_imagenet.py --output-dir /data/imagenet_jpeg --splits val
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def _save(path: Path, img) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, "JPEG", quality=95)


def extract_split(hf_split: str, out_split_dir: Path, n_workers: int) -> None:
    """Extract one HF split to <out_split_dir>/<label:04d>/<idx:08d>.jpg."""
    print(f"\nLoading HuggingFace split '{hf_split}' …")
    ds = load_dataset("ILSVRC/imagenet-1k", split=hf_split)
    n = len(ds)
    print(f"  {n:,} images → {out_split_dir}")

    CHUNK = 2000  # how many images to batch-read from Arrow at once
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        bar = tqdm(total=n, desc=hf_split, unit="img")
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            batch = ds[start:end]  # dict of lists: {"image": [...], "label": [...]}
            futs = []
            for i, (img, label) in enumerate(zip(batch["image"], batch["label"])):
                path = out_split_dir / f"{label:04d}" / f"{start + i:08d}.jpg"
                futs.append(pool.submit(_save, path, img))
            for f in as_completed(futs):
                f.result()
                bar.update(1)
        bar.close()

    print(f"  Done → {out_split_dir}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract ILSVRC/imagenet-1k HF dataset to JPEG ImageFolder on disk"
    )
    p.add_argument("--output-dir", required=True, help="Destination directory")
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        choices=["train", "validation"],
        help="Which HF splits to extract (default: both)",
    )
    p.add_argument(
        "--workers", type=int, default=16,
        help="Parallel JPEG save threads (default: 16)",
    )
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # HF uses "validation"; DALI convention (and our script) uses "val"
    split_name_map = {"train": "train", "validation": "val"}

    for hf_split in args.splits:
        out_dir = out / split_name_map[hf_split]
        extract_split(hf_split, out_dir, args.workers)

    print("\nExtraction complete.")
    print(f"Pass --data-dir {args.output_dir} to train_imagenet_qat.py to use DALI.")


if __name__ == "__main__":
    main()
