"""
benchmark_imagenet_dataloader.py — measure ImageNet data pipeline throughput.

Builds the exact same loaders examples/train_imagenet_qat.py uses (DALI when
--data-dir is set, the HuggingFace + torchvision-transforms loader otherwise,
selected by the same _build_dataloaders() function) and times how fast
batches can be produced, with batches moved to --device the same way
training_harness/trainer_v2.py does (plain .to(device), no transforms beyond
what the loader itself applies).

This measures the data pipeline in isolation -- no model forward/backward --
so the numbers reflect the ceiling on samples/sec a training run could ever
reach with this pipeline, independent of model/GPU compute cost.

By default it makes two full passes over the training set: the first (warmup)
absorbs one-time costs that a real multi-epoch training run only pays once
(DALI/worker startup, filesystem cache misses on first read, etc.) and is not
timed; the second pass is fully timed and is what's reported. This is the
realistic steady-state number -- timing only a handful of batches understates
true throughput since it never gets past the warmup cost. Use --num-batches /
--warmup-batches to time a subset instead (e.g. for a quick sanity check).

Usage examples
--------------
# DALI pipeline (ImageFolder dataset), two full epochs (warmup + measured):
python -m examples.benchmark_imagenet_dataloader --data-dir /path/to/imagenet

# HuggingFace pipeline:
python -m examples.benchmark_imagenet_dataloader --hf-dataset ILSVRC/imagenet-1k --num-workers 20

# Quick check on a subset instead of the full dataset, val loader included:
python -m examples.benchmark_imagenet_dataloader --data-dir /path/to/imagenet \\
    --include-val --num-batches 200 --warmup-batches 20
"""

from __future__ import annotations

import argparse
import time

import torch
from tqdm import tqdm

from examples.train_imagenet_qat import _build_dataloaders


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark the ImageNet data pipeline used by train_imagenet_qat.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    d = p.add_argument_group("data")
    d.add_argument(
        "--data-dir", type=str, default=None, metavar="PATH",
        help="Path to ImageFolder dataset (train/ and val/ subdirs). "
             "When set, benchmarks the DALI pipeline instead of HuggingFace.",
    )
    d.add_argument(
        "--hf-dataset", type=str, default="ILSVRC/imagenet-1k",
        help="HuggingFace dataset name (ignored when --data-dir is set)",
    )
    d.add_argument(
        "--num-workers", type=int, default=20,
        help="HuggingFace DataLoader workers (ignored when --data-dir is set)",
    )
    d.add_argument(
        "--dali-threads", type=int, default=4,
        help="DALI CPU preprocessing threads (used when --data-dir is set)",
    )
    d.add_argument("--batch-size", type=int, default=1024)
    d.add_argument(
        "--prefetch-factor", type=int, default=3,
        help="DataLoader prefetch factor (ignored when --data-dir is set)",
    )

    b = p.add_argument_group("benchmark")
    b.add_argument(
        "--num-batches", type=int, default=None,
        help="Number of batches to time, in the second (measured) pass. "
             "Default: a full epoch (every batch in the loader).",
    )
    b.add_argument(
        "--warmup-batches", type=int, default=None,
        help="Number of batches to consume in the first (untimed) warmup "
             "pass. Default: a full epoch, i.e. two full passes over the "
             "dataset in total -- one warmup, one measured.",
    )
    b.add_argument(
        "--include-val", action="store_true",
        help="Also benchmark the val loader after the train loader",
    )
    b.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device batches are moved to, same as the trainer does",
    )

    return p.parse_args()


def _run_pass(loader, device: torch.device, is_cuda: bool, limit: int | None, time_it: bool, desc: str):
    """One full (or limited) iteration over `loader`. Always starts a fresh
    `for ... in loader` so DALI's reader actually resets between the warmup
    and measured passes (unlike resuming a single live iterator with
    next())."""
    batch_times = []
    n_samples = 0
    total = limit if limit is not None else len(loader)
    start = time.perf_counter()
    pbar = tqdm(enumerate(loader), total=total, desc=desc, leave=False, dynamic_ncols=True)
    for i, (inputs, targets) in pbar:
        if limit is not None and i >= limit:
            break
        batch_start = time.perf_counter()
        inputs = inputs.to(device)
        targets = targets.to(device)
        if is_cuda:
            torch.cuda.synchronize()
        batch_time = time.perf_counter() - batch_start
        if time_it:
            batch_times.append(batch_time)
        n_samples += inputs.shape[0]
        pbar.set_postfix(samples_per_sec=f"{inputs.shape[0] / max(batch_time, 1e-9):,.0f}")
    pbar.close()
    total_time = time.perf_counter() - start
    return n_samples, total_time, batch_times


def _benchmark_loader(loader, name: str, num_batches: int | None, warmup_batches: int | None, device: torch.device) -> None:
    is_cuda = device.type == "cuda"

    if type(loader).__name__ == "DALILoader" and (num_batches is not None or warmup_batches is not None):
        print(
            f"[{name}] NOTE: DALI's reader cannot reset mid-epoch -- breaking "
            f"out early via --num-batches/--warmup-batches means the 'measured' "
            f"pass continues the same unfinished epoch instead of starting a "
            f"clean one, and may read artificially fast from already-prefetched "
            f"batches. Numbers from a limited run are a rough sanity check only; "
            f"omit both flags for an accurate full-epoch measurement."
        )

    warmup_desc = f"{warmup_batches} batches" if warmup_batches is not None else "a full epoch"
    print(f"\n[{name}] warmup pass ({warmup_desc})...")
    _run_pass(loader, device, is_cuda, warmup_batches, time_it=False, desc=f"[{name}] warmup")

    measure_desc = f"{num_batches} batches" if num_batches is not None else "a full epoch"
    print(f"[{name}] measured pass ({measure_desc})...")
    n_samples, total_time, batch_times = _run_pass(
        loader, device, is_cuda, num_batches, time_it=True, desc=f"[{name}] measured"
    )

    avg_batch_ms = sum(batch_times) / len(batch_times) * 1000
    worst_batch_ms = max(batch_times) * 1000
    samples_per_sec = n_samples / total_time

    print(f"[{name}] {n_samples} samples in {total_time:.2f}s")
    print(f"[{name}]   {samples_per_sec:,.0f} samples/sec")
    print(f"[{name}]   {avg_batch_ms:.1f} ms/batch avg, {worst_batch_ms:.1f} ms/batch worst")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("Building data loaders (same code path as train_imagenet_qat.py)...")
    train_loader, val_loader = _build_dataloaders(args)

    _benchmark_loader(train_loader, "train", args.num_batches, args.warmup_batches, device)
    if args.include_val:
        _benchmark_loader(val_loader, "val", args.num_batches, args.warmup_batches, device)


if __name__ == "__main__":
    main()
