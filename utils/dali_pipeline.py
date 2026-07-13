"""
DALI-based ImageNet data pipeline.

Replaces the HuggingFace + PIL DataLoader with NVIDIA DALI, which:
  - Decodes JPEGs on GPU via nvJPEG (or libjpeg-turbo on CPU)
  - Runs crop, flip, color jitter, and normalize on the GPU
  - Achieves 30–50k samples/sec vs ~6–13k with PIL workers

Requires images in ImageFolder format on disk:
    <data_dir>/train/<label:04d>/<filename>.jpg
    <data_dir>/val/<label:04d>/<filename>.jpg

Use scripts/extract_imagenet.py to produce this layout from the HF dataset.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Callable

from nvidia import dali
from nvidia.dali import fn, pipeline_def, types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.auto_aug import rand_augment

# Default normalization: standard ImageNet statistics (used by torchvision and
# most timm checkpoints). Some checkpoints — e.g. timm's mobilenetv1_100.ra4 —
# were trained with mean=std=0.5 instead, so mean/std are configurable per model.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HALF_MEAN = (0.5, 0.5, 0.5)
HALF_STD = (0.5, 0.5, 0.5)

# Normalization matching each model's pretrained timm checkpoint. Most use
# standard ImageNet stats, but timm's mobilenetv1_100.ra4 checkpoint was trained
# with inception-style mean=std=0.5 — feeding it ImageNet-normalized inputs
# collapses accuracy (~17% instead of ~73%).
_MODEL_NORM = {
    "resnet18":    (IMAGENET_MEAN, IMAGENET_STD),
    "resnet50":    (IMAGENET_MEAN, IMAGENET_STD),
    "mobilenetv1": (HALF_MEAN, HALF_STD),
    "mobilenetv2": (IMAGENET_MEAN, IMAGENET_STD),
}


def norm_for_model(arch: str):
    """Return the (mean, std) normalization matching a model's pretrained checkpoint."""
    return _MODEL_NORM.get(arch, (IMAGENET_MEAN, IMAGENET_STD))


# ---------------------------------------------------------------------------
# Pipeline definitions
# ---------------------------------------------------------------------------

@pipeline_def(enable_conditionals=True)  # required for auto_aug
def _train_pipeline(file_root: str, num_shards: int, shard_id: int,
                    crop: int = 224, randaugment_n: int = 2, randaugment_m: int = 7,
                    mean=IMAGENET_MEAN, std=IMAGENET_STD):
    jpegs, labels = fn.readers.file(
        file_root=file_root,
        random_shuffle=True,
        shard_id=shard_id,
        num_shards=num_shards,
        name="Reader",
    )
    images = fn.decoders.image_random_crop(
        jpegs,
        device="mixed",
        output_type=types.RGB,
        random_aspect_ratio=[3/4, 4/3],
        random_area=[0.08, 1.0],
        num_attempts=100,
    )
    images = fn.resize(
        images, device="gpu", size=[crop, crop],
        interp_type=types.INTERP_CUBIC,
    )
    images = rand_augment.rand_augment(images, n=randaugment_n, m=randaugment_m)
    images = fn.crop_mirror_normalize(
        images,
        device="gpu",
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[m * 255 for m in mean],
        std=[s * 255 for s in std],
        mirror=fn.random.coin_flip(probability=0.5),
    )
    return images, labels.gpu()


@pipeline_def
def _val_pipeline(file_root: str, num_shards: int, shard_id: int,
                  crop: int = 224, resize_shorter: int = 256,
                  mean=IMAGENET_MEAN, std=IMAGENET_STD):
    jpegs, labels = fn.readers.file(
        file_root=file_root,
        random_shuffle=False,
        shard_id=shard_id,
        num_shards=num_shards,
        pad_last_batch=True,
        name="Reader",
    )
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    images = fn.resize(images, device="gpu", resize_shorter=resize_shorter)
    images = fn.crop_mirror_normalize(
        images,
        device="gpu",
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(crop, crop),  # center crop
        mean=[m * 255 for m in mean],
        std=[s * 255 for s in std],
    )
    return images, labels.gpu()


# ---------------------------------------------------------------------------
# DataLoader-compatible wrapper
# ---------------------------------------------------------------------------

class DALILoader:
    """
    Wraps a DALI pipeline so the trainer can iterate over it like a DataLoader.

    Yields (images, labels) where both tensors are already on the GPU.
    The trainer's .to(device) calls are harmless no-ops on GPU tensors.

    When DALI encounters a corrupt/unreadable image it raises a critical error
    that permanently invalidates the pipeline object.  DALILoader handles this
    by rebuilding the pipeline from the stored factory and ending the current
    epoch early — the next epoch starts with a fresh, valid pipeline.
    """

    def __init__(
        self,
        pipeline_factory: Callable,
        dataset_size: int,
        batch_size: int,
        last_batch_policy: LastBatchPolicy = LastBatchPolicy.DROP,
    ):
        self._factory = pipeline_factory
        self._dataset_size = dataset_size
        self._batch_size = batch_size
        self._last_batch_policy = last_batch_policy
        self._build()

    def _build(self):
        pipe = self._factory()
        pipe.build()
        self._iter = DALIClassificationIterator(
            pipe,
            reader_name="Reader",
            last_batch_policy=self._last_batch_policy,
        )

    def __iter__(self):
        try:
            self._iter.reset()
        except Exception:
            # Pipeline was invalidated by a previous critical error — rebuild.
            self._build()

        try:
            for batch in self._iter:
                images = batch[0]["data"]
                labels = batch[0]["label"].squeeze(-1).long()
                yield images, labels
        except RuntimeError as e:
            msg = str(e)
            if "Critical error" in msg or "no longer valid" in msg:
                print(
                    f"\n[DALI] Warning: unreadable image encountered mid-epoch; "
                    f"skipping remaining batches and rebuilding pipeline for next epoch."
                )
                self._build()
            else:
                raise

    def __len__(self) -> int:
        return self._dataset_size // self._batch_size


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_dali_loaders(
    data_dir: str | Path,
    batch_size: int,
    num_threads: int = 4,
    device_id: int = 0,
    crop: int = 224,
    resize_shorter: int = 256,
    randaugment_n: int = 2,
    randaugment_m: int = 7,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> tuple[DALILoader, DALILoader]:
    """
    Build DALI train and val loaders from an ImageFolder directory.

    Args:
        data_dir:        Root with train/ and val/ subdirectories.
        batch_size:      Batch size (same for train and val).
        num_threads:     CPU threads for pre-processing stages before GPU.
        device_id:       CUDA device index.
        crop:            Output spatial size (default 224).
        resize_shorter:  Val resize-shorter target before center-crop (default 256,
                         matching the standard torchvision IMAGENET1K recipe).
        randaugment_n:   Number of RandAugment transforms applied per image (default 2).
        randaugment_m:   RandAugment magnitude (default 7).
        mean, std:       Per-channel normalization stats in [0, 1] (scaled by 255
                         internally). Defaults to ImageNet stats; pass (0.5, 0.5, 0.5)
                         for checkpoints trained with inception-style normalization.

    Returns:
        (train_loader, val_loader)
    """
    data_dir = Path(data_dir)
    train_dir = str(data_dir / "train")
    val_dir   = str(data_dir / "val")

    dali_kwargs = dict(batch_size=batch_size, num_threads=num_threads, device_id=device_id)

    train_factory = functools.partial(
        _train_pipeline,
        file_root=train_dir, num_shards=1, shard_id=0,
        crop=crop, randaugment_n=randaugment_n, randaugment_m=randaugment_m,
        mean=mean, std=std,
        **dali_kwargs,
    )
    val_factory = functools.partial(
        _val_pipeline,
        file_root=val_dir, num_shards=1, shard_id=0,
        crop=crop, resize_shorter=resize_shorter,
        mean=mean, std=std,
        **dali_kwargs,
    )

    # Build one probe pipeline to query epoch sizes before handing off factories.
    train_probe = train_factory()
    train_probe.build()
    n_train = train_probe.epoch_size("Reader")
    del train_probe

    val_probe = val_factory()
    val_probe.build()
    n_val = val_probe.epoch_size("Reader")
    del val_probe

    train_loader = DALILoader(train_factory, n_train, batch_size)
    val_loader   = DALILoader(val_factory,   n_val,   batch_size, LastBatchPolicy.PARTIAL)
    return train_loader, val_loader
