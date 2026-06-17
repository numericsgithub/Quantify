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

from pathlib import Path

from nvidia import dali
from nvidia.dali import fn, pipeline_def, types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy


# ---------------------------------------------------------------------------
# Pipeline definitions
# ---------------------------------------------------------------------------

@pipeline_def
def _train_pipeline(file_root: str, num_shards: int, shard_id: int, crop: int = 224):
    jpegs, labels = fn.readers.file(
        file_root=file_root,
        random_shuffle=True,
        shard_id=shard_id,
        num_shards=num_shards,
        pad_last_batch=True,
        name="Reader",
    )
    # Decode + random crop in one fused GPU op (nvJPEG)
    images = fn.decoders.image_random_crop(
        jpegs,
        device="mixed",
        output_type=types.RGB,
        random_aspect_ratio=[3 / 4, 4 / 3],
        random_area=[0.08, 1.0],
        num_attempts=100,
    )
    images = fn.resize(images, device="gpu", size=[crop, crop])
    images = fn.flip(
        images,
        device="gpu",
        horizontal=fn.random.coin_flip(probability=0.5),
    )
    images = fn.crop_mirror_normalize(
        images,
        device="gpu",
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return images, labels.gpu()


@pipeline_def
def _val_pipeline(file_root: str, num_shards: int, shard_id: int,
                  crop: int = 224, resize_shorter: int = 236):
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
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
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
    """

    def __init__(
        self,
        pipeline,
        dataset_size: int,
        batch_size: int,
        last_batch_policy: LastBatchPolicy = LastBatchPolicy.DROP,
        already_built: bool = False,
    ):
        if not already_built:
            pipeline.build()
        self._iter = DALIClassificationIterator(
            pipeline,
            reader_name="Reader",
            last_batch_policy=last_batch_policy,
        )
        self._dataset_size = dataset_size
        self._batch_size = batch_size

    def __iter__(self):
        self._iter.reset()
        for batch in self._iter:
            images = batch[0]["data"]
            labels = batch[0]["label"].squeeze(-1).long()
            yield images, labels

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
    resize_shorter: int = 236,
) -> tuple[DALILoader, DALILoader]:
    """
    Build DALI train and val loaders from an ImageFolder directory.

    Args:
        data_dir:        Root with train/ and val/ subdirectories.
        batch_size:      Batch size (same for train and val).
        num_threads:     CPU threads for pre-processing stages before GPU.
        device_id:       CUDA device index.
        crop:            Output spatial size (default 224).
        resize_shorter:  Val resize-shorter target before center-crop (default 236,
                         matching timm's crop_pct=0.95 for the RSB recipe).

    Returns:
        (train_loader, val_loader)
    """
    data_dir = Path(data_dir)
    train_dir = str(data_dir / "train")
    val_dir   = str(data_dir / "val")

    train_pipe = _train_pipeline(
        file_root=train_dir,
        num_shards=1,
        shard_id=0,
        crop=crop,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
    )
    val_pipe = _val_pipeline(
        file_root=val_dir,
        num_shards=1,
        shard_id=0,
        crop=crop,
        resize_shorter=resize_shorter,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
    )

    # Build before querying epoch size so DALI can count the files
    train_pipe.build()
    val_pipe.build()
    n_train = train_pipe.epoch_size("Reader")
    n_val   = val_pipe.epoch_size("Reader")

    train_loader = DALILoader(train_pipe, n_train, batch_size, already_built=True)
    val_loader   = DALILoader(val_pipe,   n_val,   batch_size, LastBatchPolicy.PARTIAL,
                              already_built=True)
    return train_loader, val_loader
