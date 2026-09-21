"""On-disk result format: manifest.json + memory-mappable .npy arrays.

result_dir/
  manifest.json
  layers/<layer_id>/<level>_<metric>.npy
  samples/...

See docs/llm/importance_analysis.md for the full format description.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from importance.discovery import LayerInfo
from importance.engine import EngineResult, LEVEL_PRIORITY, _reduce_grad

TOOL_VERSION = "0.1.0"
DEFAULT_SIZE_WARNING_BYTES = 2 * 1024 ** 3  # 2 GiB


def _all_levels_from_finest(finest: str) -> List[str]:
    idx = LEVEL_PRIORITY.index(finest)
    return LEVEL_PRIORITY[idx:] + ["layer"]


def _derive_level(finest_array: np.ndarray, finest: str, target: str, ndim_kernel: int) -> np.ndarray:
    """finest_array: [n_out, *finest_level_shape]. Sums down to a coarser level."""
    if target == finest:
        return finest_array
    t = torch.from_numpy(finest_array)
    if target == "layer":
        dims = tuple(range(1, t.dim()))
        return t.sum(dim=dims).numpy() if dims else t.numpy()
    # target is one of weight/kernel/filter, coarser than or equal to finest.
    # _reduce_grad's ndim_kernel means "how many trailing spatial-kernel dims
    # does the array still have" -- that's only the true layer kernel-ndim
    # when we're reducing FROM the weight level; once we're already at kernel
    # or filter granularity those dims are gone, so use 0.
    effective_ndim_kernel = ndim_kernel if finest == "weight" else 0
    reduced = _reduce_grad(t, target, effective_ndim_kernel)
    return reduced.numpy()


def _quantize_uint8(arr: np.ndarray):
    amax = float(np.abs(arr).max()) if arr.size else 0.0
    if amax == 0.0:
        scale = 1.0
    else:
        scale = amax / 127.0
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return q, scale


def _dtype_for(level: str, weight_dtype: str) -> str:
    if level == "weight":
        return weight_dtype
    return "float16" if weight_dtype == "uint8" else weight_dtype


class Result:
    """In-memory or on-disk importance analysis result."""

    def __init__(self, manifest: dict, arrays: Dict[str, Dict[str, Dict[str, np.ndarray]]],
                 samples: Optional[dict] = None, base_dir: Optional[Path] = None):
        self.manifest = manifest
        # arrays[layer_id][level][metric] -> np.ndarray, shape [n_out_total, *level_shape]
        self.arrays = arrays
        self.samples = samples or {}
        self.base_dir = Path(base_dir) if base_dir else None

    # ------------------------------------------------------------------ #
    # Construction from a fresh engine run
    # ------------------------------------------------------------------ #
    @classmethod
    def from_engine_result(
        cls, engine_result: EngineResult, layers: List[LayerInfo], config, output_spec,
        weight_dtype: str, model_summary: dict, dataset_info: dict, samples: dict,
        size_warning_bytes: int = DEFAULT_SIZE_WARNING_BYTES,
    ) -> "Result":
        levels_wanted = set(config.levels) | {"layer"}
        finest = engine_result.level
        arrays: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        layer_manifest_entries = []
        total_bytes = 0

        for layer in layers:
            acc = engine_result.accumulators[layer.id]
            layer_arrays: Dict[str, Dict[str, np.ndarray]] = {}
            levels_entry = {}

            # Raw parameter values (not output-dependent, so stored once per
            # layer rather than per output row) -- lets the viewer show the
            # actual weight magnitudes next to their importance scores.
            raw_entry = {}
            raw_weight = layer.module.weight.detach().cpu().numpy().astype(np.float32)
            layer_arrays["raw"] = {"weight": raw_weight}
            raw_entry["weight"] = {"dtype": "float32"}
            total_bytes += raw_weight.nbytes
            if layer.has_bias and layer.module.bias is not None:
                raw_bias = layer.module.bias.detach().cpu().numpy().astype(np.float32)
                layer_arrays["raw"]["bias"] = raw_bias
                raw_entry["bias"] = {"dtype": "float32"}
                total_bytes += raw_bias.nbytes

            for level in _all_levels_from_finest(finest):
                if level != "layer" and level not in levels_wanted:
                    continue
                abs_level = _derive_level(acc.abs_sum.astype(np.float32), finest, level, acc.ndim_kernel)
                signed_level = _derive_level(acc.signed_sum.astype(np.float32), finest, level, acc.ndim_kernel)
                layer_arrays.setdefault(level, {})["mean_abs_s"] = abs_level
                layer_arrays[level]["mean_s"] = signed_level
                total_bytes += abs_level.nbytes + signed_level.nbytes

                metrics_entry = {"mean_abs_s": {"dtype": "float32"}, "mean_s": {"dtype": "float32"}}
                if level == "filter":
                    if acc.has_bias:
                        layer_arrays[level]["bias_mean_abs_s"] = acc.bias_abs_sum.astype(np.float32)
                        layer_arrays[level]["bias_mean_s"] = acc.bias_signed_sum.astype(np.float32)
                        metrics_entry["bias_mean_abs_s"] = {"dtype": "float32"}
                        metrics_entry["bias_mean_s"] = {"dtype": "float32"}
                    layer_arrays[level]["act_filter"] = acc.act_abs_sum.astype(np.float32)
                    layer_arrays[level]["act_filter_signed"] = acc.act_signed_sum.astype(np.float32)
                    metrics_entry["act_filter"] = {"dtype": "float32"}
                    metrics_entry["act_filter_signed"] = {"dtype": "float32"}
                levels_entry[level] = metrics_entry

            manifest_entry = layer.to_manifest_dict()
            manifest_entry["levels"] = levels_entry
            manifest_entry["raw"] = raw_entry
            layer_manifest_entries.append(manifest_entry)
            arrays[layer.id] = layer_arrays

        size_warning = None
        if total_bytes > size_warning_bytes:
            size_warning = (
                f"Estimated importance result size is {total_bytes / 1e9:.2f} GB, "
                f"exceeding the {size_warning_bytes / 1e9:.2f} GB warning threshold."
            )
            warnings.warn(size_warning, stacklevel=2)

        manifest = {
            "tool_version": TOOL_VERSION,
            "model_summary": model_summary,
            "dataset_info": {**dataset_info, "n_samples": engine_result.n_samples,
                              "has_loss_row": engine_result.has_loss},
            "settings": {
                "device": str(config.device) if config.device else None,
                "max_samples": config.max_samples,
                "output_reduce": config.output_reduce if isinstance(config.output_reduce, str) else "callable",
                "levels": list(config.levels),
                "store_samples": config.store_samples,
                "use_quantized_weight": config.use_quantized_weight,
                "seed": config.seed,
                "path_used": engine_result.path_used,
                "weight_dtype": weight_dtype,
            },
            "output_features": {
                "count": engine_result.n_out_total,
                "names": engine_result.output_names,
                "source_shape": list(output_spec.source_shape),
                "reduce": output_spec.reduce_kind,
            },
            "layers": layer_manifest_entries,
            "metadata": {
                "total_size_bytes": int(total_bytes),
                "size_warning": size_warning,
            },
        }
        return cls(manifest, arrays, samples=samples)

    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        base = Path(path)
        (base / "layers").mkdir(parents=True, exist_ok=True)
        (base / "samples").mkdir(parents=True, exist_ok=True)

        weight_dtype = self.manifest["settings"]["weight_dtype"]
        manifest = json.loads(json.dumps(self.manifest))  # deep copy

        for layer_entry in manifest["layers"]:
            layer_id = layer_entry["id"]
            layer_dir = base / "layers" / layer_id
            layer_dir.mkdir(parents=True, exist_ok=True)
            for level, metrics in layer_entry["levels"].items():
                for metric in list(metrics.keys()):
                    arr = self.arrays[layer_id][level][metric]
                    use_uint8 = weight_dtype == "uint8" and level == "weight" and metric in (
                        "mean_abs_s", "mean_s")
                    fname = f"{level}_{metric}.npy"
                    fpath = layer_dir / fname
                    if use_uint8:
                        q, scale = _quantize_uint8(arr)
                        np.save(fpath, q)
                        metrics[metric] = {"dtype": "int8", "scale": scale, "file": f"layers/{layer_id}/{fname}",
                                            "shape": list(arr.shape)}
                    else:
                        target_dtype = np.float16 if _dtype_for(level, weight_dtype) == "float16" else np.float32
                        arr_out = arr.astype(target_dtype)
                        np.save(fpath, arr_out)
                        metrics[metric] = {"dtype": str(target_dtype.__name__ if hasattr(target_dtype, "__name__") else target_dtype),
                                            "file": f"layers/{layer_id}/{fname}", "shape": list(arr.shape)}

        for layer_entry in manifest["layers"]:
            layer_id = layer_entry["id"]
            layer_dir = base / "layers" / layer_id
            layer_dir.mkdir(parents=True, exist_ok=True)
            for name, info in layer_entry.get("raw", {}).items():
                arr = self.arrays[layer_id]["raw"][name]
                fname = f"raw_{name}.npy"
                np.save(layer_dir / fname, arr.astype(np.float32))
                info["file"] = f"layers/{layer_id}/{fname}"
                info["shape"] = list(arr.shape)

        for name, arr in self.samples.get("arrays", {}).items():
            np.save(base / "samples" / f"{name}.npy", arr)
        if "meta" in self.samples:
            with open(base / "samples" / "meta.json", "w") as f:
                json.dump(self.samples["meta"], f)

        with open(base / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        self.manifest = manifest
        self.base_dir = base

    @classmethod
    def load(cls, path: str) -> "Result":
        base = Path(path)
        with open(base / "manifest.json") as f:
            manifest = json.load(f)

        arrays: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        for layer_entry in manifest["layers"]:
            layer_id = layer_entry["id"]
            arrays[layer_id] = {}
            for level, metrics in layer_entry["levels"].items():
                arrays[layer_id][level] = {}
                for metric, info in metrics.items():
                    fpath = base / info["file"]
                    raw = np.load(fpath, mmap_mode="r")
                    if info["dtype"] == "int8":
                        arrays[layer_id][level][metric] = raw.astype(np.float32) * info["scale"]
                    else:
                        arrays[layer_id][level][metric] = raw
            arrays[layer_id]["raw"] = {}
            for name, info in layer_entry.get("raw", {}).items():
                arrays[layer_id]["raw"][name] = np.load(base / info["file"], mmap_mode="r")

        samples = {"arrays": {}, "meta": {}}
        samples_dir = base / "samples"
        if samples_dir.exists():
            for f in samples_dir.glob("*.npy"):
                samples["arrays"][f.stem] = np.load(f, mmap_mode="r")
            meta_path = samples_dir / "meta.json"
            if meta_path.exists():
                with open(meta_path) as fh:
                    samples["meta"] = json.load(fh)

        return cls(manifest, arrays, samples=samples, base_dir=base)

    # ------------------------------------------------------------------ #
    # Query API
    # ------------------------------------------------------------------ #
    def _layer_entry(self, layer_id: str) -> dict:
        for entry in self.manifest["layers"]:
            if entry["id"] == layer_id:
                return entry
        raise KeyError(f"No such layer {layer_id!r} in this result. "
                        f"Known layers: {[e['id'] for e in self.manifest['layers']]}")

    def layer_names(self) -> List[str]:
        return [e["id"] for e in self.manifest["layers"]]

    def output_index(self, output) -> int:
        if isinstance(output, int):
            return output
        names = self.manifest["output_features"]["names"]
        if output in names:
            return names.index(output)
        if output == "all":
            return 0
        raise KeyError(f"Unknown output {output!r}; known names: {names}")

    def filter(self, layer: str, output=None, metric: str = "mean_abs_s") -> np.ndarray:
        return self._level(layer, "filter", output, metric)

    def kernel(self, layer: str, output=None, metric: str = "mean_abs_s") -> np.ndarray:
        return self._level(layer, "kernel", output, metric)

    def weight(self, layer: str, output=None, metric: str = "mean_abs_s") -> np.ndarray:
        return self._level(layer, "weight", output, metric)

    def raw_weight(self, layer: str) -> np.ndarray:
        """The actual (float) parameter values for `layer`'s weight -- not an
        importance score. Shape [F, C, *k], not output-dependent."""
        return np.asarray(self.arrays[layer]["raw"]["weight"])

    def raw_bias(self, layer: str) -> Optional[np.ndarray]:
        arr = self.arrays[layer].get("raw", {}).get("bias")
        return np.asarray(arr) if arr is not None else None

    def layer_score(self, layer: str, output=None, metric: str = "mean_abs_s") -> np.ndarray:
        return self._level(layer, "layer", output, metric)

    def _level(self, layer: str, level: str, output, metric: str) -> np.ndarray:
        arr = self.arrays[layer][level][metric]
        if output is None:
            return np.asarray(arr)
        idx = self.output_index(output)
        return np.asarray(arr[idx])

    def to_dataframe(self, level: str = "filter", metric: str = "mean_abs_s"):
        import pandas as pd

        names = self.manifest["output_features"]["names"]
        rows = []
        for entry in self.manifest["layers"]:
            layer_id = entry["id"]
            if level not in self.arrays[layer_id]:
                continue
            arr = np.asarray(self.arrays[layer_id][level][metric])  # [n_out, F] or [n_out, F, C] ...
            n_out = arr.shape[0]
            flat = arr.reshape(n_out, -1)
            for out_idx in range(n_out):
                for pos in range(flat.shape[1]):
                    rows.append({
                        "layer": layer_id,
                        "output_index": out_idx,
                        "output_name": names[out_idx] if out_idx < len(names) else str(out_idx),
                        "position": pos,
                        "value": float(flat[out_idx, pos]),
                    })
        return pd.DataFrame(rows)


def load(path: str) -> Result:
    return Result.load(path)
