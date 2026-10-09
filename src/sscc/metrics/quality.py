from __future__ import annotations

from contextlib import nullcontext
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


PAIR_METRICS = ("psnr", "ms_ssim", "lpips", "dists", "pieapp")
DISTRIBUTION_METRICS = ("fid", "kid")
ALL_METRICS = PAIR_METRICS + DISTRIBUTION_METRICS
PATCH_SIZE = 256
FID_BATCH_SIZE = 128
KID_SUBSETS = 100
KID_SUBSET_SIZE = 1000


def _image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0)


class ImagePatchDataset(Dataset):
    """Read GLC FID/256 patches without materializing thousands of PNGs.

    GLC uses two non-overlapping 256x256 grids: one starting at (0, 0),
    followed by one starting at (128, 128). Incomplete border patches are
    discarded in both passes.
    """

    def __init__(self, paths: list[Path], stride: int) -> None:
        if stride != PATCH_SIZE:
            raise ValueError(
                f"GLC FID/256 requires patch_stride={PATCH_SIZE}, got {stride}."
            )
        self.paths = tuple(Path(path).expanduser().resolve() for path in paths)
        self.stride = stride
        self.items: list[tuple[Path, int, int]] = []
        self.sizes: list[tuple[int, int]] = []
        for path in self.paths:
            with Image.open(path) as image:
                width, height = image.size
            self.sizes.append((width, height))
            self.items.extend((path, x, y) for x, y in _glc_patch_positions(width, height))
        self._cached_path: Path | None = None
        self._cached_image: Image.Image | None = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> torch.Tensor:
        path, x, y = self.items[index]
        if path != self._cached_path:
            with Image.open(path) as image:
                self._cached_image = image.convert("RGB")
            self._cached_path = path
        assert self._cached_image is not None
        patch = self._cached_image.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
        values = np.asarray(patch, dtype=np.uint8).copy()
        return torch.from_numpy(values).permute(2, 0, 1)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cached_path"] = None
        state["_cached_image"] = None
        return state


def _reference_cache_name(paths: list[Path], stride: int) -> str:
    """Build a stable torch-fidelity cache key for reference patch features."""

    digest = hashlib.sha256(f"jscc-glc-fid256-v1:{PATCH_SIZE}:{stride}".encode())
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        stat = resolved.stat()
        digest.update(str(resolved).encode())
        digest.update(f":{stat.st_size}:{stat.st_mtime_ns}".encode())
    return f"jscc-ref-{digest.hexdigest()}"


class QualityMetricRunner:
    """OSD-compatible uint8-quantized image metrics with lazy model loading."""

    def __init__(self, metrics: list[str], device: str = "auto") -> None:
        unknown = sorted(set(metrics) - set(ALL_METRICS))
        if unknown:
            raise ValueError(f"Unknown metrics: {', '.join(unknown)}")
        self.metrics = tuple(metrics)
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        )
        self._models: dict[str, Any] = {}
        self.skipped: dict[str, str] = {}

    def _pyiqa(self, name: str):
        if name not in self._models:
            import pyiqa

            self._models[name] = pyiqa.create_metric(name, device=self.device).eval()
        return self._models[name]

    @torch.inference_mode()
    def evaluate_pair(self, reconstruction_path: Path, reference_path: Path) -> dict[str, float | None]:
        reconstruction = _image_tensor(reconstruction_path).to(self.device)
        reference = _image_tensor(reference_path).to(self.device)
        if reconstruction.shape != reference.shape:
            raise ValueError(
                f"Metric image shapes differ: {reconstruction.shape} vs {reference.shape}."
            )
        values: dict[str, float | None] = {}
        if "psnr" in self.metrics:
            mse = (reconstruction - reference).square().mean().item()
            values["psnr"] = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
        if "ms_ssim" in self.metrics:
            if min(reference.shape[-2:]) < 161:
                values["ms_ssim"] = None
                self.skipped["ms_ssim"] = "MS-SSIM requires a minimum image side of 161 pixels."
            else:
                try:
                    from pytorch_msssim import ms_ssim

                    values["ms_ssim"] = float(
                        ms_ssim(reconstruction, reference, data_range=1.0, size_average=True).item()
                    )
                except Exception as exc:
                    values["ms_ssim"] = None
                    self.skipped["ms_ssim"] = str(exc)
        for name in ("lpips", "dists", "pieapp"):
            if name not in self.metrics:
                continue
            try:
                values[name] = float(self._pyiqa(name)(reconstruction, reference).mean().item())
            except Exception as exc:
                values[name] = None
                self.skipped[name] = str(exc)
                self._models.pop(name, None)
        return values

    def evaluate_distribution(
        self,
        reference_paths: list[Path],
        reconstruction_paths: list[Path],
        patch_stride: int = 256,
        batch_size: int = FID_BATCH_SIZE,
    ) -> dict[str, float | None]:
        selected = [name for name in DISTRIBUTION_METRICS if name in self.metrics]
        if not selected:
            return {}
        if len(reference_paths) != len(reconstruction_paths):
            raise ValueError("Reference and reconstruction image counts differ.")
        reference_dataset = ImagePatchDataset(reference_paths, patch_stride)
        reconstruction_dataset = ImagePatchDataset(reconstruction_paths, patch_stride)
        for index, (reference_size, reconstruction_size) in enumerate(
            zip(reference_dataset.sizes, reconstruction_dataset.sizes, strict=True)
        ):
            if reference_size != reconstruction_size:
                reason = (
                    f"Image sizes differ for image {index}: "
                    f"{reference_size} vs {reconstruction_size}."
                )
                self.skipped.update({name: reason for name in selected})
                return {name: None for name in selected}
        if len(reference_dataset) != len(reconstruction_dataset):
            reason = (
                "Patch counts differ: "
                f"{len(reference_dataset)} vs {len(reconstruction_dataset)}."
            )
            self.skipped.update({name: reason for name in selected})
            return {name: None for name in selected}
        if len(reference_dataset) < 2:
            reason = "FID/KID require at least two matched 256x256 patches."
            self.skipped.update({name: reason for name in selected})
            return {name: None for name in selected}
        try:
            from torch_fidelity import calculate_metrics

            if self.device.type == "cuda":
                # Pair metrics on large images can leave sizeable cached
                # allocations. Release unused blocks before the larger FID batch.
                torch.cuda.empty_cache()
            kwargs: dict[str, Any] = {
                "input1": reconstruction_dataset,
                "input2": reference_dataset,
                "input2_cache_name": _reference_cache_name(
                    reference_paths, patch_stride
                ),
                "cuda": self.device.type == "cuda",
                "fid": "fid" in selected,
                "kid": "kid" in selected,
                "batch_size": batch_size,
                "verbose": False,
            }
            if "kid" in selected:
                kwargs.update(
                    kid_subset_size=max(2, min(KID_SUBSET_SIZE, len(reference_dataset))),
                    kid_subsets=KID_SUBSETS,
                )
            with (
                torch.cuda.device(self.device)
                if self.device.type == "cuda"
                else nullcontext()
            ):
                raw = calculate_metrics(**kwargs)
            result: dict[str, float | None] = {}
            if "fid" in selected:
                result["fid"] = max(
                    0.0, float(raw["frechet_inception_distance"])
                )
            if "kid" in selected:
                result["kid"] = float(raw["kernel_inception_distance_mean"])
                result["kid_subset_std"] = float(
                    raw["kernel_inception_distance_std"]
                )
            return result
        except Exception as exc:
            self.skipped.update({name: str(exc) for name in selected})
            return {name: None for name in selected}


def _glc_patch_positions(width: int, height: int) -> list[tuple[int, int]]:
    positions = [
        (x, y)
        for y in range(0, height - PATCH_SIZE + 1, PATCH_SIZE)
        for x in range(0, width - PATCH_SIZE + 1, PATCH_SIZE)
    ]
    half_patch = PATCH_SIZE // 2
    positions.extend(
        (x, y)
        for y in range(half_patch, height - PATCH_SIZE + 1, PATCH_SIZE)
        for x in range(half_patch, width - PATCH_SIZE + 1, PATCH_SIZE)
    )
    return positions
