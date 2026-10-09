from pathlib import Path

import numpy as np
from PIL import Image
import torch

from sscc.metrics.quality import (
    FID_BATCH_SIZE,
    ImagePatchDataset,
    KID_SUBSETS,
    KID_SUBSET_SIZE,
    QualityMetricRunner,
    _glc_patch_positions,
    _reference_cache_name,
)


def _gradient_image(path: Path, width: int = 300, height: int = 300) -> None:
    y, x = np.mgrid[:height, :width]
    values = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
    Image.fromarray(values, mode="RGB").save(path)


def test_patch_dataset_matches_direct_pil_crop(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    _gradient_image(path)
    dataset = ImagePatchDataset([path], stride=256)
    assert len(dataset) == 1
    with Image.open(path) as image:
        expected = torch.from_numpy(
            np.asarray(image.convert("RGB").crop((0, 0, 256, 256))).copy()
        ).permute(2, 0, 1)
    assert torch.equal(dataset[0], expected)


def test_glc_patch_positions_use_origin_and_diagonal_half_patch_shift() -> None:
    positions = _glc_patch_positions(width=768, height=512)
    assert positions == [
        (0, 0),
        (256, 0),
        (512, 0),
        (0, 256),
        (256, 256),
        (512, 256),
        (128, 128),
        (384, 128),
    ]


def test_glc_kodak_patch_count_is_192(tmp_path: Path) -> None:
    paths = []
    for index in range(24):
        path = tmp_path / f"kodim{index:02d}.png"
        width, height = (768, 512) if index % 2 == 0 else (512, 768)
        Image.new("RGB", (width, height)).save(path)
        paths.append(path)
    assert len(ImagePatchDataset(paths, stride=256)) == 192


def test_reference_cache_name_is_stable_and_stride_specific(tmp_path: Path) -> None:
    path = tmp_path / "reference.png"
    _gradient_image(path)
    first = _reference_cache_name([path], 256)
    assert first == _reference_cache_name([path], 256)
    assert first != _reference_cache_name([path], 128)


def test_distribution_metrics_use_online_datasets_cache_and_large_batch(
    tmp_path: Path, monkeypatch
) -> None:
    reference = tmp_path / "reference.png"
    reconstruction = tmp_path / "reconstruction.png"
    _gradient_image(reference, width=768, height=512)
    _gradient_image(reconstruction, width=768, height=512)
    calls = []

    def fake_calculate_metrics(**kwargs):
        calls.append(kwargs)
        return {
            "frechet_inception_distance": 1.25,
            "kernel_inception_distance_mean": 0.02,
            "kernel_inception_distance_std": 0.003,
        }

    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, "torch_fidelity", SimpleNamespace(calculate_metrics=fake_calculate_metrics))
    runner = QualityMetricRunner(["fid", "kid"], device="cpu")
    result = runner.evaluate_distribution([reference], [reconstruction])

    assert result == {"fid": 1.25, "kid": 0.02, "kid_subset_std": 0.003}
    assert len(calls) == 1
    call = calls[0]
    assert isinstance(call["input1"], ImagePatchDataset)
    assert isinstance(call["input2"], ImagePatchDataset)
    assert call["batch_size"] == FID_BATCH_SIZE == 128
    assert call["kid_subset_size"] == min(KID_SUBSET_SIZE, 8)
    assert call["kid_subsets"] == KID_SUBSETS == 100
    assert call["input2_cache_name"] == _reference_cache_name([reference], 256)
    assert not list(tmp_path.glob("jscc_*_patches_*"))
