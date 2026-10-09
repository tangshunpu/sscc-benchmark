import csv
import json
import os
import shutil

import numpy as np
from PIL import Image
import pytest

from sscc import resolve_channel_config, transmit_bytes
from sscc.cli.prepare import main as prepare
from sscc.cli.benchmark import main as benchmark
from sscc.cli.suite import commands
from sscc.data import ImageBitstreamDataset


@pytest.mark.integration
def test_real_ldpc_round_trip_and_symbol_accounting():
    pytest.importorskip("sionna")
    payload = bytes(range(256)) * 2
    for modulation in (4, 16, 64):
        config = resolve_channel_config(40, modulation_order=modulation, ldpc_rate=0.5)
        result = transmit_bytes(payload, config, seed=9, device="cpu")
        assert result.recovered_bytes == payload
        assert result.ber == 0
        assert result.header_valid
        assert result.channel_symbols == config.transmission_shape(len(payload))[3]


@pytest.mark.integration
def test_channel_noise_seed_is_reproducible():
    pytest.importorskip("sionna")
    config = resolve_channel_config(-5, modulation_order=16, ldpc_rate=0.75)
    payload = bytes(range(256))
    first = transmit_bytes(payload, config, seed=31, device="cpu")
    second = transmit_bytes(payload, config, seed=31, device="cpu")
    different = transmit_bytes(payload, config, seed=32, device="cpu")
    assert first == second
    assert first.bit_errors > 0
    assert (first.bit_errors, first.recovered_bytes) != (different.bit_errors, different.recovered_bytes)


@pytest.mark.integration
@pytest.mark.parametrize("method", ["BPG", "VTM"])
def test_source_channel_benchmark_and_resume(tmp_path, method):
    extra_prepare, extra_benchmark = [], []
    if method == "BPG":
        if not shutil.which("bpgenc") or not shutil.which("bpgdec"):
            pytest.skip("BPG tools not installed")
    else:
        encoder, decoder, config = (os.getenv(name) for name in ("SSCC_VTM_ENCODER", "SSCC_VTM_DECODER", "SSCC_VTM_CONFIG"))
        if not all((encoder, decoder, config)):
            pytest.skip("Set SSCC_VTM_ENCODER/DECODER/CONFIG to test VTM")
        extra_prepare = ["--vtm-encoder", encoder, "--vtm-decoder", decoder, "--vtm-config", config]
        extra_benchmark = ["--vtm-decoder", decoder]
    pytest.importorskip("sionna")
    roots = [tmp_path / "mobile_test", tmp_path / "professional_test"]
    for i, root in enumerate(roots):
        root.mkdir()
        # Same filename across CLIC subsets must be unambiguous.
        image = np.random.default_rng(i).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(image).save(root / "image.png")
    root_args = [item for root in roots for item in ("--dataset-root", str(root))]
    streams = tmp_path / "streams"
    args = ["--method", method, "--dataset", "clic", *root_args,
            "--bitstream-root", str(streams), "--qualities", "45", *extra_prepare]
    prepare(args)
    dataset = ImageBitstreamDataset(roots, streams, method, "clic")
    assert len(dataset) == 2 and not dataset.unmatched_images
    stream = dataset[0].candidates[0].path
    original, mtime = stream.read_bytes(), stream.stat().st_mtime_ns
    prepare(args)
    assert stream.stat().st_mtime_ns == mtime
    stream.write_bytes(b"damaged cache")
    prepare(args)
    assert stream.read_bytes() == original
    result_root = tmp_path / "results"
    bench_args = ["--method", method, "--dataset", "clic", *root_args,
              "--bitstream-root", str(streams), "--results-root", str(result_root),
              "--snr-db", "40", "--compression-ratio", "1", "--repeats", "1",
              "--device", "cpu", "--metrics", "psnr", *extra_benchmark]
    benchmark(bench_args)
    rows = list(csv.DictReader(next(result_root.rglob("per_image_metrics.csv")).open()))
    assert len(rows) == 2
    assert all(row["decode_success"] == "1" and row["fallback_used"] == "0" for row in rows)
    assert all(int(row["channel_symbols_used"]) <= int(row["channel_symbols_budget"]) for row in rows)
    metadata = json.loads(next(result_root.rglob("run.json")).read_text())
    assert len(metadata["images"]) == 2
    with pytest.raises(FileExistsError):
        benchmark(bench_args)


def test_suite_emits_all_methods_datasets_without_shell_interpolation():
    config = {"methods": ["BPG", "VTM"], "datasets": [
        {"name": "kodak", "roots": ["data with spaces/kodak"]},
        {"name": "clic", "roots": ["mobile", "professional"]}],
        "benchmark": {"snr_db": [0, 20], "repeats": 2}}
    jobs = list(commands(config))
    assert len(jobs) == 4
    assert "data with spaces/kodak" in jobs[0]
    assert jobs[1].count("--dataset-root") == 2
    with pytest.raises(ValueError, match="Unknown benchmark"):
        list(commands(dict(config, benchmark={"typo": 1})))
