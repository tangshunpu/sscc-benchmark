"""Reproducibility metadata kept alongside each benchmark run."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
import sys


def write_run_manifest(path, args, config, dataset):
    packages = {}
    for name in ("sscc-benchmark", "numpy", "pillow", "torch", "sionna", "compressai",
                 "pyiqa", "pytorch-msssim", "torch-fidelity"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    images = []
    for sample in dataset:
        with sample.reference_path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        streams = []
        for candidate in sample.candidates:
            with candidate.path.open("rb") as handle:
                stream_digest = hashlib.file_digest(handle, "sha256").hexdigest()
            streams.append({"quality": candidate.quality, "bytes": candidate.size_bytes,
                            "sha256": stream_digest, "path": str(candidate.path)})
        images.append({"image": str(sample.relative_path), "sha256": digest, "streams": streams})
    path.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "arguments": vars(args), "channel": asdict(config), "images": images,
        "fallback": "oracle per-channel reference mean; not transmitted",
        "snr_definition": "Es/N0 per complex symbol with unit average power"}, default=str, indent=2) + "\n")
