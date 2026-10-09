"""File-based process bridge for incompatible channel/learned-codec environments."""
from pathlib import Path
import subprocess
import tempfile


def worker_options(method, device, msillm_repo, elic_root, checkpoint_dir):
    args = ["--method", method, "--device", device]
    for flag, value in (("--msillm-torch-hub-repo", msillm_repo),
                        ("--elic-root", elic_root), ("--elic-checkpoint-dir", checkpoint_dir)):
        if value is not None:
            args.extend([flag, str(Path(value).expanduser().resolve())])
    return args


def run_worker(python, args):
    result = subprocess.run([str(python), "-m", "sscc.codec_worker", *args],
                            capture_output=True, text=True, timeout=7200)
    if result.returncode:
        raise RuntimeError(f"Codec worker failed: {result.stderr[-4000:]}")


class ProcessDecoder:
    def __init__(self, python, method, device, msillm_repo, elic_root, checkpoint_dir):
        self.python = python
        self.options = worker_options(method, device, msillm_repo, elic_root, checkpoint_dir)

    def decode(self, payload, reference_path, quality, output_path):
        with tempfile.TemporaryDirectory(prefix="sscc_codec_") as tmp:
            stream = Path(tmp) / "received.bin"
            stream.write_bytes(payload)
            run_worker(self.python, ["decode", *self.options, "--input", str(stream),
                       "--reference", str(reference_path.resolve()), "--quality", quality,
                       "--output", str(output_path.resolve())])
