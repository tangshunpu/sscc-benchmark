from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image

from .base import ensure_reference_size


class BPGDecoder:
    def __init__(self, decoder: str = "bpgdec") -> None:
        self.decoder = decoder
        if shutil.which(decoder) is None and not Path(decoder).exists():
            raise FileNotFoundError(f"BPG decoder not found: {decoder}")

    def decode(self, payload: bytes, reference_path: Path, quality: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="jscc_bpg_") as tmp:
            bitstream = Path(tmp) / "received.bpg"
            reconstruction = Path(tmp) / "reconstruction.png"
            bitstream.write_bytes(payload)
            result = subprocess.run(
                [self.decoder, "-o", str(reconstruction), str(bitstream)],
                capture_output=True,
                text=True,
                check=False,
                timeout=300,
            )
            if result.returncode != 0 or not reconstruction.is_file():
                detail = result.stderr.strip() or result.stdout.strip() or "no decoder output"
                raise RuntimeError(f"BPG decoding failed: {detail}")
            with Image.open(reconstruction) as image:
                ensure_reference_size(image.copy(), reference_path).save(output_path)
