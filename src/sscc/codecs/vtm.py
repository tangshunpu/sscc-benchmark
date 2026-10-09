from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image


DEFAULT_VTM_DECODER = "DecoderApp"


class VTMDecoder:
    def __init__(self, decoder: str = DEFAULT_VTM_DECODER) -> None:
        self.decoder = decoder
        if not Path(decoder).expanduser().exists() and shutil.which(decoder) is None:
            raise FileNotFoundError(f"VTM decoder not found: {decoder}")

    def decode(self, payload: bytes, reference_path: Path, quality: str, output_path: Path) -> None:
        with Image.open(reference_path) as reference:
            width, height = reference.size
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="jscc_vtm_") as tmp:
            bitstream = Path(tmp) / "received.266"
            decoded_yuv = Path(tmp) / "decoded_444.yuv"
            bitstream.write_bytes(payload)
            result = subprocess.run(
                [self.decoder, "-b", str(bitstream), "-o", str(decoded_yuv), "--OutputBitDepth=8"],
                capture_output=True,
                text=True,
                check=False,
                timeout=300,
            )
            if result.returncode != 0 or not decoded_yuv.is_file():
                detail = result.stderr.strip() or result.stdout.strip() or "no decoder output"
                raise RuntimeError(f"VTM decoding failed: {detail}")
            self._read_ycbcr444(decoded_yuv, width, height).save(output_path)

    @staticmethod
    def _read_ycbcr444(path: Path, width: int, height: int) -> Image.Image:
        frame_size = width * height
        values = np.fromfile(path, dtype=np.uint8)
        if values.size != frame_size * 3:
            raise RuntimeError(
                f"Expected exactly one 8-bit YCbCr 4:4:4 frame: {values.size} < {frame_size * 3} bytes."
            )
        planes = [
            Image.fromarray(values[offset * frame_size : (offset + 1) * frame_size].reshape(height, width), "L")
            for offset in range(3)
        ]
        return Image.merge("YCbCr", planes).convert("RGB")
