from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VTMResult:
    source: Path
    bitstream: Path
    reconstruction: Path
    quality: int
    bytes: int
    bits: int
    encode_seconds: float
    decode_seconds: float
    total_seconds: float


class VTMCodec:
    """Thin wrapper around VTM EncoderApp and DecoderApp for single-image coding."""

    def __init__(
        self,
        encoder: str = "EncoderApp",
        decoder: str = "DecoderApp",
        config: str = "third_party/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg",
        chroma_format: str = "444",
        internal_bit_depth: int = 10,
        profile: str = "main_10_444_still_picture",
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.chroma_format = chroma_format
        self.internal_bit_depth = internal_bit_depth
        self.profile = profile
        self._check_path_or_binary(self.encoder)
        self._check_path_or_binary(self.decoder)
        if not Path(self.config).expanduser().exists():
            raise FileNotFoundError(f"VTM config not found: {self.config}")
        if self.chroma_format != "444":
            raise ValueError("Only 4:4:4 VTM image coding is currently supported.")

    @staticmethod
    def _check_path_or_binary(name: str) -> None:
        path = Path(name).expanduser()
        if path.exists():
            return
        if shutil.which(name) is None:
            raise FileNotFoundError(f"Required VTM binary '{name}' was not found.")

    def encode_decode(
        self,
        image_path: Path,
        quality: int,
        bitstream_path: Path,
        reconstruction_path: Path,
    ) -> VTMResult:
        bitstream_path.parent.mkdir(parents=True, exist_ok=True)
        reconstruction_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
        width, height = rgb.size

        with tempfile.TemporaryDirectory(prefix="vtm_image_") as tmp:
            tmp_dir = Path(tmp)
            input_yuv = tmp_dir / "input_444.yuv"
            encoder_recon_yuv = tmp_dir / "encoder_recon_444.yuv"
            decoded_yuv = tmp_dir / "decoded_444.yuv"
            self._write_ycbcr444(rgb, input_yuv)

            enc_cmd = [
                self.encoder,
                "-c",
                self.config,
                "-i",
                str(input_yuv),
                "-b",
                str(bitstream_path),
                "-o",
                str(encoder_recon_yuv),
                "-wdt",
                str(width),
                "-hgt",
                str(height),
                "-f",
                "1",
                "-fr",
                "1",
                "-q",
                str(quality),
                f"--Profile={self.profile}",
                "--InputBitDepth=8",
                "--OutputBitDepth=8",
                f"--InternalBitDepth={self.internal_bit_depth}",
                "--InputChromaFormat=444",
                "--ChromaFormatIDC=444",
                "--TemporalSubsampleRatio=1",
                "--ConformanceWindowMode=1",
                "--PrintFrameMSE=0",
                "--PrintSequenceMSE=0",
            ]
            encode_start = time.perf_counter()
            self._run(enc_cmd)
            encode_seconds = time.perf_counter() - encode_start

            dec_cmd = [
                self.decoder,
                "-b",
                str(bitstream_path),
                "-o",
                str(decoded_yuv),
                "--OutputBitDepth=8",
            ]
            decode_start = time.perf_counter()
            self._run(dec_cmd)
            decode_seconds = time.perf_counter() - decode_start
            self._read_ycbcr444(decoded_yuv, width, height).save(reconstruction_path)

        num_bytes = bitstream_path.stat().st_size
        return VTMResult(
            source=image_path,
            bitstream=bitstream_path,
            reconstruction=reconstruction_path,
            quality=quality,
            bytes=num_bytes,
            bits=num_bytes * 8,
            encode_seconds=encode_seconds,
            decode_seconds=decode_seconds,
            total_seconds=encode_seconds + decode_seconds,
        )

    @staticmethod
    def _write_ycbcr444(rgb: Image.Image, path: Path) -> None:
        ycbcr = rgb.convert("YCbCr")
        y, cb, cr = ycbcr.split()
        with path.open("wb") as f:
            f.write(np.asarray(y, dtype=np.uint8).tobytes())
            f.write(np.asarray(cb, dtype=np.uint8).tobytes())
            f.write(np.asarray(cr, dtype=np.uint8).tobytes())

    @staticmethod
    def _read_ycbcr444(path: Path, width: int, height: int) -> Image.Image:
        frame_size = width * height
        data = np.fromfile(path, dtype=np.uint8)
        expected_size = frame_size * 3
        if data.size < expected_size:
            raise RuntimeError(
                f"Decoded YUV is too small: got {data.size} bytes, expected {expected_size}."
            )
        y = data[0:frame_size].reshape((height, width))
        cb = data[frame_size : frame_size * 2].reshape((height, width))
        cr = data[frame_size * 2 : frame_size * 3].reshape((height, width))
        return Image.merge(
            "YCbCr",
            [
                Image.fromarray(y, mode="L"),
                Image.fromarray(cb, mode="L"),
                Image.fromarray(cr, mode="L"),
            ],
        ).convert("RGB")

    @staticmethod
    def _run(cmd: list[str]) -> None:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=7200)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            detail = stderr or stdout or "no process output"
            raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{detail}")
