from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BPGResult:
    source: Path
    bitstream: Path
    reconstruction: Path
    quality: int
    bytes: int
    bits: int
    encode_seconds: float
    decode_seconds: float
    total_seconds: float


class BPGCodec:
    """Thin wrapper around the reference `bpgenc` and `bpgdec` binaries."""

    def __init__(
        self,
        encoder: str = "bpgenc",
        decoder: str = "bpgdec",
        chroma_format: str = "444",
        encoder_preset: int = 8,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.chroma_format = chroma_format
        self.encoder_preset = encoder_preset
        self._check_binary(self.encoder)
        self._check_binary(self.decoder)

    @staticmethod
    def _check_binary(name: str) -> None:
        if shutil.which(name) is None:
            raise FileNotFoundError(
                f"Required BPG binary '{name}' was not found on PATH."
            )

    def encode_decode(
        self,
        image_path: Path,
        quality: int,
        bitstream_path: Path,
        reconstruction_path: Path,
    ) -> BPGResult:
        bitstream_path.parent.mkdir(parents=True, exist_ok=True)
        reconstruction_path.parent.mkdir(parents=True, exist_ok=True)

        enc_cmd = [
            self.encoder,
            "-q",
            str(quality),
            "-f",
            self.chroma_format,
            "-m",
            str(self.encoder_preset),
            "-o",
            str(bitstream_path),
            str(image_path),
        ]
        encode_start = time.perf_counter()
        self._run(enc_cmd)
        encode_seconds = time.perf_counter() - encode_start

        dec_cmd = [
            self.decoder,
            "-o",
            str(reconstruction_path),
            str(bitstream_path),
        ]
        decode_start = time.perf_counter()
        self._run(dec_cmd)
        decode_seconds = time.perf_counter() - decode_start

        num_bytes = bitstream_path.stat().st_size
        return BPGResult(
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
    def _run(cmd: list[str]) -> None:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=7200)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            detail = stderr or stdout or "no process output"
            raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{detail}")
