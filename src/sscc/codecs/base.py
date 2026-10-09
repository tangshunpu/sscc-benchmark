from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image


class DecoderAdapter(Protocol):
    def decode(
        self,
        payload: bytes,
        reference_path: Path,
        quality: str,
        output_path: Path,
    ) -> None:
        """Decode recovered bytes and write an RGB reconstruction."""


def mean_colour_image(reference_path: Path) -> Image.Image:
    with Image.open(reference_path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    means = np.round(array.mean(axis=(0, 1))).astype(np.uint8)
    output = np.empty_like(array)
    output[...] = means
    return Image.fromarray(output, mode="RGB")


def ensure_reference_size(image: Image.Image, reference_path: Path) -> Image.Image:
    with Image.open(reference_path) as reference:
        expected = reference.size
    image = image.convert("RGB")
    if image.size != expected:
        image = image.resize(expected, Image.Resampling.LANCZOS)
    return image
