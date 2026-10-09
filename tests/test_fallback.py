from pathlib import Path

import numpy as np
from PIL import Image

from sscc.codecs.base import mean_colour_image


def test_mean_colour_fallback(tmp_path: Path):
    source = tmp_path / "source.png"
    values = np.array([[[0, 10, 20], [10, 20, 30]]], dtype=np.uint8)
    Image.fromarray(values, mode="RGB").save(source)
    result = np.asarray(mean_colour_image(source))
    assert result.shape == values.shape
    assert np.all(result == np.array([5, 15, 25], dtype=np.uint8))
