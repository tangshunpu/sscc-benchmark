from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


QUALITY_LEVELS = ("vlo1", "vlo2", "1", "2", "3", "4", "5", "6")

QUALITY_TARGET_BPP = {
    "vlo1": 0.00218,
    "vlo2": 0.00438,
    "1": 0.035,
    "2": 0.07,
    "3": 0.14,
    "4": 0.30,
    "5": 0.45,
    "6": 0.90,
}


@dataclass(frozen=True)
class MSILLMResult:
    source: Path
    bitstream: Path
    reconstruction: Path
    quality: str
    bytes: int
    bits: int
    encode_seconds: float
    decode_seconds: float
    total_seconds: float


class MSILLMCodec:
    """MS-ILLM image codec wrapper using the cached NeuralCompression torch hub repo."""

    def __init__(
        self,
        device: str | None = None,
        torch_hub_repo: Path | None = None,
    ) -> None:
        self._torch: Any | None = None
        self._models: dict[str, Any] = {}
        self.device = device
        self.torch_hub_repo = torch_hub_repo or Path(
            "~/.cache/torch/hub/facebookresearch_NeuralCompression_main"
        ).expanduser()

    def encode_decode(
        self,
        image_path: Path,
        quality: str,
        bitstream_path: Path,
        reconstruction_path: Path,
    ) -> MSILLMResult:
        quality = normalize_quality(quality)
        bitstream_path.parent.mkdir(parents=True, exist_ok=True)
        reconstruction_path.parent.mkdir(parents=True, exist_ok=True)

        model = self._get_model(quality)
        tensor = self._load_image_tensor(image_path)

        encode_start = time.perf_counter()
        with self._torch.no_grad():
            compressed = model.compress(tensor, force_cpu=False)
        with bitstream_path.open("wb") as f:
            pickle.dump(compressed, f, protocol=pickle.HIGHEST_PROTOCOL)
        encode_seconds = time.perf_counter() - encode_start

        decode_start = time.perf_counter()
        with bitstream_path.open("rb") as f:
            compressed_from_file = pickle.load(f)
        with self._torch.no_grad():
            reconstruction = model.decompress(compressed_from_file, force_cpu=False)
        self._save_reconstruction(reconstruction, reconstruction_path)
        decode_seconds = time.perf_counter() - decode_start

        num_bytes = bitstream_path.stat().st_size
        return MSILLMResult(
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

    def _get_model(self, quality: str) -> Any:
        quality = normalize_quality(quality)
        if quality in self._models:
            return self._models[quality]

        torch = self._load_torch()
        model_name = f"msillm_quality_{quality}"
        if self.torch_hub_repo.exists():
            model = torch.hub.load(
                str(self.torch_hub_repo),
                model_name,
                pretrained=True,
                source="local",
                skip_validation=True,
            )
        else:
            model = torch.hub.load(
                "facebookresearch/NeuralCompression",
                model_name,
                pretrained=True,
                skip_validation=True,
            )

        model = model.eval().to(self._device())
        model.update()
        model.update_tensor_devices("compress")
        self._models[quality] = model
        return model

    def clear_model_cache(self) -> None:
        self._models.clear()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _load_torch(self) -> Any:
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def _device(self) -> str:
        if self.device:
            return self.device
        torch = self._load_torch()
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_image_tensor(self, image_path: Path) -> Any:
        torch = self._load_torch()
        with Image.open(image_path) as img:
            array = np.array(img.convert("RGB"), dtype=np.float32, copy=True) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self._device())

    def _save_reconstruction(self, reconstruction: Any, output_path: Path) -> None:
        tensor = reconstruction.squeeze(0).detach().clamp(0.0, 1.0).cpu()
        array = tensor.permute(1, 2, 0).numpy()
        image = Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="RGB")
        image.save(output_path)


def normalize_quality(quality: str | int) -> str:
    quality_key = str(quality).strip().lower()
    if quality_key not in QUALITY_LEVELS:
        raise ValueError(
            f"MS-ILLM quality must be one of {', '.join(QUALITY_LEVELS)}, got {quality!r}"
        )
    return quality_key


def quality_sort_key(quality: str | int) -> int:
    return QUALITY_LEVELS.index(normalize_quality(quality))
