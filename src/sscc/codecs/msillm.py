from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import numpy as np
from PIL import Image


QUALITY_LEVELS = ("vlo1", "vlo2", "1", "2", "3", "4", "5", "6")


class MSILLMDecoder:
    def __init__(self, device: str = "auto", torch_hub_repo: Path | None = None) -> None:
        import torch

        self.torch = torch
        self.device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        )
        self.torch_hub_repo = torch_hub_repo or Path(
            "~/.cache/torch/hub/facebookresearch_NeuralCompression_main"
        ).expanduser()
        self.models: dict[str, Any] = {}

    @staticmethod
    def normalize_quality(quality: str) -> str:
        value = quality.strip().lower()
        if value.startswith("q"):
            value = value[1:]
        if value not in QUALITY_LEVELS:
            raise ValueError(f"Unknown MSILLM quality: {quality}")
        return value

    def _model(self, quality: str):
        quality = self.normalize_quality(quality)
        if quality in self.models:
            return self.models[quality]
        model_name = f"msillm_quality_{quality}"
        if self.torch_hub_repo.exists():
            model = self.torch.hub.load(
                str(self.torch_hub_repo), model_name, pretrained=True,
                source="local", skip_validation=True,
            )
        else:
            model = self.torch.hub.load(
                "facebookresearch/NeuralCompression", model_name,
                pretrained=True, skip_validation=True,
            )
        model = model.eval().to(self.device)
        model.update()
        model.update_tensor_devices("compress")
        self.models[quality] = model
        return model

    def decode(self, payload: bytes, reference_path: Path, quality: str, output_path: Path) -> None:
        # Bitstreams are trusted local experiment artifacts; never unpickle untrusted input.
        # Load the cached repository first so pickle resolves neuralcompression
        # classes against the same implementation that created the bitstream.
        model = self._model(quality)
        compressed = pickle.loads(payload)
        with self.torch.no_grad():
            reconstruction = model.decompress(compressed, force_cpu=False)
        tensor = reconstruction.squeeze(0).detach().clamp(0.0, 1.0).cpu()
        array = np.round(tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        image = Image.fromarray(array, mode="RGB")
        with Image.open(reference_path) as reference:
            if image.size != reference.size:
                image = image.resize(reference.size, Image.Resampling.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
