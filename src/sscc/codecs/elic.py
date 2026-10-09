from pathlib import Path
import pickle

from sscc.source.elic import ELiCCodec, normalize_quality


class ELiCDecoder:
    """Decode with the actual entropy model, without reconstruction-cache shortcuts."""

    def __init__(self, device="auto", elic_root=None, checkpoint_dir=None):
        self.codec = ELiCCodec(
            device=None if device == "auto" else device,
            elic_root=elic_root, checkpoint_dir=checkpoint_dir,
        )

    @staticmethod
    def normalize_quality(quality):
        value = quality.removeprefix("q").replace("_", ".")
        return normalize_quality(value)

    def decode(self, payload: bytes, reference_path: Path, quality: str, output_path: Path):
        from PIL import Image
        compressed = pickle.loads(payload)  # Only trusted, locally generated experiment streams.
        if not isinstance(compressed, dict) or compressed.get("format") != "elic-pickle-v1":
            raise ValueError("Unsupported ELiC bitstream container")
        model = self.codec._get_model(self.normalize_quality(quality))
        torch = self.codec._load_torch()
        with torch.no_grad():
            decoded = model.decompress(compressed["strings"], compressed["shape"])
        with Image.open(reference_path) as image:
            width, height = image.size
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.codec._save_reconstruction(decoded["x_hat"], output_path, (height, width))
