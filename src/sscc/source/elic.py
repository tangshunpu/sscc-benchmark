from __future__ import annotations

import os
import pickle
import re
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


QUALITY_LEVELS = ("0.004", "0.008", "0.016", "0.032", "0.15", "0.45")

QUALITY_ALIASES = {
    "0004": "0.004",
    "004": "0.004",
    "4": "0.004",
    "0_004": "0.004",
    "lambda0.004": "0.004",
    "0008": "0.008",
    "008": "0.008",
    "8": "0.008",
    "0_008": "0.008",
    "lambda0.008": "0.008",
    "0016": "0.016",
    "016": "0.016",
    "16": "0.016",
    "0_016": "0.016",
    "lambda0.016": "0.016",
    "0032": "0.032",
    "032": "0.032",
    "32": "0.032",
    "0_032": "0.032",
    "lambda0.032": "0.032",
    "0150": "0.15",
    "150": "0.15",
    "15": "0.15",
    "0_15": "0.15",
    "0.150": "0.15",
    "lambda0.15": "0.15",
    "0450": "0.45",
    "450": "0.45",
    "45": "0.45",
    "0_45": "0.45",
    "0.450": "0.45",
    "lambda0.45": "0.45",
}

CHECKPOINT_EXTENSIONS = (".pth.tar", ".pth", ".pt", ".ckpt")

_COMPRESSAI_PARAMETER_LIST_VERSION = (1, 2, 7)
_LEGACY_ENTROPY_KEY_RE = re.compile(
    r"^(?P<prefix>.*entropy_bottleneck)\._(?P<kind>matrix|bias|factor)(?P<index>\d+)$"
)
_CURRENT_ENTROPY_KEY_RE = re.compile(
    r"^(?P<prefix>.*entropy_bottleneck)\.(?P<kind>matrices|biases|factors)\.(?P<index>\d+)$"
)
_LEGACY_TO_CURRENT_ENTROPY_KIND = {
    "matrix": "matrices",
    "bias": "biases",
    "factor": "factors",
}
_CURRENT_TO_LEGACY_ENTROPY_KIND = {
    value: key for key, value in _LEGACY_TO_CURRENT_ENTROPY_KIND.items()
}


@dataclass(frozen=True)
class ELiCResult:
    source: Path
    bitstream: Path
    reconstruction: Path
    quality: str
    bytes: int
    bits: int
    encode_seconds: float
    decode_seconds: float
    total_seconds: float
    container_bytes: int


class ELiCCodec:
    """Wrapper around VincentChandelier/ELiC-ReImplemetation."""

    def __init__(
        self,
        device: str | None = None,
        elic_root: Path | None = None,
        checkpoint_dir: Path | None = None,
        checkpoints: Mapping[str, Path] | None = None,
        patch: int = 64,
        entropy_coder: str | None = None,
        half: bool = False,
    ) -> None:
        if patch < 1:
            raise ValueError(f"ELiC patch must be >= 1, got {patch}")
        self.device = device
        self.elic_root = resolve_elic_root(elic_root)
        self.checkpoint_dir = (
            checkpoint_dir.expanduser().resolve()
            if checkpoint_dir is not None
            else self.elic_root / "checkpoints"
        )
        self.checkpoints = {
            normalize_quality(quality): path.expanduser().resolve()
            for quality, path in (checkpoints or {}).items()
        }
        self.patch = patch
        self.entropy_coder = entropy_coder
        self.half = half

        self._torch: Any | None = None
        self._model: Any | None = None
        self._current_quality: str | None = None
        self._modules: dict[str, Any] | None = None

    def encode_decode(
        self,
        image_path: Path,
        quality: str,
        bitstream_path: Path,
        reconstruction_path: Path,
    ) -> ELiCResult:
        quality = normalize_quality(quality)
        bitstream_path.parent.mkdir(parents=True, exist_ok=True)
        reconstruction_path.parent.mkdir(parents=True, exist_ok=True)

        model = self._get_model(quality)
        torch = self._load_torch()
        tensor = self._load_image_tensor(image_path)
        x_padded, original_size = self._pad_tensor(tensor)

        encode_start = time.perf_counter()
        with torch.no_grad():
            encoded = model.compress(x_padded)
        payload_bytes = count_nested_bytes(encoded["strings"])
        self._save_bitstream(
            bitstream_path,
            strings=encoded["strings"],
            shape=encoded["shape"],
            payload_bytes=payload_bytes,
        )
        encode_seconds = time.perf_counter() - encode_start

        decode_start = time.perf_counter()
        compressed = load_elic_bitstream(bitstream_path)
        with torch.no_grad():
            decoded = model.decompress(compressed["strings"], compressed["shape"])
        self._save_reconstruction(decoded["x_hat"], reconstruction_path, original_size)
        decode_seconds = time.perf_counter() - decode_start

        return ELiCResult(
            source=image_path,
            bitstream=bitstream_path,
            reconstruction=reconstruction_path,
            quality=quality,
            bytes=payload_bytes,
            bits=payload_bytes * 8,
            encode_seconds=encode_seconds,
            decode_seconds=decode_seconds,
            total_seconds=encode_seconds + decode_seconds,
            container_bytes=bitstream_path.stat().st_size,
        )

    def clear_model_cache(self) -> None:
        self._model = None
        self._current_quality = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _get_model(self, quality: str) -> Any:
        quality = normalize_quality(quality)
        if self._model is not None and self._current_quality == quality:
            return self._model

        self.clear_model_cache()
        torch = self._load_torch()
        modules = self._load_modules()
        checkpoint_path = self.resolve_checkpoint(quality)
        state_dict = _load_checkpoint_state_dict(torch, checkpoint_path)

        model = modules["TestModel"]()
        state_dict = _convert_state_dict_for_model(model, state_dict)
        model.load_state_dict(state_dict)
        model.update(force=True)
        model.eval()
        model.to(self._device())
        if self.half:
            model.half()

        self._model = model
        self._current_quality = quality
        return model

    def resolve_checkpoint(self, quality: str) -> Path:
        quality = normalize_quality(quality)
        if quality in self.checkpoints:
            checkpoint = self.checkpoints[quality]
            if checkpoint.exists():
                return checkpoint
            raise FileNotFoundError(f"ELiC checkpoint not found for lambda {quality}: {checkpoint}")

        candidates = list(_checkpoint_name_candidates(self.checkpoint_dir, quality))
        for candidate in candidates:
            if candidate.exists():
                return candidate

        matches = sorted(
            path
            for path in self.checkpoint_dir.rglob("*")
            if path.is_file()
            and path.name.lower().endswith(CHECKPOINT_EXTENSIONS)
            and _quality_code(quality) in path.name
        ) if self.checkpoint_dir.exists() else []
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            match_list = ", ".join(str(path) for path in matches)
            raise FileNotFoundError(
                f"Multiple ELiC checkpoints match lambda {quality}: {match_list}. "
                f"Pass --checkpoint {quality}=PATH."
            )

        searched = ", ".join(str(path) for path in candidates[:8])
        raise FileNotFoundError(
            f"Could not find ELiC checkpoint for lambda {quality}. "
            f"Pass --checkpoint {quality}=PATH or place it under {self.checkpoint_dir}. "
            f"Example searched paths: {searched}"
        )

    def _load_modules(self) -> dict[str, Any]:
        if self._modules is not None:
            return self._modules

        self._ensure_import_path()
        import compressai
        from Network import TestModel

        if self.entropy_coder is not None:
            compressai.set_entropy_coder(self.entropy_coder)

        self._modules = {"TestModel": TestModel}
        return self._modules

    def _ensure_import_path(self) -> None:
        root = str(self.elic_root)
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)

    def _load_torch(self) -> Any:
        if self._torch is None:
            import torch

            torch.backends.cudnn.deterministic = True
            torch.set_num_threads(1)
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
        if self.half:
            tensor = tensor.half()
        return tensor.to(self._device())

    def _pad_tensor(self, tensor: Any) -> tuple[Any, tuple[int, int]]:
        torch = self._load_torch()
        height, width = tensor.size(2), tensor.size(3)
        patch = self.patch
        new_height = (height + patch - 1) // patch * patch
        new_width = (width + patch - 1) // patch * patch
        padding_right = new_width - width
        padding_bottom = new_height - height
        padded = torch.nn.functional.pad(
            tensor,
            (0, padding_right, 0, padding_bottom),
            mode="constant",
            value=0,
        )
        return padded, (height, width)

    def _save_reconstruction(self, reconstruction: Any, output_path: Path, size: tuple[int, int]) -> None:
        height, width = size
        tensor = reconstruction[..., :height, :width].squeeze(0).detach().clamp(0.0, 1.0).cpu()
        array = tensor.float().permute(1, 2, 0).numpy()
        image = Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="RGB")
        image.save(output_path)

    def _save_bitstream(
        self,
        bitstream_path: Path,
        *,
        strings: Any,
        shape: Any,
        payload_bytes: int,
    ) -> None:
        payload = {
            "format": "elic-pickle-v1",
            "strings": strings,
            "shape": shape,
            "payload_bytes": payload_bytes,
        }
        with bitstream_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_elic_bitstream(bitstream_path: Path) -> dict[str, Any]:
    with bitstream_path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or payload.get("format") != "elic-pickle-v1":
        raise ValueError(f"Unsupported ELiC bitstream container: {bitstream_path}")
    return payload


def elic_bitstream_payload_bytes(bitstream_path: Path) -> int:
    payload = load_elic_bitstream(bitstream_path)
    value = payload.get("payload_bytes")
    if isinstance(value, int):
        return value
    return count_nested_bytes(payload["strings"])


def count_nested_bytes(value: Any) -> int:
    if isinstance(value, bytes | bytearray):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, Mapping):
        return sum(count_nested_bytes(item) for item in value.values())
    if isinstance(value, list | tuple):
        return sum(count_nested_bytes(item) for item in value)
    raise TypeError(f"Cannot count bytes for ELiC bitstream item of type {type(value).__name__}")


def normalize_quality(quality: str | int | float) -> str:
    quality_key = str(quality).strip().lower().replace("-", "_").replace("/", "_")
    quality_key = quality_key.replace(" ", "")
    quality_key = QUALITY_ALIASES.get(quality_key, quality_key)
    try:
        quality_value = float(quality_key)
    except ValueError:
        valid = ", ".join([*QUALITY_LEVELS, *sorted(QUALITY_ALIASES)])
        raise ValueError(
            f"ELiC quality must be a positive numeric lambda or one of {valid}, got {quality!r}"
        ) from None
    if not np.isfinite(quality_value) or quality_value <= 0:
        raise ValueError(f"ELiC quality must be a positive numeric lambda, got {quality!r}")
    quality_key = f"{quality_value:g}"
    quality_key = QUALITY_ALIASES.get(quality_key, quality_key)
    return quality_key


def quality_sort_key(quality: str | int | float) -> float:
    return float(normalize_quality(quality))


def resolve_elic_root(elic_root: Path | None = None) -> Path:
    candidates: list[Path] = []
    if elic_root is not None:
        candidates.append(elic_root)

    env_root = os.environ.get("ELIC_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    project_root = Path.cwd()
    candidates.append(project_root / "third_party" / "ELiC")

    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "Network.py").exists() and (root / "ELICUtilis").is_dir():
            return root

    searched = ", ".join(str(path.expanduser()) for path in candidates)
    raise FileNotFoundError(
        "Could not find an ELiC source tree. Pass --elic-root, set ELIC_ROOT, "
        f"or place ELiC under third_party/ELiC. Searched: {searched}"
    )


def _checkpoint_name_candidates(checkpoint_dir: Path, quality: str) -> list[Path]:
    code = _quality_code(quality)
    dotless = quality.replace(".", "_")
    return [
        checkpoint_dir / f"ELIC_{code}_ft_3980_Plateau.pth.tar",
        checkpoint_dir / f"ELIC_{code}.pth.tar",
        checkpoint_dir / f"elic_{code}.pth.tar",
        checkpoint_dir / f"ELiC_{code}.pth.tar",
        checkpoint_dir / f"{code}.pth.tar",
        checkpoint_dir / f"lambda_{dotless}.pth.tar",
        checkpoint_dir / f"{quality}.pth.tar",
        checkpoint_dir / f"{dotless}.pth.tar",
        checkpoint_dir / quality / "checkpoint.pth.tar",
        checkpoint_dir / quality / f"ELIC_{code}.pth.tar",
        checkpoint_dir / code / "checkpoint.pth.tar",
        checkpoint_dir / code / f"ELIC_{code}.pth.tar",
    ]


def _quality_code(quality: str) -> str:
    return f"{int(round(float(normalize_quality(quality)) * 1000)):04d}"


def _load_checkpoint_state_dict(torch: Any, checkpoint_path: Path) -> Mapping[str, Any]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, Mapping):
        if "network" in checkpoint:
            return checkpoint["network"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def _convert_state_dict_for_model(model: Any, state_dict: Mapping[str, Any]) -> dict[str, Any]:
    model_keys = set(model.state_dict().keys())
    converted = dict(state_dict)
    model_has_module = next(iter(model_keys)).startswith("module.")
    state_has_module = next(iter(converted)).startswith("module.")
    if model_has_module and not state_has_module:
        converted = {f"module.{key}": value for key, value in converted.items()}
    elif state_has_module and not model_has_module:
        converted = {key.removeprefix("module."): value for key, value in converted.items()}

    uses_parameter_lists = _compressai_uses_parameter_lists(model_keys)
    return {
        _convert_entropy_bottleneck_key(key, model_keys, uses_parameter_lists): value
        for key, value in converted.items()
    }


def _convert_entropy_bottleneck_key(
    key: str,
    expected_keys: set[str],
    uses_parameter_lists: bool,
) -> str:
    legacy_match = _LEGACY_ENTROPY_KEY_RE.match(key)
    if uses_parameter_lists and legacy_match:
        converted = (
            f"{legacy_match.group('prefix')}."
            f"{_LEGACY_TO_CURRENT_ENTROPY_KIND[legacy_match.group('kind')]}."
            f"{legacy_match.group('index')}"
        )
        if converted in expected_keys:
            return converted

    current_match = _CURRENT_ENTROPY_KEY_RE.match(key)
    if not uses_parameter_lists and current_match:
        converted = (
            f"{current_match.group('prefix')}."
            f"_{_CURRENT_TO_LEGACY_ENTROPY_KIND[current_match.group('kind')]}"
            f"{current_match.group('index')}"
        )
        if converted in expected_keys:
            return converted

    return key


def _compressai_uses_parameter_lists(expected_keys: set[str]) -> bool:
    try:
        installed_version = _parse_version(version("compressai"))
    except PackageNotFoundError:
        installed_version = None
    if installed_version is not None:
        return installed_version >= _COMPRESSAI_PARAMETER_LIST_VERSION
    return any(".entropy_bottleneck.matrices." in key for key in expected_keys)


def _parse_version(value: str) -> tuple[int, int, int]:
    parts = []
    for part in re.split(r"[^\d]+", value):
        if part:
            parts.append(int(part))
        if len(parts) == 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])
