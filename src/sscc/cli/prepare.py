"""Encode a dataset at several quality points, with resumable content manifests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from PIL import Image
from tqdm import tqdm

from sscc.data import ImageDirectoryDataset, ImageBitstreamDataset


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def create_encoder(args):
    if args.method == "BPG":
        from sscc.source.bpg import BPGCodec
        return BPGCodec(args.bpg_encoder, args.bpg_decoder, encoder_preset=args.bpg_preset)
    if args.method == "VTM":
        from sscc.source.vtm import VTMCodec
        return VTMCodec(args.vtm_encoder, args.vtm_decoder, str(args.vtm_config))
    if args.method == "MSILLM":
        from sscc.source.msillm import MSILLMCodec
        return MSILLMCodec(None if args.device == "auto" else args.device, args.msillm_torch_hub_repo)
    from sscc.source.elic import ELiCCodec
    return ELiCCodec(device=None if args.device == "auto" else args.device,
                     elic_root=args.elic_root, checkpoint_dir=args.elic_checkpoint_dir)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["BPG", "VTM", "MSILLM", "ELiC"], required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--dataset-root", type=Path, action="append", required=True)
    p.add_argument("--bitstream-root", type=Path, default=Path("bitstreams"))
    p.add_argument("--qualities", nargs="+", help="QP values or learned-model quality/lambda values.")
    p.add_argument("--max-images", type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--force", action="store_true", help="Re-encode even with a matching manifest.")
    p.add_argument("--bpg-encoder", default="bpgenc")
    p.add_argument("--bpg-decoder", default="bpgdec")
    p.add_argument("--bpg-preset", type=int, choices=range(1, 10), default=8)
    p.add_argument("--vtm-encoder", default="EncoderApp")
    p.add_argument("--vtm-decoder", default="DecoderApp")
    p.add_argument("--vtm-config", type=Path,
                   default=Path("third_party/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg"))
    p.add_argument("--msillm-torch-hub-repo", type=Path)
    p.add_argument("--elic-root", type=Path)
    p.add_argument("--elic-checkpoint-dir", type=Path)
    args = p.parse_args(argv)
    if not args.dataset or Path(args.dataset).name != args.dataset or args.dataset in {".", ".."}:
        p.error("--dataset must be a single directory name")
    defaults = {"BPG": [24, 28, 32, 36, 40, 44, 48, 51],
                "VTM": [22, 27, 32, 37, 42, 47, 52, 57, 63],
                "MSILLM": ["1", "2", "3", "4", "5", "6"],
                "ELiC": ["0.004", "0.008", "0.016", "0.032", "0.15", "0.45"]}
    args.qualities = args.qualities or defaults[args.method]
    if args.method in {"BPG", "VTM"}:
        maximum = 51 if args.method == "BPG" else 63
        try:
            args.qualities = [int(q) for q in args.qualities]
            if any(q < 0 or q > maximum for q in args.qualities):
                raise ValueError()
        except ValueError:
            p.error(f"{args.method} qualities must be integer QPs in 0..{maximum}")
    elif args.method == "MSILLM":
        from sscc.source.msillm import normalize_quality
        args.qualities = [normalize_quality(q) for q in args.qualities]
    else:
        from sscc.source.elic import normalize_quality
        args.qualities = [normalize_quality(q) for q in args.qualities]
    return args


def main(argv=None):
    args = parse_args(argv)
    dataset = ImageDirectoryDataset(args.dataset_root, args.max_images)
    keys = [ImageBitstreamDataset._safe_stem(item.relative_path) for item in dataset]
    if len(set(k.casefold() for k in keys)) != len(keys):
        raise ValueError("Image identifiers collide after normalization; rename images or dataset roots.")
    codec = create_encoder(args)
    extension = {"BPG": ".bpg", "VTM": ".266", "MSILLM": ".pkl", "ELiC": ".elic"}[args.method]
    root = args.bitstream_root.expanduser().resolve()
    settings = {k: str(v) for k, v in vars(args).items()
                if k not in {"force", "max_images", "qualities", "dataset_root", "bitstream_root"}}
    # Track encoder binaries/configs and model checkpoints, not just their paths.
    import shutil
    for name in ("bpg_encoder", "bpg_decoder", "vtm_encoder", "vtm_decoder", "vtm_config"):
        value = str(getattr(args, name))
        candidate = Path(shutil.which(value) or value).expanduser()
        if candidate.is_file():
            settings[f"{name}_sha256"] = sha256(candidate)
    if args.elic_checkpoint_dir and args.elic_checkpoint_dir.is_dir():
        settings["checkpoint_hashes"] = {p.name: sha256(p) for p in sorted(args.elic_checkpoint_dir.glob("*.tar"))}
    for quality in args.qualities:
        quality_dir = f"q{quality}".replace(".", "_")
        for sample, key in tqdm(list(zip(dataset, keys)), desc=f"{args.method} {quality_dir}"):
            output = root / args.method / args.dataset / quality_dir / f"{key}{extension}"
            manifest = root / "_manifests" / args.method / args.dataset / quality_dir / f"{key}.json"
            provenance = {"input_sha256": sha256(sample.reference_path), "quality": str(quality),
                          "settings": settings}
            if (args.method in {"BPG", "VTM"} and output.is_file()
                    and manifest.is_file() and not args.force):
                previous = json.loads(manifest.read_text())
                if all(previous.get(k) == v for k, v in provenance.items()) and previous.get("stream_sha256") == sha256(output):
                    continue
            output.parent.mkdir(parents=True, exist_ok=True)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".encode-", dir=output.parent) as temporary:
                tmp = Path(temporary)
                # Normalize every input to RGB PNG; bpgenc only accepts PNG/JPEG.
                with Image.open(sample.reference_path) as image:
                    image.convert("RGB").save(tmp / "source.png")
                result = codec.encode_decode(tmp / "source.png", quality, tmp / f"stream{extension}", tmp / "recon.png")
                if not result.bitstream.stat().st_size:
                    raise RuntimeError("Encoder produced an empty bitstream")
                with Image.open(tmp / "recon.png") as recon, Image.open(sample.reference_path) as reference:
                    if recon.size != reference.size:
                        raise RuntimeError("Source codec reconstruction size differs from input")
                provenance.update(stream_sha256=sha256(result.bitstream), size_bytes=result.bitstream.stat().st_size,
                                  encode_seconds=result.encode_seconds, decode_seconds=result.decode_seconds)
                os.replace(result.bitstream, output)
                staged = manifest.with_suffix(".json.tmp")
                staged.write_text(json.dumps(provenance, indent=2) + "\n")
                os.replace(staged, manifest)
    print(f"Prepared {args.method}/{args.dataset}: {len(dataset)} images, {len(args.qualities)} qualities in {root}")


if __name__ == "__main__":
    main()
