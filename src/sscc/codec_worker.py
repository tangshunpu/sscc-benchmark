"""Internal entry point; install this package in the learned codec environment too."""
import argparse
from pathlib import Path

from sscc.codecs.registry import create_decoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["decode"])
    parser.add_argument("--method", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--msillm-torch-hub-repo", type=Path)
    parser.add_argument("--elic-root", type=Path)
    parser.add_argument("--elic-checkpoint-dir", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality", required=True)
    args = parser.parse_args()
    decoder = create_decoder(args.method, device=args.device,
        msillm_torch_hub_repo=args.msillm_torch_hub_repo, elic_root=args.elic_root,
        elic_checkpoint_dir=args.elic_checkpoint_dir)
    decoder.decode(args.input.read_bytes(), args.reference, args.quality, args.output)


if __name__ == "__main__":
    main()
