"""Run a reproducible multi-method dataset/SNR benchmark from a TOML file."""
import argparse
from pathlib import Path
import subprocess
import sys
import tomllib


ALLOWED = {"compression_ratio", "repeats", "seed", "device", "decoder_iterations", "max_images",
           "mcs_index", "mcs_snr_gap_db", "modulation_order", "ldpc_rate", "saved_repeats_per_image",
           "metrics", "snr_db", "skip_distribution", "fid_batch_size"}
CODECS = {"bpg_decoder", "vtm_decoder", "msillm_torch_hub_repo", "elic_root", "elic_checkpoint_dir", "codec_python"}


def commands(config):
    unknown = set(config) - {"methods", "bitstream_root", "results_root", "benchmark", "codecs", "datasets"}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    if not config.get("methods") or not config.get("datasets"):
        raise ValueError("Configuration requires non-empty methods and datasets")
    for method in config["methods"]:
        if method not in {"BPG", "VTM", "MSILLM", "ELiC"}:
            raise ValueError(f"Unknown method: {method}")
        for dataset in config["datasets"]:
            if set(dataset) != {"name", "roots"} or not dataset["roots"]:
                raise ValueError("Each dataset requires name and non-empty roots")
            cmd = [sys.executable, "-m", "sscc.cli.benchmark", "--method", method,
                   "--dataset", dataset["name"], "--bitstream-root", config.get("bitstream_root", "bitstreams"),
                   "--results-root", config.get("results_root", "results")]
            for root in dataset["roots"]:
                cmd.extend(["--dataset-root", root])
            for section, allowed in (("benchmark", ALLOWED), ("codecs", CODECS)):
                for key, value in config.get(section, {}).items():
                    if key not in allowed:
                        raise ValueError(f"Unknown {section} option: {key}")
                    if isinstance(value, bool):
                        if value:
                            cmd.append(f"--{key.replace('_', '-')}")
                    else:
                        cmd.append(f"--{key.replace('_', '-')}")
                        cmd.extend(str(v) for v in (value if isinstance(value, list) else [value]))
            yield cmd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    import shlex
    jobs = list(commands(config))  # Validate all jobs before launching any.
    for cmd in jobs:
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
