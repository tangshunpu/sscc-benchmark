from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import time

from PIL import Image
from tqdm import tqdm

from sscc.codecs import create_decoder, mean_colour_image
from sscc.codecs.vtm import DEFAULT_VTM_DECODER
from sscc.channel.mcs import resolve_channel_config
from sscc.channel.transport import TransmissionResult, transmit_bytes
from sscc.data import ImageBitstreamDataset, select_largest_fitting
from sscc.metrics import (
    ALL_METRICS,
    DISTRIBUTION_METRICS,
    FID_BATCH_SIZE,
    PAIR_METRICS,
    QualityMetricRunner,
)


DEFAULT_BITSTREAM_ROOT = Path("bitstreams")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transmit pre-generated image bitstreams through 5G LDPC/QAM/AWGN."
    )
    parser.add_argument(
        "--method", required=True, choices=("BPG", "VTM", "MSILLM", "ELiC", "ELIC")
    )
    parser.add_argument("--dataset", required=True, help="Bitstream dataset label, e.g. kodak.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        action="append",
        required=True,
        help="Image root. Repeat this option to combine roots, e.g. both CLIC test subsets.",
    )
    parser.add_argument("--bitstream-root", type=Path, default=DEFAULT_BITSTREAM_ROOT)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--snr-db",
        type=float,
        nargs="+",
        help="Explicit SNR points. By default, use 0..20 dB at 3 dB intervals.",
    )
    parser.add_argument("--snr-start", type=float, help="SNR range start (default: 0 dB).")
    parser.add_argument("--snr-stop", type=float, help="Inclusive SNR range end (default: 20 dB).")
    parser.add_argument(
        "--snr-interval",
        type=float,
        help="SNR range interval (default: 3 dB). The stop endpoint is always included.",
    )
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=96.0,
        help="CR in the channel-use budget 3*H*W/CR (default: 96).",
    )
    parser.add_argument(
        "--ldpc-rate",
        type=float,
        help="Override automatic MCS LDPC rate, e.g. 0.5 or 0.75.",
    )
    parser.add_argument(
        "--modulation-order",
        type=int,
        choices=(4, 16, 64),
        help="Override automatic MCS with QPSK/16-QAM/64-QAM.",
    )
    parser.add_argument(
        "--mcs-index",
        type=int,
        choices=range(3, 29),
        metavar="{3..28}",
        help="Use one exact supported row from 3GPP TS 38.214 MCS Table 1.",
    )
    parser.add_argument(
        "--mcs-snr-gap-db",
        type=float,
        default=3.0,
        help="Implementation/link margin for automatic SNR-to-CQI selection (default: 3 dB).",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--decoder-iterations", type=int, default=20)
    parser.add_argument("--max-images", type=int)
    parser.add_argument(
        "--saved-repeats-per-image",
        type=int,
        default=1,
        help="Keep repeat 0 for every source image under recon/; use 0 to save no images (default: 1).",
    )
    parser.add_argument("--metrics", nargs="+", choices=ALL_METRICS, default=["psnr"])
    parser.add_argument("--skip-distribution", action="store_true")
    parser.add_argument(
        "--patch-stride",
        type=int,
        default=256,
        help="Compatibility option; GLC FID/256 requires and defaults to 256.",
    )
    parser.add_argument("--fid-batch-size", type=int, default=FID_BATCH_SIZE)
    parser.add_argument("--bpg-decoder", default="bpgdec")
    parser.add_argument("--vtm-decoder", default=DEFAULT_VTM_DECODER)
    parser.add_argument("--msillm-torch-hub-repo", type=Path)
    parser.add_argument("--elic-root", type=Path)
    parser.add_argument("--elic-checkpoint-dir", type=Path)
    parser.add_argument("--codec-python", type=Path, help="Python interpreter for learned decoding.")
    parser.add_argument("--allow-existing", action="store_true", help="Explicitly permit replacing a run directory.")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.repeats <= 0:
        raise ValueError("repeats must be positive.")
    if args.decoder_iterations <= 0:
        raise ValueError("decoder_iterations must be positive.")
    if args.patch_stride != 256:
        raise ValueError("patch_stride must be 256 for the GLC FID/256 protocol.")
    if args.fid_batch_size <= 0:
        raise ValueError("fid_batch_size must be positive.")
    if len(set(args.metrics)) != len(args.metrics):
        raise ValueError("metrics cannot contain duplicates.")
    if args.saved_repeats_per_image not in (0, 1):
        raise ValueError("saved_repeats_per_image must be 0 or 1.")
    if args.mcs_index is not None and (
        args.ldpc_rate is not None or args.modulation_order is not None
    ):
        raise ValueError(
            "--mcs-index cannot be combined with --ldpc-rate or --modulation-order."
        )
    range_values = (args.snr_start, args.snr_stop, args.snr_interval)
    if args.snr_db is not None and any(value is not None for value in range_values):
        raise ValueError(
            "--snr-db cannot be combined with --snr-start, --snr-stop, or --snr-interval."
        )
    resolve_snr_values(args)


def inclusive_snr_grid(start: float, stop: float, interval: float) -> tuple[float, ...]:
    if not all(math.isfinite(value) for value in (start, stop, interval)):
        raise ValueError("SNR range values must be finite.")
    if interval <= 0:
        raise ValueError("snr_interval must be positive.")
    if start > stop:
        raise ValueError("snr_start cannot exceed snr_stop.")
    count = math.floor((stop - start) / interval + 1e-12)
    values = [start + index * interval for index in range(count + 1)]
    if not math.isclose(values[-1], stop, rel_tol=0.0, abs_tol=1e-9):
        values.append(stop)
    return tuple(float(value) for value in values)


def resolve_snr_values(args: argparse.Namespace) -> tuple[float, ...]:
    if args.snr_db is not None:
        if not args.snr_db or not all(math.isfinite(value) for value in args.snr_db):
            raise ValueError("--snr-db values must be finite and non-empty.")
        return tuple(args.snr_db)
    start = 0.0 if args.snr_start is None else args.snr_start
    stop = 20.0 if args.snr_stop is None else args.snr_stop
    interval = 3.0 if args.snr_interval is None else args.snr_interval
    return inclusive_snr_grid(start, stop, interval)


def _float_tag(value: float) -> str:
    return f"{value:g}".replace("-", "minus_").replace(".", "p")


def experiment_dir(args: argparse.Namespace, config) -> Path:
    mcs_tag = f"mcs_{config.mcs_index}_" if config.mcs_index is not None else ""
    config_name = (
        f"cr_{_float_tag(config.compression_ratio)}_{mcs_tag}qam_{config.modulation_order}_"
        f"rate_{_float_tag(config.ldpc_rate)}"
    )
    return (
        args.results_root.expanduser().resolve()
        / args.method
        / "awgn"
        / args.dataset
        / f"snr_{_float_tag(config.snr_db)}dB"
        / config_name
    )


def save_fallback(reference_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_colour_image(reference_path).save(output_path)


def format_exception(exc: Exception, max_chars: int = 2000) -> str:
    message = f"{type(exc).__name__}: {exc}".replace("\x00", "")
    if len(message) <= max_chars:
        return message
    omitted = len(message) - max_chars
    return f"{message[:max_chars]} ... [truncated {omitted} chars]"


def empty_transmission() -> dict[str, object]:
    return {
        "payload_bits": 0,
        "coded_bits": 0,
        "channel_symbols_used": 0,
        "bit_errors": 0,
        "ber": None,
        "block_errors": 0,
        "num_blocks": 0,
        "header_valid": 0,
        "actual_ldpc_rate": None,
    }


def transmission_fields(result: TransmissionResult) -> dict[str, object]:
    values = asdict(result)
    values.pop("recovered_bytes")
    values["channel_symbols_used"] = values.pop("channel_symbols")
    values["header_valid"] = int(bool(values["header_valid"]))
    return values


def finite_mean(values) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(clean) if clean else None


def finite_std(values) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.pstdev(clean) if clean else None


def aggregate_rows(
    rows: list[dict[str, object]],
    distribution_by_repeat: list[dict[str, float | None]],
    args: argparse.Namespace,
    config,
    skipped: dict[str, str],
) -> dict[str, object]:
    repeat_pair: list[dict[str, float | None]] = []
    for repeat in range(args.repeats):
        repeat_rows = [row for row in rows if row["repeat"] == repeat]
        repeat_pair.append(
            {
                name: finite_mean(row.get(name) for row in repeat_rows)
                for name in PAIR_METRICS
                if name in args.metrics
            }
        )
    summary: dict[str, object] = {
        "method": args.method,
        "channel": "awgn",
        "dataset": args.dataset,
        "snr_db": config.snr_db,
        "compression_ratio": config.compression_ratio,
        "modulation_order": config.modulation_order,
        "bits_per_symbol": config.bits_per_symbol,
        "requested_ldpc_rate": config.ldpc_rate,
        "target_code_rate_x1024": config.target_code_rate_x1024,
        "mcs_index": config.mcs_index,
        "cqi_index": config.cqi_index,
        "mcs_table": "38.214-5.1.3.1-1",
        "mcs_selection": config.mcs_selection,
        "mcs_snr_gap_db": config.mcs_snr_gap_db,
        "spectral_efficiency": config.spectral_efficiency,
        "repeats": args.repeats,
        "aggregation": "mean_over_repeats",
        "num_images": len(rows) // args.repeats,
        "decode_success_rate": finite_mean(row["decode_success"] for row in rows),
        "fallback_rate": finite_mean(row["fallback_used"] for row in rows),
        "mean_bpp": finite_mean(row["bpp"] for row in rows),
        "mean_ber": finite_mean(row["ber"] for row in rows),
        "total_bit_errors": sum(int(row["bit_errors"]) for row in rows),
        "total_payload_bits": sum(int(row["payload_bits"]) for row in rows),
        "skipped_metrics": json.dumps(skipped, sort_keys=True),
    }
    total_payload = int(summary["total_payload_bits"])
    summary["weighted_ber"] = (
        int(summary["total_bit_errors"]) / total_payload if total_payload else None
    )
    for name in args.metrics:
        source = repeat_pair if name in PAIR_METRICS else distribution_by_repeat
        values = [item.get(name) for item in source]
        summary[name] = finite_mean(values)
        summary[f"{name}_repeat_std"] = finite_std(values)
    if "kid" in args.metrics:
        summary["kid_subset_std"] = finite_mean(
            item.get("kid_subset_std") for item in distribution_by_repeat
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run_snr(
    args: argparse.Namespace,
    dataset: ImageBitstreamDataset,
    decoder,
    config,
    metric_runner: QualityMetricRunner,
    pair_metric_cache: dict[tuple[Path, str], dict[str, float | None]],
    distribution_metric_cache: dict[tuple[str, ...], dict[str, float | None]],
) -> Path:
    output_root = experiment_dir(args, config)
    if output_root.exists() and any(output_root.iterdir()) and not getattr(args, "allow_existing", False):
        raise FileExistsError(f"Run directory already exists: {output_root}. Use another --results-root or --allow-existing.")
    output_root.mkdir(parents=True, exist_ok=True)
    from sscc.provenance import write_run_manifest
    write_run_manifest(output_root / "run.json", args, config, dataset)
    rows: list[dict[str, object]] = []
    distribution_by_repeat: list[dict[str, float | None]] = []

    for repeat in range(args.repeats):
        reconstruction_paths: list[Path] = []
        reference_paths: list[Path] = []
        reconstruction_digests: list[str] = []
        repeat_root = (
            output_root / "recon"
            if repeat == 0
            else output_root / ".repeat_tmp" / f"repeat_{repeat:03d}"
        )
        iterator = tqdm(range(len(dataset)), desc=f"{args.method} {config.snr_db:g}dB repeat {repeat}")
        for image_index in iterator:
            sample = dataset[image_index]
            with Image.open(sample.reference_path) as reference:
                width, height = reference.size
            capacity_bits = config.payload_capacity_bits(width, height)
            # Enforce the actual channel-use budget, including the length header,
            # LDPC final-block padding and QAM padding.
            channel_feasible = tuple(
                candidate
                for candidate in sample.candidates
                if config.payload_fits(candidate.size_bytes, width, height)
            )
            selected = select_largest_fitting(channel_feasible, capacity_bits)
            output_path = repeat_root / sample.relative_path
            output_path = output_path.with_suffix(".png")
            seed = args.seed + repeat * 10_000_000 + image_index
            status = "ok"
            error = ""
            fallback = False
            decode_success = False
            started = time.perf_counter()
            tx_values = empty_transmission()

            if selected is None:
                status = "no_feasible_bitstream"
                fallback = True
                save_fallback(sample.reference_path, output_path)
            else:
                try:
                    payload = selected.path.read_bytes()
                    transmission = transmit_bytes(
                        payload,
                        config,
                        seed=seed,
                        device=args.device,
                        decoder_iterations=args.decoder_iterations,
                    )
                    tx_values = transmission_fields(transmission)
                    if transmission.recovered_bytes is None:
                        raise RuntimeError("The decoded length header is invalid.")
                    decoder.decode(
                        transmission.recovered_bytes,
                        sample.reference_path,
                        selected.quality,
                        output_path,
                    )
                    decode_success = True
                except Exception as exc:
                    status = "decode_failed"
                    error = format_exception(exc)
                    fallback = True
                    save_fallback(sample.reference_path, output_path)

            reconstruction_digest = file_sha256(output_path)
            pair_cache_key = (sample.reference_path, reconstruction_digest)
            if pair_cache_key not in pair_metric_cache:
                pair_metric_cache[pair_cache_key] = metric_runner.evaluate_pair(
                    output_path, sample.reference_path
                )
            pair_values = pair_metric_cache[pair_cache_key]
            source_bits = selected.size_bits if selected is not None else 0
            row: dict[str, object] = {
                "method": args.method,
                "channel": "awgn",
                "dataset": args.dataset,
                "image": str(sample.relative_path),
                "repeat": repeat,
                "seed": seed,
                "snr_db": config.snr_db,
                "compression_ratio": config.compression_ratio,
                "modulation_order": config.modulation_order,
                "bits_per_symbol": config.bits_per_symbol,
                "requested_ldpc_rate": config.ldpc_rate,
                "target_code_rate_x1024": config.target_code_rate_x1024,
                "mcs_index": config.mcs_index,
                "cqi_index": config.cqi_index,
                "mcs_table": "38.214-5.1.3.1-1",
                "mcs_selection": config.mcs_selection,
                "mcs_snr_gap_db": config.mcs_snr_gap_db,
                "spectral_efficiency": config.spectral_efficiency,
                "width": width,
                "height": height,
                "channel_symbols_budget": config.channel_symbols(width, height),
                "payload_capacity_bits": capacity_bits,
                "quality": selected.quality if selected else "",
                "source_bits": source_bits,
                "bpp": source_bits / (width * height) if source_bits else None,
                "bitstream_path": str(selected.path) if selected else "",
                "reconstruction_path": str(output_path),
                **tx_values,
                "decode_success": int(decode_success),
                "fallback_used": int(fallback),
                "status": status,
                "error": error,
                "elapsed_seconds": time.perf_counter() - started,
                **pair_values,
            }
            rows.append(row)
            reference_paths.append(sample.reference_path)
            reconstruction_paths.append(output_path)
            reconstruction_digests.append(reconstruction_digest)
        distribution_signature = tuple(reconstruction_digests)
        if distribution_signature not in distribution_metric_cache:
            distribution_metric_cache[distribution_signature] = (
                metric_runner.evaluate_distribution(
                    reference_paths,
                    reconstruction_paths,
                    patch_stride=args.patch_stride,
                    batch_size=args.fid_batch_size,
                )
            )
        distribution_by_repeat.append(distribution_metric_cache[distribution_signature])
        save_repeat = repeat < args.saved_repeats_per_image
        repeat_rows = [row for row in rows if row["repeat"] == repeat]
        for row, path in zip(repeat_rows, reconstruction_paths):
            row["reconstruction_saved"] = int(save_repeat)
            if not save_repeat:
                path.unlink(missing_ok=True)
                row["reconstruction_path"] = ""
        if not save_repeat:
            shutil.rmtree(repeat_root, ignore_errors=True)

    shutil.rmtree(output_root / ".repeat_tmp", ignore_errors=True)

    summary = aggregate_rows(rows, distribution_by_repeat, args, config, metric_runner.skipped)
    write_csv(output_root / "per_image_metrics.csv", rows)
    write_csv(output_root / "metrics.csv", [summary])
    write_csv(output_root / "average_metrics.csv", [summary])
    return output_root


def main(argv=None) -> None:
    args = parse_args(argv)
    validate_args(args)
    if args.method == "ELIC":
        args.method = "ELiC"
    # Dependency errors must fail before the per-image fallback handler.
    from sionna.phy.fec.ldpc import LDPC5GEncoder  # noqa: F401
    from sscc.channel.transport import resolve_device
    import torch
    torch.empty(1, device=resolve_device(args.device))
    dataset = ImageBitstreamDataset(
        dataset_root=args.dataset_root,
        bitstream_root=args.bitstream_root,
        method=args.method,
        dataset_name=args.dataset,
        max_images=args.max_images,
    )
    if dataset.unmatched_images:
        examples = ", ".join(path.name for path in dataset.unmatched_images[:5])
        raise RuntimeError(
            f"{len(dataset.unmatched_images)} source images have no matching bitstreams; examples: {examples}"
        )
    decoder = create_decoder(
        args.method,
        device=args.device,
        bpg_decoder=args.bpg_decoder,
        vtm_decoder=args.vtm_decoder,
        msillm_torch_hub_repo=args.msillm_torch_hub_repo,
        elic_root=args.elic_root,
        elic_checkpoint_dir=args.elic_checkpoint_dir,
        codec_python=args.codec_python,
    )
    snr_values = resolve_snr_values(args)
    print(f"SNR sweep: {', '.join(f'{value:g}' for value in snr_values)} dB")
    selected_metrics = [
        name
        for name in args.metrics
        if not (args.skip_distribution and name in DISTRIBUTION_METRICS)
    ]
    metric_runner = QualityMetricRunner(selected_metrics, args.device)
    pair_metric_cache: dict[tuple[Path, str], dict[str, float | None]] = {}
    distribution_metric_cache: dict[tuple[str, ...], dict[str, float | None]] = {}
    for snr_db in snr_values:
        config = resolve_channel_config(
            snr_db,
            compression_ratio=args.compression_ratio,
            modulation_order=args.modulation_order,
            ldpc_rate=args.ldpc_rate,
            mcs_index=args.mcs_index,
            mcs_snr_gap_db=args.mcs_snr_gap_db,
        )
        output = run_snr(
            args,
            dataset,
            decoder,
            config,
            metric_runner,
            pair_metric_cache,
            distribution_metric_cache,
        )
        print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
