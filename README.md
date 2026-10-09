# SSCC Benchmark

A standalone Python library for **image source coding + 5G LDPC + QAM + AWGN**.
Encode images, transmit actual compressed bytes and benchmark reconstructed images
on **Kodak** and **CLIC**, including multiple CLIC subsets in one experiment.

[中文快速开始](README.zh-CN.md) · [BPG/VTM installation](docs/codecs.md) ·
[Benchmark protocol](docs/benchmark.md) · [Learned codecs](docs/learned-codecs.md) ·
[Upstream projects](docs/upstream.md) · [Validation](docs/validation.md)

| Source codec | Source encoding | Channel decoding | External requirements |
|---|---|---|---|
| BPG | `sscc-prepare` | Real `bpgdec` | libbpg 0.9.8 |
| VTM | `sscc-prepare` | Real `DecoderApp` | VTM, YCbCr 4:4:4 |
| MS-ILLM | `sscc-prepare` in learned environment | NeuralCompression model | Upstream source + pretrained weights |
| ELiC | `sscc-prepare` in learned environment | Actual entropy/model decoder | ELiC reimplementation + checkpoints |

The channel supports QPSK, 16-QAM and 64-QAM, fixed or SNR-adaptive MCS,
multiple seeds/repeats and an exact complex-symbol budget including the 32-bit
length header, LDPC padding and modulation padding. Default metric: RGB PSNR.
Optional metrics: MS-SSIM, LPIPS, DISTS, PieAPP, patch FID and KID.

## Install

Linux x86-64 and Python 3.12 are the tested target. Python 3.13 is allowed but has
not been validated. Run all commands from this repository root.

```bash
uv sync --locked --extra channel --extra dev
source .venv/bin/activate
sscc-benchmark --help
```

With pip: `python -m pip install '.[channel]'`. For CPU-only PyTorch, install its
CPU wheel first, then install this package. The lockfile's default PyTorch build
can download CUDA libraries even if you plan to run on CPU.

For the optional image metrics, use `uv sync --locked --extra channel --extra metrics`.
Metrics with neural networks may download their pretrained weights on first use.
The default PSNR path does not download metric models.

Sionna 2 requires NumPy 2; CompressAI 1.2.8 requires NumPy 1. Keep the `channel`
and `learned` extras in **separate virtual environments**. No dependency override
or `--no-deps` workaround is needed; see [learned codecs](docs/learned-codecs.md).

## Install BPG and VTM

Upstreams: [BPG / Fabrice Bellard](https://bellard.org/bpg/),
[VTM / JVET](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM).

```bash
# System prerequisites on Ubuntu/Debian (run this yourself if needed).
sudo apt-get update
sudo apt-get install -y build-essential cmake git curl yasm pkg-config \
  libpng-dev libjpeg-dev zlib1g-dev libssl-dev

bash scripts/install_bpg.sh
bash scripts/install_vtm.sh
export PATH="$PWD/.local/bin:$PATH"
```

Scripts build upstream source under `third_party/` and install to `.local/bin/`;
no root access is needed after OS prerequisites. Default VTM tag: `VTM-23.14`.
Use `JOBS=4` to change compilation parallelism. Full manual instructions,
configuration flags and troubleshooting: [docs/codecs.md](docs/codecs.md).

## Datasets

Download from the dataset providers; images are not redistributed in this repo:

- [Kodak lossless true-color images](https://r0k.us/graphics/kodak/): 24 PNG files.
- [CLIC](https://www.compression.cc/): select a year/split explicitly. The example
  below uses CLIC 2020 test, mobile + professional. Extract downloaded archives
  so that the following directories contain the images (nested folders work).

```text
data/
  kodak/kodim01.png ... kodim24.png
  clic2020/mobile_test/*.png
  clic2020/professional_test/*.png
```

You may use any other location with `--dataset-root`. Repeating this option
combines subsets. Use the **same ordered roots** for preparation and benchmark;
root names are included in identifiers when more than one root is supplied.

## Quick smoke run: Kodak + BPG

```bash
sscc-prepare --method BPG --dataset kodak --dataset-root data/kodak \
  --qualities 40 48 51 --max-images 1

sscc-benchmark --method BPG --dataset kodak --dataset-root data/kodak \
  --snr-db 20 --compression-ratio 96 --repeats 1 --max-images 1 \
  --device cpu --metrics psnr
```

Check `decode_success_rate`, `fallback_rate` and `status` in the CSVs; a completed
run can legitimately have no feasible stream at a tight budget. Use a denser QP
grid or a smaller compression ratio if needed. VTM encoding can take minutes per
large image and QP, so start with `--max-images 1`.

## Full Kodak + CLIC benchmark

```bash
for method in BPG VTM; do
  sscc-prepare --method "$method" --dataset kodak --dataset-root data/kodak
  sscc-prepare --method "$method" --dataset clic2020_test \
    --dataset-root data/clic2020/mobile_test \
    --dataset-root data/clic2020/professional_test
done

sscc-suite examples/kodak-clic.toml --dry-run
sscc-suite examples/kodak-clic.toml
sscc-summarize results --output benchmark.csv
```

The example runs 2 codecs × 2 datasets × 8 SNRs × 3 repeats. It uses compression
ratio 96 and SNRs `0, 3, 6, 9, 12, 15, 18, 20 dB`. Edit the TOML for paths,
methods, metrics, repeats or device. Paths in TOML are relative to the current
working directory. BPG/VTM preparation resumes only when the source hash, encoder settings
and output hash match. Learned streams are regenerated to avoid silently reusing
streams after model/checkpoint changes. Benchmark runs reject existing nonempty
output directories; use a new `results_root` for a new experiment, or explicitly
pass `--allow-existing` to a single benchmark command.

To benchmark your existing streams, omit `sscc-prepare` and pass
`--bitstream-root /path/to/bitstreams`:

```text
bitstreams/METHOD/DATASET/QUALITY/IMAGE.bpg|.266|.pkl|.elic
```

`QUALITY` can be `q40`, `q1`, `q0_004`, etc. Image paths are flattened into safe
identifiers (`mobile_test/example.png` → `mobile_test__example`). See the
[protocol and compatibility notes](docs/benchmark.md) before importing old streams.

## Outputs

```text
results/METHOD/awgn/DATASET/snr_20dB/cr_96_mcs_28_qam_64_rate_0p925781/
  run.json                 # arguments, versions, channel config, image/stream hashes
  per_image_metrics.csv    # selected quality, BER, symbols, status, image metrics
  average_metrics.csv      # dataset mean and repeat variation
  metrics.csv              # same aggregate for compatibility
  recon/                   # first repeat's reconstructions by default
```

The fallback is the original implementation's **per-channel reference mean
image**, which uses oracle side information not charged to the channel budget.
Always report fallback rate and this convention with scores. This is an explicit
research comparison convention, not an operational receiver model.

SNR is **Es/N0 per complex symbol**, not Eb/N0. The SNR→MCS mapping is a
Shannon-based heuristic with a default 3 dB margin; 3GPP defines the MCS rows,
not this SNR mapping. No retransmissions, OFDM, pilots, fading or packet CRC are
modeled. See [benchmark.md](docs/benchmark.md) for exact definitions.

## Library API

```python
from sscc import resolve_channel_config, transmit_bytes
from sscc.source.bpg import BPGCodec

config = resolve_channel_config(20, modulation_order=16, ldpc_rate=0.5)
result = transmit_bytes(b"hello channel" * 32, config, seed=42, device="cpu")
print(result.ber, result.channel_symbols, result.recovered_bytes)
```

`BPGCodec`, `VTMCodec`, `MSILLMCodec`, `ELiCCodec` expose `encode_decode(...)`.
`sscc.codecs.create_decoder(...)` decodes recovered byte containers. Learned
codec APIs must run in the learned environment, or use `codec_python` with the
factory to delegate decoding to that environment.

## Development and license

```bash
uv sync --locked --extra channel --extra dev
uv run --no-sync ruff check src tests
uv run --no-sync pytest -q
uv run --no-sync python -m build
```

Tests include actual CPU LDPC transmission and optional real BPG/VTM round trips.
See [CONTRIBUTING.md](CONTRIBUTING.md). The Python harness is MIT-licensed;
external software, pretrained weights and datasets retain their own terms.
No upstream binaries, model weights, datasets or generated results are committed.
Upstream links and extraction history are in [docs/upstream.md](docs/upstream.md)
and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
