# Validation record

Validated on 2026-10-09 on Linux x86-64 / Python 3.12.13.

## Build and installation

- Independent channel environment: `uv sync --locked --extra channel --extra dev`.
- Independent learned environment: `UV_PROJECT_ENVIRONMENT=.venv-learned uv sync --locked --extra learned`.
- `uv pip check` reports compatible dependencies in both environments.
- BPG 0.9.8 was downloaded from the original release, compiled and installed by `scripts/install_bpg.sh`.
- VTM was freshly cloned at VTM-23.14, compiled and installed by `scripts/install_vtm.sh`.
- The BPG installer disables NUMA to resolve the reproduced upstream x265/libbpg link mismatch.

Channel environment versions:

- numpy: 2.5.3
- torch: 2.14.1
- sionna: 2.0.1
- pillow: 12.3.0

Upstream source revisions:

- VVCSoftware_VTM: `ba166fd31acc050329d40012cd945b52023fdf57`
- NeuralCompression: `3f122808255a2278eea049119367eb3abc434862`
- ELiC: `92a9ece1e6e1a188a12dfd7a58f9b51c554f9f2d`

ELiC uses the two documented import edits from `scripts/patch_elic.py`.

## Automated checks

- 34 tests passed in the independent channel environment, using the freshly built BPG/VTM binaries.
- Tests exercise all three QAM orders, exact LDPC byte recovery, seeded noisy transmission,
  symbol accounting, CLIC multi-root naming, source cache recovery and overwrite protection.
- Ruff lint and Bash syntax checks passed.

## End-to-end smoke runs

Each row below is one image and one repeat. These are functionality checks,
**not full-dataset performance claims**. BPG/VTM real-image checks use CR=96,
CPU, default adaptive MCS; learned checks use a synthetic 192×192 RGB image,
CR=1 and SNR=40 dB to test the process bridge with feasible streams.

| Method | Dataset | SNR (dB) | PSNR (dB) | Status |
|---|---|---:|---:|---|
| BPG | clic2020_test | 20 | 28.2894 | ok |
| BPG | kodak | 0 | 16.0872 | no_feasible_bitstream |
| BPG | kodak | 20 | 23.2137 | ok |
| ELiC | synthetic | 40 | 10.8896 | ok |
| MSILLM | synthetic | 40 | 10.6257 | ok |
| VTM | clic2020_test | 20 | 28.4464 | ok |
| VTM | kodak | 20 | 23.6332 | ok |

Real images: Kodak `kodim01.png`; CLIC2020 mobile test
`00b64869422e0011ff5bb492e56042cc.png`.
The full-resolution CLIC smoke used both mobile/professional roots with `--max-images 1`;
both-root enumeration and duplicate-filename handling are additionally exercised by synthetic integration tests.
At 0 dB the selected BPG QP grid had no stream within budget, so the explicit fallback was expected.
ELiC lambda 0.004 and MS-ILLM quality 1 used existing trusted model weights.
Both learned tests were repeated successfully using fresh upstream clones and separate environments.

BPG/VTM real-image rows used an existing VTM build `VTM-23.14-7-g0f1e36d7f`;
the final native integration tests used the newly installed VTM-23.14 build.

## Scope

Full Kodak/CLIC sweeps, GPU execution, every quality/checkpoint, optional pretrained
metric inference and Python 3.13 were not exhaustively validated. The provided
suite configuration can run full dataset experiments. Source/wheel packaging and
wheel-only CLI verification are recorded in the local build logs.
