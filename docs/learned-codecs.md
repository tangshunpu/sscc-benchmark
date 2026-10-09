# MS-ILLM and ELiC

These are optional adapters. Their source and checkpoints are obtained from
upstream; weights are not included. The default BPG/VTM installation does not
need them. Old learned-codec bitstreams may not decode with a new entropy runtime;
prepare fresh streams and benchmark with the same environment and checkpoints.

## Two environments with compatible dependencies

The channel runtime depends on Sionna 2.0.1 (NumPy ≥2.2.6). The learned runtime
depends on CompressAI 1.2.8 (NumPy <2). Both environments install this package:

```bash
uv sync --locked --extra channel --extra dev
UV_PROJECT_ENVIRONMENT=.venv-learned uv sync --locked --extra learned
```

Use `.venv-learned/bin/sscc-prepare` for source coding and
`.venv/bin/sscc-benchmark --codec-python "$PWD/.venv-learned/bin/python"`
for channel evaluation. The receiver passes recovered bytes through a temporary
file to the isolated decoder. A worker is launched for each image, so model
loading overhead is included in elapsed time; this version targets correctness
rather than learned-decoder throughput. No Sionna import occurs in the worker.

Pip equivalent, if uv is unavailable:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install '.[channel]'
python3.12 -m venv .venv-learned
.venv-learned/bin/python -m pip install '.[learned]'
```

Do not install `.[channel,learned]` together. Optional full-metric dependencies
also belong in the channel environment.

## MS-ILLM

Original implementation:
[facebookresearch/NeuralCompression](https://github.com/facebookresearch/NeuralCompression),
[MS-ILLM project](https://github.com/facebookresearch/NeuralCompression/tree/main/projects/illm).

```bash
mkdir -p third_party
git clone https://github.com/facebookresearch/NeuralCompression.git third_party/NeuralCompression
git -C third_party/NeuralCompression rev-parse HEAD

.venv-learned/bin/sscc-prepare --method MSILLM --dataset kodak \
  --dataset-root data/kodak --qualities 1 --max-images 1 --device cpu \
  --msillm-torch-hub-repo third_party/NeuralCompression

.venv/bin/sscc-benchmark --method MSILLM --dataset kodak \
  --dataset-root data/kodak --max-images 1 --snr-db 20 --repeats 1 \
  --device cpu --metrics psnr \
  --msillm-torch-hub-repo third_party/NeuralCompression \
  --codec-python "$PWD/.venv-learned/bin/python"
```

Torch Hub loads source from the specified local clone and may download pretrained
weights on first use. Record the clone commit and cached weight SHA-256. The
default preparation qualities are 1–6. Upstream also provides the very-low-rate
`vlo1` and `vlo2` models; pass them explicitly with `--qualities vlo1 vlo2 1`.
Source code and pretrained weights have different licenses; upstream states MIT
for code and CC-BY-NC 4.0 for released weights.

## ELiC

The adapter uses the third-party reimplementation
[VincentChandelier/ELiC-ReImplemetation](https://github.com/VincentChandelier/ELiC-ReImplemetation).
This is a **reimplementation**, not a claim to be the paper authors' original code.
Use the checkpoint links and instructions in its README.

```bash
git clone https://github.com/VincentChandelier/ELiC-ReImplemetation.git third_party/ELiC
.venv-learned/bin/python scripts/patch_elic.py third_party/ELiC
mkdir -p third_party/ELiC/checkpoints
```

Download the upstream pretrained files into the checkpoint directory. Names
recognized include `ELIC_0004.pth.tar`, `ELIC_0008.pth.tar`,
`ELIC_0016.pth.tar`, `ELIC_0032.pth.tar`, `ELIC_0150.pth.tar`,
`ELIC_0450.pth.tar` and the corresponding upstream `_ft_3980_Plateau` names.
These map to lambda `0.004, 0.008, 0.016, 0.032, 0.15, 0.45`.
The compatibility script edits two CompressAI imports in the cloned source;
our adapter converts old entropy-bottleneck checkpoint key names when necessary.
It does not alter the network architecture.

```bash
.venv-learned/bin/sscc-prepare --method ELiC --dataset kodak \
  --dataset-root data/kodak --qualities 0.004 --max-images 1 --device cpu \
  --elic-root third_party/ELiC --elic-checkpoint-dir third_party/ELiC/checkpoints

.venv/bin/sscc-benchmark --method ELiC --dataset kodak \
  --dataset-root data/kodak --max-images 1 --snr-db 20 --repeats 1 \
  --device cpu --metrics psnr \
  --elic-root third_party/ELiC --elic-checkpoint-dir third_party/ELiC/checkpoints \
  --codec-python "$PWD/.venv-learned/bin/python"
```

If the lowest model quality cannot fit CR=96 at a given SNR, the correct outcome
is `no_feasible_bitstream`. A smaller `--compression-ratio` can be used to verify
runtime functionality, but must be reported and matched across comparison methods.
This adapter always uses model decompression; there is no reconstruction cache.

For CLIC, use `--dataset clic2020_test` and both dataset roots, as in the main
README. Full-resolution learned encoding can require substantial RAM/VRAM.

Only process trusted checkpoints and locally generated pickle bitstreams. Pin
upstream revisions and weight hashes when publishing results. The repository does
not redistribute those assets or grant rights beyond their upstream terms.
