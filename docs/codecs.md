# BPG and VTM: source, build, install, verify

## BPG 0.9.8

The original distribution is [Fabrice Bellard's BPG page](https://bellard.org/bpg/).
[mirrorer/libbpg](https://github.com/mirrorer/libbpg) is an **unofficial mirror**.
Use the original release archive for the commands below.

Prerequisites on Ubuntu/Debian:

```bash
sudo apt-get install -y build-essential cmake curl yasm pkg-config \
  libpng-dev libjpeg-dev zlib1g-dev
```

From this repository root:

```bash
bash scripts/install_bpg.sh
export PATH="$PWD/.local/bin:$PATH"
```

Equivalent manual build:

```bash
mkdir -p third_party .local/bin
curl -fL https://bellard.org/bpg/libbpg-0.9.8.tar.gz -o third_party/libbpg-0.9.8.tar.gz
tar -xzf third_party/libbpg-0.9.8.tar.gz -C third_party
make -C third_party/libbpg-0.9.8 -j2 USE_BPGVIEW= CMAKE_OPTS=-DENABLE_LIBNUMA=OFF bpgenc bpgdec
install -m755 third_party/libbpg-0.9.8/bpgenc third_party/libbpg-0.9.8/bpgdec .local/bin/
```

Only the CLI encoder/decoder are built. The SDL viewer is disabled, so SDL and
SDL_image development packages are unnecessary. The default upstream x265
backend is used. NUMA support is disabled because the old libbpg linker command
does not include libnuma even when x265 auto-detects it. To check both executables on an actual image:

```bash
mkdir -p results/codec-check
bpgenc -q 40 -f 444 -m 8 -o results/codec-check/image.bpg data/kodak/kodim01.png
bpgdec -o results/codec-check/image.png results/codec-check/image.bpg
```

This repository normalizes inputs to RGB PNG before source coding. BPG uses
8-bit input, YCbCr 4:4:4 and encoder preset 8; `--bpg-preset` can change speed.
QP range is 0–51 (larger means fewer bits/lower quality).

## VTM

Original repository: [JVET/VVCSoftware_VTM](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM).
The build follows the repository's CMake instructions. This harness uses the
reference VTM encoder, not VVenC. The script selects release tag `VTM-23.14`:

```bash
sudo apt-get install -y build-essential cmake git libssl-dev
bash scripts/install_vtm.sh
export PATH="$PWD/.local/bin:$PATH"
```

Equivalent manual steps:

```bash
git clone --depth 1 --branch VTM-23.14 \
  https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git third_party/VVCSoftware_VTM
cmake -S third_party/VVCSoftware_VTM -B third_party/VVCSoftware_VTM/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build third_party/VVCSoftware_VTM/build --config Release -j2
find third_party/VVCSoftware_VTM/bin -type f -name '*App*'
```

CMake versions/platforms may emit `EncoderApp`, `DecoderApp`,
`EncoderAppStatic`, `DecoderAppStatic`, or nested build directories. The install
script locates executables and installs stable names into `.local/bin`.
To use an existing installation, pass the executable paths directly:

```bash
sscc-prepare --method VTM --dataset kodak --dataset-root data/kodak \
  --qualities 47 --max-images 1 \
  --vtm-encoder /path/to/EncoderAppStatic \
  --vtm-decoder /path/to/DecoderAppStatic \
  --vtm-config /path/to/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg
sscc-benchmark --method VTM --dataset kodak --dataset-root data/kodak \
  --snr-db 20 --max-images 1 --repeats 1 \
  --vtm-decoder /path/to/DecoderAppStatic
```

Profile: `main_10_444_still_picture`, one intra frame, planar YCbCr 4:4:4,
8-bit input/output, 10-bit internal coding, automatic conformance-window padding.
RGB↔YCbCr conversion uses Pillow. The decoder requires exactly one 8-bit 4:4:4
frame at the reference dimensions. **Do not import arbitrary 4:2:0, 10-bit raw
output or RGB-coded VVC streams** into this adapter. QPs supported by the CLI:
0–63. Use the same VTM build for encoding and decoding.

## Troubleshooting

- `bpgenc` / `DecoderApp` not found: export `.local/bin` on PATH, or supply an
  absolute path. Installing the Python package does not install native codecs.
- Missing PNG/JPEG headers or yasm: install the OS dependencies above.
- Old bundled x265 with CMake 4: build with CMake 3.x if its obsolete CMake
  minimum causes configuration errors. The validation host uses CMake 3.x.
- VTM build runs out of memory: use `JOBS=1` or `cmake --build ... -j1`.
- A recent compiler treats a warning as an error: inspect the actual failing
  warning and upstream build flags; do not silently replace VTM with another codec.
- CLIC encoding is slow: this is reference software; reduce the initial QP grid
  and use one image to verify setup. Preparation has a two-hour per-process timeout.
- The installers reuse existing source trees; remove/rename them yourself or
  select the required revision before rebuilding. `VTM_REF` selects the initial
  clone revision only. Save `.local/vtm-commit.txt` with benchmark provenance.

BPG includes components under BSD, LGPL and GPL licenses; VTM has its own
software license. Read the upstream license files. Neither binary is bundled
with the MIT-licensed Python harness.
