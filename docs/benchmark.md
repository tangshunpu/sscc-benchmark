# Benchmark protocol

## Channel and rate matching

For an RGB image of size H×W and compression ratio CR, the budget is
`floor(3*H*W/CR)` **complex channel symbols**, equivalently CBR ≈ 1/CR.
The image codec contributes a complete byte container. A 32-bit big-endian byte
length precedes the payload. The LDPC implementation balances information bits
across blocks and accounts for code-block and QAM padding before selecting the
largest feasible stream. Container headers and padding are charged to the budget.

The transport is Sionna 2.0.1's 5G LDPC encoder, QAM constellation, AWGN channel,
APP soft demapper and iterative LDPC decoder (20 iterations by default).
Average constellation energy is one; complex noise variance is `10**(-snr_db/10)`.
Thus SNR means **Es/N0**, not Eb/N0. Repeats use deterministic per-image seeds;
Sionna's generator and PyTorch's generator are both seeded.

Automatic adaptation uses rows of 3GPP TS 38.214 Table 5.1.3.1-1, with QPSK,
16-QAM and 64-QAM. Available spectral efficiency is estimated from Shannon
capacity after a configurable 3 dB implementation margin, quantized through
CQI Table 1 and mapped to an MCS row. This mapping is a benchmark heuristic,
not a 3GPP-mandated receiver adaptation algorithm. MCS 0–2 are excluded because
the current rate matcher requires rate ≥1/5. Use `--mcs-index 3..28`, or both
`--modulation-order` and `--ldpc-rate`, to control the channel explicitly.

There is no CRC, retransmission, OFDM, fading, pilot or channel-estimation cost.
The length field is not independently protected. Byte errors do not by themselves
force failure: recovered bytes go to the source decoder, which may succeed or fail.
This is a byte-container AWGN simulation rather than a complete cellular stack.

## Quality selection and failure

Quality is selected by **largest actual stream size within the symbol budget**,
not by highest PSNR or a rate-distortion oracle. A sparse quality grid can leave
unused capacity. Compare methods with documented grids and report actual BPP.
No feasible stream, invalid header or source decode error causes fallback.
For compatibility with the original experiments, fallback is a constant image of
the reference's rounded per-channel RGB means. Those means come from the source
image and are not transmitted; report this oracle assumption and fallback rate.

Inspect `decode_success_rate`, `fallback_rate`, `status`, `error` and
`skipped_metrics`, not just PSNR. Missing metrics are empty fields, not zero.
Scores include failed images reconstructed by the fallback convention.
The original finite-only aggregation excludes nonfinite values, including infinite
PSNR for an exact reconstruction; per-image CSV preserves the value. This is
relevant for lossless or constant-image tests.

## Metrics and repetitions

Images are evaluated as 8-bit RGB PNGs. Pair metrics are averaged over images
within each repeat, then over repeats; repeat standard deviations are population
standard deviations, not confidence intervals. PSNR is mean per-image PSNR,
not PSNR of the pooled dataset MSE. BER is reported both as image mean and as
payload-bit-weighted BER. The first repeat's reconstruction is kept by default.

MS-SSIM requires a minimum side of 161 pixels. FID/KID use 256×256 patches from
two grids starting at (0,0) and (128,128), stride 256; incomplete border patches
are discarded. KID uses 100 subsets of up to 1,000 patches. These are patch metrics,
not full-image FID/KID; sample counts and dataset/split must accompany results.

## Existing bitstreams

Expected layout: `ROOT/METHOD/DATASET/QUALITY/IDENTIFIER.EXT`.
The method and dataset directory lookup is case-insensitive. Extensions recognized:
`.bpg`, `.266`, `.pkl`, `.pickle`, `.elic`, `.bin`. Ancillary files are excluded.

Paths are normalized consistently by prepare and benchmark. For one image root,
`scene/a.png` becomes `scene__a`. For multiple roots, the root directory name is
prepended: `mobile_test/a.png` becomes `mobile_test__a`. Collisions after replacing
non-ASCII/special characters are rejected. Root order defines image order/seeds.
Choose roots and labels consistently between runs, and avoid overlapping roots.

Old BPG/VTM streams from `image_compression/results/bitstreams` can be passed by
path without copying. Learned streams must use the **same upstream code, weights,
CompressAI version, device and entropy settings** for encoding and decoding.
Regenerate old ELiC streams if they were made with a different entropy runtime;
this repo does not use old cached PNG reconstructions as a substitute for decoding.
MS-ILLM also supports the upstream `vlo1`/`vlo2` models when selected explicitly;
the default preparation grid uses qualities 1–6.

Learned bitstreams use Python pickle containers, and checkpoint loading can also
execute Python objects. Only use trusted locally produced streams and weights;
this package is not a service for accepting uploaded or untrusted media containers.

## Reproduction records

Each run's `run.json` stores package versions, arguments, image hashes, source
stream hashes and resolved MCS. Source preparation records input/output hashes,
settings and available encoder/config hashes under `bitstreams/_manifests/`.
For publications, also record native codec revisions, model weight hashes,
upstream source commit, hardware and the separate learned environment's package
freeze. Metadata records the host paths for debugging; inspect it before publishing
result files. Generated manifests/data/results are ignored by Git.
