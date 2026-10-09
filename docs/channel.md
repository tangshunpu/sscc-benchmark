# Channel coding library API

The channel module transmits **any nonempty byte payload** through a simulated
5G LDPC/QAM/AWGN link. It can be used independently of the image codecs and
dataset benchmark. Its implementation is in
[`sscc/channel/mcs.py`](../src/sscc/channel/mcs.py) and
[`sscc/channel/transport.py`](../src/sscc/channel/transport.py).

## Installation

```bash
uv sync --locked --extra channel
source .venv/bin/activate
```

Pip alternative: `python -m pip install '.[channel]'`.
BPG/VTM executables and pretrained models are unnecessary for byte transmission.

## Processing chain

```text
bytes → 32-bit length header + payload bits → balanced blocks + zero padding
      → 5G LDPC encoding → QAM padding → QPSK / 16-QAM / 64-QAM
      → AWGN → APP soft demapping → iterative LDPC decoding
      → length parsing → recovered bytes
```

`snr_db` is **Es/N0 per complex symbol**. The constellation has unit average
energy; the complex noise variance is `10**(-snr_db/10)`. The implementation
currently supports AWGN. There is no fading, OFDM, pilot overhead, CRC or ARQ.

## Transmit a byte payload

```python
from sscc import resolve_channel_config, transmit_bytes

payload = bytes(range(256))
config = resolve_channel_config(
    snr_db=30,
    modulation_order=16,
    ldpc_rate=0.5,
)
result = transmit_bytes(
    payload,
    config,
    seed=42,
    device="cpu",
    decoder_iterations=20,
)

print("Recovered:", result.recovered_bytes == payload)
print("BER:", result.ber)
print("Symbols:", result.channel_symbols)
```

Noise can cause errors even when `recovered_bytes` is present. In a simulation,
compare recovered bytes with the original to check exact recovery. At the
receiver, a valid decoded container and successful source decoding are separate
conditions. `transmit_bytes` does not itself decode images or generate fallback
images; that behavior belongs to the benchmark runner.

## Configure modulation and code rate

Use `resolve_channel_config(...)` to validate parameters and construct the
`ChannelConfig` dataclass. Direct dataclass construction does not perform these
checks.

```python
from sscc import resolve_channel_config

# Automatic SNR → CQI → MCS selection; default implementation margin is 3 dB.
automatic = resolve_channel_config(snr_db=10, compression_ratio=96)

# Fix one supported MCS table row regardless of SNR.
fixed_mcs = resolve_channel_config(snr_db=10, mcs_index=10)

# Fix modulation and requested LDPC rate explicitly.
manual = resolve_channel_config(
    snr_db=10, modulation_order=16, ldpc_rate=0.5,
)

for config in (automatic, fixed_mcs, manual):
    print(config.mcs_selection, config.mcs_index,
          config.modulation_order, config.ldpc_rate)
```

| Argument | Default | Meaning |
|---|---|---|
| `snr_db` | Required | Finite Es/N0 in dB |
| `compression_ratio` | `96.0` | Positive finite CR used for image symbol budgets |
| `modulation_order` | Automatic | `4` = QPSK, `16` = 16-QAM, `64` = 64-QAM |
| `ldpc_rate` | Automatic | Target information/codeword bit ratio in `[0.2, 948/1024]` |
| `mcs_index` | `None` | Fixed MCS row `3..28`; mutually exclusive with modulation/rate overrides |
| `mcs_snr_gap_db` | `3.0` | Finite margin used only by automatic MCS selection |

When only modulation **or** rate is overridden, the other remains selected by
SNR. Set both for a fixed modulation/rate comparison. Codeword rounding makes
`result.actual_ldpc_rate` potentially slightly smaller than the requested rate.

Automatic selection uses the MCS/CQI tables embedded in the package and a
Shannon-capacity heuristic, with MCS 3 as its supported lower floor. The mapping
is not a measured BLER threshold table and does not guarantee reliable decoding.
See [benchmark protocol](benchmark.md#channel-and-rate-matching).

Useful `ChannelConfig` attributes include `bits_per_symbol`,
`spectral_efficiency`, `mcs_index`, `cqi_index` and `mcs_selection`.

## Check an image's channel-use budget

```python
from sscc import resolve_channel_config

config = resolve_channel_config(20, compression_ratio=96)
width, height = 768, 512
payload_bytes = 2_000

budget = config.channel_symbols(width, height)
k, n, blocks, symbols = config.transmission_shape(payload_bytes)
print("Budget:", budget, "complex symbols")
print("Information bits/block:", k, "coded bits/block:", n)
print("Blocks:", blocks, "required symbols:", symbols)
print("Fits:", config.payload_fits(payload_bytes, width, height))
```

| Method | Result |
|---|---|
| `channel_symbols(width, height)` | `floor(3*width*height/CR)` complex symbols |
| `payload_capacity_bits(width, height)` | Nominal capacity estimate; excludes header/padding overhead |
| `transmission_shape(payload_bytes)` | `(k, n, blocks, symbols)` including the length header, LDPC padding and QAM padding |
| `payload_fits(payload_bytes, width, height)` | Whether actual required symbols fit the image budget |

Use positive image dimensions and a positive payload length in **bytes**.
Use `payload_fits` for the final acceptance decision; nominal capacity alone can
overestimate what fits. **`transmit_bytes` has no image dimensions and does not
enforce an image budget.** `compression_ratio` affects budget calculations, not
the physical noise level or the transmission of a given payload. Check before
calling it, as the benchmark runner does.

## Transmission arguments and result fields

```python
transmit_bytes(payload, config, seed, device="auto", decoder_iterations=20)
```

`payload` is a nonempty `bytes` object; its length must fit the 32-bit header.
`seed` is required. `device` accepts `"auto"`, `"cpu"` or a PyTorch CUDA device
such as `"cuda:0"`; auto chooses CUDA when available. `decoder_iterations` must
be a positive integer. Invalid configuration, unavailable dependencies/devices,
and transport errors raise exceptions.

The returned `TransmissionResult` dataclass has these fields:

| Field | Meaning |
|---|---|
| `recovered_bytes` | Recovered byte container, or `None` if the length cannot be used to extract it |
| `payload_bits` | Original payload length × 8, excluding the length header |
| `coded_bits` | Total transmitted LDPC codeword bits, excluding QAM padding |
| `channel_symbols` | Actual number of complex QAM symbols transmitted |
| `bit_errors` | Errors in the original payload bit positions, excluding header and padding |
| `ber` | `bit_errors / payload_bits` |
| `block_errors` | Blocks with any wrong decoded information bit, including header/padding positions |
| `num_blocks` | Number of LDPC blocks |
| `header_valid` | Whether the decoded length passes the current plausibility check; not a checksum |
| `actual_ldpc_rate` | Realized per-block `k/n`, including information padding |

`header_valid` checks `0 < decoded_length <= 2 * original_payload_length`.
Extraction also requires that the indicated payload fits inside the decoded
information buffer; consequently `header_valid=True` can coexist with
`recovered_bytes=None`. It does not establish equality to the original length or
payload. BER and this plausibility check use transmitted information known to
the simulator; they are not standalone receiver-side integrity tests.

## Reproducible SNR sweeps

```python
from sscc import resolve_channel_config, transmit_bytes

payload = bytes(range(128))
for snr_db in (0, 10, 20):
    config = resolve_channel_config(snr_db, modulation_order=16, ldpc_rate=0.5)
    for repeat in range(2):
        result = transmit_bytes(payload, config, seed=42 + repeat, device="cpu")
        print(snr_db, repeat, result.ber, result.recovered_bytes == payload)
```

This example holds modulation/rate fixed. Omit those overrides for an adaptive
MCS comparison. The function resets both Sionna's RNG and PyTorch's global RNG
on each call. Identical parameters, seeds and runtime reproduce the simulation;
cross-device/version bitwise identity is not guaranteed. Avoid concurrently
calling it from threads that share those generators; separate processes provide
independent simulation state. Each call currently constructs its channel and
LDPC objects anew.

## Connect a source codec to the channel

This BPG example needs an installed `bpgenc`/`bpgdec` and a Kodak image at the
indicated path. Run from the repository root after [codec installation](codecs.md).

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image
from sscc import resolve_channel_config, transmit_bytes
from sscc.source.bpg import BPGCodec
from sscc.codecs import create_decoder

reference = Path("data/kodak/kodim01.png")
with TemporaryDirectory() as directory:
    root = Path(directory)
    source = BPGCodec().encode_decode(
        reference, 48, root / "source.bpg", root / "source-recon.png",
    )
    payload = source.bitstream.read_bytes()
    with Image.open(reference) as image:
        width, height = image.size
    config = resolve_channel_config(20, compression_ratio=96)
    if not config.payload_fits(len(payload), width, height):
        raise ValueError("Choose a smaller source stream or a larger symbol budget.")
    received = transmit_bytes(payload, config, seed=42, device="cpu")
    if received.recovered_bytes is None:
        raise RuntimeError("No extractable payload was recovered.")
    reconstruction = Path("results/library-api/reconstruction.png")
    create_decoder("BPG").decode(
        received.recovered_bytes, reference, "q48", reconstruction,
    )
    print("Saved:", reconstruction, "BER:", received.ber)
```

Source decoding may still raise on damaged bytes. The benchmark CLI records this
and applies its documented fallback. The library caller can choose its own error
handling. MS-ILLM and ELiC use the same byte transport, with the
[isolated learned-codec environment](learned-codecs.md) for source coding/decoding.
