from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np

from .mcs import ChannelConfig


@dataclass(frozen=True)
class TransmissionResult:
    recovered_bytes: bytes | None
    payload_bits: int
    coded_bits: int
    channel_symbols: int
    bit_errors: int
    ber: float
    block_errors: int
    num_blocks: int
    header_valid: bool
    actual_ldpc_rate: float


def bytes_to_bits(data: bytes):
    import torch

    values = np.frombuffer(data, dtype=np.uint8)
    return torch.from_numpy(np.unpackbits(values).astype(np.float32))


def bits_to_bytes(bits) -> bytes:
    values = bits.detach().cpu().numpy().round().astype(np.uint8)
    padding = (-len(values)) % 8
    if padding:
        values = np.pad(values, (0, padding))
    return np.packbits(values).tobytes()


def resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def transmit_bytes(
    payload: bytes,
    config: ChannelConfig,
    seed: int,
    device: str = "auto",
    decoder_iterations: int = 20,
) -> TransmissionResult:
    """Transmit a complete byte container through 5G LDPC, QAM and AWGN."""
    import torch
    from sionna.phy import config as sionna_config
    from sionna.phy.channel import AWGN
    from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
    from sionna.phy.mapping import Constellation, Demapper, Mapper

    if not payload:
        raise ValueError("payload cannot be empty.")
    if decoder_iterations <= 0:
        raise ValueError("decoder_iterations must be positive.")

    target_device = resolve_device(device)
    sionna_device = str(target_device)
    # Sionna 2.x uses its own per-device torch.Generator. Seeding only the
    # global PyTorch RNG would therefore repeat the same channel realization
    # whenever AWGN is reconstructed for a new image/repeat.
    sionna_config.seed = seed
    torch.manual_seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    payload_bits = bytes_to_bits(payload)
    length_bits = bytes_to_bits(struct.pack(">I", len(payload)))
    information = torch.cat((length_bits, payload_bits))

    k, n, num_blocks, expected_symbols = config.transmission_shape(len(payload))
    if n <= k:
        raise ValueError(
            f"The requested LDPC rate produces invalid parameters k={k}, n={n}."
        )
    padded_length = num_blocks * k
    if len(information) < padded_length:
        information = torch.cat((information, torch.zeros(padded_length - len(information))))
    information_blocks = information.reshape(num_blocks, k).to(target_device)

    # Sionna keeps some internal tensors outside normal Module parameters, so
    # constructor-level device selection is required; Module.to() alone is not enough.
    encoder = LDPC5GEncoder(k=k, n=n, device=sionna_device).to(target_device)
    decoder = LDPC5GDecoder(
        encoder, num_iter=decoder_iterations, device=sionna_device
    ).to(target_device)
    constellation = Constellation(
        "qam", num_bits_per_symbol=config.bits_per_symbol, device=sionna_device
    ).to(target_device)
    mapper = Mapper(constellation=constellation, device=sionna_device).to(target_device)
    demapper = Demapper(
        "app", constellation=constellation, device=sionna_device
    ).to(target_device)
    channel = AWGN(device=sionna_device).to(target_device)
    noise_variance = 10.0 ** (-config.snr_db / 10.0)

    decoded_blocks = []
    coded_bits = 0
    transmitted_symbols = 0
    block_errors = 0
    for block in information_blocks:
        block = block.unsqueeze(0)
        codeword = encoder(block)
        codeword_length = codeword.shape[1]
        coded_bits += codeword_length
        modulation_padding = (-codeword_length) % config.bits_per_symbol
        if modulation_padding:
            codeword = torch.cat(
                (codeword, torch.zeros((1, modulation_padding), device=target_device)), dim=1
            )
        symbols = mapper(codeword)
        transmitted_symbols += symbols.numel()
        received = channel(symbols, noise_variance)
        llr = demapper(received, noise_variance)
        if modulation_padding:
            llr = llr[:, :codeword_length]
        decoded = decoder(llr)
        decoded_blocks.append(decoded)
        if torch.any(decoded.round() != block):
            block_errors += 1

    decoded_information = torch.cat(decoded_blocks, dim=1).squeeze(0).round()
    if transmitted_symbols != expected_symbols:
        raise RuntimeError(
            f"Internal channel-symbol mismatch: {transmitted_symbols} != {expected_symbols}."
        )
    decoded_payload_bits = decoded_information[32 : 32 + len(payload_bits)]
    bit_errors = int((decoded_payload_bits != payload_bits.to(target_device)).sum().item())
    ber = bit_errors / len(payload_bits)

    header_bytes = bits_to_bytes(decoded_information[:32])
    recovered_length = struct.unpack(">I", header_bytes[:4])[0]
    header_valid = 0 < recovered_length <= len(payload) * 2
    recovered = None
    if header_valid:
        end = 32 + recovered_length * 8
        if end <= len(decoded_information):
            recovered = bits_to_bytes(decoded_information[32:end])[:recovered_length]

    return TransmissionResult(
        recovered_bytes=recovered,
        payload_bits=len(payload_bits),
        coded_bits=coded_bits,
        channel_symbols=transmitted_symbols,
        bit_errors=bit_errors,
        ber=ber,
        block_errors=block_errors,
        num_blocks=num_blocks,
        header_valid=header_valid,
        actual_ldpc_rate=k / n,
    )
