"""Run after installing .[channel]: python examples/transmit.py."""
from sscc import resolve_channel_config, transmit_bytes

config = resolve_channel_config(30, modulation_order=4, ldpc_rate=0.5)
payload = b"Separate source and channel coding" * 8
result = transmit_bytes(payload, config, seed=42, device="cpu")
assert result.recovered_bytes == payload
print(f"Recovered {len(payload)} bytes, BER={result.ber}, symbols={result.channel_symbols}")
