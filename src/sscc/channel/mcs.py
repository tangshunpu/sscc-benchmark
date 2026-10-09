from __future__ import annotations

from dataclasses import dataclass
import math


SUPPORTED_MODULATION_ORDERS = (4, 16, 64)
MIN_5G_LDPC_RATE = 1 / 5
MAX_5G_LDPC_RATE = 948 / 1024
DEFAULT_MCS_SNR_GAP_DB = 3.0


@dataclass(frozen=True)
class MCSEntry:
    """One row from 3GPP TS 38.214 Table 5.1.3.1-1."""

    index: int
    modulation_order_bits: int
    target_code_rate_x1024: int

    @property
    def modulation_order(self) -> int:
        return 1 << self.modulation_order_bits

    @property
    def code_rate(self) -> float:
        return self.target_code_rate_x1024 / 1024

    @property
    def spectral_efficiency(self) -> float:
        return self.modulation_order_bits * self.code_rate


def _mcs(index: int, qm: int, rate: int) -> MCSEntry:
    return MCSEntry(index, qm, rate)


# 3GPP TS 38.214 Table 5.1.3.1-1, reserved rows 29--31 omitted.
MCS_TABLE_1 = (
    _mcs(0, 2, 120),
    _mcs(1, 2, 157),
    _mcs(2, 2, 193),
    _mcs(3, 2, 251),
    _mcs(4, 2, 308),
    _mcs(5, 2, 379),
    _mcs(6, 2, 449),
    _mcs(7, 2, 526),
    _mcs(8, 2, 602),
    _mcs(9, 2, 679),
    _mcs(10, 4, 340),
    _mcs(11, 4, 378),
    _mcs(12, 4, 434),
    _mcs(13, 4, 490),
    _mcs(14, 4, 553),
    _mcs(15, 4, 616),
    _mcs(16, 4, 658),
    _mcs(17, 6, 438),
    _mcs(18, 6, 466),
    _mcs(19, 6, 517),
    _mcs(20, 6, 567),
    _mcs(21, 6, 616),
    _mcs(22, 6, 666),
    _mcs(23, 6, 719),
    _mcs(24, 6, 772),
    _mcs(25, 6, 822),
    _mcs(26, 6, 873),
    _mcs(27, 6, 910),
    _mcs(28, 6, 948),
)
MCS_BY_INDEX = {entry.index: entry for entry in MCS_TABLE_1}


@dataclass(frozen=True)
class CQIEntry:
    index: int
    spectral_efficiency: float


# 3GPP TS 38.214 CQI Table 1 spectral-efficiency levels. CQI 0 is OOR.
CQI_TABLE_1 = (
    CQIEntry(1, 0.1523),
    CQIEntry(2, 0.2344),
    CQIEntry(3, 0.3770),
    CQIEntry(4, 0.6016),
    CQIEntry(5, 0.8770),
    CQIEntry(6, 1.1758),
    CQIEntry(7, 1.4766),
    CQIEntry(8, 1.9141),
    CQIEntry(9, 2.4063),
    CQIEntry(10, 2.7305),
    CQIEntry(11, 3.3223),
    CQIEntry(12, 3.9023),
    CQIEntry(13, 4.5234),
    CQIEntry(14, 5.1152),
    CQIEntry(15, 5.5547),
)


@dataclass(frozen=True)
class AutoMCSSelection:
    entry: MCSEntry
    cqi_index: int
    usable_spectral_efficiency: float


@dataclass(frozen=True)
class ChannelConfig:
    snr_db: float
    modulation_order: int
    ldpc_rate: float
    compression_ratio: float = 96.0
    mcs_index: int | None = None
    cqi_index: int | None = None
    mcs_snr_gap_db: float = DEFAULT_MCS_SNR_GAP_DB
    mcs_selection: str = "manual_override"

    @property
    def bits_per_symbol(self) -> int:
        return int(math.log2(self.modulation_order))

    @property
    def target_code_rate_x1024(self) -> float:
        return self.ldpc_rate * 1024

    @property
    def spectral_efficiency(self) -> float:
        return self.bits_per_symbol * self.ldpc_rate

    def channel_symbols(self, width: int, height: int) -> int:
        """Complex channel-use budget, following TF_SemCom's 3HW/CR rule."""
        return int(3 * width * height / self.compression_ratio)

    def payload_capacity_bits(self, width: int, height: int) -> int:
        return int(self.channel_symbols(width, height) * self.spectral_efficiency)

    def transmission_shape(self, payload_bytes: int) -> tuple[int, int, int, int]:
        """Return (k, n, blocks, QAM symbols) including the 32-bit length header."""
        if payload_bytes <= 0:
            raise ValueError("payload_bytes must be positive.")
        information_bits = 32 + payload_bytes * 8
        # Balance information across code blocks instead of fixing k=8448 and
        # heavily padding the tail block. Also keep n=ceil(k/R) within the 5G
        # LDPC rate-matching limit used by Sionna.
        # Sionna follows TS 38.212 base-graph selection: rates <=1/3 must
        # remain within BG2's k<=3824 region because BG1 rejects R<1/3.
        max_k_by_base_graph = 3824 if self.ldpc_rate <= 1 / 3 else 8448
        max_k_for_rate = min(
            max_k_by_base_graph,
            math.floor((68 * 384) * self.ldpc_rate),
        )
        blocks = max(1, math.ceil(information_bits / max_k_for_rate))
        k = max(12, math.ceil(information_bits / blocks))
        # Ceil keeps the realized k/n at or below the 3GPP target code rate.
        n = min(math.ceil(k / self.ldpc_rate), 68 * 384)
        symbols_per_block = (n + self.bits_per_symbol - 1) // self.bits_per_symbol
        return k, n, blocks, blocks * symbols_per_block

    def payload_fits(self, payload_bytes: int, width: int, height: int) -> bool:
        return self.transmission_shape(payload_bytes)[3] <= self.channel_symbols(width, height)


def usable_spectral_efficiency(snr_db: float, snr_gap_db: float) -> float:
    """AWGN Shannon capacity after an explicit implementation/link margin."""
    effective_snr_linear = 10 ** ((snr_db - snr_gap_db) / 10)
    return math.log2(1 + effective_snr_linear)


def select_auto_mcs(
    snr_db: float,
    snr_gap_db: float = DEFAULT_MCS_SNR_GAP_DB,
) -> AutoMCSSelection:
    """Map SNR to CQI Table 1, then to a supported MCS Table 1 row.

    3GPP specifies the CQI and MCS tables but deliberately does not standardize
    an SNR-to-CQI receiver algorithm. This deterministic baseline uses an AWGN
    capacity estimate with a configurable SNR gap and quantizes it through CQI.
    """
    if not math.isfinite(snr_db) or not math.isfinite(snr_gap_db):
        raise ValueError("snr_db and snr_gap_db must be finite.")
    usable = usable_spectral_efficiency(snr_db, snr_gap_db)
    cqi = CQI_TABLE_1[0]
    for candidate in CQI_TABLE_1:
        if candidate.spectral_efficiency <= usable:
            cqi = candidate

    supported = [
        entry
        for entry in MCS_TABLE_1
        if entry.code_rate >= MIN_5G_LDPC_RATE
        and entry.spectral_efficiency <= cqi.spectral_efficiency + 1e-4
    ]
    # Sionna's 5G LDPC rate matcher supports R>=1/5, so MCS 0--2 cannot
    # currently be represented. MCS 3 is the conservative supported floor.
    entry = supported[-1] if supported else MCS_BY_INDEX[3]
    return AutoMCSSelection(entry, cqi.index, usable)


def auto_mcs(
    snr_db: float,
    snr_gap_db: float = DEFAULT_MCS_SNR_GAP_DB,
) -> tuple[int, float]:
    selection = select_auto_mcs(snr_db, snr_gap_db)
    return selection.entry.modulation_order, selection.entry.code_rate


def resolve_channel_config(
    snr_db: float,
    compression_ratio: float = 96.0,
    modulation_order: int | None = None,
    ldpc_rate: float | None = None,
    mcs_index: int | None = None,
    mcs_snr_gap_db: float = DEFAULT_MCS_SNR_GAP_DB,
) -> ChannelConfig:
    if mcs_index is not None and (modulation_order is not None or ldpc_rate is not None):
        raise ValueError(
            "mcs_index cannot be combined with modulation_order or ldpc_rate overrides."
        )

    auto = select_auto_mcs(snr_db, mcs_snr_gap_db)
    if mcs_index is not None:
        if mcs_index not in MCS_BY_INDEX:
            raise ValueError(f"mcs_index must be in [0, 28], got {mcs_index}.")
        selected = MCS_BY_INDEX[mcs_index]
        if selected.code_rate < MIN_5G_LDPC_RATE:
            raise ValueError(
                f"MCS {mcs_index} has R={selected.code_rate:.4f}; the current "
                "Sionna 5G LDPC transport supports MCS indices 3--28 (R>=1/5)."
            )
        order = selected.modulation_order
        rate = selected.code_rate
        resolved_mcs_index = selected.index
        cqi_index = None
        selection_name = "explicit_3gpp_mcs"
    else:
        selected = auto.entry
        order = selected.modulation_order if modulation_order is None else modulation_order
        rate = selected.code_rate if ldpc_rate is None else ldpc_rate
        has_override = modulation_order is not None or ldpc_rate is not None
        resolved_mcs_index = None if has_override else selected.index
        cqi_index = auto.cqi_index
        selection_name = "manual_override" if has_override else "auto_3gpp_cqi"

    if order not in SUPPORTED_MODULATION_ORDERS:
        raise ValueError(
            f"modulation_order must be one of {SUPPORTED_MODULATION_ORDERS}, got {order}."
        )
    if not MIN_5G_LDPC_RATE <= rate <= MAX_5G_LDPC_RATE:
        raise ValueError(
            "ldpc_rate must be supported by 5G LDPC "
            f"([{MIN_5G_LDPC_RATE:g}, {MAX_5G_LDPC_RATE:g}]), got {rate}."
        )
    if compression_ratio <= 0:
        raise ValueError("compression_ratio must be positive.")
    return ChannelConfig(
        snr_db=float(snr_db),
        modulation_order=order,
        ldpc_rate=float(rate),
        compression_ratio=float(compression_ratio),
        mcs_index=resolved_mcs_index,
        cqi_index=cqi_index,
        mcs_snr_gap_db=float(mcs_snr_gap_db),
        mcs_selection=selection_name,
    )
