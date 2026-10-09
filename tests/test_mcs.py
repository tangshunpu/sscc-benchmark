import pytest

from sscc.channel.mcs import MCS_BY_INDEX, auto_mcs, resolve_channel_config, select_auto_mcs


def test_auto_mcs_boundaries():
    assert select_auto_mcs(0).entry.index == 3
    assert select_auto_mcs(5).entry.index == 8
    assert select_auto_mcs(10).entry.index == 15
    assert select_auto_mcs(15).entry.index == 22
    assert select_auto_mcs(20).entry.index == 28
    assert auto_mcs(10) == (16, 616 / 1024)


def test_3gpp_mcs_table_1_rows():
    assert MCS_BY_INDEX[0].spectral_efficiency == pytest.approx(0.234375)
    assert MCS_BY_INDEX[10].modulation_order == 16
    assert MCS_BY_INDEX[10].target_code_rate_x1024 == 340
    assert MCS_BY_INDEX[28].modulation_order == 64
    assert MCS_BY_INDEX[28].code_rate == 948 / 1024


def test_explicit_parameters_override_auto_mcs():
    config = resolve_channel_config(0, compression_ratio=48, modulation_order=64, ldpc_rate=0.75)
    assert config.modulation_order == 64
    assert config.bits_per_symbol == 6
    assert config.ldpc_rate == 0.75
    assert config.channel_symbols(8, 4) == 2
    assert config.payload_capacity_bits(8, 4) == 9
    assert config.mcs_index is None
    assert config.mcs_selection == "manual_override"


def test_explicit_standard_mcs_index():
    config = resolve_channel_config(0, mcs_index=10)
    assert config.mcs_index == 10
    assert config.modulation_order == 16
    assert config.ldpc_rate == 340 / 1024
    assert config.mcs_selection == "explicit_3gpp_mcs"


def test_unsupported_low_rate_mcs_is_rejected():
    with pytest.raises(ValueError, match="indices 3--28"):
        resolve_channel_config(0, mcs_index=2)


def test_mcs_index_and_manual_override_are_mutually_exclusive():
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_channel_config(10, mcs_index=10, ldpc_rate=0.5)


def test_transmission_shape_includes_header_and_padding():
    config = resolve_channel_config(20, modulation_order=16, ldpc_rate=0.5)
    k, n, blocks, symbols = config.transmission_shape(2000)
    assert (k, n, blocks) == (8016, 16032, 2)
    assert symbols == 8016


def test_low_rate_segmentation_respects_maximum_codeword_length():
    config = resolve_channel_config(0, mcs_index=3)
    k, n, blocks, _ = config.transmission_shape(2000)
    assert blocks == 5
    assert k <= 3824
    assert k <= int((68 * 384) * config.ldpc_rate)
    assert n <= 68 * 384


@pytest.mark.parametrize("rate", [0, 0.1, 0.99, 1.1])
def test_invalid_ldpc_rate(rate):
    with pytest.raises(ValueError):
        resolve_channel_config(0, ldpc_rate=rate)
