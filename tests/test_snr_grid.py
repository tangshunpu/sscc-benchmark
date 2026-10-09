from argparse import Namespace

import pytest

from sscc.cli.benchmark import inclusive_snr_grid, resolve_snr_values


def test_default_three_db_grid_includes_twenty_db_endpoint():
    assert inclusive_snr_grid(0, 20, 3) == (0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 20.0)


def test_aligned_grid_does_not_duplicate_endpoint():
    assert inclusive_snr_grid(0, 18, 3) == (0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0)


def test_explicit_snr_values_are_preserved():
    args = Namespace(snr_db=[1.0, 4.5], snr_start=None, snr_stop=None, snr_interval=None)
    assert resolve_snr_values(args) == (1.0, 4.5)


@pytest.mark.parametrize("start,stop,interval", [(1, 0, 1), (0, 1, 0), (0, 1, -1)])
def test_invalid_grid_is_rejected(start, stop, interval):
    with pytest.raises(ValueError):
        inclusive_snr_grid(start, stop, interval)
