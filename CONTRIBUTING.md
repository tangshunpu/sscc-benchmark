# Contributing

Use Python 3.12 and `uv sync --locked --extra channel --extra dev`.

```bash
uv run --no-sync ruff check src tests
uv run --no-sync pytest -q
uv run --no-sync python -m build
```

The BPG integration test runs when `bpgenc` and `bpgdec` are on PATH. Set
`SSCC_VTM_ENCODER`, `SSCC_VTM_DECODER`, `SSCC_VTM_CONFIG` to enable the real VTM
test. Channel integration tests run on CPU and do not require images or model
downloads. Synthetic images are generated inside the test temporary directory.
CI tests the CPU channel and package; native codec integration requires an
installation and is explicitly skipped when unavailable.

Preserve exact channel-use accounting, record metric/failure convention changes,
and add behavioral tests for new codec formats, naming rules or MCS calculations.
Do not commit datasets, private paths, checkpoints, source streams or native
binaries. Keep optional imports lazy. A new method must document upstream
source, checkpoint requirements, container format and compatibility constraints.

Before publishing a benchmark, record the dataset split, image count, source QP
grid, CR, SNR definition, MCS settings, repeats/seeds, fallback rate, dependency
versions and weight/native-codec revisions. Do not present small smoke runs as
full-dataset benchmark results.
