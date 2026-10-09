"""Collect benchmark averages into one CSV without merging distinct runs."""
import argparse
import csv
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_root", type=Path)
    p.add_argument("--output", type=Path, default=Path("benchmark.csv"))
    args = p.parse_args(argv)
    rows = []
    for path in sorted(args.results_root.rglob("average_metrics.csv")):
        with path.open(newline="") as handle:
            rows.extend(dict(row, run_path=str(path.parent)) for row in csv.DictReader(handle))
    if not rows:
        p.error("No average_metrics.csv files found")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
