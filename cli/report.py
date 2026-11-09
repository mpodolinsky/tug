"""python -m cli.report [--datasets ...] [--csv results/areas.csv] [--sizes-csv results/split_sizes.csv]

Read the evaluation summaries under results/ (python -m cli.evaluate first) and print: the number of episodes in every part of every split
(train, val, calibration, seen, unseen, with failures and successes); the marginal means per factor and the architecture chosen on
validation for every label variant, under each calibrator; and the calibrators side by side. Writes the long table of areas (one row per
run x calibrator x evaluation set) and the split sizes as CSV.
"""

import argparse
import csv
from pathlib import Path

from fp.calibration.protocol import MAX_CONFORMAL, SAFE_BAND
from fp.config import RESULTS
from fp.data.inputs import load_inputs
from fp.eval.collect import collect_rows, write_csv
from fp.eval.compute import collect_costs
from fp.eval.report import report, split_sizes
from fp.grid import DATASETS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=RESULTS)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--csv", type=Path, default=RESULTS / "areas.csv")
    p.add_argument("--sizes-csv", type=Path, default=RESULTS / "split_sizes.csv")
    p.add_argument("--calibrators", nargs="+", default=[MAX_CONFORMAL, SAFE_BAND], choices=[MAX_CONFORMAL, SAFE_BAND])
    args = p.parse_args()
    rows = collect_rows(args.root, tuple(args.datasets))
    if not rows:
        raise SystemExit("no evaluated runs found; run python -m cli.evaluate first")
    sizes = []
    for dataset in sorted({r["dataset"] for r in rows}, key=DATASETS.index):
        inputs = load_inputs(dataset)
        sizes += split_sizes(dataset, inputs.splits, inputs.episodes)
    write_csv(rows, args.csv)
    with open(args.sizes_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sizes[0]))
        writer.writeheader()
        writer.writerows(sizes)
    print(report(rows, tuple(args.calibrators), sizes, collect_costs(args.root, tuple(args.datasets))))
    print(f"\nlong table: {args.csv} ({len(rows)} rows); split sizes: {args.sizes_csv} ({len(sizes)} splits)")


if __name__ == "__main__":
    main()
