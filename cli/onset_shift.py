"""python -m cli.onset_shift [--root results] [--chosen results/figures/chosen.csv] [--out results/figures] [--seconds 0 1 2]

How the early-detection area of the methods of the main table changes when the failure time of every failed episode is moved later by --seconds (fp/eval/onset_shift.py):
the saved scores are scored again against the shifted failure times, with the configurations the main table chose (--chosen, written by python -m cli.figures) and their
own calibrators, so no model is run and nothing is retrained. Writes table_onset_shift.csv (mean and standard deviation over the splits of each setting, for the seen and
the unseen episodes) and the LaTeX tables of the means, table_onset_shift.tex (unseen) and table_onset_shift_seen.tex.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from cli.figures import write_csv
from fp.config import RESULTS
from fp.data.inputs import load_inputs
from fp.eval import figures, onset_shift, tables


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=RESULTS)
    p.add_argument("--chosen", type=Path, default=RESULTS / "figures" / "chosen.csv")
    p.add_argument("--out", type=Path, default=RESULTS / "figures")
    p.add_argument("--seconds", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    args = p.parse_args()

    with open(args.chosen) as f:
        chosen = {(r["setting"], r["entry"]): r["method"] for r in csv.DictReader(f)}
    inputs = {}
    records = []
    for setting in figures.SETTINGS:
        inputs[setting.dataset] = inputs.get(setting.dataset) or load_inputs(setting.dataset)
        for label in figures.TABLE_LABELS:
            method = chosen.get((setting.label, label))
            if method is None:
                continue
            areas = onset_shift.shifted_areas(inputs[setting.dataset], args.root, method, setting, tuple(args.seconds))
            for (seconds, eval_set), values in areas.items():
                if values:
                    records.append({"setting": setting.label, "method": label, "set": eval_set, "shift (s)": seconds, "steps": onset_shift.shift_steps(seconds, setting.dataset),
                                    "area mean": round(float(np.mean(values)), 3), "area std": round(float(np.std(values)), 3), "splits": len(values)})
            print(f"{setting.label:18s} {label:18s} done", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "table_onset_shift.csv", records)
    settings, shifts = [s.label for s in figures.SETTINGS], list(args.seconds)
    for eval_set, stem in (("unseen", "table_onset_shift"), ("seen", "table_onset_shift_seen")):
        rows = onset_shift.shift_rows(records, settings, list(figures.TABLE_LABELS), shifts, eval_set)
        (args.out / f"{stem}.tex").write_text(tables.onset_shift_latex_table(rows, settings, shifts, eval_set, label=f"tab:{stem[len('table_'):]}"))
    print("wrote", args.out / "table_onset_shift.csv", "table_onset_shift.tex (the unseen episodes) and table_onset_shift_seen.tex")


if __name__ == "__main__":
    main()
