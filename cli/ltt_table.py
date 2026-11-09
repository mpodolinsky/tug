"""python -m cli.ltt_table --delta 0.1 [--dataset liberox_groot] [--method lstm_modes_clip__onset_pre] [--n-starts 40] [--fpr-targets 0.1 0.2 ...]
                          [--groups all ...] [--out results/figures]

LTT against max-conformal at the same FPR targets (fp/eval/ltt_compare.py), from the saved scores of one probe over its runs in the dataset
(python -m cli.train first; no model is run and no evaluation is needed). Writes table_ltt.tex and table_ltt.csv, and table_ltt_compact.tex (the
rows of three FPR targets, no PPV columns, the mean rate with the share of runs in which the target holds in parentheses; the full table is for the appendix):

    for each FPR target, an LTT row (certifying the FPR target together with the highest TPR-early target that at least half of the runs certify)
    and a max-conformal row (no TPR-early target); per row the runs with a threshold and, on the seen and on the unseen episodes, the share of runs
    in which each target holds and the mean and spread of FPR and TPR early; and the PPV: the lower bound the two targets give on the probability of
    failure given an alarm (an LTT row only), and on each set the realised PPV and the share of runs in which it is at least that bound (fp/metrics/ppv.py).

and figure_ppv (.png and .pdf): the bound against the failure prevalence for three of the rows, with the realised PPV of the seen and the unseen set.

--delta has no default, as in  python -m cli.evaluate --ltt: the confidence is a decision. The number of walks (--n-starts) defaults to 40.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from cli.figures import write_csv
from fp.config import RESULTS
from fp.data.inputs import load_inputs
from fp.eval import ltt_compare, plots, tables
from fp.eval.figures import OURS_ARCHITECTURE
from fp.eval.select_config import group_of
from fp.grid import DATASETS
from fp.results import iter_runs, read_scores


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=RESULTS)
    p.add_argument("--out", type=Path, default=RESULTS / "figures")
    p.add_argument("--dataset", default="liberox_groot", choices=DATASETS)
    p.add_argument("--method", default=f"{OURS_ARCHITECTURE}__onset_pre", help="the probe, e.g. lstm_modes_clip__onset_pre")
    p.add_argument("--delta", type=float, required=True, help="the confidence of the LTT guarantee is 1 - delta")
    p.add_argument("--n-starts", type=int, default=ltt_compare.N_STARTS, help="number of fixed-sequence walks of LTT")
    p.add_argument("--fpr-targets", type=float, nargs="+", default=list(ltt_compare.FPR_TARGETS))
    p.add_argument("--groups", nargs="+", help="only the splits of these groups (fp.eval.select_config.group_of), e.g. all loto_seeds on the real robot; "
                                               "default: every split with a run")
    args = p.parse_args()

    inputs = load_inputs(args.dataset)
    selected = [(split, path) for _, split, method, path in iter_runs(args.root, (args.dataset,))
                if method == args.method and (args.groups is None or group_of(split) in args.groups)]
    if not selected:
        sys.exit(f"no runs of {args.method} on {args.dataset} under {args.root}")
    runs = [ltt_compare.prepare(read_scores(path), inputs.splits[split], inputs.episodes, inputs.window) for split, path in selected]
    kinds = ltt_compare.kinds_text(Counter(group_of(split) for split, _ in selected))
    rows = ltt_compare.compare(runs, args.fpr_targets, delta=args.delta, n_starts=args.n_starts)

    args.out.mkdir(parents=True, exist_ok=True)
    source = f"python -m cli.ltt_table --dataset {args.dataset} --method {args.method} --delta {args.delta:g} --n-starts {args.n_starts}"
    (args.out / "table_ltt.tex").write_text(tables.ltt_latex_table(rows, args.delta, args.n_starts, source=source, kinds=kinds))
    write_csv(args.out / "table_ltt.csv", tables.ltt_csv_rows(rows))
    print(f"{len(runs)} runs of {args.method} on {args.dataset}; wrote", args.out / "table_ltt.tex", "and table_ltt.csv")
    if tables.ltt_compact_rows(rows):
        (args.out / "table_ltt_compact.tex").write_text(tables.ltt_compact_latex_table(rows, args.delta, source=source))
        print("wrote", args.out / "table_ltt_compact.tex", f"(the FPR targets {', '.join(f'{100 * t:g}%' for t in tables.COMPACT_FPR_TARGETS)})")
    else:
        print("no compact table: none of its FPR targets is among --fpr-targets")
    try:
        print("wrote", *plots.save(plots.ppv_figure(rows), "figure_ppv", args.out))
    except ValueError as error:
        print(f"no PPV figure: {error}")


if __name__ == "__main__":
    main()
