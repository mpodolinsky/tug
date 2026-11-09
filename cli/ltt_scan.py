"""python -m cli.ltt_scan [--variants onset_pre] [--delta 0.1]

Which LTT targets (alpha_tpr_early, alpha_fpr) can be certified? For each pair on a small grid, over the saved runs of the chosen label
variants: the share of runs for which a threshold is certified on the calibration episodes, and what that threshold does on the seen
and unseen episodes (mean TPR early and FPR over the certified runs). Nothing is written. The guarantee holds for episodes exchangeable
with the calibration set, so the unseen columns show how far it carries to held-out tasks; they are not part of the guarantee.
The targets are a decision: pick them from this table and pass them to  python -m cli.evaluate --ltt ...
"""

import argparse
from pathlib import Path

import numpy as np

from fp.calibration.protocol import parse_probe_name
from fp.config import RESULTS
from fp.data.inputs import load_inputs
from fp.eval.run_eval import scan_ltt
from fp.eval.select_config import group_of
from fp.grid import DATASETS
from fp.results import iter_runs, read_scores

TPR_EARLY_ALPHAS = (0.5, 0.6, 0.7, 0.8)  # certify TPR early >= 1 - alpha: 50%, 40%, 30%, 20%
FPR_ALPHAS = (0.2, 0.3, 0.4)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=RESULTS)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--variants", nargs="+", default=["onset_pre"])
    p.add_argument("--delta", type=float, default=0.1, help="a scan setting only; the evaluation itself has no default")
    args = p.parse_args()
    targets = [(a, b) for a in TPR_EARLY_ALPHAS for b in FPR_ALPHAS]
    cells: dict[tuple, list[dict]] = {}
    for dataset in args.datasets:
        inputs = load_inputs(dataset)
        for _, split_name, method, path in iter_runs(args.root, (dataset,)):
            probe = parse_probe_name(method)
            if probe is None or probe["variant"] not in args.variants:
                continue
            outs = scan_ltt(read_scores(path), inputs.splits[split_name], inputs.episodes, inputs.window, targets, args.delta)
            for target, out in zip(targets, outs):
                cells.setdefault((dataset, group_of(split_name), target), []).append(out)
    print(f"LTT scan, delta {args.delta}, label variants {args.variants}")
    for dataset, group in sorted({(d, g) for d, g, _ in cells}):
        print(f"\n{dataset} / {group}  ({len(cells[(dataset, group, targets[0])])} runs per row)\n"
              f"  {'TPRe >=':>8s}{'FPR <=':>8s}{'certified':>11s}{'seen TPRe':>11s}{'seen FPR':>10s}{'unseen TPRe':>13s}{'unseen FPR':>12s}")
        for a, b in targets:
            outs = cells[(dataset, group, (a, b))]
            ok = [o for o in outs if o["certified"]]
            mean = lambda s, k: np.mean([o["sets"][s][k] for o in ok]) if ok else float("nan")
            print(f"  {100 * (1 - a):7.0f}%{100 * b:7.0f}%{100 * len(ok) / len(outs):10.0f}%{mean('seen', 'tpr_early'):11.1f}{mean('seen', 'fpr'):10.1f}"
                  f"{mean('unseen', 'tpr_early'):13.1f}{mean('unseen', 'fpr'):12.1f}")


if __name__ == "__main__":
    main()
