"""python -m cli.evaluate [--datasets ...] [--splits ...] [--methods ...] [--workers N] [--secondary] [--force]

Evaluate the saved runs under results/ (no model is run): calibrate on each split's calibration successes, sweep alpha on its validation,
seen and unseen episodes, and write curve_<calibrator>.json and summary.json into every run folder. An evaluation made by the current
code is skipped unless --force. Our probes get the protocol's max-conformal calibrator; --secondary adds the SAFE band for them.

LTT is a separate evaluation and its targets have no defaults:
    python -m cli.evaluate --ltt --alpha-tpr-early 0.3 --alpha-fpr 0.2 --delta 0.1
writes ltt.json into every run folder (python -m cli.ltt_scan shows which targets are certifiable first).
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from fp.config import RESULTS
from fp.data.inputs import Inputs, load_inputs
from fp.eval.run_eval import evaluate_ltt, evaluate_saved_run
from fp.grid import DATASETS
from fp.results import iter_runs, read_scores, write_json

_INPUTS: dict[str, Inputs] = {}


def inputs_of(dataset: str) -> Inputs:
    if dataset not in _INPUTS:
        _INPUTS[dataset] = load_inputs(dataset)
    return _INPUTS[dataset]


def work(job: tuple) -> tuple[str, str]:
    dataset, split, method, path, secondary, force, ltt = job
    try:
        if ltt is None:
            return str(path), evaluate_saved_run(dataset, split, method, Path(path), inputs_of(dataset), secondary, force)
        inputs = inputs_of(dataset)
        out = evaluate_ltt(read_scores(Path(path)), inputs.splits[split], inputs.episodes, inputs.window, **ltt)
        write_json(Path(path) / "ltt.json", out)
        return str(path), "done"
    except Exception as error:  # one broken run must not stop the others
        return str(path), f"FAILED {type(error).__name__}: {error}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=RESULTS)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--splits", nargs="+", help="only these split names")
    p.add_argument("--methods", nargs="+", help="only these methods, e.g. lstm_plain__onset_pre")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--secondary", action="store_true", help="also evaluate our probes with the other calibrator (SAFE band)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--ltt", action="store_true")
    p.add_argument("--alpha-tpr-early", type=float)
    p.add_argument("--alpha-fpr", type=float)
    p.add_argument("--delta", type=float)
    args = p.parse_args()

    ltt = None
    if args.ltt:
        if None in (args.alpha_tpr_early, args.alpha_fpr, args.delta):
            sys.exit("--ltt needs --alpha-tpr-early, --alpha-fpr and --delta: they have no defaults")
        ltt = {"alpha_tpr_early": args.alpha_tpr_early, "alpha_fpr": args.alpha_fpr, "delta": args.delta}
    jobs = [(d, s, m, str(path), args.secondary, args.force, ltt) for d, s, m, path in iter_runs(args.root, tuple(args.datasets))
            if (not args.splits or s in args.splits) and (not args.methods or m in args.methods)]
    print(f"{len(jobs)} runs to evaluate with {args.workers} workers", flush=True)
    start, counts = time.time(), {}
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (path, status) in enumerate(pool.map(work, jobs, chunksize=4), 1):
            counts[status.split(" ")[0]] = counts.get(status.split(" ")[0], 0) + 1
            if status.startswith("FAILED"):
                print(path, status, flush=True)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)} ({time.time() - start:.0f}s)", flush=True)
    print(f"finished in {time.time() - start:.0f}s: {counts}")


if __name__ == "__main__":
    main()
