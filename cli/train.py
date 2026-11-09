"""python -m cli.train [--datasets ...] [--variants ...] [--splits ...] [--probes ...] [--worker I/N] [--dry-run] [--preflight-only]

Train the probe grid (16 architectures x 4 label variants) on every split of every dataset and write each run under results/:
scores of every episode, the outputs of the mode and CLIP heads, weights and history. Runs that exist are skipped, so the command can be repeated to resume.
Several workers can share the grid: --worker 0/3, --worker 1/3, --worker 2/3 take every third unit of the run order.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from fp.config import RESULTS
from fp.data.features import feature_lengths
from fp.data.inputs import load_inputs
from fp.grid import DATASETS, VARIANTS, all_split_names, units
from fp.models.ours import all_configs
from fp.results import is_done, run_dir
from fp.train.preflight import check_episodes, check_split
from fp.train.sweep import Settings, train_unit


def log(message: str) -> None:
    print(f"{datetime.now():%m-%d %H:%M:%S} {message}", flush=True)


def worker_arg(text: str) -> tuple[int, int]:
    index, count = (int(part) for part in text.split("/"))
    if not 0 <= index < count:
        raise argparse.ArgumentTypeError("--worker is I/N with 0 <= I < N")
    return index, count


def preflight(datasets: list[str], variants: tuple[str, ...]) -> list[str]:
    problems = []
    for dataset in datasets:
        inputs = load_inputs(dataset)
        problems += [f"{dataset}: {p}" for p in check_episodes(inputs.episodes, inputs.targets, feature_lengths(dataset))]
        for split in inputs.splits.values():
            problems += [f"{dataset}: {p}" for p in check_split(split, inputs.episodes, inputs.targets, variants, inputs.window)]
    return problems


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--splits", nargs="+", help="only these split names, e.g. task_seed0 paper_seed1")
    p.add_argument("--probes", nargs="+", help="only these architectures, e.g. lstm_plain mlp_modes_clip")
    p.add_argument("--out", type=Path, default=RESULTS)
    p.add_argument("--worker", type=worker_arg, default=(0, 1), help="I/N: this process takes units I, I+N, I+2N, ... of the run order")
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--no-weights", action="store_true")
    p.add_argument("--no-heads", action="store_true", help="do not save the outputs of the mode and CLIP heads")
    p.add_argument("--lam-mode", type=float, default=1.0, help="weight of the mode term in the loss (the score term has 1)")
    p.add_argument("--lam-clip", type=float, default=1.0, help="weight of the CLIP term in the loss")
    p.add_argument("--lam-score", type=float, default=1.0, help="weight of the score term in the loss; 0 trains a mode-only probe (it needs a mode head; the epoch is chosen on the validation mode term)")
    p.add_argument("--min-free-gb", type=float, default=20.0)
    p.add_argument("--dry-run", action="store_true", help="print the units of this worker in run order and stop")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--skip-preflight", action="store_true")
    args = p.parse_args()

    names = all_split_names(tuple(args.datasets))
    if args.splits:
        names = {d: [s for s in ss if s in args.splits] for d, ss in names.items()}
        unknown = set(args.splits) - {s for ss in names.values() for s in ss}
        if unknown:
            sys.exit(f"unknown splits for these datasets: {sorted(unknown)}")
    if args.probes:
        known = {c.name for c in all_configs()}
        if set(args.probes) - known:
            sys.exit(f"unknown probes {sorted(set(args.probes) - known)}; choose from {sorted(known)}")
    index, count = args.worker
    mine = units(names, tuple(args.variants))[index::count]

    if args.dry_run:
        for i, u in enumerate(mine):
            methods = [m for m in u.methods if not args.probes or m.split("__")[0] in args.probes]
            done = sum(is_done(run_dir(u.dataset, u.split, m, args.out)) for m in methods)
            print(f"{i:3d}  {u.dataset:15s} {u.split:24s} {u.variant:10s} {done}/{len(methods)} probes done")
        print(f"{len(mine)} units, {sum(len(u.methods) for u in mine)} probes (worker {index}/{count})")
        return

    if not args.skip_preflight:
        problems = preflight(args.datasets, tuple(args.variants))
        for problem in problems:
            print("PREFLIGHT:", problem)
        if problems:
            sys.exit(f"{len(problems)} preflight problems; nothing was trained")
        log(f"preflight passed for {args.datasets}")
    if args.preflight_only:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    settings = Settings(args.out, args.max_epochs, args.patience, not args.no_weights, not args.no_heads, args.min_free_gb, args.lam_mode, args.lam_clip, args.lam_score)
    log(f"worker {index}/{count}: {len(mine)} units on {device}, writing to {args.out}")
    inputs, started, finished_units, written = {}, time.time(), 0, 0
    for u in mine:
        if u.dataset not in inputs:
            inputs[u.dataset] = load_inputs(u.dataset)
        written += train_unit(u, inputs[u.dataset], settings, device, log, args.probes)
        finished_units += 1
        per_unit = (time.time() - started) / finished_units
        log(f"unit {finished_units}/{len(mine)} done ({written} runs written); {per_unit / 60:.1f} min per unit on average, "
            f"about {per_unit * (len(mine) - finished_units) / 3600:.1f} h left for this worker")


if __name__ == "__main__":
    main()
