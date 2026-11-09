"""python -m cli.run_baselines [--families ...] [--datasets ...] [--splits ...] [--methods ...] [--worker I/N] [--dry-run]

Run baseline families on every split of every dataset and write each run under results/<dataset>/<split>/<method>/, next to the probes' runs.
Families: mahalanobis, euclidean, pca_kmeans (SAFE's 26 distance configurations; the default), and safe_mlp, safe_lstm (SAFE's 12 configurations
each, early stopping with patience 50) with safe_mlp_pre, safe_lstm_pre (the same trained on rollouts cut at the onset; not SAFE as published),
and stac (6 configurations) and stac_single (2), which read the sampled action chunks of data/<dataset>/actions.npz (python -m cli.prepare_actions)
and train nothing, and actprobe (one configuration), which trains a small network on them and needs data/<dataset>/task_embeddings.npz
(python -m cli.embed_tasks), and failurespot_mlp, failurespot_lstm (nine configurations each: the paper's weak-supervision weights are not stated,
so each pair is its own method), which train a detector in two stages on the pooled features and the action chunks. `python -m cli.evaluate` then evaluates them with the calibrator the protocol gives them. Runs that exist are skipped, so the command can be repeated to resume; several workers can share the
list: --worker 0/2, --worker 1/2.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from fp.baselines.registry import FAMILIES, Context, build
from fp.baselines.sweep import LazyActions, LazyFeatures, Settings, baseline_units, run_unit
from fp.config import RESULTS
from fp.data.features import feature_lengths
from fp.data.inputs import load_inputs
from fp.grid import DATASETS, all_split_names
from fp.results import is_done, run_dir
from fp.train.preflight import check_episodes, check_split

DEFAULT_FAMILIES = ["mahalanobis", "euclidean", "pca_kmeans"]


def log(message: str) -> None:
    print(f"{datetime.now():%m-%d %H:%M:%S} {message}", flush=True)


def worker_arg(text: str) -> tuple[int, int]:
    index, count = (int(part) for part in text.split("/"))
    if not 0 <= index < count:
        raise argparse.ArgumentTypeError("--worker is I/N with 0 <= I < N")
    return index, count


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--splits", nargs="+", help="only these split names, e.g. task_seed0 paper_seed1")
    p.add_argument("--methods", nargs="+", help="only these methods, e.g. euclidean__k10_cumsum pca_kmeans__d128_c64_cumsum")
    p.add_argument("--out", type=Path, default=RESULTS)
    p.add_argument("--worker", type=worker_arg, default=(0, 1), help="I/N: this process takes units I, I+N, I+2N, ... of the run order")
    p.add_argument("--min-free-gb", type=float, default=20.0)
    p.add_argument("--dry-run", action="store_true", help="print the units of this worker and how many of their runs exist, then stop")
    p.add_argument("--skip-preflight", action="store_true")
    args = p.parse_args()

    import fp.baselines.sweep  # noqa: F401  (registers the families)

    unknown = set(args.families) - set(FAMILIES)
    if unknown:
        sys.exit(f"unknown families {sorted(unknown)}; registered: {sorted(FAMILIES)}")
    names = all_split_names(tuple(args.datasets))
    if args.splits:
        names = {d: [s for s in ss if s in args.splits] for d, ss in names.items()}
        if set(args.splits) - {s for ss in names.values() for s in ss}:
            sys.exit(f"unknown splits {sorted(set(args.splits) - {s for ss in names.values() for s in ss})} for these datasets")
    index, count = args.worker
    units = baseline_units(names, args.families)[index::count]
    inputs = {d: load_inputs(d) for d in dict.fromkeys(u.dataset for u in units)}

    if args.dry_run:
        for i, u in enumerate(units):
            keys = list(inputs[u.dataset].episodes)  # (nothing is loaded: the lazy mappings read only when asked for a value)
            context = Context(inputs[u.dataset], LazyFeatures(u.dataset, keys), actions=LazyActions(u.dataset, keys))
            try:
                built = [s.name for s in build(u.family, context, inputs[u.dataset].splits[u.split])]
            except NotImplementedError as reason:
                print(f"{i:3d}  {u.dataset:15s} {u.split:28s} {u.family:12s} skipped: {reason}")
                continue
            methods = [m for m in built if not args.methods or m in args.methods]
            done = sum(is_done(run_dir(u.dataset, u.split, m, args.out)) for m in methods)
            print(f"{i:3d}  {u.dataset:15s} {u.split:28s} {u.family:12s} {done}/{len(methods)} runs done")
        print(f"{len(units)} units (worker {index}/{count})")
        return

    if not args.skip_preflight:
        problems = []
        for dataset, data in inputs.items():
            problems += [f"{dataset}: {p_}" for p_ in check_episodes(data.episodes, data.targets, feature_lengths(dataset))]
            for name in names[dataset]:
                problems += [f"{dataset}: {p_}" for p_ in check_split(data.splits[name], data.episodes, data.targets, (), data.window)]
        for problem in problems:
            print("PREFLIGHT:", problem)
        if problems:
            sys.exit(f"{len(problems)} preflight problems; nothing was run")
        log("preflight passed")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    settings = Settings(args.out, min_free_gb=args.min_free_gb)
    log(f"worker {index}/{count}: {len(units)} units on {device}, writing to {args.out}")
    started, written, loaded, current = time.time(), 0, [], None
    for done, u in enumerate(units, 1):
        if u.dataset != current:  # a dataset's units are adjacent: drop the previous dataset's features and action chunks
            for lazy in loaded:
                lazy.release()
            current = u.dataset
            keys = list(inputs[u.dataset].episodes)
            loaded = [LazyFeatures(u.dataset, keys), LazyActions(u.dataset, keys)]
        written += run_unit(u, Context(inputs[u.dataset], loaded[0], device, loaded[1]), settings, log, args.methods)
        per_unit = (time.time() - started) / done
        log(f"unit {done}/{len(units)} done ({written} runs written); {per_unit / 60:.1f} min per unit on average, "
            f"about {per_unit * (len(units) - done) / 3600:.1f} h left for this worker")


if __name__ == "__main__":
    main()
