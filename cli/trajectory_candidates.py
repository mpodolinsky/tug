"""python -m cli.trajectory_candidates --points results/pca3_train/liberox_groot/probe__task_seed0__lstm_modes_clip__onset_pre/points.npz [--split task_seed0] [--from unseen]
                                       [--successes 5] [--failures 5] [--seed 0] [--exclude EARLIER.csv ...] [--out results/paper_figures/trajectory_candidates/lstm_unseen]

Candidate example trajectories on the PCA plane of saved points (fp/eval/trajectories.py), to choose one success and one failure from: a contact sheet with the successes (S1...) above
the failures (F1...), each an arrow over the points and the Gaussians of the pre window of the subset --from (unseen, seen, seen_held_out, all or train), and a CSV with the episode
key, task, failure mode and failure time of every candidate. The candidates are random successes and failures of the most common failure modes of the subset (--seed draws others).
--exclude takes the CSVs of earlier sheets: their episodes are not drawn again, the labels continue from theirs (S6..., F6...) and the failure modes they showed come last.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from fp.config import RESULTS, get
from fp.data.inputs import load_inputs
from fp.eval import class_fits, tsne
from fp.eval.trajectories import write_candidates


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--points", type=Path, required=True)
    p.add_argument("--split")
    p.add_argument("--from", dest="subset", choices=(*tsne.SUBSETS, "train"), default="unseen", help="the episodes the candidates come from and the background shows")
    p.add_argument("--successes", type=int, default=5)
    p.add_argument("--failures", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude", nargs="*", type=Path, default=[], help="the CSVs of earlier sheets: their episodes are not chosen again, the numbering continues after them")
    p.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=["png"])
    p.add_argument("--out", type=Path, default=RESULTS / "paper_figures" / "trajectory_candidates" / "candidates")
    args = p.parse_args()
    points = tsne.load_points(args.points)
    config = points["config"]
    split = args.split or config["split"]
    if split is None:
        p.error("the VLA embedding has no split of its own: pass --split")
    inputs = load_inputs(config["dataset"])
    keys = points["keys"].tolist()
    masks = {**tsne.subset_masks(keys, inputs.splits[split]), "train": tsne.train_mask(keys, inputs.splits[split])}
    mask = masks[args.subset][points["episode"]]
    index = {k: i for i, k in enumerate(keys)}
    earlier = [row for path in args.exclude for row in csv.DictReader(open(path))]
    start = (1 + sum(r["outcome"] == "success" for r in earlier), 1 + sum(r["outcome"] == "failure" for r in earlier))
    successes, failures = class_fits.trajectory_candidates(points, mask, args.successes, args.failures, np.random.default_rng(args.seed), {index[r["key"]] for r in earlier})
    spec = get(config["dataset"])
    for file in write_candidates(points, mask, successes, failures, spec.replan_steps / spec.control_hz, args.out, tuple(args.formats), start=start):
        print(f"wrote {file}")
    print(open(args.out.with_suffix(".csv")).read())


if __name__ == "__main__":
    main()
