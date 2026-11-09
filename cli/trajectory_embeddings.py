"""python -m cli.trajectory_embeddings --csv results/paper_figures/trajectory_candidates/lstm_unseen_set2.csv --labels S10 F8 [--split task_seed0] [--from unseen]
                                       [--dir results/pca3_train/liberox_groot] [--out results/paper_figures/trajectory_candidates/s10_f8_embeddings]

Chosen candidate trajectories (the labels of the CSV of a contact sheet, python -m cli.trajectory_candidates) drawn in the PCA planes of all four embeddings of a split in one figure
(fp/eval/trajectories.py): the VLA backbone embedding, the probe's LSTM state, the hidden layer of its failure score head and that of its failure mode head (whose background has no
successes, since the head is not trained on them). Each panel has the points and Gaussians of the pre window of the subset --from and the paths, on the axes of its own space.
"""

import argparse
import csv
from pathlib import Path

from fp.config import RESULTS, get
from fp.data.inputs import load_inputs
from fp.eval import tsne
from fp.eval.trajectories import write_embeddings

SPACES = (("VLA backbone embedding", "vla", False), ("LSTM state", "probe__{split}__lstm_modes_clip__onset_pre", False),
          ("Failure score head", "probe_score_hidden__{split}__lstm_modes_clip__onset_pre", False), ("Failure mode head (failures only)", "probe_mode_hidden__{split}__lstm_modes_clip__onset_pre", True))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True, help="the table of a contact sheet of candidates")
    p.add_argument("--labels", nargs="+", required=True, help="the labels of the candidates to draw, e.g. S10 F8")
    p.add_argument("--split", default="task_seed0")
    p.add_argument("--from", dest="subset", choices=(*tsne.SUBSETS, "train"), default="unseen")
    p.add_argument("--dir", type=Path, default=RESULTS / "pca3_train" / "liberox_groot", help="the folder of the saved PCA points of the four embeddings")
    p.add_argument("--note", default="", help="small text under the legend (default: none)")
    p.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=["png"])
    p.add_argument("--out", type=Path, default=RESULTS / "paper_figures" / "trajectory_candidates" / "embeddings")
    args = p.parse_args()
    loaded = [(title, tsne.load_points(args.dir / folder.format(split=args.split) / "points.npz"), failures_only) for title, folder, failures_only in SPACES]
    config = loaded[0][1]["config"]
    inputs = load_inputs(config["dataset"])
    keys = loaded[0][1]["keys"].tolist()
    split = inputs.splits[args.split]
    mask = {**tsne.subset_masks(keys, split), "train": tsne.train_mask(keys, split)}[args.subset]
    rows = {r["label"]: r for r in csv.DictReader(open(args.csv))}
    missing = [label for label in args.labels if label not in rows]
    if missing:
        p.error(f"not in {args.csv}: {missing}")
    chosen = [(label, keys.index(rows[label]["key"])) for label in args.labels]
    panels = [(title, points, mask[points["episode"]], failures_only) for title, points, failures_only in loaded]
    spec = get(config["dataset"])
    for file in write_embeddings(panels, chosen, spec.replan_steps / spec.control_hz, args.out, tuple(args.formats), note=args.note):
        print(f"wrote {file}")


if __name__ == "__main__":
    main()
