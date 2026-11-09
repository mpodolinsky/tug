"""python -m cli.class_fits --points results/pca/liberox_groot/probe__task_seed0__lstm_modes_clip__onset_pre/points.npz [--split task_seed0] [--fit-dim 256] [--out FILE]

How far apart the classes of saved t-SNE or PCA points are (python -m cli.tsne), in the full space and in the plane (fp/eval/class_fits.py). The states of the points are computed again
(the VLA embedding is reduced to its first --fit-dim principal components; the 256-d probe state is used whole), and for every subset (all, seen, seen_held_out, unseen), window
(late, all) and kind of class (outcome, mode) the Mahalanobis distance between the class means under the pooled within-class covariance is written to a CSV (default
<points dir>/class_fits.csv) next to the same distance in the plane, and the distance between successes and failures is printed. The ellipses are drawn by
python -m cli.tsne_plots (outcome_ellipses_late, ...). One point per episode and subsampled steps are not supported (the states are recomputed for every step).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from fp.config import RESULTS
from fp.data.inputs import load_inputs
from fp.eval import class_fits, tsne

FIT_DIM = 256


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--points", type=Path, required=True)
    p.add_argument("--split")
    p.add_argument("--root", type=Path, default=RESULTS, help="where the runs of the probes are")
    p.add_argument("--fit-dim", type=int, default=FIT_DIM)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    points = tsne.load_points(args.points)
    config = points["config"]
    if config.get("unit") == "episode" or config.get("every", 1) != 1:
        sys.exit("the points must be every step of every episode")
    split = args.split or config["split"]
    if split is None:
        p.error("the VLA embedding has no split of its own: pass --split")
    inputs = load_inputs(config["dataset"])
    keys = sorted(inputs.episodes)
    if keys != points["keys"].tolist():
        sys.exit("the episodes of the points are not those of the dataset")
    x, episode, step = tsne.states(config["dataset"], keys, config["embedding"], config["split"], config["method"], args.root, layer=config.get("layer") or "hidden")
    if not (np.array_equal(episode, points["episode"]) and np.array_equal(step, points["step"])):
        sys.exit("the states do not line up with the saved points")
    which = config.get("fit_set") or "all"  # the axes of the plots and the reduction of the VLA embedding use the same rows
    fit = tsne.fit_rows(keys, episode, step, points["onset"], inputs.splits[config.get("fit_split") or config["split"] or split], which)
    reduced, _ = tsne.pca_reduce(x, args.fit_dim, config["seed"], fit=fit)
    masks = tsne.subset_masks(keys, inputs.splits[split])
    rows = []
    for subset in tsne.SUBSETS:
        for window in class_fits.WINDOWS:
            for kind in class_fits.KINDS:
                chosen = masks[subset][points["episode"]] & class_fits.window_mask(points, window)
                ids, k = class_fits.class_ids(points, kind)[chosen], class_fits.n_classes(kind)
                full = class_fits.class_stats(reduced[chosen], ids, k)
                plane = class_fits.class_stats(points["xy"][chosen], ids, k)
                if (full[0] >= 2).sum() < 2:
                    continue
                d_full, d_plane = (class_fits.pairwise_distance(*stats) for stats in (full, plane))
                for a in range(k):
                    for b in range(a + 1, k):
                        if np.isfinite(d_full[a, b]):
                            rows.append({"subset": subset, "window": window, "kind": kind, "class_a": class_fits.class_name(kind, a), "class_b": class_fits.class_name(kind, b),
                                         "points_a": int(full[0][a]), "points_b": int(full[0][b]), "distance_full": round(float(d_full[a, b]), 3), "distance_plane": round(float(d_plane[a, b]), 3)})
    out = args.out or args.points.parent / "class_fits.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    for r in rows:
        if r["kind"] == "outcome":
            print(f"{r['subset']:14s} {r['window']:5s} success vs failure: {r['distance_full']:.2f} in the full space ({reduced.shape[1]} dimensions), {r['distance_plane']:.2f} in the plane")


if __name__ == "__main__":
    main()
