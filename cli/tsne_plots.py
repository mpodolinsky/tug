"""python -m cli.tsne_plots --points results/tsne/liberox_groot/vla/points.npz [--split task_seed0] [--colorings task skill outcome mode outcome_lines mode_lines] [--out DIR]
                    [--titles with|without|both] [--formats png pdf] [--tight]

Draws the figures of saved t-SNE points (python -m cli.tsne), one PNG per colouring and subset (all, seen, seen_held_out, unseen), all on the same axes, without repeating the
t-SNE: <points dir>/figures/<split>/<coloring>__<subset>.png. --split is the split whose seen and unseen episodes the subsets are (the probe's own for a probe; any for
the VLA embedding). Colourings: task, skill, outcome (success: light to dark blue; failure: light blue to red at the failure step), mode (failure: light blue to the colour of
its failure mode); outcome_ellipses_late/_all and mode_ellipses_late/_all fit a Gaussian to each class (fp/eval/class_fits.py) and draw its one-standard-deviation ellipse;
outcome_gaussians_late/_all/_pre and mode_gaussians_late/_all/_pre draw the same ellipses without the points; outcome_ellipses_path and mode_ellipses_path add the path of the
most typical successful episode as a black arrow, outcome_ellipses_paths and mode_ellipses_paths also that of a failure of --path-mode (dashed, ending in a cross);
outcome_ellipses_pairs (only when asked for) draws --pairs figures per subset, each with a random success and a random failure of any mode; outcome_3d_ellipsoids, outcome_3d_gaussians,
mode_3d_ellipsoids and mode_3d_gaussians draw 3D Gaussians as ellipsoids with and without the points;
outcome_3d and mode_3d draw the first three principal axes of PCA points from three directions;
outcome_ellipses_pre_train_unseen and mode_ellipses_pre_train_unseen (only when asked for) draw the points and Gaussians of the pre window of the
training episodes and of the unseen tasks side by side in one figure without a title, on the same axes;
outcome_lines and mode_lines draw a sample of episodes as lines in those colours over the other points in grey, to follow the progression of an
episode. See fp/eval/tsne_plots.py.

--failures-only leaves the successes out of the failure-mode Gaussian and ellipse figures (points, fits, legend): the mode head is not supervised on successes.

For a paper: --titles both writes every figure with and without its title into DIR/with_title and DIR/without_title (needs --out), --formats png pdf writes a PNG and a PDF
(vector graphics) of each, and --tight crops the blank margins around the figure.
"""

import argparse
from pathlib import Path

from fp.data.targets import MODE_NAMES
from fp.eval import tsne_plots


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--points", type=Path, required=True)
    p.add_argument("--split")
    p.add_argument("--colorings", nargs="+", choices=tsne_plots.FIGURES + tuple(tsne_plots.PAIRS) + tuple(tsne_plots.TRAIN_UNSEEN))
    p.add_argument("--seed", type=int, default=0, help="the seed of the random episodes of outcome_ellipses_pairs and of the order of the points; another seed draws other pairs (use another --out to keep both)")
    p.add_argument("--pairs", type=int, default=tsne_plots.N_PAIRS, help="the number of random pairs of a success and a failure drawn by outcome_ellipses_pairs")
    p.add_argument("--out", type=Path)
    p.add_argument("--subsets", nargs="+", choices=tsne_plots.SUBSETS, default=list(tsne_plots.SUBSETS), help="the subsets of the episodes to draw (default: all of them)")
    p.add_argument("--titles", choices=("with", "without", "both"), default="with", help="draw the figures with their title, without it, or both (into DIR/with_title and DIR/without_title)")
    p.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=["png"], help="the file formats of every figure (pdf is vector graphics)")
    p.add_argument("--tight", action="store_true", help="crop the blank margins around the figures")
    p.add_argument("--font-scale", type=float, default=1.0, help="multiply every font size of the figures (1.8 is a large text for a paper); the legends of the Gaussian figures take fewer columns and more room")
    p.add_argument("--failures-only", action="store_true", help="draw only the failure-mode Gaussian and ellipse figures (mode_*_gaussians, mode_*_ellipses, mode_3d_*) without the successes, as <figure>_failures: for a layer that is not supervised on successes, like the mode head")
    p.add_argument("--path-mode", choices=MODE_NAMES, default=tsne_plots.PATH_MODE, help="the failure mode of the failure drawn by outcome_ellipses_paths and mode_ellipses_paths")
    args = p.parse_args()
    if tsne_plots.load_points(args.points)["config"]["embedding"] == "vla" and not args.split:
        p.error("the VLA embedding has no split of its own: pass --split")
    if args.titles == "both" and not args.out:
        p.error("--titles both needs --out (the folders with_title and without_title are made in it)")
    titles = {"with": (True,), "without": (False,), "both": (True, False)}[args.titles]
    for path in tsne_plots.render(args.points, args.split, args.out, args.colorings, subsets=tuple(args.subsets), seed=args.seed, path_mode=args.path_mode, n_pairs=args.pairs,
                                titles=titles, formats=tuple(args.formats), tight=args.tight,
                                failures_only=args.failures_only, font_scale=args.font_scale):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
