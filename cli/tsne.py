"""python -m cli.tsne --dataset liberox_groot --embedding vla|probe [--split task_seed0] [--method lstm_modes_clip__onset_pre] [--out results/tsne]
                    [--perplexity 50] [--every 1] [--unit step|episode] [--reducer tsne|pca] [--fit-set all|train_pre] [--layer hidden|mode_hidden|score_hidden|clip] [--pca-dim 50] [--seed 0] [--n-jobs 8] [--device cpu] [--force]

The t-SNE of the embeddings of the VLA (--embedding vla: the 6144-d pooled embedding the probe reads, one point per policy step) or of a trained probe (--embedding probe: the
256-d state of its LSTM, for the probe of --method on --split), computed once and saved with everything needed to colour the points later (fp/eval/tsne.py):
<out>/<dataset>/vla/points.npz and <out>/<dataset>/probe__<split>__<method>/points.npz. An existing file is not overwritten without --force. The figures are drawn from the
saved points by python -m cli.tsne_plots. --every K keeps the steps 0, K, 2K, ... of every episode (a perplexity above the episode length then spans several episodes);
--unit episode embeds one point per episode instead (the mean of its states up to the failure step, the whole episode for a success; use a perplexity of about 30);
--fit-set train_pre fits the axes on the pre-failure steps of the training episodes of --split only and projects every episode onto them.
--reducer pca projects on the first three principal axes instead of running t-SNE (a linear projection: an episode traces a path; no perplexity; xy is the first two
axes and xyz, saved too, the three of the 3D figures);
use another --out for each setting, the file name does not say which was used (it is in the saved settings).
"""

import argparse
import sys
import time
from pathlib import Path

from fp.config import RESULTS
from fp.data.inputs import load_inputs
from fp.eval import tsne


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--embedding", choices=("vla", "probe"), required=True)
    p.add_argument("--split", help="the split whose probe is run (--embedding probe)")
    p.add_argument("--method", default="lstm_modes_clip__onset_pre", help="the probe (--embedding probe)")
    p.add_argument("--root", type=Path, default=RESULTS, help="where the runs of the probes are")
    p.add_argument("--out", type=Path, default=RESULTS / "tsne")
    p.add_argument("--perplexity", type=float, default=tsne.PERPLEXITY)
    p.add_argument("--every", type=int, default=1, help="keep every K-th step of every episode")
    p.add_argument("--unit", choices=("step", "episode"), default="step", help="a point per policy step or per episode (the mean up to the failure step)")
    p.add_argument("--layer", choices=tsne.LAYERS, default="hidden", help="the activations of the probe to embed (--embedding probe): its 256-d LSTM state, the 64-unit hidden layer of its mode head or of its score head, or the 64-d output of its CLIP head")
    p.add_argument("--reducer", choices=("tsne", "pca"), default="tsne", help="t-SNE, or the first two principal axes")
    p.add_argument("--fit-set", choices=tsne.FIT_SETS, default="all", help="the rows the axes of --reducer pca are fitted on: all points, or train_pre: the training episodes of --split up to their failure step, so that the unseen tasks play no part in the axes (the other episodes are projected onto them)")
    p.add_argument("--pca-dim", type=int, default=tsne.PCA_DIM)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.embedding == "probe" and not args.split:
        sys.exit("--embedding probe needs --split")
    if args.unit == "episode" and args.every != 1:
        sys.exit("--every is for --unit step")
    if args.fit_set != "all" and (args.reducer != "pca" or args.unit != "step" or not args.split):
        sys.exit("--fit-set train_pre needs --reducer pca, --unit step and --split")

    name = "vla" if args.embedding == "vla" else f"probe{'' if args.layer == 'hidden' else '_' + args.layer}__{args.split}__{args.method}"
    path = args.out / args.dataset / name / "points.npz"
    if path.exists() and not args.force:
        sys.exit(f"{path} exists (--force to replace it)")
    inputs = load_inputs(args.dataset)
    keys = sorted(inputs.episodes)
    start = time.time()
    x, episode, step = tsne.states(args.dataset, keys, args.embedding, args.split, args.method, args.root, args.device, args.layer)
    table = tsne.episode_table(inputs, keys)
    if args.unit == "episode":
        x, episode, step = tsne.episode_means(x, episode, table)
    else:
        kept = tsne.every_kth_step(step, args.every)
        x, episode, step = x[kept], episode[kept], step[kept]
    print(f"{len(x)} points of {x.shape[1]} dimensions from {len(keys)} episodes ({time.time() - start:.0f} s)", flush=True)
    explained, extra = None, None
    if args.reducer == "pca":
        fit = tsne.fit_rows(keys, episode, step, table["onset"], inputs.splits[args.split], args.fit_set) if args.fit_set != "all" else None
        xyz, explained = tsne.pca_axes(x, 3, args.seed, fit=fit)
        xy, extra = xyz[:, :2], {"xyz": xyz}
        print("variance explained by the three axes: " + ", ".join(f"{v:.3f}" for v in explained), flush=True)
    else:
        xy = tsne.embed_2d(x, args.seed, args.perplexity, args.pca_dim, n_jobs=args.n_jobs, verbose=True)
    config = {"dataset": args.dataset, "embedding": args.embedding, "split": args.split, "method": args.method if args.embedding == "probe" else None,
              "perplexity": args.perplexity, "every": args.every, "unit": args.unit, "layer": args.layer if args.embedding == "probe" else None, "reducer": args.reducer, "fit_set": args.fit_set, "fit_split": args.split if args.fit_set != "all" else None, "explained_variance": None if explained is None else explained.tolist(), "pca_dim": args.pca_dim, "pca_fit_points": tsne.PCA_FIT_POINTS, "seed": args.seed, "points": len(x), "dimensions": x.shape[1]}
    tsne.save_points(path, xy, episode, step, keys, table, config, extra)
    print(f"wrote {path} ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
