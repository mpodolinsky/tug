"""python -m cli.label_example --folder <episode folder> [<episode folder> ...] [--out results/figures/label_example/example] [--windows 6] [--formats png pdf] [--fps 20]

Labeled rollouts for a paper figure (fp/eval/label_example.py), one row per episode folder from top to bottom: agent-view frames of the first windows of the episode, the semantic
description of every 3-second window under its frame, and a time axis of the windows with the failure time and failure mode of a failed episode (or "success"). The episode folders
of the released datasets have the labels and the videos next to each other (<collect>/<level>/<task>/ep000/labels.json, rollout.mp4).
"""

import argparse
from pathlib import Path

from fp.config import RESULTS
from fp.eval import label_example


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--folder", type=Path, nargs="+", required=True, help="one episode folder per row, from top to bottom")
    p.add_argument("--out", type=Path, default=RESULTS / "figures" / "label_example" / "example", help="the file name without the format")
    p.add_argument("--windows", type=int, default=6, help="how many of the 3-second windows of the timeline to show")
    p.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=["png", "pdf"])
    p.add_argument("--fps", type=float, default=20.0, help="frames per second of the videos (one frame per control step)")
    args = p.parse_args()
    rows = []
    for folder in args.folder:
        rollout = label_example.read_labels(folder)
        windows = rollout.windows[: args.windows]
        rows.append((rollout, label_example.read_frames(folder / "rollout.mp4", label_example.frame_indices(windows, args.fps))))
    for file in label_example.write(label_example.build(rows, args.windows), args.out, tuple(args.formats)):
        print(f"wrote {file}")


if __name__ == "__main__":
    main()
