"""Labeled rollouts as the dataset provides them, for a paper figure: one row per rollout with agent-view frames of its first windows, the semantic description of each 3-second
window under its frame, and a time axis of the windows with the failure time and failure mode of a failed rollout (or "success"). A frame sits in the 3-second slot of its window
(a shorter last window gets a slot of the same size), while the bar under it has the true length of the window.

The labels are read from the ``labels.json`` of an episode folder (``semantic_timeline``: a keyword phrase per window; ``vlm_failure``: the failure mode and the failure time in
seconds, which the probe is trained on at the resolution of a policy step) and the frames from its ``rollout.mp4`` (one frame per control step, 20 fps). The frame shown for a
window is the one at its centre. The rows share the time axis, so that a success and a failure of the same task can be compared.
"""

import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

FIG_WIDTH = 7.0  # inches: a figure over two columns; the height follows from the number of rows
PAD = 0.3  # seconds of margin on both sides of the time axis
WINDOW_SECONDS = 3.0
FRAME_SIZE = 2.8  # width of a frame in seconds of the time axis (the frames are square): 1 second of the axis is about 0.38 inches
ROW_HEIGHT = 6.2  # the height of a row in the units of the time axis
FRAME_BOTTOM, PHRASE_TOP, BRACKET, TAG, BAR_BOTTOM, BAR_HEIGHT, ANNOTATION, TICKS = 3.25, 3.1, 2.15, 1.8, 1.1, 0.3, 0.65, 0.2  # y of the parts of a row, from its bottom
BRACKET_TICK = 0.15  # the height of the ends of the bracket under the descriptions
PHRASE_WIDTH = 19  # characters before a description wraps onto a second line
PHRASE_FONT, ANNOTATION_FONT, TICK_FONT = 8, 9, 8  # points: the descriptions, the failure time and mode (or "success"), the time axis
INK, FAIL_COLOR, FAIL_FILL = "#222222", "#b03a2e", (1.0, 0.72, 0.72)
SUCCESS_COLOR, SUCCESS_FILL, WINDOW_FILL = "#2e7d32", (0.78, 0.92, 0.78), (0.9, 0.9, 0.9)


@dataclass(frozen=True)
class Window:
    start: float  # seconds
    end: float
    phrase: str  # the keyword description of what the robot does in the window


@dataclass(frozen=True)
class LabeledRollout:
    instruction: str
    windows: list[Window]
    mode: str | None  # the failure mode (None for a success)
    onset: float | None  # the failure time in seconds (None for a success)


def read_labels(folder: Path) -> LabeledRollout:
    """The labels of the episode folder ``folder`` (``labels.json``)."""
    labels = json.loads((Path(folder) / "labels.json").read_text())
    windows = [Window(float(s["t_start_sec"]), float(s["t_end_sec"]), str(s["phrase"])) for s in labels["semantic_timeline"]]
    if labels["success"]:
        return LabeledRollout(labels["instruction"], windows, None, None)
    failure = labels["vlm_failure"]
    return LabeledRollout(labels["instruction"], windows, failure["vlm_failure_mode"], float(failure["vlm_failure_onset_seconds"]))


def failure_text(onset: float, mode: str) -> str:
    """The line under the bar of a failed rollout, with the names of the two labels in bold: ``Failure time 10.3 s, Failure mode: grasp failure``."""
    return rf"$\mathbf{{Failure\ time}}$ {onset:.1f} s, $\mathbf{{Failure\ mode}}$: {mode.replace('_', ' ')}"


def timeline_span(windows: list[Window]) -> tuple[float, float]:
    """The x range of the bracket under the descriptions of a row: from the first slot to the last (a slot is 3 seconds wide), a little inside the outer edges."""
    return windows[0].start + 0.1, windows[-1].start + WINDOW_SECONDS - 0.1


def frame_indices(windows: list[Window], fps: float) -> list[int]:
    """The index of the video frame at the centre of each window."""
    return [int((w.start + w.end) / 2 * fps) for w in windows]


def read_frames(video: Path, indices: list[int]) -> list[np.ndarray]:
    """The frames of the video at ``indices`` as RGB arrays (the video is read in order, since seeking in an H.264 file is not frame-exact)."""
    import cv2

    wanted, frames, capture, i = set(indices), {}, cv2.VideoCapture(str(video)), 0
    while len(frames) < len(wanted):
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"{video} has {i} frames, the example needs frame {max(wanted)}")
        if i in wanted:
            frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    capture.release()
    return [frames[i] for i in indices]


def build(rows: list[tuple[LabeledRollout, list[np.ndarray]]], n_windows: int):
    """The figure of the rollouts of ``rows`` (each with one frame per window, in the order of its windows), from top to bottom, with their first ``n_windows`` windows. Each row
    has its frames, the descriptions under them, a bar of its windows (the one that holds the failure time tinted red, those of a success green) with the failure time marked, and
    under it the failure time and mode, or "success"; the keywords "Timeline" (below a bracket under the descriptions), "Failure time" and "Failure mode" name the labels. The time axis is shared and labeled under the last row."""
    shown = [(r.windows[:n_windows], frames, r) for r, frames in rows]
    for windows, frames, _ in shown:
        if len(frames) != len(windows):
            raise ValueError(f"{len(windows)} windows but {len(frames)} frames")
    end = math.ceil(max(w[-1].end for w, _, _ in shown) / WINDOW_SECONDS) * WINDOW_SECONDS  # the time axis, to a whole window
    units = len(shown) * ROW_HEIGHT
    fig = plt.figure(figsize=(FIG_WIDTH, units * FIG_WIDTH / (end + 2 * PAD)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    for k, (windows, frames, rollout) in enumerate(shown):
        base = (len(shown) - 1 - k) * ROW_HEIGHT
        failing = None if rollout.onset is None else next((i for i, w in enumerate(windows) if w.start <= rollout.onset < w.end), None)
        x0, x1 = timeline_span(windows)  # a bracket under the descriptions with the keyword of the timeline below it
        ax.plot([x0, x0, x1, x1], [base + BRACKET + BRACKET_TICK, base + BRACKET, base + BRACKET, base + BRACKET + BRACKET_TICK], color="0.35", linewidth=0.8, solid_capstyle="butt")
        ax.text((x0 + x1) / 2, base + TAG, "Timeline", ha="center", va="center", fontsize=ANNOTATION_FONT, fontweight="bold", color=INK)
        for i, (w, frame) in enumerate(zip(windows, frames)):
            centre = w.start + WINDOW_SECONDS / 2  # the frame sits in the slot of its window, also if the last window is shorter
            left, bottom = centre - FRAME_SIZE / 2, base + FRAME_BOTTOM
            ax.imshow(frame, extent=(left, left + FRAME_SIZE, bottom, bottom + FRAME_SIZE), aspect="equal", zorder=1)
            ax.add_patch(Rectangle((left, bottom), FRAME_SIZE, FRAME_SIZE, fill=False, edgecolor=FAIL_COLOR if i == failing else "0.55", linewidth=1.5 if i == failing else 0.5, zorder=2))
            ax.text(centre, base + PHRASE_TOP, textwrap.fill(w.phrase, PHRASE_WIDTH), ha="center", va="top", fontsize=PHRASE_FONT, style="italic", color=INK, linespacing=1.1)
            fill = FAIL_FILL if i == failing else SUCCESS_FILL if rollout.onset is None else WINDOW_FILL
            ax.add_patch(Rectangle((w.start + 0.05, base + BAR_BOTTOM), w.end - w.start - 0.1, BAR_HEIGHT, facecolor=fill, edgecolor="none"))
        if rollout.onset is not None:
            ax.plot([rollout.onset] * 2, [base + BAR_BOTTOM - 0.1, base + BAR_BOTTOM + BAR_HEIGHT + 0.2], color=FAIL_COLOR, linewidth=1.6, solid_capstyle="butt")
            ax.text(rollout.onset, base + ANNOTATION, failure_text(rollout.onset, rollout.mode), ha="center", va="center", fontsize=ANNOTATION_FONT, color=FAIL_COLOR)
        else:
            ax.text(end / 2, base + ANNOTATION, "success", ha="center", va="center", fontsize=ANNOTATION_FONT, color=SUCCESS_COLOR)  # in the middle of the figure
    for t in np.arange(0, end + 1e-9, WINDOW_SECONDS):
        ax.text(t, TICKS, f"{t:g}" + (" s" if t == end else ""), ha="center", va="center", fontsize=TICK_FONT, color="0.35")
    ax.set_xlim(-PAD, end + PAD)
    ax.set_ylim(0, units)
    ax.set_aspect("equal")
    return fig


def write(fig, stem: Path, formats: tuple[str, ...] = ("png", "pdf"), dpi: int = 300) -> list[Path]:
    """Write the figure as ``<stem>.<format>`` for each format, with the blank margins cropped, and close it; returns the files."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    files = [stem.with_suffix(f".{fmt}") for fmt in formats]
    for file in files:
        fig.savefig(file, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return files
