"""Per-step training labels for each supervision variant.

Let o be the onset step, w the on-time window (policy steps), lo = max(o - w, 0) and n the rollout length.

    variant        positive steps       steps kept
    onset          [lo, o]              o + 1          (rollout cut at the onset)
    onset_pre      [0, o]               o + 1          (the whole pre-onset trajectory)
    onset_post     [lo, n - 1]          n              (window and everything after it)
    onset_traj     [0, n - 1]           n              (every step of a failure; SAFE's label)
    onset_pm_cut   [lo, min(o+w, n-1)]  min(o+w, n-1) + 1   (window on both sides of the onset)

Successes are all zeros and keep every step.
"""

from dataclasses import dataclass

import numpy as np

VARIANTS = ("onset", "onset_pre", "onset_post", "onset_traj", "onset_pm_cut")


@dataclass(frozen=True)
class LabelledEpisode:
    y: np.ndarray  # (n,) 0/1 training label; len(y) is the number of steps kept
    n: int  # steps kept (rollout cut applied)


def build_labels(variant: str, n_steps: int, onset: int | None, window: int) -> LabelledEpisode:
    """Labels and kept length for one episode. ``onset`` is None for a success."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown label variant {variant!r}; choose from {VARIANTS}")
    if n_steps < 1:
        raise ValueError("an episode needs at least one step")
    if onset is None:
        return LabelledEpisode(np.zeros(n_steps, dtype=np.float32), n_steps)
    if not 0 <= onset < n_steps:
        raise ValueError(f"onset {onset} outside [0, {n_steps - 1}]")

    lo = max(onset - window, 0)
    if variant == "onset":
        first, last, keep = lo, onset, onset + 1
    elif variant == "onset_pre":
        first, last, keep = 0, onset, onset + 1
    elif variant == "onset_post":
        first, last, keep = lo, n_steps - 1, n_steps
    elif variant == "onset_traj":
        first, last, keep = 0, n_steps - 1, n_steps
    else:  # onset_pm_cut
        last = min(onset + window, n_steps - 1)
        first, keep = lo, last + 1
    y = np.zeros(keep, dtype=np.float32)
    y[first : last + 1] = 1.0
    return LabelledEpisode(y, keep)
