"""Episode table: one row per rollout (outcome, task, onset step, rollout length).

The onset is defined here, once. Every other module takes ``Episode.onset`` as given.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from ..config import get


@dataclass(frozen=True)
class Episode:
    key: str  # rollout directory name (the manifest's ``dir``), the key into features.npz
    success: bool
    task: str
    n_steps: int  # policy steps in the rollout
    onset: int | None  # policy step of the failure onset; None for successes
    condition: str = ""  # stratification label: scene variant (LIBERO-10) or level (LIBERO-X)


def read_manifest(path: Path) -> list[Episode]:
    """Episodes of a ``manifest.csv``.

    The onset of a failure is the labelled failure frame divided by ``replan_steps`` (frames are control steps, onsets are policy
    steps). A failure without a labelled frame, or with an onset outside [0, n_policy - 1], is an error; successes have no onset.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    episodes, no_onset, out_of_range = [], [], []
    for r in rows:
        success = r["success"] == "True"
        n_steps = int(r["n_policy"])
        onset = None
        if not success:
            frame = r["vlm_failure_onset_frame"].strip()
            if not frame:
                no_onset.append(r["dir"])
                continue
            onset = int(float(frame)) // int(r["replan_steps"])
            if not 0 <= onset < n_steps:
                out_of_range.append(r["dir"])
                continue
        condition = r["level"] if "level" in r else r["scene_variant"]
        episodes.append(Episode(r["dir"], success, r["task"], n_steps, onset, condition))
    if no_onset:
        raise ValueError(f"{len(no_onset)} failed episodes have no labelled failure frame, e.g. {no_onset[:3]}")
    if out_of_range:
        raise ValueError(f"{len(out_of_range)} failed episodes have an onset outside the rollout, e.g. {out_of_range[:3]}")
    return episodes


def load_manifest(dataset: str) -> list[Episode]:
    return read_manifest(get(dataset).data_dir / "manifest.csv")
