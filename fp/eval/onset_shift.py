"""How the early-detection area changes when the failure time is moved later (the table of ``cli.onset_shift``; no model is run).

The failure time of an episode is a human-reviewed label: the earliest point at which the mistake responsible for the failure becomes apparent. It is already at the
limit of what can be seen, so it can only be too early, never too late; and a later failure time is the direction in which the early-detection metric gets
*easier* (an alarm counts as early when it comes at or before the failure time). This asks how much the areas, and the order of the methods, depend on that choice.

The shift is applied to the evaluation only: the saved scores are scored again against failure times moved ``seconds`` later, and each method keeps the threshold it
was calibrated to (the calibrators of the protocol use the calibration successes alone, so they do not depend on the failure times). A shift of ``s`` seconds is a
shift of floor(s * control rate / replan steps) policy steps: an alarm is early when ``alarm step - failure step`` is at most ``s / step period``, and the alarm step is an
integer, so a fraction of a step does not count (a policy step lasts 0.4 s on LIBERO-X, 0.25 s for pi0.5 and 0.12 s on the real robot: 1 s is 2, 4 and 8 steps, 2 s is
5, 8 and 16). The training labels are not changed (the probes are not retrained).
"""

import math
from dataclasses import replace
from typing import Mapping

import numpy as np

from ..config import get
from ..data.manifest import Episode
from ..results import run_dir
from .figures import Setting
from .run_eval import evaluate_run
from .tables import ShiftRow
from .select_config import group_of
from ..calibration.protocol import calibrator_name


def shift_steps(seconds: float, dataset: str) -> int:
    """The shift of ``seconds`` in policy steps: the whole steps that fit in it (see above)."""
    spec = get(dataset)
    return int(math.floor(seconds * spec.control_hz / spec.replan_steps + 1e-9))


def shift_episodes(episodes: Mapping[str, Episode], steps: int) -> dict[str, Episode]:
    """The episodes with the failure time of every failed one moved ``steps`` policy steps later (successes are unchanged)."""
    return {k: e if e.success else replace(e, onset=e.onset + steps) for k, e in episodes.items()}


def shift_rows(records: list[dict], settings: list[str], labels: list[str], shifts: list[float], eval_set: str) -> list[ShiftRow]:
    """The rows of the LaTeX table from the records of ``cli.onset_shift`` (setting, method, set, shift (s), area mean): a row per label of ``labels`` that has a
    record, with the area for each setting and shift (None where there is none)."""
    area = {(r["setting"], r["method"], float(r["shift (s)"])): float(r["area mean"]) for r in records if r["set"] == eval_set}
    return [ShiftRow(label, label.startswith("Ours"), [area.get((s, label, x)) for s in settings for x in shifts])
            for label in labels if any((s, label, x) in area for s in settings for x in shifts)]


def shifted_areas(inputs, root, method: str, setting: Setting, seconds: tuple[float, ...]) -> dict[tuple[float, str], list[float]]:
    """{(shift in seconds, set): the area of each split of the setting}, for the ``seen`` and ``unseen`` sets, with the calibrator the protocol gives ``method``.
    A split without a run or an evaluation (a failed run) is left out."""
    calibrator = calibrator_name(method)
    out = {(s, e): [] for s in seconds for e in ("seen", "unseen")}
    for name in (n for n in inputs.splits if group_of(n) in (setting.group, *setting.also)):
        folder = run_dir(inputs.dataset, name, method, root)
        if not (folder / "scores.npz").exists():
            continue
        for s in seconds:
            episodes = shift_episodes(inputs.episodes, shift_steps(s, inputs.dataset))
            result = evaluate_run(folder, method, inputs.splits[name], episodes, inputs.window)[calibrator]
            if "error" in result:
                continue
            for eval_set in ("seen", "unseen"):
                area = result["sets"][eval_set].area
                if not np.isnan(area):
                    out[(s, eval_set)].append(area)
    return out
