"""Choose the best configuration on the VALIDATION episodes, never on the unseen tasks.

For one family of configurations (e.g. Euclidean with its 6 configurations, or our 64 probes) and one dataset with several runs (splits):
  1. per run, the configuration's scores on the run's calibration and validation episodes (fitted on the spot, or read from saved runs);
  2. calibrate with its protocol calibrator on the run's calibration successes;
  3. sweep alpha on the run's validation episodes and take the area under TPR early vs FPR up to 40%;
  4. the best configuration is the one with the highest MEAN validation area over the dataset's runs
     (ties: the first in grid order).

Steps 2-3 are ``validation_area_from_scores`` and step 4 is ``choose_best``: the fitted-scorer path (``select_best``) and the saved-scores
path call the same two functions, so every family is selected by one rule.
The only episodes this module ever reads scores of are the ``calibration`` and ``val`` keys of each split; ``seen`` and ``unseen``
are never read here (a test checks this).
"""

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from ..calibration.protocol import make_calibrator
from ..data.manifest import Episode
from ..data.splits import Split
from ..metrics.curves import evaluate_set
from ..models.base import Scorer


@dataclass(frozen=True)
class Selection:
    best: str  # name of the winning configuration
    mean_val_area: dict[str, float]  # configuration name -> mean validation area over the runs (grid order)
    per_run: dict[str, list[float]]  # configuration name -> validation area of each run (same run order for every configuration)


def validation_area_from_scores(scores: Mapping[str, np.ndarray], method: str, split: Split, episodes: Mapping[str, Episode], window: int) -> float:
    """Validation area of a method whose scores on the split's calibration and validation episodes are given; the calibrator is the
    protocol's for ``method``, fitted on the split's calibration successes."""
    calibrator = make_calibrator(method, seed=split.seed).fit([scores[k] for k in split.calibration if episodes[k].success])
    result = evaluate_set(calibrator, scores, [episodes[k] for k in split.val], window)
    if np.isnan(result.area):
        raise ValueError(f"the validation episodes of {split.name} need both failures and successes to have an area")
    return result.area


def validation_area(scorer: Scorer, split: Split, episodes: Mapping[str, Episode], window: int) -> float:
    """Fit ``scorer`` on the split's training episodes and return its validation area (calibrator from the protocol)."""
    n_fail = sum(not episodes[k].success for k in split.val)
    if n_fail == 0 or n_fail == len(split.val):
        raise ValueError(f"the validation episodes of {split.name} need both failures and successes to have an area")
    scorer.fit(split.train, split.val)
    scores = scorer.score(split.calibration + split.val)
    return validation_area_from_scores(scores, scorer.name, split, episodes, window)


LIBEROX_KINDS = ("task", "signature", "skill")


def group_of(split_name: str) -> str:
    """The held-out set a split belongs to: 'task', 'signature' or 'skill' for LIBERO-X splits (named task_seed0,
    signature_seed1, skill_<name>), 'full' for the real robot's splits with nothing held out (full_seed0), 'loto_seeds' for its extra leave-one-task-out
    seeds (lotoseed1_<task>), 'all' for every other split (the LIBERO-10 and real-robot seed-0 leave-one-task-out runs are pooled)."""
    first = split_name.split("_")[0]
    if first.startswith("lotoseed"):  # the extra seeds of the real robot's leave-one-task-out runs, kept apart from the pooled 'all' group
        return "loto_seeds"
    return first if first in LIBEROX_KINDS + ("full",) else "all"


def split_group(split: Split) -> str:
    return group_of(split.name)


def choose_best(per_run: Mapping[str, list[float]]) -> Selection:
    """The configuration with the highest mean validation area over the runs; ties go to the earlier configuration.
    ``per_run`` maps a configuration name, in grid order, to its validation area on each run (the same runs for every configuration)."""
    lengths = {len(v) for v in per_run.values()}
    if len(lengths) != 1:
        raise ValueError("every configuration needs a validation area on the same runs")
    if any(np.isnan(v).any() for v in per_run.values()):
        raise ValueError("a validation area is NaN; the configurations cannot be compared")
    mean = {name: float(np.mean(v)) for name, v in per_run.items()}
    best = max(mean, key=lambda n: (mean[n], -list(mean).index(n)))
    return Selection(best, mean, {name: list(v) for name, v in per_run.items()})


def select_best_per_group(
    make_scorers: Callable[[], list[Scorer]],
    splits: list[Split],
    episodes: Mapping[str, Episode],
    window: int,
    group: Callable[[Split], str] = split_group,
) -> dict[str, "Selection"]:
    """One selection per group of splits (LIBERO-X: the 3 task seeds, the 3 signature seeds, the 8 skill folds; LIBERO-10: all runs)."""
    groups: dict[str, list[Split]] = {}
    for s in splits:
        groups.setdefault(group(s), []).append(s)
    return {g: select_best(make_scorers, members, episodes, window) for g, members in groups.items()}


def select_best(
    make_scorers: Callable[[], list[Scorer]],
    splits: list[Split],
    episodes: Mapping[str, Episode],
    window: int,
) -> Selection:
    """``make_scorers`` returns fresh, unfitted scorers (one per configuration, always in the same order) for each split."""
    per_run: dict[str, list[float]] = {}
    for split in splits:
        for scorer in make_scorers():
            per_run.setdefault(scorer.name, []).append(validation_area(scorer, split, episodes, window))
    return choose_best(per_run)
