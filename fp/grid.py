"""The training grid: every probe of the suite on every split of every dataset, in the order the runs are made.

A *unit* is the 16 architectures of one label variant on one split: they share one standardised copy of the split's features. The order
decides what exists if a run is cut short: the pre-onset label variant (the key plot) goes first, on every split, seed 0 of each split kind
before the other seeds; the other label variants follow.
"""

import re
from dataclasses import dataclass
from typing import Mapping

from .data.splits import make_splits
from .models.ours import all_configs

VARIANTS = ("onset_pre", "onset", "onset_post", "onset_traj")  # in priority order
DATASETS = ("liberox_groot", "libero10_groot", "libero10_pi05", "real8_groot")
KIND_RANK = {"task": 0, "paper": 0, "loto": 0, "full": 0, "signature": 1, "skill": 2}


@dataclass(frozen=True)
class Unit:
    dataset: str
    split: str
    variant: str

    @property
    def methods(self) -> list[str]:
        return [f"{cfg.name}__{self.variant}" for cfg in all_configs()]


def split_priority(name: str) -> tuple[int, int, str]:
    """Seed first, then split kind (task / paper / loto, signature, skill); the skill and loto folds all have seed 0."""
    extra = re.fullmatch(r"lotoseed(\d+)", name.split("_")[0])
    if extra:  # an extra seed of the leave-one-task-out runs: lotoseed<seed>_<task>
        return (int(extra.group(1)), KIND_RANK["loto"], name)
    seed = re.search(r"_seed(\d+)$", name)
    return (int(seed.group(1)) if seed else 0, KIND_RANK[name.split("_")[0]], name)


def all_split_names(datasets: tuple[str, ...] = DATASETS) -> dict[str, list[str]]:
    return {d: [s.name for s in make_splits(d)] for d in datasets}


def units(split_names: Mapping[str, list[str]], variants: tuple[str, ...] = VARIANTS) -> list[Unit]:
    """Every unit in run order: label variant, then dataset (in ``DATASETS`` order), then split by priority."""
    return [Unit(d, s, v) for v in variants for d in DATASETS if d in split_names for s in sorted(split_names[d], key=split_priority)]
