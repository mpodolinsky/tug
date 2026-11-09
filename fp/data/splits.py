"""Train / val / calibration / seen-eval / unseen episode splits (pure logic).

One algorithm for every dataset:
  1. episodes of the held-out groups (tasks, signatures or skills) go to ``unseen`` (none held out: ``unseen`` is empty);
  2. the rest is split 65/15/20 into train / val / test, stratified by (outcome, condition);
  3. the test part is halved into ``calibration`` and ``seen`` with the same strata.
``condition`` is the scene variant (LIBERO-10), the level (LIBERO-X) or the task (real robot). Same seed, same split.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Hashable

from ..config import get
from .manifest import Episode, load_manifest

FRACTIONS = (("train", 0.65), ("val", 0.15), ("test", 0.20))
N_UNSEEN_TASKS = 3  # LIBERO-10: tasks held out per seed
PAPER_SEEDS = (0, 1, 2)  # LIBERO-10: three random held-out task triples
LIBEROX_SEEDS = (0, 1, 2)  # LIBERO-X task and signature splits; the skill folds use seed 0
FULL_SEEDS = (0, 1, 2)  # real robot: nothing held out, three shuffles of all tasks
LOTO_EXTRA_SEEDS = (1, 2, 3, 4)  # real robot: the leave-one-task-out runs again (seed 0 is the loto_<task> runs); the held-out task is the same, the seen episodes are shuffled differently


@dataclass(frozen=True)
class Split:
    name: str  # e.g. "paper_seed0", "skill_pick", "task_seed1", "loto_basket"
    train: list[str]
    val: list[str]
    calibration: list[str]  # first half of the seen-task test part (fits the calibrators)
    seen: list[str]  # second half: seen tasks, never used by a calibrator
    unseen: list[str]  # every episode of the held-out groups
    seed: int  # the seed the split was drawn with; also the shuffle seed of the SAFE band calibrated on this split


def stratum(e: Episode) -> tuple[bool, str]:
    return (e.success, e.condition)


def stratified_partition(
    episodes: list[Episode],
    fractions: tuple[tuple[str, float], ...],
    seed: int,
) -> dict[str, list[str]]:
    """Per stratum: sort keys, shuffle with Random(seed), cut at rounded fractions; the last part takes the remainder."""
    rng = random.Random(seed)
    strata: dict[Hashable, list[str]] = {}
    for e in episodes:
        strata.setdefault(stratum(e), []).append(e.key)
    out: dict[str, list[str]] = {name: [] for name, _ in fractions}
    last = fractions[-1][0]
    for s in sorted(strata):
        keys = sorted(strata[s])
        rng.shuffle(keys)
        n, pos = len(keys), 0
        for name, frac in fractions:
            cnt = n - pos if name == last else min(int(round(frac * n)), n - pos)
            out[name].extend(keys[pos : pos + cnt])
            pos += cnt
    return out


def build_split(
    episodes: list[Episode],
    name: str,
    seed: int,
    group_of: Callable[[Episode], str],
    unseen_groups: set[str],
    calibration_fraction: float = 0.5,
) -> Split:
    unseen = [e for e in episodes if group_of(e) in unseen_groups]
    seen = [e for e in episodes if group_of(e) not in unseen_groups]
    if unseen_groups and not unseen:  # (no groups asked for is a split with nothing held out)
        raise ValueError(f"no episode belongs to the held-out groups {sorted(unseen_groups)}")
    parts = stratified_partition(seen, FRACTIONS, seed)
    test = [e for e in seen if e.key in set(parts["test"])]
    halves = stratified_partition(
        test, (("calibration", calibration_fraction), ("seen", 1.0 - calibration_fraction)), seed
    )
    return Split(
        name=name,
        train=sorted(parts["train"]),
        val=sorted(parts["val"]),
        calibration=sorted(halves["calibration"]),
        seen=sorted(halves["seen"]),
        unseen=sorted(e.key for e in unseen),
        seed=seed,
    )


def pick_unseen_tasks(tasks: list[str], seed: int, n: int = N_UNSEEN_TASKS) -> set[str]:
    """LIBERO-10 rule: shuffle the sorted task names with Random(seed + 1000) and take the first n."""
    names = sorted(set(tasks))
    random.Random(seed + 1000).shuffle(names)
    return set(names[:n])


def read_unseen_tasks(path: Path) -> set[str]:
    """Task names of a LIBERO-X split file (one ``<task>.bddl`` per line)."""
    return {line.strip()[: -len(".bddl")] for line in open(path) if line.strip()}


def split_plan(dataset: str, tasks: list[str]) -> list[tuple[str, int, set[str]]]:
    """(name, seed, held-out tasks) of every run of the dataset. The names are what ``select_config.split_group`` reads."""
    spec = get(dataset)
    plan = []
    if "paper" in spec.split_kinds:
        plan += [(f"paper_seed{s}", s, pick_unseen_tasks(tasks, s)) for s in PAPER_SEEDS]
    files = spec.data_dir / "splits"
    if "task" in spec.split_kinds:
        plan += [(f"task_seed{s}", s, read_unseen_tasks(files / "leave_task_out" / "test.txt")) for s in LIBEROX_SEEDS]
    if "signature" in spec.split_kinds:
        plan += [(f"signature_seed{s}", s, read_unseen_tasks(files / "leave_signature_out" / "test.txt")) for s in LIBEROX_SEEDS]
    if "skill" in spec.split_kinds:
        skills = sorted(p.name for p in (files / "leave_skill_out").iterdir() if p.is_dir())
        plan += [(f"skill_{k}", 0, read_unseen_tasks(files / "leave_skill_out" / k / "test.txt")) for k in skills]
    if "loto" in spec.split_kinds:  # real robot: leave one task out, every task once
        plan += [(f"loto_{t}", 0, {t}) for t in sorted(set(tasks))]
    if "full" in spec.split_kinds:  # real robot: every task in training, no unseen set (for the model that is shared, not for a result on unseen tasks)
        plan += [(f"full_seed{s}", s, set()) for s in FULL_SEEDS]
    if "lotoseeds" in spec.split_kinds:  # real robot: more seeds of the leave-one-task-out runs, named lotoseed<seed>_<task>
        plan += [(f"lotoseed{s}_{t}", s, {t}) for s in LOTO_EXTRA_SEEDS for t in sorted(set(tasks))]
    return plan


def make_splits(dataset: str) -> list[Split]:
    """Every run of the dataset (LIBERO-10: 3 paper seeds; LIBERO-X: 3 task seeds, 3 signature seeds and one fold
    per skill; real robot: one fold per task and three splits with nothing held out), from data/<dataset>/manifest.csv and, for LIBERO-X, data/<dataset>/splits/."""
    episodes = load_manifest(dataset)
    tasks = {e.task for e in episodes}
    splits = []
    for name, seed, unseen in split_plan(dataset, sorted(tasks)):
        missing = unseen - tasks
        if missing:
            raise ValueError(f"split {name}: held-out tasks not in the manifest, e.g. {sorted(missing)[:2]}")
        splits.append(build_split(episodes, name, seed, lambda e: e.task, unseen))
    return splits
