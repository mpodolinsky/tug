"""Cheap checks, made before any training, for the things that would crash a job hours into a run.

For a dataset: every episode's feature length equals its number of steps and the length of its phrase ids; every failure has a mode.
For every split: it partitions the episodes; training and validation hold both outcomes; and for every label variant the training steps hold
both classes and at least ``clip_dim`` distinct phrases (fewer would make the PCA target narrower than the semantic head).
"""

from typing import Mapping

import numpy as np

from ..data.labels import build_labels
from ..data.manifest import Episode
from ..data.splits import Split
from ..data.targets import MODE_NAMES, Targets
from ..models.ours import ProbeConfig


def check_episodes(episodes: Mapping[str, Episode], targets: Targets, feature_lengths: Mapping[str, int]) -> list[str]:
    problems = []
    for key, e in episodes.items():
        if feature_lengths.get(key) != e.n_steps:
            problems.append(f"{key}: {feature_lengths.get(key)} feature steps, the manifest says {e.n_steps}")
        ids = targets.phrase_ids.get(key)
        if ids is None or len(ids) != e.n_steps:
            problems.append(f"{key}: {None if ids is None else len(ids)} phrase ids, the manifest says {e.n_steps}")
        if not e.success and not 0 <= targets.modes.get(key, -1) < len(MODE_NAMES):
            problems.append(f"{key}: a failed episode without a failure mode")
    return problems


def check_split(split: Split, episodes: Mapping[str, Episode], targets: Targets, variants: tuple[str, ...], window: int,
                min_phrases: int = ProbeConfig().clip_dim) -> list[str]:
    problems = []
    parts = (split.train, split.val, split.calibration, split.seen, split.unseen)
    keys = [k for part in parts for k in part]
    if len(keys) != len(set(keys)) or set(keys) != set(episodes):
        problems.append(f"{split.name}: train, val, calibration, seen and unseen do not partition the episodes")
    for name, part in (("train", split.train), ("val", split.val)):
        outcomes = {episodes[k].success for k in part}
        if outcomes != {True, False}:
            problems.append(f"{split.name}: {name} holds only {'successes' if outcomes == {True} else 'failures'}")
    for variant in variants:
        labels = [build_labels(variant, episodes[k].n_steps, episodes[k].onset, window) for k in split.train]
        y = np.concatenate([lab.y for lab in labels])
        if y.sum() == 0 or y.sum() == len(y):
            problems.append(f"{split.name}/{variant}: the training steps do not hold both classes")
        phrases = {int(p) for k, lab in zip(split.train, labels) for p in targets.phrase_ids[k][: lab.n] if p >= 0}
        if len(phrases) < min_phrases:
            problems.append(f"{split.name}/{variant}: {len(phrases)} distinct training phrases, fewer than the {min_phrases} of the semantic head")
    return problems
