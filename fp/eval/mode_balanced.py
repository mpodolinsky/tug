"""Failure-mode prediction scored with a balanced accuracy over merged classes (a table of ``cli.figures``; no model is run).

Plain top-k accuracy (``fp.eval.modes``) rewards predicting the most common mode: on the real robot, 57% of the failures are grasp failures, so a constant
prediction gets 53% top-1. Here every class counts equally, and rare modes are merged so that no class is too small to estimate.

Classes. The 10 modes of the taxonomy are merged, once per dataset, until every class holds at least ``FLOOR`` of the failures of the dataset
(``merge_groups``): the two smallest classes are merged repeatedly. The rule uses label frequencies only, never the performance of a probe, and a mode that
never occurs is not a class. The mode head is not retrained: a merged class gets the sum of the probabilities of its modes, so the evaluation of a trained
head only changes in how its output is read (training the head on the merged classes, with its class weights, is left to future work).

Score. Each failed episode gets its prediction at the anchor of ``fp.eval.modes`` (the probe's first alarm, or the true onset): the merged class with the
highest mean probability over the on-time window, or the k highest for top-k. The balanced accuracy is the mean over the classes of the share of
the failed episodes of that class whose prediction is a hit, in percent, so a constant prediction scores 100 / K (K classes) for top-1. It is computed on the
predictions pooled over the splits of a setting, not split by split: a split has few failures of a rare class (a real-robot fold has 24 to 32 failures in all),
the pooled count is what the 5% floor refers to, and a class absent from a split's set is simply not in the pool. Episodes that several splits evaluate (the
unseen episodes are the same in the three seeds of the LIBERO-X task and signature splits) count once per split. The baselines are scored in the same classes:
most common (the k most common training classes of the split) and random (the expected hit of a uniform guess among the training classes).
"""

from collections import defaultdict
from typing import Mapping

import numpy as np

from ..calibration.protocol import make_calibrator
from ..results import read_scores, run_dir
from . import modes
from .figures import ENTRIES, Entry
from .select_config import group_of

FLOOR = 0.05  # no class holds less than this share of the failures of its dataset


def merge_groups(counts: Mapping[int, int], floor: float = FLOOR) -> list[tuple[int, ...]]:
    """The classes (tuples of mode indices, sorted) of failure ``counts`` {mode: n}: modes with n = 0 are dropped, then the two smallest classes are merged
    (ties to the lower mode index) until every class has at least ``floor`` of the failures or one class is left."""
    total = sum(counts.values())
    classes = [((m,), n) for m, n in sorted(counts.items()) if n > 0]
    while len(classes) > 1 and min(n for _, n in classes) < floor * total:
        classes.sort(key=lambda c: (c[1], c[0]))
        (a, na), (b, nb), *rest = classes
        classes = [(tuple(sorted(a + b)), na + nb), *rest]
    return sorted(c for c, _ in classes)


def class_index(groups: list[tuple[int, ...]]) -> dict[int, int]:
    """The class of each mode that belongs to one."""
    return {m: i for i, g in enumerate(groups) for m in g}


def class_probs(probs: np.ndarray, groups: list[tuple[int, ...]]) -> np.ndarray:
    """(n_classes,) probability of each merged class: the sum over its modes."""
    return np.array([probs[list(g)].sum() for g in groups])


def balanced(pairs: list[tuple[int, float]]) -> float | None:
    """The mean over the classes present in ``pairs`` of (class, hit) of the mean hit of that class, in percent; None for no pair."""
    by_class = defaultdict(list)
    for c, h in pairs:
        by_class[c].append(h)
    return 100 * float(np.mean([np.mean(h) for h in by_class.values()])) if by_class else None


def episode_hits(logits: Mapping[str, np.ndarray], scores: Mapping[str, np.ndarray], episodes, true_class: Mapping[str, int], keys: list[str],
                 calibrator, alpha: float, window: int, groups: list[tuple[int, ...]]) -> list[dict]:
    """For each failed episode among ``keys``: its class, whether it has an alarm, and the top-k hits (k of ``modes.TOP_K``) of the merged prediction at the
    true onset (``oracle``) and, when there is an alarm, at the alarm (``detected``)."""
    failed, alarms = modes.alarm_anchors(scores, episodes, keys, calibrator, alpha)
    out = []
    for e in failed:
        def hits(anchor: int) -> dict[int, float]:
            p = class_probs(modes.window_probs(logits[e], anchor, window), groups)
            return {k: float(modes.hit(p, true_class[e], k)) for k in modes.TOP_K}

        out.append({"class": true_class[e], "alarm": alarms[e] is not None, "oracle": hits(episodes[e].onset),
                    "detected": None if alarms[e] is None else hits(alarms[e])})
    return out


def dataset_groups(inputs, floor: float = FLOOR) -> list[tuple[int, ...]]:
    """The classes of a dataset: ``merge_groups`` of the modes of all its failed episodes."""
    counts = defaultdict(int)
    for k, e in inputs.episodes.items():
        if not e.success:
            counts[int(inputs.targets.modes[k])] += 1
    return merge_groups(counts, floor)


def balanced_results(inputs, root, entries: list[Entry] | None = None, settings=modes.LIBEROX, alpha: float = modes.ALPHA,
                     floor: float = FLOOR) -> dict[tuple, float]:
    """{(rule, label, metric, setting label, set): percent} as ``modes.mode_results`` (same keys and rules: majority and random without a label; for each
    entry oracle, detected, oracle_on_detected and the share of failures with an alarm), but the balanced accuracy over the merged classes, pooled over the
    splits of the setting. Metric 'top1', 'top3' or 'share'."""
    entries = [e for e in ENTRIES if e.variant] if entries is None else entries
    groups = dataset_groups(inputs, floor)
    index = class_index(groups)
    pooled, alarmed = defaultdict(list), defaultdict(lambda: [0, 0])
    for setting in settings:
        for name in (n for n in inputs.splits if group_of(n) in (setting.group, *setting.also)):
            split = inputs.splits[name]
            train_modes, sets, true_modes = modes.split_modes(inputs, split)
            true_class = {k: index[m] for k, m in true_modes.items()}
            train_class = [index[m] for m in train_modes]
            for eval_set, keys in sets.items():
                failed = [true_class[k] for k in keys if k in true_class]
                for k in modes.TOP_K:
                    majority, random = modes.baseline_hits(train_class, failed, k)
                    pooled[("majority", None, f"top{k}", setting.label, eval_set)] += list(zip(failed, majority))
                    pooled[("random", None, f"top{k}", setting.label, eval_set)] += list(zip(failed, random))
            for entry in entries:
                folder = run_dir(inputs.dataset, name, f"{entry.architecture}__{entry.variant}", root)
                scores = read_scores(folder)
                calibrator = make_calibrator(folder.name, seed=split.seed).fit([scores[k] for k in split.calibration if inputs.episodes[k].success])
                with np.load(folder / "mode_logits.npz") as z:
                    logits = {k: z[k] for k in true_class}
                for eval_set, keys in sets.items():
                    for r in episode_hits(logits, scores, inputs.episodes, true_class, keys, calibrator, alpha, inputs.window, groups):
                        alarmed[("detected", entry.label, "share", setting.label, eval_set)][0] += r["alarm"]
                        alarmed[("detected", entry.label, "share", setting.label, eval_set)][1] += 1
                        for k in modes.TOP_K:
                            pooled[("oracle", entry.label, f"top{k}", setting.label, eval_set)].append((r["class"], r["oracle"][k]))
                            if r["alarm"]:
                                pooled[("detected", entry.label, f"top{k}", setting.label, eval_set)].append((r["class"], r["detected"][k]))
                                pooled[("oracle_on_detected", entry.label, f"top{k}", setting.label, eval_set)].append((r["class"], r["oracle"][k]))
    results = {key: balanced(pairs) for key, pairs in pooled.items() if pairs}
    results |= {key: 100 * n / total for key, (n, total) in alarmed.items() if total}
    return results


def group_rows(inputs, floor: float = FLOOR) -> list[dict]:
    """One record per class of the dataset: its modes and its share of the failures, in percent (for the CSV of the classes)."""
    from ..data.targets import MODE_NAMES

    counts = defaultdict(int)
    for k, e in inputs.episodes.items():
        if not e.success:
            counts[int(inputs.targets.modes[k])] += 1
    total = sum(counts.values())
    return [{"dataset": inputs.dataset, "class": i + 1, "modes": " + ".join(MODE_NAMES[m] for m in g), "failures": sum(counts[m] for m in g),
             "share %": round(100 * sum(counts[m] for m in g) / total, 1)} for i, g in enumerate(dataset_groups(inputs, floor))]
