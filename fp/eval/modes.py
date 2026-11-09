"""Failure-mode prediction of our probes on LIBERO-X and on the real robot, from the saved runs (no model is run).

The mode head of a probe gives, at every policy step, logits over the 10 failure modes (``fp.data.targets.MODE_NAMES``). A failed episode has one
true mode, so the head is scored per failed episode: the mean softmax over a window of steps, ``window`` steps before an anchor step plus the anchor
itself (the on-time window of the detection metric, 2.4 s: 7 steps on LIBERO-X, 21 on the real robot), and the mode(s) with the highest mean probability are the prediction. Top-k accuracy is
the share of failed episodes whose true mode is among the k highest. Two anchors:

    oracle onset   the true failure onset: an upper bound, since a deployed system does not know when the failure starts
    own detection  the first step at which the probe's own score reaches its calibrated threshold (max-conformal at ``ALPHA``, the protocol's
                   calibrator for our probes): what a deployed system could do. It exists only for the failed episodes that get an alarm, so
                   accuracy is over those, with the share they make up (``coverage``) reported next to it.

``oracle_on_detected`` is the oracle rule on exactly the episodes of own detection, which isolates the effect of the timing from that of the
episodes. Baselines: always predicting the most common training mode(s) of the split, and a uniform guess among the training modes.

The numbers are per split, then the mean over the splits of a setting (every split counts once), for the seen and the unseen episodes, in percent.
"""

from collections import Counter, defaultdict
from typing import Mapping

import numpy as np

from ..calibration.protocol import make_calibrator
from ..metrics.outcomes import alarm_step
from ..results import read_scores, run_dir
from .figures import ENTRIES, SETTINGS, Entry
from .select_config import group_of
from .tables import ModeRow

ALPHA = 0.2  # the operating point of own detection: at most 20% false alarms on exchangeable successes
TOP_K = (1, 3)
SETS = ("seen", "unseen")
LIBEROX = tuple(s for s in SETTINGS if s.dataset == "liberox_groot")
REAL = tuple(s for s in SETTINGS if s.dataset == "real8_groot")  # the real robot: leave one task out, the splits of its extra seeds averaged in (``Setting.also``)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def window_probs(logits: np.ndarray, anchor: int, window: int) -> np.ndarray:
    """(n_modes,) mean softmax over the steps ``anchor - window`` ... ``anchor`` (clipped at the first step)."""
    return softmax(logits[max(anchor - window, 0) : anchor + 1]).mean(axis=0)


def hit(probs: np.ndarray, true: int, k: int) -> bool:
    """Whether the true mode is among the k most probable (ties are broken by the lower mode index)."""
    return bool(true in np.argsort(-probs, kind="stable")[:k])


def baseline_hits(train_modes: list[int], true: list[int], k: int) -> tuple[list[float], list[float]]:
    """(majority, random) top-k hit of each true mode of ``true``: 1 or 0 for always predicting the k most common training modes; the probability of a hit for a
    uniform guess of k distinct modes among those that occur in the training failures (a true mode that never does can never be hit)."""
    order = [m for m, _ in Counter(train_modes).most_common()]
    present = set(train_modes)
    return [float(t in order[:k]) for t in true], [(t in present) * min(k, len(present)) / len(present) for t in true]


def baseline_accuracy(train_modes: list[int], true: list[int], k: int) -> tuple[float, float]:
    """(majority, random) top-k accuracy over the true modes ``true`` (the mean of ``baseline_hits``)."""
    majority, random = baseline_hits(train_modes, true, k)
    return float(np.mean(majority)), float(np.mean(random))


def alarm_anchors(scores: Mapping[str, np.ndarray], episodes, keys: list[str], calibrator, alpha: float) -> tuple[list[str], dict[str, int | None]]:
    """The failed episodes among ``keys`` and the alarm step of each (None without an alarm) at the calibrated threshold of level ``alpha``. Needs at least one
    failed episode."""
    failed = [k for k in keys if not episodes[k].success]
    threshold = calibrator.threshold(alpha, max(len(scores[k]) for k in failed))
    return failed, {k: alarm_step(scores[k], threshold) for k in failed}


def evaluate_split(logits: Mapping[str, np.ndarray], scores: Mapping[str, np.ndarray], episodes, true_modes: Mapping[str, int], keys: list[str], calibrator,
                   alpha: float, window: int) -> dict:
    """Top-k accuracies of one probe on the failed episodes among ``keys``: ``oracle`` (all of them), ``detected`` and ``oracle_on_detected`` (those
    with an alarm; NaN if none), and ``coverage`` (the share with an alarm), for each k of ``TOP_K``. Needs at least one failed episode."""
    failed, alarms = alarm_anchors(scores, episodes, keys, calibrator, alpha)
    detected = [k for k in failed if alarms[k] is not None]

    def accuracy(selected: list[str], anchor, k: int) -> float:
        return float(np.mean([hit(window_probs(logits[e], anchor(e), window), true_modes[e], k) for e in selected])) if selected else float("nan")

    onset = lambda e: episodes[e].onset  # noqa: E731
    out = {"coverage": len(detected) / len(failed)}
    for k in TOP_K:
        out[f"oracle{k}"] = accuracy(failed, onset, k)
        out[f"detected{k}"] = accuracy(detected, lambda e: alarms[e], k)
        out[f"oracle_on_detected{k}"] = accuracy(detected, onset, k)
    return out


def split_modes(inputs, split) -> tuple[list[int], dict[str, list[str]], dict[str, int]]:
    """What the mode metrics of one split need: the modes of its training failures, its evaluation sets that have at least one failure (a set without a failure
    has no mode to predict) as {set: episode keys}, and the true mode of every failed episode of those sets."""
    train_modes = [int(inputs.targets.modes[k]) for k in split.train if not inputs.episodes[k].success]
    sets = {e: list(getattr(split, e)) for e in SETS}
    sets = {e: keys for e, keys in sets.items() if any(not inputs.episodes[k].success for k in keys)}
    true_modes = {k: int(inputs.targets.modes[k]) for keys in sets.values() for k in keys if not inputs.episodes[k].success}
    return train_modes, sets, true_modes


def mode_results(inputs, root, entries: list[Entry] | None = None, settings=LIBEROX, alpha: float = ALPHA) -> dict[tuple, float]:
    """{(rule, label, metric, setting label, set): percent}, the mean over the splits of the setting. Rules: majority, random (no label) and, for each
    of our entries, oracle, detected, oracle_on_detected and coverage (metric 'top1', 'top3', or 'share')."""
    entries = [e for e in ENTRIES if e.variant] if entries is None else entries
    per_split = defaultdict(list)
    for setting in settings:
        for name in (n for n in inputs.splits if group_of(n) in (setting.group, *setting.also)):
            split = inputs.splits[name]
            train_modes, sets, true_modes = split_modes(inputs, split)
            for eval_set, keys in sets.items():
                for k in TOP_K:
                    majority, random = baseline_accuracy(train_modes, [true_modes[e] for e in keys if e in true_modes], k)
                    per_split[("majority", None, f"top{k}", setting.label, eval_set)].append(majority)
                    per_split[("random", None, f"top{k}", setting.label, eval_set)].append(random)
            for entry in entries:
                folder = run_dir(inputs.dataset, name, f"{entry.architecture}__{entry.variant}", root)
                scores = read_scores(folder)
                calibrator = make_calibrator(folder.name, seed=split.seed).fit([scores[k] for k in split.calibration if inputs.episodes[k].success])
                with np.load(folder / "mode_logits.npz") as z:
                    logits = {k: z[k] for k in true_modes}
                for eval_set, keys in sets.items():
                    result = evaluate_split(logits, scores, inputs.episodes, true_modes, keys, calibrator, alpha, inputs.window)
                    per_split[("detected", entry.label, "share", setting.label, eval_set)].append(result["coverage"])
                    for k in TOP_K:
                        for rule in ("oracle", "detected", "oracle_on_detected"):
                            per_split[(rule, entry.label, f"top{k}", setting.label, eval_set)].append(result[f"{rule}{k}"])
    return {key: 100 * float(np.nanmean(v)) for key, v in per_split.items() if not np.all(np.isnan(v))}


def mode_rows(results: Mapping[tuple, float], entries: list[Entry] | None = None, settings=LIBEROX) -> list[ModeRow]:
    """The rows of the failure-mode table from ``mode_results``: the baselines, then our probe, one row per label variant, at its own alarm and at
    the oracle onset, and (for the CSV only) the oracle rule on the failures that have an alarm."""
    entries = [e for e in ENTRIES if e.variant] if entries is None else entries
    columns = [(s.label, e) for s in settings for e in SETS]

    def values(rule: str, label: str | None, metric: str) -> list[float | None]:
        return [results.get((rule, label, metric, s, e)) for s, e in columns]

    rows = [ModeRow("Most common", "baseline", values("majority", None, "top1"), values("majority", None, "top3")),
            ModeRow("Random", "baseline", values("random", None, "top1"), values("random", None, "top3"))]
    for rule in ("detected", "oracle"):
        rows += [ModeRow(e.variant, rule, values(rule, e.label, "top1"), values(rule, e.label, "top3"),
                         share=values("detected", e.label, "share") if rule == "detected" else None) for e in entries]
    rows += [ModeRow(e.variant, "oracle_on_detected", values("oracle_on_detected", e.label, "top1"), values("oracle_on_detected", e.label, "top3"), in_latex=False)
             for e in entries]
    return rows
