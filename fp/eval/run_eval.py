"""Evaluate saved scores: calibrate on the split's calibration successes, sweep alpha on the validation, seen and unseen episodes.

``evaluate_scores`` is the one function that does the work, on scores alone (no model, no simulator): the calibrator is fitted on the
successes of ``split.calibration`` and nothing else, and the same fitted calibrator is applied to each evaluation set. The sets are

    val      the validation episodes (seen tasks); the probe's early stopping used them, so they are optimistic -- they are what
             configurations are chosen on, never what is reported as a result
    seen     seen tasks, never used by a calibrator or by training
    unseen   every episode of the held-out tasks, signatures or skills

Each set gets the alpha sweep and the area under TPR early vs FPR up to 40 (``metrics.curves.evaluate_set``). The calibrator of a method is
the one the protocol gives it (``calibration.protocol``); our probes can also be evaluated with the other calibrator as a secondary table.
LTT is a separate evaluation with targets that must be given (``evaluate_ltt``).
"""

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from ..calibration.ltt import calibrate_ltt
from ..calibration.protocol import MAX_CONFORMAL, SAFE_BAND, calibrator_by_name, calibrator_name, parse_probe_name
from ..data.inputs import Inputs
from ..data.manifest import Episode
from ..data.splits import Split
from ..metrics.curves import ALPHAS, SetResult, evaluate_set
from ..metrics.outcomes import classify
from ..metrics.rates import rates_from_outcomes
from ..results import code_version, read_json, read_scores, write_json

EVAL_SETS = ("val", "seen", "unseen")


def evaluate_scores(scores: Mapping[str, np.ndarray], split: Split, episodes: Mapping[str, Episode], window: int, calibrator,
                    alphas: np.ndarray = ALPHAS) -> dict[str, SetResult]:
    """Fit ``calibrator`` on the calibration successes, then sweep each of the evaluation sets."""
    calibrator.fit([scores[k] for k in split.calibration if episodes[k].success])
    results = {}
    for name in EVAL_SETS:
        keys = getattr(split, name)
        if not keys:
            results[name] = SetResult([], float("nan"), False, 0, 0)
            continue
        results[name] = evaluate_set(calibrator, scores, [episodes[k] for k in keys], window, alphas)
    return results


def calibrator_names(method: str, secondary: bool = False) -> list[str]:
    """The method's protocol calibrator first; with ``secondary`` our probes also get the other one."""
    primary = calibrator_name(method)
    if secondary and parse_probe_name(method) is not None:
        return [primary] + [name for name in (MAX_CONFORMAL, SAFE_BAND) if name != primary]
    return [primary]


def evaluate_run(path: Path, method: str, split: Split, episodes: Mapping[str, Episode], window: int, secondary: bool = False,
                 alphas: np.ndarray = ALPHAS) -> dict[str, dict]:
    """Evaluate one saved run with each of its calibrators. A calibrator that cannot be fitted (for example the SAFE band with too few
    calibration successes) is recorded as {"error": message}, not raised."""
    scores = read_scores(path)
    out = {}
    for name in calibrator_names(method, secondary):
        try:
            out[name] = {"sets": evaluate_scores(scores, split, episodes, window, calibrator_by_name(name, split.seed), alphas)}
        except ValueError as error:
            out[name] = {"error": str(error)}
    return out


def write_evaluation(path: Path, evaluation: dict[str, dict], dataset: str, split: Split, method: str, episodes: Mapping[str, Episode],
                     window: int, alphas: np.ndarray = ALPHAS) -> None:
    """curve_<calibrator>.json for each calibrator and summary.json, each replaced atomically."""
    calibration = [episodes[k] for k in split.calibration]
    summary = {
        "dataset": dataset, "split": split.name, "method": method, "window": window, "code": code_version(),
        "alphas": {"first": float(alphas[0]), "last": float(alphas[-1]), "n": len(alphas)},
        "n_calibration_successes": sum(e.success for e in calibration), "n_calibration_failures": sum(not e.success for e in calibration),
        "calibrators": {},
    }
    for name, result in evaluation.items():
        if "error" in result:
            summary["calibrators"][name] = result
            continue
        sets = result["sets"]
        write_json(path / f"curve_{name}.json", {"calibrator": name, "sets": {s: r.rows for s, r in sets.items()}})
        summary["calibrators"][name] = {"sets": {s: {"area": r.area, "extended": r.extended, "n_fail": r.n_fail, "n_succ": r.n_succ} for s, r in sets.items()}}
    write_json(path / "summary.json", summary)


def is_current(path: Path) -> bool:
    """True if ``path`` holds an evaluation made by the current code."""
    summary = Path(path) / "summary.json"
    return summary.exists() and read_json(summary).get("code") == code_version()


def evaluate_saved_run(dataset: str, split_name: str, method: str, path: Path, inputs: Inputs, secondary: bool = False, force: bool = False) -> str:
    """Evaluate a finished run and write its evaluation files; 'skipped' if they are current and ``force`` is off."""
    if not force and is_current(path):
        return "skipped"
    split = inputs.splits[split_name]
    write_evaluation(path, evaluate_run(path, method, split, inputs.episodes, inputs.window, secondary), dataset, split, method, inputs.episodes, inputs.window)
    return "done"


def rates_at(tau: float | Callable[[int], np.ndarray], scores: Mapping[str, np.ndarray], episodes: list[Episode], window: int) -> dict:
    """Rates of the threshold ``tau`` on ``episodes`` (alarm where score >= tau): a constant, or a function giving the threshold of each step of an episode
    of a given length (a band that varies with the step)."""
    at = tau if callable(tau) else lambda n: np.full(n, tau)
    outcomes = [classify(scores[e.key], at(len(scores[e.key])), e.onset, window) for e in episodes]
    n_fail = sum(not e.success for e in episodes)
    return {**asdict(rates_from_outcomes(outcomes)), "n_fail": n_fail, "n_succ": len(episodes) - n_fail}


def evaluate_ltt(scores: Mapping[str, np.ndarray], split: Split, episodes: Mapping[str, Episode], window: int, *, alpha_tpr_early: float,
                 alpha_fpr: float, delta: float, n_starts: int = 5) -> dict:
    """The Learn-then-Test operating point chosen on the calibration episodes and its rates on each evaluation set.

    The targets have no defaults: they are a decision about what to guarantee. The guarantee is over the draw of the calibration set and
    holds for episodes exchangeable with it; it is not a statement about false alarms on unseen tasks.
    """
    cal = [episodes[k] for k in split.calibration]
    failures = [e for e in cal if not e.success]
    result = calibrate_ltt([scores[e.key] for e in cal if e.success], [scores[e.key] for e in failures], [e.onset for e in failures],
                           alpha_tpr_early=alpha_tpr_early, alpha_fpr=alpha_fpr, delta=delta, n_starts=n_starts)
    out = {"certified": result.certified, "tau": result.tau, "n_accepted": result.n_accepted,
           "targets": {"alpha_tpr_early": alpha_tpr_early, "alpha_fpr": alpha_fpr, "delta": delta, "n_starts": n_starts}}
    if result.certified:
        out["sets"] = {name: rates_at(result.tau, scores, [episodes[k] for k in getattr(split, name)], window) for name in EVAL_SETS if getattr(split, name)}
    return out


def scan_ltt(scores: Mapping[str, np.ndarray], split: Split, episodes: Mapping[str, Episode], window: int, targets: list[tuple[float, float]],
             delta: float) -> list[dict]:
    """``evaluate_ltt`` for each (alpha_tpr_early, alpha_fpr) pair, to see which targets can be certified with this calibration set and
    what the chosen threshold does on the other episodes. For choosing targets; it decides nothing."""
    return [evaluate_ltt(scores, split, episodes, window, alpha_tpr_early=a, alpha_fpr=b, delta=delta) for a, b in targets]
