"""LTT against max-conformal at the same false-positive-rate targets (the table of ``cli.ltt_table``), from the saved scores of one probe.

For each FPR target f and each run (a split), two constant thresholds (and the functional conformal band) are chosen on the run's calibration episodes:

    max-conformal   level alpha = f, from the calibration successes alone. It promises an FPR level and nothing about TPR, so it has no TPR-early
                    target, and it always returns a threshold (+inf, which never alarms, when alpha is below 1 / (n + 1)).
    functional      the functional conformal band (FC, ``calibration.safe_band``, computed as SAFE does) at level alpha = f from the same calibration successes:
                    a threshold that varies with the step, with no TPR-early target either.
    LTT             certifies FPR <= f together with a TPR-early target t, with ``n_starts`` fixed-sequence walks at level delta. t is the highest
                    target on a grid that at least ``MIN_SHARE`` of the runs certify, so that the row describes a typical run; a run that cannot
                    certify it has no threshold (LTT abstains).

Each threshold is measured on the run's held-out seen episodes and on its unseen episodes (``run_eval.rates_at``): the share of runs in which each
target holds, and the mean and standard deviation over the runs of the FPR and of the TPR early. LTT's shares are over the runs it certifies,
max-conformal's over all runs. The LTT guarantee is for episodes exchangeable with the calibration set, i.e. the seen episodes.

The two guaranteed targets also bound the probability of failure given an alarm (``metrics.ppv``): PPV >= beta pi / (beta pi + alpha (1 - pi)) with beta the
TPR-early target, alpha the FPR target and pi the failure prevalence. Each LTT row reports that bound at the calibration set's prevalence; each evaluated set
reports the realised PPV (the failed rollouts among those with an alarm, late alarms included) and the share of runs in which it is at least the bound, computed
with the calibration set's prevalence and with the evaluated set's own. Max-conformal has no TPR-early target, so it has no bound (its PPV is reported).
"""

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..calibration.ltt import RiskCounts, fixed_sequence_test, joint_p_values, risk_counts, select_threshold
from ..calibration.max_conformal import MaxConformal
from ..calibration.safe_band import SafeBand
from ..data.manifest import Episode
from ..data.splits import Split
from ..metrics.ppv import ppv_bound, ppv_from_rates
from .run_eval import rates_at
from .tables import LttCells, LttRow

N_STARTS = 40
MIN_SHARE = 0.5
TPR_TARGETS = tuple(round(0.10 + 0.05 * i, 2) for i in range(18))  # the grid the TPR-early target is taken from: 10% ... 95%
FPR_TARGETS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
LTT, MAX_CONFORMAL, FUNCTIONAL_CONFORMAL = "LTT", "Max conformal (MC)", "Functional conformal (FC)"  # the codes of the main table
KIND_NAMES = {"task": "task seeds", "signature": "signature seeds", "skill": "skill folds"}  # what the splits of each group (``select_config.group_of``) are


def kinds_text(kinds: Mapping[str, int]) -> str:
    """What the runs are, for the caption: the number of splits of each group, the LIBERO-X ones first and by their names ("3 task seeds, 3 signature seeds, 8 skill
    folds"), any other group as "<n> <group> splits"."""
    order = lambda group: list(KIND_NAMES).index(group) if group in KIND_NAMES else len(KIND_NAMES)
    return ", ".join(f"{kinds[g]} {KIND_NAMES.get(g, g + ' splits')}" for g in sorted(kinds, key=order))


@dataclass(frozen=True)
class Run:
    """One run prepared for both calibrators: what both fit on (the calibration episodes) and the episodes they are measured on."""
    scores: Mapping[str, np.ndarray]
    counts: RiskCounts
    max_conformal: MaxConformal
    safe_band: SafeBand
    seen: list[Episode]
    unseen: list[Episode]
    window: int

    @property
    def prevalence(self) -> float:
        """The failure prevalence of the calibration set (a fraction)."""
        return self.counts.n_fail / (self.counts.n_fail + self.counts.n_succ)


def prepare(scores: Mapping[str, np.ndarray], split: Split, episodes: Mapping[str, Episode], window: int) -> Run:
    cal = [episodes[k] for k in split.calibration]
    successes, failures = [e for e in cal if e.success], [e for e in cal if not e.success]
    success_scores = [scores[e.key] for e in successes]
    counts = risk_counts(success_scores, [scores[e.key] for e in failures], [e.onset for e in failures])
    return Run(scores, counts, MaxConformal().fit(success_scores), SafeBand(seed=split.seed).fit(success_scores),
               [episodes[k] for k in split.seen], [episodes[k] for k in split.unseen], window)


def ltt_threshold(run: Run, tpr_target: float, fpr_target: float, delta: float, n_starts: int) -> float | None:
    """The LTT threshold certifying TPR early >= ``tpr_target`` and FPR <= ``fpr_target`` (fractions), None if the run cannot certify it. The
    same steps as ``calibrate_ltt``, on the counts of the run, which do not depend on the targets."""
    p = joint_p_values(run.counts, 1 - tpr_target, fpr_target)
    return select_threshold(fixed_sequence_test(p, delta, n_starts), run.counts).tau


def strongest_tpr_target(runs: list[Run], fpr_target: float, delta: float, n_starts: int, tpr_targets=TPR_TARGETS, min_share: float = MIN_SHARE) -> float | None:
    """The highest TPR-early target on ``tpr_targets`` that at least ``min_share`` of the runs certify together with ``fpr_target``; None if there is none."""
    needed = math.ceil(min_share * len(runs))
    certified = [t for t in tpr_targets if sum(ltt_threshold(r, t, fpr_target, delta, n_starts) is not None for r in runs) >= needed]
    return max(certified, default=None)


def set_cells(thresholds: list[tuple[Run, float]], fpr_target: float, tpr_target: float | None, eval_set: str) -> LttCells | None:
    """The cells of one evaluation set, over the runs that have a threshold and episodes in the set (None when there are none). The PPV is that of
    the set at the run's threshold, over the runs that have an alarm in it; its bound (``metrics.ppv``) is computed from the two targets and the failure
    prevalence of the calibration set, and of the evaluated set itself, and exists only with a TPR-early target."""
    measured = [(run, rates_at(tau, run.scores, getattr(run, eval_set), run.window)) for run, tau in thresholds if getattr(run, eval_set)]
    if not measured:
        return None
    rates = [r for _, r in measured]
    fpr, tpr = np.array([r["fpr"] for r in rates]), np.array([r["tpr_early"] for r in rates])
    own = np.array([r["n_fail"] / (r["n_fail"] + r["n_succ"]) for r in rates])
    ppv = np.array([ppv_from_rates(r["tpr"], r["fpr"], r["n_fail"], r["n_succ"]) for r in rates])
    alarmed = ~np.isnan(ppv)
    holds_calibration = holds_own = None
    if tpr_target is not None and alarmed.any():
        at_calibration = np.array([ppv_bound(tpr_target, fpr_target, run.prevalence) for run, _ in measured])
        at_own = np.array([ppv_bound(tpr_target, fpr_target, p) for p in own])
        holds_calibration = 100 * float(np.mean(ppv[alarmed] / 100 >= at_calibration[alarmed] - 1e-9))
        holds_own = 100 * float(np.mean(ppv[alarmed] / 100 >= at_own[alarmed] - 1e-9))
    return LttCells(100 * float(np.mean(fpr <= 100 * fpr_target)), None if tpr_target is None else 100 * float(np.mean(tpr >= 100 * tpr_target)),
                    (float(fpr.mean()), float(fpr.std())), (float(tpr.mean()), float(tpr.std())),
                    float(ppv[alarmed].mean()) if alarmed.any() else None, holds_calibration, holds_own, float(own.mean()))


def compare(runs: list[Run], fpr_targets=FPR_TARGETS, *, delta: float, n_starts: int = N_STARTS, tpr_targets=TPR_TARGETS,
            min_share: float = MIN_SHARE) -> list[LttRow]:
    """Three rows per FPR target: LTT, max conformal (MC) and functional conformal (FC). When no TPR-early target is certified by enough runs the LTT row has no target and no cells."""
    rows = []
    for f in fpr_targets:
        t = strongest_tpr_target(runs, f, delta, n_starts, tpr_targets, min_share)
        ltt = [] if t is None else [(r, tau) for r in runs if (tau := ltt_threshold(r, t, f, delta, n_starts)) is not None]
        bound = 100 * float(np.mean([ppv_bound(t, f, run.prevalence) for run, _ in ltt])) if ltt else None  # the PPV bound at the calibration prevalence, mean over the runs
        rows.append(LttRow(f, t, LTT, True, len(ltt), len(runs), *(set_cells(ltt, f, t, s) if ltt else None for s in ("seen", "unseen")), ppv_bound=bound))
        conformal = [(r, float(r.max_conformal.threshold(f, 1)[0])) for r in runs]
        rows.append(LttRow(f, None, MAX_CONFORMAL, False, len(runs), len(runs), *(set_cells(conformal, f, None, s) for s in ("seen", "unseen"))))
        band = [(r, functional_threshold(r, f)) for r in runs]
        rows.append(LttRow(f, None, FUNCTIONAL_CONFORMAL, False, len(runs), len(runs), *(set_cells(band, f, None, s) for s in ("seen", "unseen"))))
    return rows


def functional_threshold(run: Run, fpr_target: float):
    """The functional conformal band of the run at level ``fpr_target``, as a function of the episode length (``rates_at``): computed once at the longest rollout of the run."""
    band = run.safe_band.threshold(fpr_target, max(len(s) for s in run.scores.values()))
    return lambda n: band[:n]
