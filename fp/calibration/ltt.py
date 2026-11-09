"""Learn-then-Test threshold (ours), reported as a separate evaluation.

A constant threshold tau is picked on a fixed sigmoid grid. Two risks are controlled together:

    miss_early(tau)  = a calibration failure is NOT flagged at or before its onset   target: TPR_early >= 1 - alpha_tpr_early
    false_alarm(tau) = a calibration success is flagged                              target: FPR <= alpha_fpr

Each risk gets a binomial p-value for H0: risk > alpha; the joint p-value is the larger of the two (an
intersection-union test of "either risk is violated"). A multi-start fixed-sequence test (``n_starts`` walks along
the ascending grid, level delta / n_starts) decides which taus are accepted: with probability >= 1 - delta over the
calibration sample, every accepted tau meets both targets. Among accepted taus the one with the highest calibration
TPR_early is returned (ties: lowest calibration FPR, then the lowest tau). If nothing is accepted the run is not certified.
Structure ported from the previous LTT script; the risks differ (the previous one tested an on-time risk
and only minimised FPR).

The scores must be probabilities in (0, 1): the grid is sigmoid(z), z in [-9, 9].

Unlike the other calibrators this one needs failure episodes (with onsets) as well as successes,
so it does not implement the alpha-based ``Calibrator`` interface.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

# Data-independent grid tau = sigmoid(z), z in [-9, 9], 181 points (ascending).
TAUS = 1.0 / (1.0 + np.exp(-np.linspace(-9.0, 9.0, 181)))


@dataclass(frozen=True)
class LttResult:
    tau: float | None  # None when no threshold was accepted (not certified)
    n_accepted: int

    @property
    def certified(self) -> bool:
        return self.tau is not None


@dataclass(frozen=True)
class RiskCounts:
    """Violations of each risk at every tau of the grid, on the calibration episodes."""

    n_miss_early: np.ndarray  # (n_taus,) failures NOT flagged at or before their onset
    n_false_alarm: np.ndarray  # (n_taus,) successes flagged
    n_fail: int
    n_succ: int

    @property
    def tpr_early(self) -> np.ndarray:
        return 1.0 - self.n_miss_early / self.n_fail

    @property
    def fpr(self) -> np.ndarray:
        return self.n_false_alarm / self.n_succ


def first_crossing(scores: np.ndarray, taus: np.ndarray = TAUS) -> np.ndarray:
    """Step of the first score >= tau for every tau (``len(scores)`` when it never crosses)."""
    return np.searchsorted(np.maximum.accumulate(scores), taus, side="left")


def risk_counts(success_scores: list[np.ndarray], failure_scores: list[np.ndarray], onsets: list[int]) -> RiskCounts:
    """Count, at every tau, the failures not caught at or before the onset and the successes with an alarm."""
    n_fail, n_succ = len(failure_scores), len(success_scores)
    if n_fail == 0 or n_succ == 0:
        raise ValueError("LTT needs calibration failures and successes")
    if len(onsets) != n_fail:
        raise ValueError("one onset per failure")
    cross_f = np.stack([first_crossing(s) for s in failure_scores])  # (n_fail, n_taus)
    early = cross_f <= np.array(onsets)[:, None]  # a crossing is < length and the onset is < length, so this implies flagged
    cross_s = np.stack([first_crossing(s) for s in success_scores])
    false_alarm = cross_s < np.array([len(s) for s in success_scores])[:, None]
    return RiskCounts((~early).sum(axis=0), false_alarm.sum(axis=0), n_fail, n_succ)


def joint_p_values(counts: RiskCounts, alpha_tpr_early: float, alpha_fpr: float) -> np.ndarray:
    """p-value of 'tau violates a target': max of the two binomial tails P[Bin(n, alpha) <= observed violations]."""
    p_early = binom.cdf(counts.n_miss_early, counts.n_fail, alpha_tpr_early)
    p_fpr = binom.cdf(counts.n_false_alarm, counts.n_succ, alpha_fpr)
    return np.maximum(p_early, p_fpr)


def fixed_sequence_test(p: np.ndarray, delta: float, n_starts: int) -> np.ndarray:
    """Boolean mask of accepted hypotheses. From each of ``n_starts`` equally spaced grid points walk
    forward (ascending tau), accepting while p <= delta / n_starts and stopping that walk at its first
    failure. Valid without monotonicity."""
    n = len(p)
    starts = np.unique(np.round(np.linspace(0, n - 1, n_starts)).astype(int))
    level = delta / len(starts)
    accepted = np.zeros(n, dtype=bool)
    for j in starts:
        while j < n and p[j] <= level:
            accepted[j] = True
            j += 1
    return accepted


def select_threshold(accepted: np.ndarray, counts: RiskCounts) -> LttResult:
    """Among accepted taus: highest calibration TPR_early, then lowest FPR, then lowest tau."""
    if not accepted.any():
        return LttResult(None, 0)
    idx = np.flatnonzero(accepted)
    order = np.lexsort((idx, counts.fpr[idx], -counts.tpr_early[idx]))
    return LttResult(float(TAUS[idx[order[0]]]), int(accepted.sum()))


def calibrate_ltt(
    success_scores: list[np.ndarray],
    failure_scores: list[np.ndarray],
    onsets: list[int],
    *,
    alpha_tpr_early: float,
    alpha_fpr: float,
    delta: float = 0.10,
    n_starts: int = 5,
) -> LttResult:
    """Guarantee TPR_early >= 1 - alpha_tpr_early and FPR <= alpha_fpr, jointly with probability >= 1 - delta."""
    counts = risk_counts(success_scores, failure_scores, onsets)
    p = joint_p_values(counts, alpha_tpr_early, alpha_fpr)
    return select_threshold(fixed_sequence_test(p, delta, n_starts), counts)
