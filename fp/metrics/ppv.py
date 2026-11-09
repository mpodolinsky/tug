"""Positive predictive value of an alarm threshold: P(failure | the score exceeds the threshold at some step), per rollout.

With the failure prevalence pi, the rollout-level true positive rate (an alarm at any step of a failed rollout) and the false positive rate (an alarm at any step of a
successful one), Bayes' rule gives  PPV = TPR pi / (TPR pi + FPR (1 - pi)). An alarm before the failure onset is one of the alarms, so TPR >= TPR early, and PPV
rises with TPR and falls with FPR: from the guarantees TPR early >= beta and FPR <= alpha (Learn-then-Test, ``calibration.ltt``)

    PPV >= beta pi / (beta pi + alpha (1 - pi)),

for every pi, since both rates are conditional on the outcome. With no TPR-early guarantee (beta = 0) the bound is 0.
"""

import math


def ppv_bound(beta: float, alpha: float, prevalence: float) -> float:
    """The lower bound on the PPV from TPR early >= ``beta`` and FPR <= ``alpha`` (fractions in [0, 1]) at the failure prevalence ``prevalence``."""
    if not (0 <= beta <= 1 and 0 <= alpha <= 1 and 0 <= prevalence <= 1):
        raise ValueError("beta, alpha and the prevalence are fractions in [0, 1]")
    detected, false_alarms = beta * prevalence, alpha * (1 - prevalence)
    return detected / (detected + false_alarms) if detected + false_alarms > 0 else math.nan


def ppv_from_rates(tpr: float, fpr: float, n_fail: int, n_succ: int) -> float:
    """The realised PPV of a set, in percent: failed rollouts with an alarm over all rollouts with an alarm. ``tpr`` and ``fpr`` are percentages (``rates_at``).
    NaN when no rollout has an alarm."""
    detected, false_alarms = tpr / 100 * n_fail, fpr / 100 * n_succ
    return 100 * detected / (detected + false_alarms) if detected + false_alarms > 0 else math.nan
