import math

import pytest

from fp.metrics.ppv import ppv_bound, ppv_from_rates


def test_the_bound_by_hand():
    # beta 0.45, alpha 0.30, prevalence 0.58: 0.45 * 0.58 = 0.261 detected, 0.30 * 0.42 = 0.126 false alarms, 0.261 / 0.387
    assert ppv_bound(0.45, 0.30, 0.58) == pytest.approx(0.261 / 0.387)
    assert ppv_bound(0.5, 0.5, 0.5) == pytest.approx(0.5)  # a coin: as many detections as false alarms at an even prevalence


def test_the_bound_rises_with_beta_and_the_prevalence_and_falls_with_alpha():
    base = ppv_bound(0.45, 0.30, 0.58)
    assert ppv_bound(0.60, 0.30, 0.58) > base and ppv_bound(0.45, 0.20, 0.58) > base and ppv_bound(0.45, 0.30, 0.70) > base
    assert ppv_bound(0.30, 0.30, 0.58) < base and ppv_bound(0.45, 0.40, 0.58) < base and ppv_bound(0.45, 0.30, 0.40) < base


def test_the_edges_no_tpr_early_guarantee_gives_zero_no_false_alarms_gives_one_and_nothing_defined_is_nan():
    assert ppv_bound(0.0, 0.3, 0.5) == 0.0  # max-conformal has no TPR-early floor: the bound says nothing
    assert ppv_bound(0.4, 0.0, 0.5) == 1.0
    assert ppv_bound(0.4, 0.3, 0.0) == 0.0 and ppv_bound(0.4, 0.3, 1.0) == 1.0  # nothing fails: no detection is right; everything fails: every one is
    assert math.isnan(ppv_bound(0.0, 0.0, 0.5))  # no detections and no false alarms promised


def test_the_inputs_are_fractions():
    for args in ((1.2, 0.3, 0.5), (0.4, -0.1, 0.5), (0.4, 0.3, 1.5)):
        with pytest.raises(ValueError):
            ppv_bound(*args)


def test_the_realised_ppv_counts_the_failed_rollouts_with_an_alarm_over_all_rollouts_with_one():
    # 50 failures at TPR 80%: 40 detected; 50 successes at FPR 10%: 5 false alarms; 40 / 45
    assert ppv_from_rates(80.0, 10.0, 50, 50) == pytest.approx(100 * 40 / 45)
    assert ppv_from_rates(100.0, 0.0, 10, 90) == 100.0 and ppv_from_rates(0.0, 100.0, 10, 90) == 0.0
    assert math.isnan(ppv_from_rates(0.0, 0.0, 10, 90))  # no alarm at all: no PPV


def test_the_realised_ppv_equals_the_bound_when_the_rates_sit_on_the_targets_and_every_alarm_is_early():
    # the bound is Bayes' rule at the targets, so rates exactly at beta and alpha reproduce it (n_fail / n gives the prevalence)
    n_fail, n_succ = 58, 42
    assert ppv_from_rates(45.0, 30.0, n_fail, n_succ) == pytest.approx(100 * ppv_bound(0.45, 0.30, n_fail / (n_fail + n_succ)))
