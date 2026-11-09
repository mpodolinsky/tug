import math

import pytest

from fp.metrics.outcomes import Outcome as O
from fp.metrics.rates import rates_from_outcomes


def test_known_mix():
    out = [O.EARLY, O.EARLY, O.ON_TIME, O.LATE, O.MISSED, O.MISSED, O.MISSED, O.MISSED, O.MISSED, O.MISSED,
           O.FALSE_ALARM, O.CORRECT_REJECT, O.CORRECT_REJECT, O.CORRECT_REJECT]
    r = rates_from_outcomes(out)
    assert r.tpr == pytest.approx(40.0)  # 4 of 10 failures flagged
    assert r.tpr_early == pytest.approx(30.0)  # early + on-time
    assert r.tpr_on_time == pytest.approx(10.0)
    assert r.ldr == pytest.approx(100 * 1 / 3)  # late / (early + late)
    assert r.fpr == pytest.approx(25.0)


def test_tpr_early_counts_on_time_but_not_late():
    r = rates_from_outcomes([O.ON_TIME, O.LATE])
    assert r.tpr_early == pytest.approx(50.0) and r.tpr == pytest.approx(100.0)


def test_ldr_is_nan_when_everything_is_on_time_or_missed():
    assert math.isnan(rates_from_outcomes([O.ON_TIME, O.MISSED]).ldr)


def test_empty_groups_give_nan_not_zero():
    only_failures = rates_from_outcomes([O.EARLY])
    assert math.isnan(only_failures.fpr)
    only_successes = rates_from_outcomes([O.CORRECT_REJECT])
    assert math.isnan(only_successes.tpr) and math.isnan(only_successes.tpr_early)
