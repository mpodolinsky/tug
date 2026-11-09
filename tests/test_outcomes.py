import numpy as np
import pytest

from fp.metrics.outcomes import Outcome, alarm_step, classify

W = 5
N = 40
THR = np.full(N, 0.5)


def spike(at):
    s = np.zeros(N)
    s[at] = 1.0
    return s


def test_alarm_step_is_first_crossing():
    s = np.zeros(N)
    s[[10, 20]] = 1.0
    assert alarm_step(s, THR) == 10


def test_threshold_equal_to_score_alarms():
    assert alarm_step(np.full(N, 0.5), THR) == 0


def test_no_alarm_returns_none():
    assert alarm_step(np.zeros(N), THR) is None


def test_short_threshold_is_an_error():
    with pytest.raises(ValueError):
        alarm_step(np.zeros(N), np.zeros(N - 1))


@pytest.mark.parametrize(
    "alarm, expected",
    [
        (19, Outcome.EARLY),  # delay -11 < -5
        (24, Outcome.EARLY),  # delay -6, one step before the window
        (25, Outcome.ON_TIME),  # delay exactly -window
        (28, Outcome.ON_TIME),
        (30, Outcome.ON_TIME),  # delay 0: the onset itself is on time
        (31, Outcome.LATE),
    ],
)
def test_failure_buckets_at_the_window_edges(alarm, expected):
    assert classify(spike(alarm), THR, onset=30, window=W) == expected


def test_failure_without_alarm_is_missed():
    assert classify(np.zeros(N), THR, onset=30, window=W) == Outcome.MISSED


def test_success_alarm_is_false_alarm():
    assert classify(spike(3), THR, onset=None, window=W) == Outcome.FALSE_ALARM
    assert classify(np.zeros(N), THR, onset=None, window=W) == Outcome.CORRECT_REJECT


def test_time_varying_threshold_is_respected():
    thr = np.full(N, 0.5)
    thr[10:] = 2.0  # band rises above the score after step 10
    s = spike(15)
    assert classify(s, thr, onset=30, window=W) == Outcome.MISSED
