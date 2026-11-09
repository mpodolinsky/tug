import numpy as np
import pytest

from fp.calibration.max_conformal import MaxConformal


def cal(maxima):
    return MaxConformal().fit([np.array([0.0, m]) for m in maxima])


def test_threshold_is_the_kth_smallest_maximum():
    m = cal([0.9, 0.1, 0.5, 0.3, 0.7, 0.2, 0.6, 0.4, 0.8])  # n = 9, unsorted on purpose
    # alpha = 0.2 -> k = ceil(10 * 0.8) = 8 -> the 8th smallest maximum
    assert m.rank(0.2) == 8
    assert m.threshold(0.2, 5)[0] == pytest.approx(0.8)
    # alpha = 0.5 -> k = ceil(10 * 0.5) = 5
    assert m.threshold(0.5, 5)[0] == pytest.approx(0.5)


def test_threshold_is_an_observed_value_not_an_interpolation():
    m = cal([0.1, 0.2, 0.4, 0.8])
    for alpha in (0.1, 0.3, 0.45, 0.6, 0.9):
        assert m.threshold(alpha, 1)[0] in (0.1, 0.2, 0.4, 0.8, np.inf)


def test_no_finite_threshold_below_one_over_n_plus_one():
    m = cal([0.1, 0.5, 0.9])  # n = 3: alpha < 1/4 has k > n
    assert m.rank(0.2) == 4 and np.isinf(m.threshold(0.2, 3)[0])
    assert m.threshold(0.26, 3)[0] == pytest.approx(0.9)  # k = ceil(4 * 0.74) = 3, the largest maximum


def test_threshold_is_constant_and_has_requested_length():
    t = cal([0.2, 0.4, 0.6, 0.8]).threshold(0.3, 7)
    assert t.shape == (7,) and np.all(t == t[0])


def test_larger_alpha_gives_a_threshold_that_is_not_higher():
    m = cal(np.linspace(0.1, 0.9, 20))
    ts = [m.threshold(a, 1)[0] for a in (0.05, 0.1, 0.2, 0.4, 0.8)]
    assert ts == sorted(ts, reverse=True)


def test_uses_the_per_episode_maximum_not_the_final_score():
    m = MaxConformal().fit([np.array([0.9, 0.1]), np.array([0.8, 0.0])])
    assert m.threshold(0.4, 1)[0] == pytest.approx(0.9)  # n = 2, k = ceil(3 * 0.6) = 2: the larger maximum (0.9, not 0.1 or 0.0)


def test_false_alarm_rate_is_at_most_alpha_on_average_over_calibration_draws():
    """The guarantee is marginal over the calibration sample: averaged over many calibration sets of n = 50 the
    false-alarm rate on fresh exchangeable episodes is <= alpha, and not far below it (the bound is nearly tight)."""
    rng = np.random.default_rng(0)
    max_of_30 = lambda size: rng.random((size, 30)).max(axis=1)  # per-episode maximum of a 30-step i.i.d. score trajectory
    for alpha in (0.05, 0.1, 0.2, 0.4):
        rates = []
        for _ in range(400):
            m = MaxConformal().fit([np.array([v]) for v in max_of_30(50)])
            rates.append(np.mean(max_of_30(500) >= m.threshold(alpha, 1)[0]))
        assert alpha - 0.02 <= np.mean(rates) <= alpha + 0.005, (alpha, np.mean(rates))


def test_threshold_never_increases_with_alpha_so_the_false_alarm_rate_never_decreases():
    """What makes the alpha sweep (and its stop at FPR >= 40) safe for max-conformal: a lower threshold alarms on a superset."""
    rng = np.random.default_rng(3)
    m = cal(rng.random(37))
    grid = np.round(np.arange(0.005, 1.0, 0.005), 3)
    thr = np.array([m.threshold(a, 1)[0] for a in grid])
    assert np.all(thr[1:] <= thr[:-1])  # compared pairwise: np.diff of two infinities is NaN
    new_success_maxima = rng.random(500)
    fpr = np.array([np.mean(new_success_maxima >= t) for t in thr])
    assert np.all(np.diff(fpr) >= 0)


def test_rank_uses_the_exact_integer_not_the_floating_point_ceil():
    m = cal(np.arange(149) / 149.0)  # n = 149, alpha = 0.18: 150 * 0.82 = 123 exactly
    assert m.rank(0.18) == 123


def test_tied_maxima_at_the_threshold_break_the_guarantee_with_the_greater_or_equal_rule():
    """Documented limitation: if every calibration maximum saturates at the same value, the threshold is that value and
    every new success that saturates alarms, whatever alpha is. Not seen on the saved scores (all maxima distinct)."""
    m = cal([1.0] * 30)
    thr = m.threshold(0.1, 1)[0]
    assert thr == 1.0
    new_success_maxima = np.ones(100)  # exchangeable with the calibration episodes: they saturate too
    assert np.mean(new_success_maxima >= thr) == 1.0 > 0.1


def test_nan_or_infinite_scores_are_rejected_instead_of_silently_disabling_alarms():
    with pytest.raises(ValueError):
        MaxConformal().fit([np.array([0.1, np.nan, 0.3]), np.array([0.2, 0.4])])
    with pytest.raises(ValueError):
        MaxConformal().fit([np.array([0.1, np.inf])])


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        MaxConformal().fit([])
    with pytest.raises(ValueError):
        cal([0.1, 0.2]).threshold(1.0, 3)
