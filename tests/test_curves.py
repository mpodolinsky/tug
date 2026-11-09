import numpy as np
import pytest

from fp.calibration.base import Calibrator
from fp.data.manifest import Episode
from fp.metrics.curves import ALPHAS, ALPHAS_FULL, FPR_MAX, area_to_fpr, evaluate_set, sweep


def test_perfect_curve_scores_100():
    a = area_to_fpr(np.array([0.0, 10.0, 40.0]), np.array([100.0, 100.0, 100.0]))
    assert a.value == pytest.approx(100.0) and not a.extended


def test_diagonal_scores_20():
    """y = fpr over 0..40 has area 800; divided by 40 that is 20."""
    x = np.linspace(0, 40, 9)
    assert area_to_fpr(x, x).value == pytest.approx(20.0)


def test_curve_that_stops_short_is_held_flat_and_flagged():
    a = area_to_fpr(np.array([0.0, 20.0]), np.array([0.0, 50.0]))
    # triangle 0..20 (area 500) plus flat 50 over 20..40 (area 1000) -> 1500 / 40
    assert a.value == pytest.approx(37.5) and a.extended


def test_non_monotone_curve_is_made_monotone():
    a = area_to_fpr(np.array([10.0, 20.0, 40.0]), np.array([60.0, 30.0, 70.0]))
    b = area_to_fpr(np.array([10.0, 20.0, 40.0]), np.array([60.0, 60.0, 70.0]))
    assert a.value == pytest.approx(b.value)


def test_points_beyond_the_limit_are_cut_by_interpolation():
    a = area_to_fpr(np.array([0.0, 80.0]), np.array([0.0, 80.0]))  # y = x, only 0..40 counts
    assert a.value == pytest.approx(20.0) and not a.extended


def test_shared_fpr_keeps_the_best_point():
    a = area_to_fpr(np.array([0.0, 0.0, 40.0]), np.array([10.0, 70.0, 70.0]))
    assert a.value == pytest.approx(70.0)


def test_nan_points_are_ignored():
    a = area_to_fpr(np.array([0.0, np.nan, 40.0]), np.array([100.0, 5.0, 100.0]))
    assert a.value == pytest.approx(100.0)


def test_alpha_grid_is_inside_zero_one_and_ascending():
    assert ALPHAS[0] > 0 and ALPHAS[-1] < 1 and np.all(np.diff(ALPHAS) > 0)


def test_the_headline_grid_ends_where_the_plotted_range_ends_and_the_full_grid_is_kept_for_diagnostics():
    assert ALPHAS[0] == 0.005 and ALPHAS[-1] == FPR_MAX / 100 == 0.4 and len(ALPHAS) == 80  # alpha is the promised FPR: nothing above 0.4 promises <= 40%
    assert ALPHAS_FULL[-1] == 0.995 and len(ALPHAS_FULL) == 199 and np.array_equal(ALPHAS_FULL[: len(ALPHAS)], ALPHAS)


class OracleAtLargeAlpha(Calibrator):
    """Never alarms for alpha <= 0.4. For a larger alpha it happens to separate the classes perfectly on the evaluation episodes."""

    name = "oracle"

    def fit(self, success_scores):
        return self

    def threshold(self, alpha, length):
        return np.full(length, np.inf if alpha <= 0.4 + 1e-9 else 0.5)


def test_a_point_from_an_alpha_above_the_plotted_range_is_not_used_however_good_it_looks():
    eps = [Episode(f"s{i}", True, "t", 20, None) for i in range(10)] + [Episode(f"f{i}", False, "t", 20, 12) for i in range(10)]
    sc = {e.key: np.full(20, 0.1 if e.success else 0.9) for e in eps}
    kept = evaluate_set(OracleAtLargeAlpha(), sc, eps, window=5)  # default grid: only alphas that promise FPR <= 40%
    assert kept.area == 0.0 and kept.extended  # no alpha up to 0.4 ever alarms: the curve is the single point (0, 0), held flat
    peeking = evaluate_set(OracleAtLargeAlpha(), sc, eps, window=5, alphas=ALPHAS_FULL)  # the old full grid credits the alpha = 0.5 point at FPR 0
    assert peeking.area == pytest.approx(100.0)


class ConstantByAlpha(Calibrator):
    """Threshold falls linearly with alpha, so FPR rises with alpha on the fixture below."""

    name = "fake"

    def fit(self, success_scores):
        return self

    def threshold(self, alpha, length):
        return np.full(length, 1.0 - alpha)


def episodes_and_scores():
    eps, sc = [], {}
    for i in range(10):  # successes peak at i/10
        eps.append(Episode(f"s{i}", True, "t", 20, None))
        sc[f"s{i}"] = np.full(20, i / 10)
    for i in range(10):  # failures: score jumps at the onset
        eps.append(Episode(f"f{i}", False, "t", 20, 12))
        s = np.zeros(20)
        s[12:] = 0.95
        sc[f"f{i}"] = s
    return eps, sc


def test_sweep_rows_have_the_documented_fields_and_rise_with_alpha():
    eps, sc = episodes_and_scores()
    rows = sweep(ConstantByAlpha(), sc, eps, window=5, stop_fpr=None)
    assert set(rows[0]) == {"alpha", "fpr", "tpr", "tpr_early", "tpr_on_time", "ldr"}
    fprs = [r["fpr"] for r in rows]
    assert fprs == sorted(fprs)


def test_sweep_runs_the_whole_alpha_grid_by_default():
    eps, sc = episodes_and_scores()
    rows = sweep(ConstantByAlpha(), sc, eps, window=5)
    assert len(rows) == len(ALPHAS) and rows[-1]["alpha"] == ALPHAS[-1]


class SteepByAlpha(ConstantByAlpha):
    def threshold(self, alpha, length):
        return np.full(length, 1.0 - 2 * alpha)  # FPR reaches 40% at alpha 0.2, well inside the grid


def test_sweep_stops_at_the_first_row_that_reaches_the_target_fpr():
    eps, sc = episodes_and_scores()
    rows = sweep(SteepByAlpha(), sc, eps, window=5, stop_fpr=40.0)
    assert rows[-1]["fpr"] >= 40.0 and all(r["fpr"] < 40.0 for r in rows[:-1])
    assert len(rows) < len(ALPHAS)
