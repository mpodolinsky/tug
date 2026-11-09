import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from fp.calibration.safe_band import SafeBand, tfunc_modulation, upper_band
from fp.metrics.outcomes import alarm_step

CONFORMAL = "baselines/SAFE/failure_prob/utils/conformal"


def find_safe_dir() -> Path:
    """SAFE_CONFORMAL_DIR, else the SAFE checkout in the enclosing repo (the main checkout when run from a worktree)."""
    if os.environ.get("SAFE_CONFORMAL_DIR"):
        return Path(os.environ["SAFE_CONFORMAL_DIR"])
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / CONFORMAL).exists():
            return p / CONFORMAL
        if p.name == "worktrees" and p.parent.name == ".claude":
            return p.parent.parent / CONFORMAL
    return here.parent / "missing"


SAFE_DIR = find_safe_dir()


def successes(n=40, length=30, seed=0):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.random(length) * 0.05) + rng.normal(0, 0.02, length) for _ in range(n)]


def test_same_seed_gives_same_band():
    a = SafeBand(seed=3).fit(successes()).threshold(0.1, 30)
    b = SafeBand(seed=3).fit(successes()).threshold(0.1, 30)
    assert np.array_equal(a, b)


def test_different_seed_changes_the_shuffle():
    a = SafeBand(seed=1).fit(successes()).threshold(0.1, 30)
    b = SafeBand(seed=2).fit(successes()).threshold(0.1, 30)
    assert not np.array_equal(a, b)


def test_band_lies_above_the_mean_trajectory():
    band = SafeBand().fit(successes(200)).threshold(0.1, 30)
    mean = np.mean(successes(200), axis=0)
    assert np.all(band >= mean - 1e-9)


def test_smaller_alpha_gives_a_higher_band():
    m = SafeBand().fit(successes(200))
    assert m.threshold(0.02, 30).mean() > m.threshold(0.3, 30).mean()


def test_shorter_episodes_are_edge_padded_to_the_longest():
    s = successes(40, 30)
    s[0] = s[0][:20]
    m = SafeBand().fit(s)
    assert m.fit_part.shape[1] == 30 == m.cal_part.shape[1]


def test_band_extends_with_last_value_and_truncates():
    m = SafeBand().fit(successes())
    long, short = m.threshold(0.1, 45), m.threshold(0.1, 10)
    assert len(long) == 45 and np.all(long[30:] == long[29])
    assert np.array_equal(short, m.threshold(0.1, 30)[:10])


def test_too_few_successes_raise():
    with pytest.raises(ValueError):
        SafeBand().fit(successes(3))


@pytest.mark.skipif(not SAFE_DIR.exists(), reason="SAFE checkout not available")
@pytest.mark.parametrize("alpha", [0.01, 0.1, 0.3])
def test_matches_safes_own_functional_predictor(alpha):
    """The ported band equals SAFE's FunctionalPredictor.get_one_sided_prediction_band on the same split."""
    pkg = types.ModuleType("safe_conformal_test")
    pkg.__path__ = [str(SAFE_DIR)]
    sys.modules["safe_conformal_test"] = pkg
    mods = {}
    for name in ("abstract_predictor", "functional_predictor"):
        spec = importlib.util.spec_from_file_location(f"safe_conformal_test.{name}", SAFE_DIR / f"{name}.py")
        mods[name] = importlib.util.module_from_spec(spec)
        sys.modules[f"safe_conformal_test.{name}"] = mods[name]
        spec.loader.exec_module(mods[name])
    fp = mods["functional_predictor"]
    m = SafeBand(seed=5).fit(successes(60))
    ref = fp.FunctionalPredictor(fp.ModulationType.Tfunc, fp.RegressionType.Mean).get_one_sided_prediction_band(
        m.fit_part, m.cal_part, alpha, lower_bound=False
    )[0]
    assert np.allclose(m.threshold(alpha, 30), ref)


def test_finite_sample_width_is_the_kth_smallest_conformity_score():
    m = SafeBand(seed=1).fit(successes(40))
    fit, cal = m.fit_part, m.cal_part  # 12 fit trajectories, 28 conformal trajectories
    alpha = 0.2
    mu = fit.mean(axis=0, keepdims=True)
    sigma = tfunc_modulation(fit, mu, alpha)
    r = np.sort([np.max((c - mu) / sigma) for c in cal])
    k = int(np.ceil((len(cal) + 1) * (1 - alpha)))
    expected = (mu + r[k - 1] * sigma)[0]
    assert np.allclose(upper_band(fit, cal, alpha, finite_sample=True), expected)


def test_finite_sample_band_is_infinite_when_the_rank_exceeds_the_conformal_part():
    m = SafeBand(finite_sample=True).fit(successes(20))  # conformal part m = 14: alpha < 1/15 has rank > m
    assert np.isinf(m.threshold(0.05, 30)).all()
    assert np.isfinite(m.threshold(0.2, 30)).all()


def false_alarm_rate(n_cal, alpha, finite_sample, reps=120, n_test=200):
    rates = []
    for seed in range(reps):
        band = SafeBand(seed=seed, finite_sample=finite_sample).fit(successes(n_cal, 30, seed=1000 + seed))
        thr = band.threshold(alpha, 30)
        tests = successes(n_test, 30, seed=5000 + seed)
        rates.append(np.mean([alarm_step(t, thr) is not None for t in tests]))
    return np.mean(rates)


def test_finite_sample_rule_keeps_the_false_alarm_rate_at_most_alpha():
    for alpha in (0.1, 0.2):
        assert false_alarm_rate(26, alpha, finite_sample=True) <= alpha + 0.015


def test_safes_interpolated_rule_exceeds_alpha_by_about_one_minus_two_alpha_over_m_plus_one():
    """n = 26 gives a conformal part of m = 19; at alpha = 0.05 the formula predicts 0.095, i.e. nearly double the nominal level."""
    rate = false_alarm_rate(26, 0.05, finite_sample=False)
    assert 0.07 < rate < 0.115
