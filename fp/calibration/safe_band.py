"""SAFE functional conformal band (SAFE: Gu et al., NeurIPS 2025, arXiv 2506.09937), one-sided upper band with Tfunc
modulation and mean regression. The band is the one of "The Importance of Being a Band: Finite-Sample Exact
Distribution-Free Prediction Sets for Functional Data" (arXiv 2102.06746), as cited in SAFE's source.

Calibration successes are padded to a common length (edge padding), shuffled with a seeded RNG
and split 30/70: the first part (``fit``) gives the mean trajectory and the modulation function,
the second (``cal``) gives the conformal quantile. The few lines of SAFE's FunctionalPredictor that
this uses are ported here unchanged in behaviour (baselines/SAFE/failure_prob/utils/conformal/
functional_predictor.py), so the module has no dependency on the SAFE checkout.

Quantile rule (``finite_sample``): SAFE's one-sided band takes a plain ``np.quantile(scores, 1 - alpha)`` (the default,
``finite_sample=False``, which reproduces SAFE). With ``finite_sample=True`` the width is the order statistic
r_(ceil((m + 1)(1 - alpha))) of the m conformal scores, which makes the false-alarm bound alpha valid (+inf, i.e. never
alarm, if that rank exceeds m).

An alarm fires where score >= band. For episodes longer than the calibration data the band is
extended with its last value: beyond the longest fit episode mean and modulation are constant, so this is exactly the
band that padding the calibration data further would give.
"""

import numpy as np

from .base import Calibrator
from .ranks import conformal_rank

EPS = 1e-8


def tfunc_modulation(train: np.ndarray, mean_traj: np.ndarray, alpha: float) -> np.ndarray:
    """SAFE's Tfunc modulation: pointwise max deviation over the (1 - alpha) best-behaved training trajectories."""
    n = train.shape[0]
    dev = np.abs(train - mean_traj)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return dev.max(axis=0, keepdims=True) + EPS
    gamma = np.sort(dev.max(axis=1))[k - 1]
    return dev[dev.max(axis=1) <= gamma].max(axis=0, keepdims=True) + EPS


def upper_band(fit: np.ndarray, cal: np.ndarray, alpha: float, finite_sample: bool = False) -> np.ndarray:
    """(length,) upper band from the 30% part (``fit``) and the 70% part (``cal``)."""
    mean_traj = fit.mean(axis=0, keepdims=True)
    mod = tfunc_modulation(fit, mean_traj, alpha)
    scores = np.array([np.max((c - mean_traj) / mod) for c in cal])
    if finite_sample:
        k = conformal_rank(len(scores), alpha)
        width = np.sort(scores)[k - 1] if k <= len(scores) else np.inf
    else:
        width = np.quantile(scores, 1.0 - alpha)
    return (mean_traj + width * mod)[0]


class SafeBand(Calibrator):
    name = "safe_band"

    def __init__(self, seed: int = 0, fit_fraction: float = 0.3, finite_sample: bool = False):
        self.seed = seed
        self.fit_fraction = fit_fraction
        self.finite_sample = finite_sample

    def fit(self, success_scores: list[np.ndarray]) -> "SafeBand":
        length = max(len(s) for s in success_scores)
        padded = np.array([np.pad(s, (0, length - len(s)), mode="edge") for s in success_scores], dtype=float)
        np.random.RandomState(self.seed).shuffle(padded)
        n1 = int(len(padded) * self.fit_fraction)
        if n1 < 2 or len(padded) - n1 < 1:
            raise ValueError(f"{len(padded)} calibration successes are too few for a {self.fit_fraction:.0%} / {1 - self.fit_fraction:.0%} split")
        self.fit_part, self.cal_part = padded[:n1], padded[n1:]
        return self

    def threshold(self, alpha: float, length: int) -> np.ndarray:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        band = upper_band(self.fit_part, self.cal_part, alpha, self.finite_sample)
        if length <= len(band):
            return band[:length]
        return np.pad(band, (0, length - len(band)), mode="edge")
