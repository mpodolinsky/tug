"""Max-conformal threshold: split-conformal order statistic of the per-episode maximum score of calibration successes.

With n calibration successes and maxima m_(1) <= ... <= m_(n), the threshold for level alpha is
m_(k), k = ceil((n + 1) * (1 - alpha)). A new success then raises an alarm (score >= threshold) with
probability at most alpha when it is exchangeable with the calibration episodes and the maxima have no ties
(with ties at the threshold the ">=" alarm rule can exceed alpha; see docs/CALIBRATION.md).
If k > n, i.e. alpha < 1 / (n + 1), no finite threshold has that guarantee and the threshold is +inf:
the detector never alarms. One constant threshold for all steps.
"""

import numpy as np

from .base import Calibrator
from .ranks import conformal_rank


class MaxConformal(Calibrator):
    name = "max_conformal"

    def fit(self, success_scores: list[np.ndarray]) -> "MaxConformal":
        if not success_scores:
            raise ValueError("max-conformal needs at least one calibration success")
        maxima = np.array([np.max(s) for s in success_scores], dtype=float)
        if not np.isfinite(maxima).all():
            raise ValueError("a calibration success has a NaN or infinite score; a NaN maximum would silently disable the alarm")
        self.maxima = np.sort(maxima)
        return self

    def rank(self, alpha: float) -> int:
        """k = ceil((n + 1)(1 - alpha)); the threshold is the k-th smallest maximum (1-indexed)."""
        return conformal_rank(len(self.maxima), alpha)

    def threshold(self, alpha: float, length: int) -> np.ndarray:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        k = self.rank(alpha)
        value = self.maxima[k - 1] if k <= len(self.maxima) else np.inf
        return np.full(length, value)
