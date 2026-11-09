"""Calibration interface.

A calibrator is fitted once on the scores of calibration successes and then, for any alpha,
returns a threshold that can be applied to a score trajectory. Per-step thresholds are
``(T,)`` arrays so that a constant threshold and SAFE's time-varying band share one type.
"""

from abc import ABC, abstractmethod

import numpy as np


class Calibrator(ABC):
    name: str

    @abstractmethod
    def fit(self, success_scores: list[np.ndarray]) -> "Calibrator":
        """success_scores: one per-step score array per calibration success."""

    @abstractmethod
    def threshold(self, alpha: float, length: int) -> np.ndarray:
        """Per-step threshold of shape (length,); an alarm fires where score >= threshold."""
