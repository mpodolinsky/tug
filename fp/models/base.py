"""Common interface for everything that turns an episode into a per-step failure score.

A ``Scorer`` is anything that, after fitting on training episodes, returns for every episode a
1-D array of per-step scores (higher = more likely to fail). Calibration and evaluation only
ever see these arrays, so learned probes, distance baselines and STAC are interchangeable.
"""

from abc import ABC, abstractmethod

import numpy as np


class Scorer(ABC):
    name: str
    saves_weights = False  # True for a scorer with learned weights: the baseline driver then writes ``save()`` into the run
    input_kind = "raw"  # what the scorer reads, recorded in the history as ``features``: the raw pooled features unless it says otherwise

    @abstractmethod
    def fit(self, train_keys: list[str], val_keys: list[str]) -> None:
        """Fit on training episodes; ``val_keys`` is used only for early stopping."""

    @abstractmethod
    def score(self, keys: list[str]) -> dict[str, np.ndarray]:
        """Per-step scores for every key, over the full rollout (no cut)."""

    def training_details(self) -> dict | None:
        """How a trained scorer was fitted (learning curves, best epoch, stopping rule, hyperparameters), or None if it was not trained by
        iteration. Merged into the history of its run, so every deviation from a reference recipe is recorded with the numbers."""
        return None

    def score_with_extras(self, keys: list[str]) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
        """The scores and, optionally, arrays files to store with the run (file name -> per-episode arrays), such as the distances a score
        is made of. The default has no extras."""
        return self.score(keys), {}

    def state_bytes(self) -> int | None:
        """Bytes of everything a fitted scorer must keep to score a new episode (learned weights, stored training steps, statistics), or
        None if it does not say. Recorded in the history of its run, to compare what different methods need at inference."""
        return None

    def save(self, path: str) -> None:
        """Write learned weights/statistics. Required for every method that has any."""
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError
