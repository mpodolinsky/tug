"""The configurations of the embedding-distance baselines that SAFE sweeps (scripts/batch_training/submit_pi0fast_libero.bash).

SAFE runs, with ``use_success_only=False`` throughout:
    euclid       topk in {1, 5, 10}  x  cumsum in {False, True}
    mahala       cumsum in {False, True}
    pca_kmeans   pca_dim in {32, 64, 128}  x  n_clusters in {16, 32, 64}  x  cumsum in {False, True}
It also sweeps the feature (encoded / pre_logits and the token index); here the feature is fixed (per-modality pooled features), so
that axis is not part of the grid. 26 configurations in all.
"""

import itertools
from typing import Mapping

import numpy as np

from ..data.manifest import Episode
from ..data.splits import Split
from .distance import DISTANCES, DistanceScorer
from .registry import Context, register

GRID: dict[str, dict[str, tuple]] = {
    "mahala": dict(cumsum=(False, True)),
    "euclid": dict(topk=(1, 5, 10), cumsum=(False, True)),
    "pca_kmeans": dict(pca_dim=(32, 64, 128), n_clusters=(16, 32, 64), cumsum=(False, True)),
}


def grid_configs(distance: str) -> list[dict]:
    """Every keyword combination SAFE sweeps for ``distance``, as dicts for ``DistanceScorer``."""
    axes = GRID[distance]
    return [dict(zip(axes, values)) for values in itertools.product(*axes.values())]


def grid_scorers(distance: str, episodes: Mapping[str, Episode], features: Mapping[str, np.ndarray], seed: int = 0,
                 device: str = "cpu") -> list[DistanceScorer]:
    """One unfitted scorer per configuration, in grid order (the order the best-configuration choice breaks ties in); each has a
    unique ``name`` like 'euclidean__k5_cumsum'. Unfitted scorers only point at the shared features; a fitted one holds its training
    steps, so fit and release them one at a time."""
    return [DistanceScorer(distance, episodes, features, seed=seed, device=device, **cfg) for cfg in grid_configs(distance)]


def all_grid_scorers(episodes, features, seed: int = 0) -> list[DistanceScorer]:
    return [s for d in DISTANCES for s in grid_scorers(d, episodes, features, seed)]


def method_names(family: str) -> list[str]:
    """The methods of a distance family ('euclidean', 'mahalanobis', 'pca_kmeans') in grid order, e.g. euclidean__k1_step ... euclidean__k10_cumsum."""
    distance = {name: key for key, name in DistanceScorer.NAMES.items()}[family]
    return [s.name for s in grid_scorers(distance, {}, {})]


def _family(distance: str):
    def build(context: Context, split: Split) -> list[DistanceScorer]:
        return grid_scorers(distance, context.inputs.episodes, context.features, seed=split.seed, device=context.device)

    return build


for _distance, _name in DistanceScorer.NAMES.items():  # the families "mahalanobis", "euclidean" and "pca_kmeans"
    register(_name)(_family(_distance))
