"""Gaussian fits of the classes of the saved points (``fp.eval.tsne``): where the successes and the failures (or each failure mode) lie, and how far apart they are.

Classes: ``outcome`` (success, failure) or ``mode`` (success and each failure mode). A class is fitted on the points of one subset (all, seen, seen_held_out, unseen) of an
episode window:

    late   the second half of each episode's colour ramp: a success from its middle to its last step, a failure from half-way to its failure step to the failure step
    all    every step of every episode (a failure's steps after its failure step too)
    post   a failure: only its steps after its failure step; a success: the same second half as ``late`` (it has no steps after a failure)
    pre    a failure: all its steps up to its failure step (``late`` is the second half of these); a success: all its steps

The ellipse of a class on the 2D plot is the one-standard-deviation region of the Gaussian of its points (mean and covariance of the 2D coordinates, which is the projection of
the Gaussian fitted in the full space; it holds 39% of the points of a Gaussian class). A class with fewer than ``MIN_EPISODES`` episodes or ``MIN_POINTS`` points in the subset is not fitted. The distance between two classes is the Mahalanobis
distance between their means under the pooled within-class covariance (a ridge of ``RIDGE`` times its mean variance keeps it invertible), in the full space and in the plane.
"""

from dataclasses import dataclass

import numpy as np

from ..data.targets import MODE_NAMES
from .tsne import progress

WINDOWS = ("late", "all", "post", "pre")
KINDS = ("outcome", "mode")
MIN_EPISODES, MIN_POINTS = 5, 200
RADIUS = 1.0  # the Mahalanobis radius of the drawn ellipse, in standard deviations: it holds 1 - exp(-1/2) = 39% of a two-dimensional Gaussian
RIDGE = 1e-3


@dataclass(frozen=True)
class ClassFit:
    id: int
    name: str
    mean: np.ndarray  # (2,) in the plane, (3,) in the space of the three principal axes
    cov: np.ndarray  # (2, 2) or (3, 3)
    rows: np.ndarray  # the points of the class
    episodes: int


def window_mask(points: dict, window: str) -> np.ndarray:
    """Mask over the points that belong to ``window`` (``late``, ``all``, ``post`` or ``pre``)."""
    if window not in WINDOWS:
        raise ValueError(f"unknown window {window!r}; choose from {WINDOWS}")
    if window == "all":
        return np.ones(len(points["step"]), bool)
    ep = points["episode"]
    failure = ~points["success"][ep]
    after_failure = failure & (points["step"] > points["onset"][ep])
    if window == "pre":
        return ~after_failure
    if window == "post":
        return ((progress(points) >= 0.5) & ~failure) | after_failure
    return (progress(points) >= 0.5) & ~after_failure


def class_ids(points: dict, kind: str) -> np.ndarray:
    """The class of every point: ``outcome``: 0 success, 1 failure; ``mode``: 0 success, 1 + the failure mode of the episode."""
    success = points["success"][points["episode"]]
    if kind == "outcome":
        return np.where(success, 0, 1)
    if kind == "mode":
        return np.where(success, 0, 1 + np.maximum(points["mode"][points["episode"]], 0))
    raise ValueError(f"unknown kind {kind!r}; choose from {KINDS}")


def class_name(kind: str, class_id: int) -> str:
    if class_id == 0:
        return "success"
    return "failure" if kind == "outcome" else MODE_NAMES[class_id - 1]


def mode_class_id(mode: str) -> int:
    """The class id of a failure mode in the classes of kind ``mode`` (0 is the successes)."""
    return 1 + MODE_NAMES.index(mode)


def n_classes(kind: str) -> int:
    return 2 if kind == "outcome" else 1 + len(MODE_NAMES)


def fit_classes(points: dict, kind: str, subset: np.ndarray, window: str, columns: str = "xy") -> list[ClassFit]:
    """The Gaussians of the classes of ``kind`` among the points of ``subset`` (a mask over points) in ``window``, without the classes that are too small: in the plane
    (``columns`` xy) or in the space of the first three principal axes (xyz, PCA points only)."""
    keep = subset & window_mask(points, window)
    ids = class_ids(points, kind)
    fits = []
    for i in np.unique(ids[keep]):
        rows = np.flatnonzero(keep & (ids == i))
        episodes = len(np.unique(points["episode"][rows]))
        if len(rows) < MIN_POINTS or episodes < MIN_EPISODES:
            continue
        coordinates = points[columns][rows].astype(np.float64)
        fits.append(ClassFit(int(i), class_name(kind, int(i)), coordinates.mean(axis=0), np.cov(coordinates.T), rows, episodes))
    return fits


def ellipse(mean: np.ndarray, cov: np.ndarray, n: int = 120) -> np.ndarray:
    """(n, 2) points of the ellipse of Mahalanobis radius ``RADIUS`` of a two-dimensional Gaussian."""
    values, vectors = np.linalg.eigh(cov)
    angle = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.cos(angle), np.sin(angle)])
    return (mean[:, None] + vectors @ (RADIUS * np.sqrt(np.maximum(values, 0))[:, None] * circle)).T


def ellipsoid(mean: np.ndarray, cov: np.ndarray, n: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The x, y and z grids (n, n) of the surface of the ellipsoid of Mahalanobis radius ``RADIUS`` of a three-dimensional Gaussian."""
    values, vectors = np.linalg.eigh(cov)
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n))
    sphere = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)])
    surface = mean[:, None, None] + np.einsum("ij,jkl->ikl", vectors, RADIUS * np.sqrt(np.maximum(values, 0))[:, None, None] * sphere)
    return surface[0], surface[1], surface[2]


def typical_episode(points: dict, subset: np.ndarray, kind: str, class_id: int, window: str = "late") -> int | None:
    """The episode of the class ``class_id`` (of ``kind``) among the points of ``subset`` (a mask over points) whose path lies closest in the plane to the Gaussian of the class in
    ``window``: the lowest mean squared Mahalanobis distance of the points of its path to it, among the episodes whose path is at least as long as the median of them (a success
    that ends after one step has no path and can sit anywhere by chance). The path of a success is all its points, that of a failure its points up to its failure step. None if the
    class is too small to be fitted."""
    fits = [f for f in fit_classes(points, kind, subset, window) if f.id == class_id]
    if not fits:
        return None
    ep = points["episode"]
    drawn = subset & np.isin(ep, np.unique(ep[fits[0].rows])) & (points["success"][ep] | (points["step"] <= points["onset"][ep]))
    rows = np.flatnonzero(drawn)
    diff = points["xy"][rows].astype(np.float64) - fits[0].mean
    distance = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(fits[0].cov), diff)
    episodes, index = np.unique(ep[rows], return_inverse=True)
    lengths = np.bincount(index)
    mean_distance = np.bincount(index, weights=distance) / lengths
    mean_distance[lengths < np.median(lengths)] = np.inf
    return int(episodes[np.argmin(mean_distance)])


def random_pairs(points: dict, subset: np.ndarray, n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Up to ``n`` pairs (success, failure) of distinct random episodes among those of ``subset`` (a mask over points), the successes and the failures each drawn without
    replacement among the episodes whose path is at least as long as the median of their class (the path of a success is all its points, that of a failure its points up to its
    failure step). Fewer pairs if a class has fewer such episodes."""
    ep = points["episode"]
    episodes, index = np.unique(ep[subset & (points["success"][ep] | (points["step"] <= points["onset"][ep]))], return_inverse=True)
    lengths = np.bincount(index)
    success = points["success"][episodes]
    picks = []
    for wins in (True, False):
        in_class = success == wins
        pool = episodes[in_class & (lengths >= np.median(lengths[in_class]))] if in_class.any() else episodes[:0]
        picks.append(rng.permutation(pool)[:n])
    return [(int(a), int(b)) for a, b in zip(*picks)]


def trajectory_candidates(points: dict, subset: np.ndarray, n_successes: int, n_failures: int, rng: np.random.Generator,
                          exclude: frozenset[int] | set[int] = frozenset()) -> tuple[list[int], list[int]]:
    """Episodes of ``subset`` (a mask over points) to choose example trajectories from: ``n_successes`` random successes and ``n_failures`` failures, one for each of the most common
    failure modes of the subset (ties by mode index), then random ones if there are fewer modes. Only episodes whose path is at least as long as the median of their class are
    eligible (the path of a success is all its points, that of a failure its points up to its failure step). Fewer are returned if a class has fewer eligible episodes. The episodes of
    ``exclude`` (those of an earlier draw) are not chosen again, and the failure modes they have are taken last, so that a second draw shows the modes the first one did not. Only failures
    of a mode that has a Gaussian in the subset (at least ``MIN_EPISODES`` episodes of it) are chosen, so that every failure can be compared with the ellipse of its mode."""
    ep = points["episode"]
    episodes, index = np.unique(ep[subset & (points["success"][ep] | (points["step"] <= points["onset"][ep]))], return_inverse=True)
    lengths = np.bincount(index)
    success = points["success"][episodes]
    pools = {}
    for wins in (True, False):
        in_class = success == wins
        pools[wins] = episodes[in_class & (lengths >= np.median(lengths[in_class])) & ~np.isin(episodes, list(exclude))] if in_class.any() else episodes[:0]
    successes = [int(e) for e in rng.permutation(pools[True])[:n_successes]]
    failed_modes = points["mode"][episodes[~success]]  # the mode of every failure of the subset
    fitted = {int(m) for m in set(failed_modes.tolist()) if int((failed_modes == m).sum()) >= MIN_EPISODES}
    pool = pools[False][np.isin(points["mode"][pools[False]], list(fitted))]
    modes = points["mode"][pool]
    used = {int(points["mode"][e]) for e in exclude if not points["success"][e]}
    by_frequency = sorted(set(modes.tolist()), key=lambda m: (m in used, -int((modes == m).sum()), m))
    failures = [int(rng.choice(pool[modes == m])) for m in by_frequency[:n_failures]]
    rest = [int(e) for e in rng.permutation(pool) if int(e) not in failures]
    return successes, failures + rest[: max(n_failures - len(failures), 0)]


def class_stats(x: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(counts (k,), means (k, d), covariances (k, d, d)) of the rows of ``x`` of each class 0..k-1; a class with fewer than two rows has NaN mean and covariance."""
    counts = np.array([(labels == i).sum() for i in range(k)])
    means, covs = np.full((k, x.shape[1]), np.nan), np.full((k, x.shape[1], x.shape[1]), np.nan)
    for i in np.flatnonzero(counts >= 2):
        rows = x[labels == i].astype(np.float64)
        means[i], covs[i] = rows.mean(axis=0), np.cov(rows.T)
    return counts, means, covs


def pairwise_distance(counts: np.ndarray, means: np.ndarray, covs: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    """(k, k) Mahalanobis distances between the class means under the pooled within-class covariance plus ``ridge`` times its mean variance; NaN for a class without a fit."""
    valid = counts >= 2
    pooled = sum((counts[i] - 1) * covs[i] for i in np.flatnonzero(valid)) / (counts[valid].sum() - valid.sum())
    pooled = pooled + ridge * np.trace(pooled) / len(pooled) * np.eye(len(pooled))
    out = np.full((len(counts), len(counts)), np.nan)
    for a in np.flatnonzero(valid):
        diff = means[valid] - means[a]
        out[a, valid] = np.sqrt(np.einsum("ij,ji->i", diff, np.linalg.solve(pooled, diff.T)))
    return out
