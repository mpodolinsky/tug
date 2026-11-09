"""Alpha sweep and the headline area.

``sweep`` evaluates one fitted calibrator on a set of episodes for increasing alpha (a looser
threshold each time) and returns one row per alpha: alpha, fpr, tpr, tpr_early, tpr_on_time, ldr.
By default the whole headline grid ``ALPHAS`` is run (0.005 to FPR_MAX / 100 = 0.40; see the comment at ``ALPHAS`` for why not
beyond), with no early stop, which is immune to a calibrator whose FPR is not monotone in alpha (the SAFE band); ``stop_fpr`` is an
optional early stop at the first alpha whose empirical FPR reaches it. Rows beyond 40% FPR (the evaluation FPR can exceed alpha on
held-out tasks) are stored and ignored by the area. A curve that never reaches 40% FPR within the grid is held flat and flagged.

``evaluate_set`` is the one place that turns a fitted calibrator and a set of episodes into rows and an area.

``area_to_fpr`` is the one area function: area under TPR_early vs FPR from 0 to ``fpr_max``,
divided by ``fpr_max`` (so a curve at 100% everywhere scores 100). The curve starts at (0, 0), is
made monotone with a running maximum, and is held flat past its last point; the returned flag says
whether that extension was needed.
"""

from dataclasses import dataclass

import numpy as np

from ..calibration.base import Calibrator
from ..data.manifest import Episode
from .outcomes import classify
from .rates import rates_from_outcomes

FPR_MAX = 40.0
# The headline grid: 0.005 ... 0.40. A calibrator at level alpha promises "FPR <= alpha" (for episodes like the calibration ones), so only an
# alpha of at most FPR_MAX / 100 promises an FPR inside the plotted range 0 ... FPR_MAX. An operating point from a larger alpha is never used
# to fill in the area below FPR_MAX, however low its FPR happens to be on the evaluation episodes.
ALPHAS = np.round(np.arange(0.005, FPR_MAX / 100 + 1e-9, 0.005), 3)
# Every alpha strictly inside (0, 1): for diagnostics (how FPR moves with alpha), not for the headline area.
ALPHAS_FULL = np.round(np.arange(0.005, 1.0, 0.005), 3)


@dataclass(frozen=True)
class Area:
    value: float  # in [0, 100] when y is a percentage
    extended: bool  # True if the curve never reached fpr_max


def curve_points(fpr: np.ndarray, y: np.ndarray, running_max: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """The curve as unique ascending x (FPR) with one y each, starting at (0, 0); NaN points are dropped.

    With ``running_max`` (the headline area's rule, for TPR early, which a looser threshold can only raise) y is made monotone and the best of
    several points at one FPR is the last. Without it (a quantity a looser threshold can lower, such as on-time TPR: an alarm moves from the
    window to before it) several points at one FPR are averaged and nothing is flattened."""
    fpr, y = np.asarray(fpr, dtype=float), np.asarray(y, dtype=float)
    if fpr.shape != y.shape:
        raise ValueError("fpr and y must have the same shape")
    keep = ~(np.isnan(fpr) | np.isnan(y))
    fpr, y = np.concatenate([[0.0], fpr[keep]]), np.concatenate([[0.0], y[keep]])
    order = np.argsort(fpr, kind="stable")
    fpr, y = fpr[order], y[order]
    xs = np.unique(fpr)
    if running_max:
        y = np.maximum.accumulate(y)  # several operating points can share one FPR; after the running max the last of them is the best
        return xs, np.array([y[fpr == x][-1] for x in xs])
    return xs, np.array([y[fpr == x].mean() for x in xs])


def curve_on_grid(fpr: np.ndarray, y: np.ndarray, grid: np.ndarray, running_max: bool = True) -> np.ndarray:
    """The curve of ``curve_points`` at the FPR values in ``grid``: linear between points, held flat past the last one."""
    xs, ys = curve_points(fpr, y, running_max)
    return np.interp(grid, xs, ys)


def area_to_fpr(fpr: np.ndarray, y: np.ndarray, fpr_max: float = FPR_MAX) -> Area:
    xs, ys = curve_points(fpr, y)
    extended = bool(xs[-1] < fpr_max)
    y_end = float(np.interp(fpr_max, xs, ys))  # np.interp holds the last value flat past xs[-1]
    knots_x, knots_y = np.append(xs[xs < fpr_max], fpr_max), np.append(ys[xs < fpr_max], y_end)
    area = float(np.sum(np.diff(knots_x) * (knots_y[1:] + knots_y[:-1]) / 2.0))
    return Area(area / fpr_max, extended)


def sweep(
    calibrator: Calibrator,
    scores: dict[str, np.ndarray],
    episodes: list[Episode],
    window: int,
    alphas: np.ndarray = ALPHAS,
    stop_fpr: float | None = None,
) -> list[dict]:
    """One row per alpha, ascending. ``calibrator`` is already fitted on the calibration successes."""
    length = max(len(scores[e.key]) for e in episodes)
    rows = []
    for alpha in alphas:
        thr = calibrator.threshold(float(alpha), length)
        r = rates_from_outcomes([classify(scores[e.key], thr, e.onset, window) for e in episodes])
        rows.append(dict(alpha=float(alpha), fpr=r.fpr, tpr=r.tpr, tpr_early=r.tpr_early, tpr_on_time=r.tpr_on_time, ldr=r.ldr))
        if stop_fpr is not None and r.fpr >= stop_fpr:
            break
    return rows


@dataclass(frozen=True)
class SetResult:
    """The alpha sweep and headline area of one calibrated detector on one set of episodes."""

    rows: list[dict]
    area: float  # NaN when the set has no failures or no successes: there is no curve to integrate
    extended: bool
    n_fail: int
    n_succ: int


def evaluate_set(calibrator: Calibrator, scores: dict[str, np.ndarray], episodes: list[Episode], window: int, alphas: np.ndarray = ALPHAS) -> SetResult:
    """Sweep the fitted ``calibrator`` over ``episodes`` and take the area to FPR 40 of TPR early. The one place that does this, for the
    validation episodes used to choose a configuration as much as for the episodes whose results are reported.

    A set without successes has no FPR and a set without failures no TPR; ``area_to_fpr`` would drop the NaN points and report an area
    of 0, so the area is NaN there instead.
    """
    n_fail = sum(not e.success for e in episodes)
    n_succ = len(episodes) - n_fail
    rows = sweep(calibrator, scores, episodes, window, alphas)
    if n_fail == 0 or n_succ == 0:
        return SetResult(rows, float("nan"), False, n_fail, n_succ)
    area = area_to_fpr(np.array([r["fpr"] for r in rows]), np.array([r["tpr_early"] for r in rows]))
    return SetResult(rows, area.value, area.extended, n_fail, n_succ)
