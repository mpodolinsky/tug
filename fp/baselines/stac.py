"""STAC (Agia et al., CoRL 2024, arXiv 2410.04640; code: github.com/agiachris/sentinel, MIT): how much the policy changes its mind about the
actions it had already planned.

At every policy call the policy plans H actions and executes the first ``replan_steps`` of them. The unexecuted rest of the previous call's plan
and the first actions of the new plan cover the same future steps (the *overlap*, H - replan_steps actions); STAC measures how far apart the two
are. With K sampled plans per call (K = 10 here) the two sides are sets of K plans and the distance is the maximum mean discrepancy (MMD) with an
RBF kernel exp(-||x - y||^2) between them ("STAC"); with only the executed plan (sample 0) it is the mean L2 distance of the overlapping actions
("STAC-Single"). Nothing is learned: a score depends on one episode only, and fitting does nothing.

Methods (each its own name, all run; the best of a family is chosen on validation like every other baseline):

    stac__mmd7_step      MMD over all 7 action dimensions (position, rotation, gripper): SAFE's re-implementation of STAC, used in the earlier results
    stac__mmd6_step      MMD over position and rotation, without the gripper: the authors' "mmd_rbf_all" setting
    stac__mmd3_step      MMD over position only: the authors' "mmd_rbf_pos" setting
    stac__<tag>_cumsum   the same, accumulated over the episode (the authors accumulate; scores are non-negative, so the maximum over steps is the final value)
    stac_single__step, stac_single__cumsum    SAFE's STAC-Single: the L2 norm over the 7 dimensions, averaged over the overlap

The first step of an episode has no previous call and scores 0. The overlap is the full one (the prediction horizon H minus the execution horizon),
as in SAFE's code; the authors' code takes the prediction horizon as a setting, which is H here. Dimension layout of our actions (LIBERO): position
0:3, rotation 3:6, gripper 6. The authors' own filters assume their robot's layout (gripper first), so their variants are defined here by what
they drop, and ``tests/test_stac.py`` checks them against the authors' code with the dimensions mapped into their layout.
"""

from typing import Mapping

import numpy as np

from .. import config
from ..data.splits import Split
from ..models.base import Scorer
from .registry import Context, register

DIMENSIONS = {7: slice(0, 7), 6: slice(0, 6), 3: slice(0, 3)}  # position 0:3, rotation 3:6, gripper 6
DIMENSION_MEANING = {7: "position, rotation and gripper (SAFE's re-implementation)", 6: "position and rotation (the authors' mmd_rbf_all)",
                     3: "position only (the authors' mmd_rbf_pos)"}
FAMILIES = ("stac", "stac_single")


def overlap(chunks: np.ndarray, exec_horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """The previous call's unexecuted actions and the new call's first actions, for every consecutive pair of calls: (T-1, ..., H-e, A) each."""
    if chunks.shape[-2] <= exec_horizon:
        raise ValueError(f"the chunks (H = {chunks.shape[-2]}) must be longer than the execution horizon ({exec_horizon})")
    return chunks[:-1, ..., exec_horizon:, :], chunks[1:, ..., :-exec_horizon, :]


def mmd_rbf(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """(T-1,) squared MMD between the K plans of two consecutive calls, kernel exp(-||x - y||^2); ``old`` and ``new`` are (T-1, K, h, A)."""
    x, y = (a.reshape(*a.shape[:2], a.shape[2] * a.shape[3]).astype(np.float64) for a in (old, new))  # (T-1 may be 0, so no -1 here)

    def mean_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.exp(-((a[:, :, None, :] - b[:, None, :, :]) ** 2).sum(-1)).mean(axis=(1, 2))

    return mean_kernel(x, x) + mean_kernel(y, y) - 2 * mean_kernel(x, y)


def stac_scores(chunks: np.ndarray, exec_horizon: int, dims: int = 7) -> np.ndarray:
    """(T,) per-step STAC of an episode: ``chunks`` is (T, K, H, A); 0 at the first step."""
    old, new = overlap(chunks[..., DIMENSIONS[dims]], exec_horizon)
    return np.concatenate([[0.0], mmd_rbf(old, new)]).astype(np.float32)


def stac_single_scores(chunk: np.ndarray, exec_horizon: int) -> np.ndarray:
    """(T,) per-step STAC-Single of an episode: ``chunk`` is the executed plan, (T, H, A); mean over the overlap of the L2 norm over the dimensions."""
    old, new = overlap(chunk, exec_horizon)
    return np.concatenate([[0.0], np.linalg.norm(old.astype(np.float64) - new, axis=-1).mean(axis=-1)]).astype(np.float32)


class StacScorer(Scorer):
    input_kind = "sampled action chunks"

    def __init__(self, family: str, actions: Mapping[str, np.ndarray], replan_steps: int, *, dims: int = 7, cumsum: bool = False):
        if family not in FAMILIES or dims not in DIMENSIONS:
            raise ValueError(f"family must be one of {FAMILIES} and dims one of {tuple(DIMENSIONS)}")
        self.family, self.actions, self.replan_steps, self.dims, self.cumsum = family, actions, replan_steps, dims, cumsum
        accumulation = "cumsum" if cumsum else "step"
        self.name = f"stac__mmd{dims}_{accumulation}" if family == "stac" else f"stac_single__{accumulation}"

    @property
    def config(self) -> dict:
        return {"family": self.family, "dimensions": self.dims if self.family == "stac" else 7, "cumsum": self.cumsum, "exec_horizon": self.replan_steps,
                "kernel": "exp(-||x-y||^2)" if self.family == "stac" else None, "overlap": "full (H - exec_horizon)",
                "meaning": DIMENSION_MEANING[self.dims] if self.family == "stac" else "L2 norm over the 7 dimensions, mean over the overlap"}

    def fit(self, train_keys: list[str], val_keys: list[str]) -> None:
        """Nothing is learned."""

    def score(self, keys: list[str]) -> dict[str, np.ndarray]:
        out = {}
        for key in keys:
            chunks = self.actions[key]
            step = stac_scores(chunks, self.replan_steps, self.dims) if self.family == "stac" else stac_single_scores(chunks[:, 0], self.replan_steps)
            out[key] = np.cumsum(step, dtype=np.float64).astype(np.float32) if self.cumsum else step
        return out

    def state_bytes(self) -> int | None:
        return 0  # nothing is kept: a new episode is scored from its own action chunks


def grid_scorers(family: str, actions: Mapping[str, np.ndarray], replan_steps: int) -> list[StacScorer]:
    """The scorers of a family in grid order (the order ties are broken in): by dimensions (7, 6, 3), step before cumsum."""
    if family == "stac":
        return [StacScorer("stac", actions, replan_steps, dims=d, cumsum=c) for d in DIMENSIONS for c in (False, True)]
    return [StacScorer("stac_single", actions, replan_steps, cumsum=c) for c in (False, True)]


def method_names(family: str) -> list[str]:
    return [s.name for s in grid_scorers(family, {}, 1)]


def _family(family: str):
    def build(context: Context, split: Split) -> list[StacScorer]:
        if context.actions is None:
            raise ValueError("STAC needs the sampled action chunks: python -m cli.prepare_actions <dataset> <rollout_root>")
        return grid_scorers(family, context.actions, config.get(context.inputs.dataset).replan_steps)

    return build


for _name in FAMILIES:
    register(_name, needs_actions=True)(_family(_name))
