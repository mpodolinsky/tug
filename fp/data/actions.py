"""Raw rollouts -> the sampled action chunks of every episode (the input of STAC, ActProbe and FailureSpot).

Output is ``data/<dataset>/actions.npz`` with one float32 array (n_steps, K, H, A) per episode, keyed by the manifest ``dir``: at every policy
call the policy sampled K = 10 action chunks of H steps (GR00T H = 16, pi0.5 H = 10) of A = 7 values (position 0:3, rotation 3:6, gripper 6).
Sample 0 is the chunk that was executed; it equals the ``predicted_action_chunks`` of the rollout, which is therefore not stored again.

Only the one array is read from each ``rollout.npz`` (the file also holds the token embeddings, tens of MB). Like the feature pooling, the
file is written episode by episode to a temporary name and renamed once every episode has passed its checks; an existing file is never
overwritten.
"""

from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..config import get
from .features import write_arrays
from .manifest import load_manifest

ACTION_DIM = 7  # position (3), rotation (3), gripper (1)


def read_rollout_actions(path: Path, replan_steps: int) -> np.ndarray:
    """The (T, K, H, A) sampled chunks of one ``rollout.npz``, after checking what the baselines rely on."""
    with np.load(path) as z:
        sampled, executed = z["sampled_action_chunks"], int(z["executed_sample_index"])
        predicted, replan = z["predicted_action_chunks"], int(z["replan_steps"])
    if sampled.ndim != 4 or sampled.shape[3] != ACTION_DIM or sampled.shape[2] <= replan_steps or not np.isfinite(sampled).all():
        raise ValueError(f"{path}: sampled chunks of shape {sampled.shape} (need finite (T, K, H > {replan_steps}, {ACTION_DIM}))")
    if replan != replan_steps:
        raise ValueError(f"{path}: replan_steps {replan}, the dataset says {replan_steps}")
    if executed != 0 or not np.array_equal(sampled[:, 0], predicted):
        raise ValueError(f"{path}: sample 0 is not the executed chunk (executed_sample_index {executed}); the baselines assume it is")
    return sampled.astype(np.float32)


def _read_one(args: tuple[str, Path, int]) -> tuple[str, np.ndarray]:
    key, path, replan_steps = args
    return key, read_rollout_actions(path, replan_steps)


def prepare_actions(dataset: str, rollout_root: str | Path, workers: int = 4) -> Path:
    """Write data/<dataset>/actions.npz from ``rollout_root/<dir>/rollout.npz`` for every episode of the manifest."""
    spec = get(dataset)
    out = spec.data_dir / "actions.npz"
    if out.exists():
        raise FileExistsError(f"{out} already exists; remove it to prepare again")
    episodes = load_manifest(dataset)
    jobs = [(e.key, Path(rollout_root) / e.key / "rollout.npz", spec.replan_steps) for e in episodes]
    missing = [str(p) for _, p, _ in jobs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} rollouts not found under {rollout_root}, e.g. {missing[:2]}")
    n_steps = {e.key: e.n_steps for e in episodes}
    shapes = set()

    def checked():
        for key, x in pool.imap(_read_one, jobs):
            if len(x) != n_steps[key]:
                raise ValueError(f"{key}: {len(x)} policy calls, the manifest says {n_steps[key]}")
            shapes.add(x.shape[1:])
            yield key, x
        if len(shapes) != 1:  # after the last episode, before the file is renamed
            raise ValueError(f"the episodes of {dataset} do not share one (K, H, A): {sorted(shapes)}")

    with Pool(workers) as pool:
        write_arrays(out, checked())
    return out


def load_actions(dataset: str, keys: list[str]) -> dict[str, np.ndarray]:
    """{episode key: (T, K, H, A) float32} for the given keys, from data/<dataset>/actions.npz."""
    path = get(dataset).data_dir / "actions.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; python -m cli.prepare_actions {dataset} <rollout_root>")
    with np.load(path) as z:
        absent = [k for k in keys if k not in z.files]
        if absent:
            raise KeyError(f"{len(absent)} episodes are not in {path.name}, e.g. {absent[:3]}")
        out = {k: z[k] for k in keys}
    for k, x in out.items():
        if x.ndim != 4 or x.shape[3] != ACTION_DIM or x.dtype != np.float32:
            raise ValueError(f"actions of {k} have shape {x.shape} and dtype {x.dtype}; expected (T, K, H, {ACTION_DIM}) float32")
    return out
