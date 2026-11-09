"""Raw rollout tokens -> per-modality pooled features.

This is the only module that reads the raw ``rollout.npz`` files. Output is ``data/<dataset>/features.npz`` with one float32
array (n_steps, 6144) per episode, keyed by the manifest ``dir`` and laid out as [base | wrist | language]:

    base image : mean over its tokens          columns    0:2048
    wrist image: mean over its tokens          columns 2048:4096
    language   : mean over the unpadded tokens columns 4096:6144

The backbone state features some rollouts also carry are not used.
"""

from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..config import FEATURE_DIM, get
from .features import write_arrays
from .manifest import load_manifest


def pool_modalities(base: np.ndarray, wrist: np.ndarray, language: np.ndarray, language_mask: np.ndarray) -> np.ndarray:
    """(T, 3 * D) float32 from base, wrist (T, tokens, D), language (T, tokens, D) and language_mask (T, tokens) bool.

    Image tokens are all valid; language tokens are averaged where the mask is True (a step with no valid token gives zeros).
    """
    base_mean = base.astype(np.float32).mean(axis=1)
    wrist_mean = wrist.astype(np.float32).mean(axis=1)
    mask = language_mask.astype(np.float32)
    language_mean = (language.astype(np.float32) * mask[..., None]).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    return np.concatenate([base_mean, wrist_mean, language_mean], axis=1).astype(np.float32)


def pool_rollout(path: Path) -> np.ndarray:
    """Pool one ``rollout.npz``. Rollouts that record ``has_features`` must have it set."""
    with np.load(path) as z:
        if "has_features" in z.files and not bool(z["has_features"]):
            raise ValueError(f"{path} has no backbone features")
        return pool_modalities(z["base_image"], z["wrist_image"], z["language"], z["language_mask"])


def _pool_one(args: tuple[str, Path]) -> tuple[str, np.ndarray]:
    key, path = args
    return key, pool_rollout(path)


def prepare_dataset(dataset: str, rollout_root: str | Path, workers: int = 4) -> Path:
    """Pool every episode of ``dataset``'s manifest from ``rollout_root/<dir>/rollout.npz`` into data/<dataset>/features.npz.

    The file is written episode by episode (no full copy in memory) to a temporary name and renamed once every episode is in it and
    has the number of steps the manifest records. An existing features.npz is never overwritten.
    """
    out = get(dataset).data_dir / "features.npz"
    if out.exists():
        raise FileExistsError(f"{out} already exists; remove it to pool again")
    episodes = load_manifest(dataset)
    jobs = [(e.key, Path(rollout_root) / e.key / "rollout.npz") for e in episodes]
    missing = [str(p) for _, p in jobs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} rollouts not found under {rollout_root}, e.g. {missing[:2]}")

    n_steps = {e.key: e.n_steps for e in episodes}

    def checked():
        for key, x in pool.imap(_pool_one, jobs):  # in manifest order
            if x.shape != (n_steps[key], FEATURE_DIM):
                raise ValueError(f"{key}: pooled shape {x.shape}, the manifest says {n_steps[key]} steps of {FEATURE_DIM}")
            yield key, x

    with Pool(workers) as pool:
        write_arrays(out, checked())
    return out
