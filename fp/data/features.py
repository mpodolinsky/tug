"""Load pooled features and standardise them with training-split statistics."""

import os
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.lib import format as npy_format

from ..config import FEATURE_DIM, get


def write_arrays(out: Path, items: Iterable[tuple[str, np.ndarray]]) -> None:
    """Write ``items`` (episode key, array) to the .npz ``out`` one array at a time (no full copy in memory), to a temporary name that is renamed
    once every item is in. ``items`` may be a generator that checks what it yields: an exception in it leaves ``out`` untouched."""
    tmp = out.with_name(out.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
            for key, x in items:
                with zf.open(f"{key}.npy", "w", force_zip64=True) as f:
                    npy_format.write_array(f, x, allow_pickle=False)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)


def feature_lengths(dataset: str) -> dict[str, int]:
    """{episode key: number of steps} from the array headers of data/<dataset>/features.npz, without loading the arrays."""
    path = get(dataset).data_dir / "features.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    lengths = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as f:
                major, _ = np.lib.format.read_magic(f)
                read_header = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
                shape, _, _ = read_header(f)
            lengths[name.removesuffix(".npy")] = shape[0]
    return lengths


def load_features(dataset: str, keys: list[str]) -> dict[str, np.ndarray]:
    """{episode key: (T, 6144) float32} for the given keys, from data/<dataset>/features.npz."""
    path = get(dataset).data_dir / "features.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    with np.load(path) as z:
        absent = [k for k in keys if k not in z.files]
        if absent:
            raise KeyError(f"{len(absent)} episodes are not in {path.name}, e.g. {absent[:3]}")
        out = {k: z[k] for k in keys}
    for k, x in out.items():
        if x.ndim != 2 or x.shape[1] != FEATURE_DIM or x.dtype != np.float32:
            raise ValueError(f"features of {k} have shape {x.shape} and dtype {x.dtype}; expected (T, {FEATURE_DIM}) float32")
    return out


def fit_standardiser(train: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and std over all steps of the training episodes (float64 sums; 1e-6 is added to the std)."""
    total = square = 0.0
    steps = 0
    for x in train:
        x = x.astype(np.float64)
        total, square, steps = total + x.sum(axis=0), square + (x**2).sum(axis=0), steps + len(x)
    mean = total / steps
    return mean, np.sqrt(np.maximum(square / steps - mean**2, 0.0)) + 1e-6


def standardise(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)
