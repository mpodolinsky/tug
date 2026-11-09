"""Where results go and how they are read back, so plots never need to re-run anything.

    results/<dataset>/<split>/<method>/
        scores.npz            per-step failure probabilities of every episode of the dataset (key = episode key)
        weights.pt            learned parameters (every trained method)
        mode_logits.npz       probes with a mode head: (T, 10) logits per step of every episode
        semantic.npz          probes with a CLIP head: (T, 64) raw semantic vectors per step of every episode
        semantic_space.npz    the CLIP head's target space (``target`` (P, 64) per phrase, ``bank_index`` (P,)), to interpret semantic.npz
        history.json         what the run was: configs, per-term learning curves, best epoch, seconds, split hash, code version
        curve_<calibrator>.json   the alpha sweep on the validation, seen and unseen episodes: alpha, fpr, tpr, tpr_early, tpr_on_time, ldr
        summary.json          per calibrator and episode set: area to FPR 40, extended flag, numbers of failures and successes; code version
        ltt.json              the Learn-then-Test operating point and its rates, when LTT was evaluated

A run directory is written under a temporary name and renamed when complete, so a directory that exists is a finished run and a killed
job leaves nothing that looks finished. An existing run is never overwritten. The evaluation files are derived from the scores and may
be overwritten; each file is replaced atomically, and ``summary.json`` records the code version that produced it.
"""

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .config import RESULTS
from .data.splits import Split


def run_dir(dataset: str, split: str, method: str, root: Path = RESULTS) -> Path:
    return Path(root) / dataset / split / method


def is_done(path: Path) -> bool:
    return (Path(path) / "history.json").exists()


def write_run(path: Path, scores: dict[str, np.ndarray], info: dict, save_weights: Callable[[str], None] | None = None,
              extras: Mapping[str, Mapping[str, np.ndarray]] | None = None) -> None:
    """Write scores, history, weights (if a saver is given) and any extra arrays files (``extras``: file name -> arrays) into ``path``, atomically."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; runs are never overwritten")
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    np.savez(partial / "scores.npz", **scores)
    for name, arrays in (extras or {}).items():
        np.savez(partial / f"{name}.npz", **arrays)
    if save_weights is not None:
        save_weights(str(partial / "weights.pt"))
    (partial / "history.json").write_text(json.dumps(info))
    os.replace(partial, path)


def read_scores(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path) / "scores.npz") as z:
        return {k: z[k] for k in z.files}


def read_history(path: Path) -> dict:
    return json.loads((Path(path) / "history.json").read_text())


def split_hash(split: Split) -> str:
    """Identifies the exact episodes of a split, so a result can be matched to the split file it was trained on."""
    digest = hashlib.sha256()
    for part in (split.train, split.val, split.calibration, split.seen, split.unseen):
        digest.update(("\n".join(part) + "\x00").encode())
    return digest.hexdigest()


def code_version() -> dict:
    """The git commit and a hash of the source of ``fp/``, which also covers changes that are not committed yet."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for p in sorted((root / "fp").rglob("*.py")):
        digest.update(p.relative_to(root).as_posix().encode() + p.read_bytes())
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return {"git_head": head.stdout.strip() or None, "fp_sha256": digest.hexdigest()}


def run_header(dataset: str, split: Split, method: str, window: int, device: str, seconds: float) -> dict:
    """The fields every run's history starts with, for probes and baselines alike: what ran where, with which seed and split, and when."""
    return {"method": method, "dataset": dataset, "split": split.name, "seed": split.seed, "window": window, "seconds": round(seconds, 1),
            "split_sha256": split_hash(split), "device": device, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def hardware_info() -> dict:
    """What a run's time and memory numbers were measured on."""
    import platform

    import torch

    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    ram_kb = next((int(line.split()[1]) for line in open("/proc/meminfo") if line.startswith("MemTotal")), 0)
    return {"gpu": gpu.name if gpu else None, "gpu_memory_gb": round(gpu.total_memory / 1e9, 1) if gpu else None, "cpu_threads": os.cpu_count(),
            "ram_gb": round(ram_kb / 1e6, 1), "torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform()}


def reset_peak_rss() -> None:
    """Restart the process's peak-memory counter (Linux): ``peak_rss_gb`` then reports the peak since this call."""
    try:
        Path("/proc/self/clear_refs").write_text("5")
    except OSError:
        pass  # not Linux or not permitted: peak_rss_gb then reports the peak over the whole process


def peak_rss_gb() -> float:
    """Peak resident memory of this process in GB (VmHWM), including the data it holds for later runs."""
    return next((int(line.split()[1]) / 1e6 for line in open("/proc/self/status") if line.startswith("VmHWM")), float("nan"))


def write_json(path: Path, obj: dict) -> None:
    """Write ``obj`` to ``path`` atomically, replacing an existing file (for derived files such as the evaluation outputs)."""
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(obj))
    os.replace(partial, path)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def iter_runs(root: Path = RESULTS, datasets: tuple[str, ...] | None = None):
    """Every finished run under ``root`` as (dataset, split, method, path), in sorted order."""
    for history in sorted(Path(root).glob("*/*/*/history.json")):
        path = history.parent
        if datasets is None or path.parents[1].name in datasets:
            yield path.parents[1].name, path.parent.name, path.name, path
