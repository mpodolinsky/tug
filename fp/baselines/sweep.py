"""Run the baseline families on the splits and write their runs under results/, in the same folders as the probes' runs.

A unit is one family on one split of one dataset (the distance families have 2, 6 and 18 configurations, in SAFE's grid order). The dataset's
features are loaded once, the first time a unit needs them, and shared by all its splits. The scorers of a unit are fitted one at a time and
released before the next, because a fitted distance scorer holds its training steps (about 3 GB for LIBERO-X). Each fits on the split's train
(and val) keys, scores every episode of the dataset, and writes ``scores.npz`` and ``history.json`` (and ``weights.pt`` if it has weights).
Runs that exist are skipped, so a killed sweep resumes; a scorer that fails is logged and skipped, never marked finished.

The baselines use the RAW pooled features (no standardisation), as SAFE's embedding baselines use their embeddings; the history says so.

Each history also records what the run cost, so that methods can be compared on cost as well as accuracy: seconds to fit and to score, the
number of steps scored (so milliseconds per step), peak GPU memory, peak resident memory of the process, the size of what a fitted scorer
must keep to score a new episode (``state_mb``), and the hardware. Peak RSS includes the features the process keeps loaded for the
dataset's other runs, which is part of what the method needs.
"""

import gc
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping

import numpy as np
import torch

from ..data.actions import load_actions
from ..data.features import load_features
from ..data.inputs import Inputs
from ..data.splits import Split
from ..grid import DATASETS, split_priority
from ..models.base import Scorer
from ..results import code_version, free_gb, hardware_info, is_done, peak_rss_gb, reset_peak_rss, run_dir, run_header, write_run
from . import actprobe, distance_grid, failurespot, safe_grid, stac  # noqa: F401  (register the distance, SAFE, STAC, ActProbe and FailureSpot families)
from .registry import Context, build


@dataclass(frozen=True)
class BaselineUnit:
    dataset: str
    split: str
    family: str


def baseline_units(split_names: Mapping[str, list[str]], families: list[str]) -> list[BaselineUnit]:
    """Every unit in run order: dataset (in ``DATASETS`` order), then split by priority, then family. A dataset's units are adjacent, so a
    worker loads its features once."""
    return [BaselineUnit(d, s, f) for d in DATASETS if d in split_names for s in sorted(split_names[d], key=split_priority) for f in families]


@dataclass(frozen=True)
class Settings:
    out: Path
    save_weights: bool = True
    min_free_gb: float = 20.0  # below this much free disk, scores and history are still written but weights are not


class LazyFeatures(Mapping):
    """The raw pooled features of a dataset, read from disk the first time one is asked for (a unit whose runs all exist never loads them)."""

    def __init__(self, dataset: str, keys: list[str]):
        self.dataset, self.keys, self._data = dataset, keys, None

    def _load(self) -> dict[str, np.ndarray]:
        return load_features(self.dataset, self.keys)

    def __getitem__(self, key: str) -> np.ndarray:
        if self._data is None:
            self._data = self._load()
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys)

    def __len__(self) -> int:
        return len(self.keys)

    def release(self) -> None:
        self._data = None


class LazyActions(LazyFeatures):
    """The sampled action chunks of a dataset (T, K, H, 7), read from disk the first time one is asked for."""

    def _load(self) -> dict[str, np.ndarray]:
        return load_actions(self.dataset, self.keys)


def run_scorer(scorer: Scorer, family: str, split: Split, inputs: Inputs, device: str) -> tuple[dict[str, np.ndarray], dict, dict]:
    """Fit on the split's train (and val) keys and score every episode; returns the scores, the extra arrays files the scorer offers (for the
    distance baselines the distances to the success and failure sets, ``distances.npz``) and the run's history."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    reset_peak_rss()
    start = time.time()
    scorer.fit(split.train, split.val)
    fit_seconds = time.time() - start
    scores, extras = scorer.score_with_extras(list(inputs.episodes))
    score_seconds = time.time() - start - fit_seconds
    all_scores = np.concatenate(list(scores.values()))
    if not np.isfinite(all_scores).all():
        raise ValueError(f"{scorer.name} produced non-finite scores on {split.name}")
    steps = sum(len(s) for s in scores.values())
    state = scorer.state_bytes()
    info = {**run_header(inputs.dataset, split, scorer.name, inputs.window, device, time.time() - start), "family": family, "features": scorer.input_kind,
            "config": getattr(scorer, "config", {}), **(scorer.training_details() or {}),
            "fit_seconds": round(fit_seconds, 1), "score_seconds": round(score_seconds, 1),
            "scored_steps": steps, "score_ms_per_step": round(1000 * score_seconds / steps, 4),
            "n_train_episodes": len(split.train), "n_train_steps": sum(inputs.episodes[k].n_steps for k in split.train),
            "score_min": float(all_scores.min()), "score_max": float(all_scores.max()),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
            "peak_rss_gb": round(peak_rss_gb(), 2), "state_mb": None if state is None else round(state / 1e6, 2), "hardware": hardware_info()}
    return scores, extras, info


def run_unit(unit: BaselineUnit, context: Context, settings: Settings, log: Callable[[str], None] = print, methods: list[str] | None = None) -> int:
    """Run the scorers of the unit that have no run yet (all, or only the named methods); returns how many runs were written."""
    split = context.inputs.splits[unit.split]
    try:
        built = build(unit.family, context, split)
    except NotImplementedError as reason:
        log(f"{unit.dataset}/{unit.split}/{unit.family}: skipped, {reason}")
        return 0
    scorers = [s for s in built
               if (methods is None or s.name in methods) and not is_done(run_dir(unit.dataset, unit.split, s.name, settings.out))]
    del built  # a second reference would keep every fitted scorer alive until the unit ends
    if not scorers:
        return 0
    settings.out.mkdir(parents=True, exist_ok=True)
    log(f"{unit.dataset}/{unit.split}/{unit.family}: {len(scorers)} runs to make")
    written = 0
    while scorers:
        scorer = scorers.pop(0)  # the only reference left to it is `scorer`, so releasing it below frees its training steps
        try:
            scores, extras, info = run_scorer(scorer, unit.family, split, context.inputs, context.device)
            room = free_gb(settings.out) >= settings.min_free_gb
            weights = scorer.saves_weights and settings.save_weights and room
            write_run(run_dir(unit.dataset, unit.split, scorer.name, settings.out), scores,
                      {**info, "code": code_version(), "weights_saved": weights, "extras_saved": sorted(extras) if room else []},
                      scorer.save if weights else None, extras if room else None)
            written += 1
            log(f"  {scorer.name}: fit {info['fit_seconds']:.0f}s, score {info['score_seconds']:.0f}s, "
                f"scores in [{info['score_min']:.3g}, {info['score_max']:.3g}]")
        except Exception:
            log(f"  FAILED {unit.dataset}/{unit.split}/{scorer.name}\n{traceback.format_exc()}")
        finally:
            scorer = None
            gc.collect()
            torch.cuda.empty_cache()
    return written
