"""Train the probes of a unit (one label variant on one split) and write their runs under results/.

The split's features are loaded and standardised once and shared by the 16 architectures of the unit. Each probe is trained with its
split's seed, then runs on every episode of the dataset; scores, weights, history and (for the architectures that have those heads) the mode and CLIP
head outputs go to ``results/<dataset>/<split>/<method>/``. Runs that
exist are skipped, so a killed sweep resumes; a probe that fails is logged and skipped, never marked finished.
"""

import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ..data.features import load_features
from ..data.inputs import Inputs
from ..data.splits import Split
from ..grid import Unit
from ..models.ours import ProbeConfig, all_configs
from ..results import code_version, free_gb, hardware_info, is_done, peak_rss_gb, reset_peak_rss, run_dir, run_header, write_run
from .loop import TrainConfig, default_lr
from .ours import OursScorer
from .standardised import StandardisedFeatures, standardise_features


@dataclass(frozen=True)
class Settings:
    out: Path
    max_epochs: int = 100
    patience: int = 15
    save_weights: bool = True
    save_heads: bool = True  # outputs of the mode and CLIP heads, for the architectures that have them
    min_free_gb: float = 20.0  # below this much free disk, scores and history are still written but weights and head outputs are not
    lam_mode: float = 1.0  # weight of the mode term in the loss
    lam_clip: float = 1.0  # weight of the CLIP term in the loss
    lam_score: float = 1.0  # weight of the score term; 0 trains a mode-only probe (the epoch is then chosen on the validation mode term)


def head_arrays(scorer: OursScorer, outputs: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    """The arrays files of the mode and CLIP heads (only the architectures that have them), as ``write_run`` takes them."""
    extras = {}
    if "mode" in outputs:
        extras["mode_logits"] = outputs["mode"]
    if "sem" in outputs:
        extras["semantic"] = outputs["sem"]
        extras["semantic_space"] = {"target": scorer.space.target, "bank_index": scorer.space.bank_index}
    return extras


def train_probe(cfg: ProbeConfig, variant: str, split: Split, inputs: Inputs, features: StandardisedFeatures, settings: Settings,
                device: str) -> tuple[OursScorer, dict, dict]:
    """Fit one probe and run it on every episode; returns the scorer, the outputs of all its heads and the run's history."""
    train_cfg = TrainConfig(lr=default_lr(cfg.trunk), max_epochs=settings.max_epochs, patience=settings.patience, seed=split.seed)
    scorer = OursScorer(cfg, variant, inputs.episodes, features, inputs.targets, inputs.window, train_cfg, device, settings.lam_mode, settings.lam_clip, settings.lam_score)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    reset_peak_rss()
    start = time.time()
    scorer.fit(split.train, split.val)
    fit_seconds = time.time() - start
    outputs = scorer.outputs(list(inputs.episodes))
    score_seconds = time.time() - start - fit_seconds
    r = scorer.result
    steps = sum(len(s) for s in outputs["score"].values())
    info = {
        **run_header(inputs.dataset, split, scorer.name, inputs.window, device, time.time() - start),
        "variant": variant, "arch": asdict(cfg), "lam_mode": settings.lam_mode, "lam_clip": settings.lam_clip, "lam_score": settings.lam_score, "train_cfg": asdict(train_cfg), "best_epoch": r.best_epoch, "epochs": len(r.val["score"]),
        "stopped_early": r.stopped_early, "train": r.train, "val": r.val,
        # What the run cost, as the baselines record it (see baselines/sweep.py). The GPU peak includes the split's standardised features,
        # which the sweep keeps on the device for all its probes; benchmark a single probe for the cost of the method alone.
        "fit_seconds": round(fit_seconds, 1), "score_seconds": round(score_seconds, 1), "scored_steps": steps,
        "score_ms_per_step": round(1000 * score_seconds / steps, 4), "n_params": sum(p.numel() for p in scorer.model.parameters()),
        "n_train_episodes": len(split.train), "state_mb": round(scorer.state_bytes() / 1e6, 2),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
        "peak_rss_gb": round(peak_rss_gb(), 2), "hardware": hardware_info(),
    }
    return scorer, outputs, info


def train_unit(unit: Unit, inputs: Inputs, settings: Settings, device: str, log: Callable[[str], None] = print,
               architectures: list[str] | None = None) -> int:
    """Train the probes of the unit (all 16 architectures, or only the named ones) that have no run yet; returns how many runs were written."""
    split = inputs.splits[unit.split]
    todo = [cfg for cfg in all_configs() if (architectures is None or cfg.name in architectures)
            and not is_done(run_dir(unit.dataset, unit.split, f"{cfg.name}__{unit.variant}", settings.out))]
    if not todo:
        return 0
    settings.out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    features = standardise_features(load_features(unit.dataset, list(inputs.episodes)), split.train, device)
    log(f"{unit.dataset}/{unit.split}/{unit.variant}: features ready in {time.time() - start:.0f}s, {len(todo)} probes to train")
    code = code_version()
    written = 0
    for cfg in todo:
        method = f"{cfg.name}__{unit.variant}"
        try:
            scorer, outputs, info = train_probe(cfg, unit.variant, split, inputs, features, settings, device)
            room = free_gb(settings.out) >= settings.min_free_gb
            weights, heads = settings.save_weights and room, settings.save_heads and room
            write_run(run_dir(unit.dataset, unit.split, method, settings.out), outputs["score"],
                      {**info, "code": code, "weights_saved": weights, "heads_saved": heads}, scorer.save if weights else None,
                      head_arrays(scorer, outputs) if heads else None)
            written += 1
            log(f"  {method}: {info['epochs']} epochs, best {info['best_epoch']}{' (stopped early)' if info['stopped_early'] else ''}, "
                f"val score {info['val']['score'][info['best_epoch']]:.4f}, {info['seconds']:.0f}s{'' if room else ', weights and heads NOT saved (disk)'}")
        except Exception:
            log(f"  FAILED {unit.dataset}/{unit.split}/{method}\n{traceback.format_exc()}")
        finally:
            scorer = None
            torch.cuda.empty_cache()
    return written
