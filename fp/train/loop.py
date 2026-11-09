"""The single generic training loop.

Whole rollouts are batched (one gradient update per batch of ``batch`` rollouts, shuffled every epoch with a seeded generator) and the model
is trained with Adam. After every epoch the validation items are scored, and the weights of the epoch with the lowest value of one named
validation term (``select``) are restored at the end; training stops after ``patience`` epochs without a new best. Methods differ only in
the step function they pass in.

A step function returns the loss to backpropagate and, for reporting, every loss term as a (numerator, denominator) pair. Pairs are pooled
over the batches of an epoch (add both, divide once), so each reported value is the mean of the term over all its eligible steps of the
set, whatever the batch boundaries are.
"""

import copy
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 3e-4
    batch: int = 32  # whole rollouts per gradient update
    max_epochs: int = 100
    patience: int = 15
    seed: int = 0
    # If set, a batch is run through the model in chunks of this many rollouts and their gradients are accumulated, each chunk weighted by its
    # share of the batch. The update is the same as for the whole batch only if the step function's loss is a MEAN over the rollouts it is given
    # (plus terms that do not depend on the data, which a mean preserves), as SAFE's monitor loss with its L2 penalty is; it saves memory and
    # nothing else. Leave it None for any other loss.
    micro_batch: int | None = None


def default_lr(trunk: str) -> float:
    """3e-4 for the LSTM trunk, 1e-3 for the MLP trunk (the earlier experiments' values)."""
    return 3e-4 if trunk == "lstm" else 1e-3


@dataclass(frozen=True)
class Step:
    loss: torch.Tensor  # backpropagated in training
    parts: dict[str, tuple[float, float]]  # term name -> (numerator, denominator) over the batch, for reporting


@dataclass(frozen=True)
class FitResult:
    best_epoch: int  # 0-based
    train: dict[str, list[float]]  # per term and epoch: pooled mean over the epoch's training batches (the weights change meanwhile)
    val: dict[str, list[float]]  # per term and epoch: pooled mean over the validation items; ``val[select]`` decides the best epoch
    stopped_early: bool


def _add(pool: dict[str, list[float]], parts: dict[str, tuple[float, float]]) -> None:
    for name, (num, den) in parts.items():
        total = pool.setdefault(name, [0.0, 0.0])
        total[0], total[1] = total[0] + num, total[1] + den


def _means(pool: dict[str, list[float]]) -> dict[str, float]:
    """Numerator over denominator per term; NaN for a term that had no eligible step."""
    return {name: num / den if den > 0 else float("nan") for name, (num, den) in pool.items()}


def _evaluate(model: torch.nn.Module, items: Sequence, step_fn: Callable, batch: int) -> dict[str, float]:
    """Pooled mean of every term over all items; evaluation mode, no gradient."""
    model.eval()
    pool: dict[str, list[float]] = {}
    with torch.no_grad():
        for i in range(0, len(items), batch):
            _add(pool, step_fn(model, items[i : i + batch]).parts)
    return _means(pool)


def fit(model: torch.nn.Module, train: Sequence, val: Sequence, step_fn: Callable[[torch.nn.Module, Sequence], Step], cfg: TrainConfig,
        select: str = "score") -> FitResult:
    """Train ``model`` in place and leave it with the weights of the epoch with the lowest validation value of the term ``select``."""
    if not train or not val:
        raise ValueError("training and validation sets must not be empty")
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    best, best_epoch, best_state, bad, stopped = float("inf"), -1, None, 0, False
    train_hist: dict[str, list[float]] = {}
    val_hist: dict[str, list[float]] = {}
    for epoch in range(cfg.max_epochs):
        model.train()
        order = rng.permutation(len(train))
        pool: dict[str, list[float]] = {}
        for i in range(0, len(order), cfg.batch):
            batch = [train[j] for j in order[i : i + cfg.batch]]
            optimiser.zero_grad()
            for k in range(0, len(batch), cfg.micro_batch or len(batch)):
                chunk = batch[k : k + (cfg.micro_batch or len(batch))]
                step = step_fn(model, chunk)
                (step.loss * (len(chunk) / len(batch))).backward()  # a whole batch is one chunk with weight 1
                _add(pool, step.parts)
            optimiser.step()
        for hist, means in ((train_hist, _means(pool)), (val_hist, _evaluate(model, val, step_fn, cfg.batch))):
            for name, value in means.items():
                hist.setdefault(name, []).append(value)
        if select not in val_hist:
            raise ValueError(f"the step function reports no term {select!r} (it reports {sorted(val_hist)})")
        value = val_hist[select][-1]
        if np.isnan(value):
            raise ValueError(f"no validation step has the term {select!r}")
        if value < best:
            best, best_epoch, best_state, bad = value, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                stopped = True
                break
    model.load_state_dict(best_state)
    return FitResult(best_epoch, train_hist, val_hist, stopped)
