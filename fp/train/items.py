"""One rollout as a training example, and padded batches of them.

``make_item`` applies a label variant (see ``fp.data.labels``): the rollout is cut where the variant cuts it and every kept step gets its 0/1
training label. The phrase target of a step is its timeline phrase (-1 if none); the mode target belongs to the whole failed episode and is
applied, in ``collate``, on the steps whose training label is 1 only.
"""

from dataclasses import dataclass

import numpy as np
import torch

from ..data.labels import build_labels
from ..data.manifest import Episode


@dataclass(frozen=True)
class Item:
    x: torch.Tensor  # (T, d) standardised features of the kept steps
    y: torch.Tensor  # (T,) float32 training label
    phrase: torch.Tensor  # (T,) int64 index into the phrases, -1 where there is none
    mode: int  # failure-mode index of a failed episode, -1 for a success


@dataclass(frozen=True)
class Batch:
    x: torch.Tensor  # (B, T, d) zero-padded
    y: torch.Tensor  # (B, T)
    valid: torch.Tensor  # (B, T) bool, False on padding
    phrase: torch.Tensor  # (B, T) int64, -1 on padding and where there is none
    mode: torch.Tensor  # (B, T) int64, the episode's mode on steps with y == 1 of a failed episode, -1 elsewhere


def make_item(episode: Episode, x: torch.Tensor, phrase_ids: np.ndarray | None, mode: int | None, variant: str, window: int) -> Item:
    """``x`` and ``phrase_ids`` cover all steps of the episode; the item keeps the steps the label variant keeps (``x`` is not copied).
    ``phrase_ids`` may be None for a model that has no semantic target: every step then has no phrase."""
    labels = build_labels(variant, episode.n_steps, episode.onset, window)
    keep = labels.n
    phrase = np.full(keep, -1, dtype=np.int64) if phrase_ids is None else phrase_ids[:keep].astype(np.int64)
    return Item(x[:keep], torch.from_numpy(labels.y), torch.from_numpy(phrase), -1 if mode is None else int(mode))


def collate(items: list[Item], device: str | torch.device = "cpu") -> Batch:
    """Pad to the longest item, building the batch on ``device`` (the items' features may already live there)."""
    n, steps, d = len(items), max(len(it.y) for it in items), items[0].x.shape[1]
    x, y = torch.zeros(n, steps, d, device=device), torch.zeros(n, steps, device=device)
    valid = torch.zeros(n, steps, dtype=torch.bool, device=device)
    phrase, mode = (torch.full((n, steps), -1, dtype=torch.long, device=device) for _ in range(2))
    for i, it in enumerate(items):
        k = len(it.y)
        x[i, :k], y[i, :k], valid[i, :k], phrase[i, :k] = it.x, it.y, True, it.phrase
        if it.mode >= 0:
            mode[i, :k] = torch.where(it.y == 1, it.mode, -1)
    return Batch(x, y, valid, phrase, mode)
