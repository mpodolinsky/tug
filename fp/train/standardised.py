"""The features of a split, standardised once with the training episodes' statistics and kept on the device.

All probes trained on a split share one ``StandardisedFeatures``, so the 6144-d features of the split are standardised and copied to the
GPU once, not once per probe.
"""

from dataclasses import dataclass

import numpy as np
import torch

from ..data.features import fit_standardiser


@dataclass(frozen=True)
class StandardisedFeatures:
    mean: np.ndarray  # (d,) float64, over all steps of the training episodes
    std: np.ndarray  # (d,) float64, plus 1e-6
    train_keys: frozenset[str]  # the episodes the statistics come from
    arrays: dict[str, torch.Tensor]  # episode key -> (T, d) float32 on the device


def standardise_features(raw: dict[str, np.ndarray], train_keys: list[str], device: str | torch.device = "cpu") -> StandardisedFeatures:
    """Standardise every episode with the statistics of ``train_keys``. Consumes ``raw``: an array is dropped as soon as its
    standardised copy exists, so the raw and the standardised features are never both held in memory.
    The arithmetic is done in float64 and rounded to float32, as ``fp.data.features.standardise`` does."""
    mean, std = fit_standardiser([raw[k] for k in train_keys])
    mean_t, std_t = (torch.from_numpy(a).to(device) for a in (mean, std))
    arrays = {}
    for key in list(raw):
        x = torch.from_numpy(raw.pop(key)).to(device)
        arrays[key] = ((x.double() - mean_t) / std_t).float()
    return StandardisedFeatures(mean, std, frozenset(train_keys), arrays)
