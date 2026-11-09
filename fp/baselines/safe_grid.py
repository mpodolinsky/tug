"""The configurations of SAFE-MLP and SAFE-LSTM that SAFE sweeps (scripts/batch_training/submit_pi0fast_libero.bash).

    SAFE-MLP:   learning rate in {1e-5, 1e-4, 3e-4, 1e-3}  x  lambda_reg in {1e-3, 1e-2, 1e-1}      12 configurations
    SAFE-LSTM:  learning rate in {3e-5, 1e-4, 3e-4, 1e-3}  x  lambda_reg in {1e-3, 1e-2, 1e-1}      12 configurations
SAFE also sweeps the feature (which embedding, which token) and the seed; here the feature is fixed (the pooled per-modality vector, raw)
and the seed is the split's.

Four families, each with its own name so that a selection never mixes them:
    safe_mlp, safe_lstm            SAFE's own labels (the whole failed rollout is positive)
    safe_mlp_pre, safe_lstm_pre    SAFE's loss on rollouts cut at the failure onset (our ``onset_pre``): NOT SAFE as published, an ablation of
                                   what the labels contribute
A method is named ``<family>__lr<learning rate>_reg<lambda_reg>``, e.g. ``safe_lstm__lr3e-4_reg1e-2``. The order of ``method_names`` is the
order ties are broken in when the best configuration is chosen: learning rate first, then lambda_reg, both ascending.
"""

import itertools
from typing import Mapping

import numpy as np

from ..data.manifest import Episode
from ..data.splits import Split
from ..train.safe import KINDS, SafeScorer
from .registry import Context, register

LEARNING_RATES = {"mlp": (1e-5, 1e-4, 3e-4, 1e-3), "lstm": (3e-5, 1e-4, 3e-4, 1e-3)}
LAMBDA_REGS = (1e-3, 1e-2, 1e-1)
FAMILIES = {"safe_mlp": ("mlp", "safe"), "safe_lstm": ("lstm", "safe"), "safe_mlp_pre": ("mlp", "pre"), "safe_lstm_pre": ("lstm", "pre")}


def grid_configs(kind: str) -> list[dict]:
    assert kind in KINDS
    return [dict(lr=lr, lambda_reg=reg) for lr, reg in itertools.product(LEARNING_RATES[kind], LAMBDA_REGS)]


def grid_scorers(family: str, episodes: Mapping[str, Episode], features: Mapping[str, np.ndarray], window: int, seed: int = 0,
                 device: str = "cpu") -> list[SafeScorer]:
    """One unfitted scorer per configuration, in grid order. They only point at the shared features; fit and release them one at a time."""
    kind, labels = FAMILIES[family]
    return [SafeScorer(kind, episodes, features, window, labels=labels, seed=seed, device=device, **cfg) for cfg in grid_configs(kind)]


def method_names(family: str) -> list[str]:
    """The methods of a family in grid order."""
    return [s.name for s in grid_scorers(family, {}, {}, 0)]


def _family(family: str):
    def build(context: Context, split: Split) -> list[SafeScorer]:
        return grid_scorers(family, context.inputs.episodes, context.features, context.inputs.window, split.seed, context.device)

    return build


for _family_name in FAMILIES:
    register(_family_name)(_family(_family_name))
