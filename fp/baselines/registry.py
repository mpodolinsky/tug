"""Name -> scorers for every baseline family.

A family is run on one split of one dataset: ``build(name, context, split)`` returns its scorers (the distance baselines return one per
configuration of SAFE's sweep, in grid order; a single-model family returns one). Which calibrator a method gets is decided in one place,
``calibration/protocol.py``; the registry only builds scorers.

A baseline that cannot be built for a dataset raises ``NotImplementedError`` with the reason, and the driver records the family as
skipped, so nothing is dropped silently. A family registers itself when its module is imported (``baselines/sweep.py`` imports them all).
"""

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .. import config
from ..data.inputs import Inputs
from ..data.splits import Split
from ..models.base import Scorer


@dataclass(frozen=True)
class Context:
    """What baselines are built from: loaded once per dataset and shared by all its splits and families."""

    inputs: Inputs
    features: Mapping[str, np.ndarray]  # RAW pooled features (T, 6144) float32 of every episode, as the SAFE baselines use them
    device: str = "cpu"
    actions: Mapping[str, np.ndarray] | None = None  # the sampled action chunks (T, K, H, 7) of every episode, for the action-based baselines


FAMILIES: dict[str, Callable[[Context, Split], list[Scorer]]] = {}
NEEDS_ACTIONS: set[str] = set()  # the families that read data/<dataset>/actions.npz


def register(name: str, needs_actions: bool = False):
    def wrap(factory):
        FAMILIES[name] = factory
        if needs_actions:
            NEEDS_ACTIONS.add(name)
        return factory

    return wrap


def build(name: str, context: Context, split: Split) -> list[Scorer]:
    if name not in FAMILIES:
        raise KeyError(f"unknown baseline family {name!r}; registered: {sorted(FAMILIES)}")
    if name in NEEDS_ACTIONS and not config.get(context.inputs.dataset).has_actions:
        raise NotImplementedError(f"{name} needs the planned action chunks of every call, which {context.inputs.dataset} does not have (its saved plans hold only the executed actions)")
    return FAMILIES[name](context, split)
