"""Everything of a dataset that training and evaluation need besides the features: episodes, targets, splits and the on-time window."""

from dataclasses import dataclass

from .. import config
from .manifest import Episode, load_manifest
from .splits import Split, make_splits
from .targets import Targets, load_targets


@dataclass(frozen=True)
class Inputs:
    dataset: str
    episodes: dict[str, Episode]
    targets: Targets
    splits: dict[str, Split]
    window: int  # on-time window in policy steps


def load_inputs(dataset: str) -> Inputs:
    episodes = {e.key: e for e in load_manifest(dataset)}
    return Inputs(dataset, episodes, load_targets(dataset), {s.name: s for s in make_splits(dataset)}, config.get(dataset).window_steps)
