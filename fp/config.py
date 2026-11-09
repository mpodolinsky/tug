"""Dataset registry. The only place where per-dataset constants live (no environment variables).

The on-time window is given in policy steps and converted to seconds as
``window_steps * replan_steps / control_hz`` so that the two time bases stay explicit.
"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# Per-modality pooling: [base | wrist | language], 2048 each.
MODALITY_DIM = 2048
FEATURE_DIM = 3 * MODALITY_DIM


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    policy: str  # "groot" | "pi05"
    suite: str  # "libero10_occ" | "liberox_level123" | "real8" (real robot)
    replan_steps: int  # control steps executed per policy call
    control_hz: float
    window_steps: int  # on-time window, in policy steps
    has_actions: bool  # the whole planned action chunk of every call is saved (LIBERO: K = 10 sampled chunks); the action-based baselines need it
    split_kinds: tuple[str, ...]
    hf_repo: str  # Hugging Face dataset with the raw per-token embeddings, labels, manifest and (LIBERO-X) split files; empty if the data is kept local

    @property
    def window_seconds(self) -> float:
        return self.window_steps * self.replan_steps / self.control_hz

    @property
    def data_dir(self) -> Path:
        return DATA / self.name

    @property
    def results_dir(self) -> Path:
        return RESULTS / self.name


REGISTRY: dict[str, DatasetSpec] = {
    s.name: s
    for s in (
        DatasetSpec("libero10_groot", "groot", "libero10_occ", 8, 20.0, 6, True, ("paper",), "podolinsky/groot-libero-10-features-v3"),
        DatasetSpec("libero10_pi05", "pi05", "libero10_occ", 5, 20.0, 10, True, ("paper",), "podolinsky/pi0.5-libero-10-features-v3"),
        DatasetSpec("liberox_groot", "groot", "liberox_level123", 8, 20.0, 6, True, ("task", "signature", "skill"), "podolinsky/gr00t-libero-x-features-v3"),
        # real robot, 8 tasks: about 50 Hz, 6 control steps per call, the 2.4 s window of the other GR00T datasets (20 calls); the saved plans hold only
        # the executed actions (the rest is NaN), so there are no action chunks; the data stays local
        DatasetSpec("real8_groot", "groot", "real8", 6, 50.0, 20, False, ("loto", "full", "lotoseeds"), ""),
    )
}


def get(name: str) -> DatasetSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name]
