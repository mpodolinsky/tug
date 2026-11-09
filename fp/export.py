"""Export of a saved probe run for the Hub, shared by the scripts that push models (scripts/push_models_to_hub.py, push_real_model_to_hub.py).

A run's ``weights.pt`` is a pickled dict (our own format); the Hub gets ``model.safetensors`` instead: the weights, the feature mean and std the
probe was trained with, and ``calibration_maxima``, the sorted maxima of the scores of the split's calibration successes, from which the alarm
threshold of any level is read (``fp.calibration.max_conformal``). ``verify_tensors`` reads the file back before anything is uploaded.
Needs ``safetensors`` (the ``hub`` extra) and, for ``push_folder``, ``huggingface_hub``.
"""

from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .data.manifest import Episode
from .data.splits import Split
from .models.ours import Probe, ProbeConfig

EXTRA_TENSORS = ("feature_mean", "feature_std", "calibration_maxima")


def load_checkpoint(run: Path) -> dict:
    """The dict of a run's weights.pt: cfg, variant, state, mean, std, best_epoch. A pickle: only for files this code wrote."""
    return torch.load(Path(run) / "weights.pt", map_location="cpu", weights_only=False)


def calibration_maxima(run: Path, split: Split, episodes: Mapping[str, Episode]) -> np.ndarray:
    """Sorted per-episode maximum score of the split's calibration successes, from the run's scores.npz."""
    scores = np.load(Path(run) / "scores.npz")
    return np.sort([float(scores[k].max()) for k in split.calibration if episodes[k].success])


def write_tensors(saved: dict, maxima: np.ndarray, out_dir: Path) -> dict[str, torch.Tensor]:
    """Write out_dir/model.safetensors from a checkpoint and the calibration maxima; returns the tensors written."""
    from safetensors.torch import save_file

    clash = set(saved["state"]) & set(EXTRA_TENSORS)
    if clash:
        raise ValueError(f"the state dict already has tensors named {sorted(clash)}")
    tensors = {name: value.detach().cpu().contiguous() for name, value in saved["state"].items()}
    tensors.update({"feature_mean": torch.from_numpy(np.asarray(saved["mean"], dtype=np.float64)), "feature_std": torch.from_numpy(np.asarray(saved["std"], dtype=np.float64)),
                    "calibration_maxima": torch.from_numpy(np.asarray(maxima, dtype=np.float64))})
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    return tensors


def verify_tensors(saved: dict, out_dir: Path, label: str) -> None:
    """Read model.safetensors back: the tensors equal the checkpoint's, and a probe built from them gives the original outputs. Raises on any difference."""
    from safetensors.torch import load_file

    loaded = load_file(str(out_dir / "model.safetensors"))
    state = {k: v for k, v in loaded.items() if k not in EXTRA_TENSORS}
    if set(state) != set(saved["state"]) or not all(torch.equal(state[k], saved["state"][k]) for k in state):
        raise ValueError(f"{label}: the weights read back differ from the saved ones")
    if not (np.array_equal(loaded["feature_mean"].numpy(), saved["mean"]) and np.array_equal(loaded["feature_std"].numpy(), saved["std"])):
        raise ValueError(f"{label}: the feature statistics read back differ")
    cfg = ProbeConfig(**saved["cfg"])
    original, restored = Probe(cfg).eval(), Probe(cfg).eval()
    original.load_state_dict(saved["state"])
    restored.load_state_dict(state)
    x = torch.randn(2, 5, cfg.d_in, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        a, b = original(x), restored(x)
    if not all(torch.equal(a[k], b[k]) for k in a):
        raise ValueError(f"{label}: the probe built from the file gives other outputs")


def hub_api():
    from huggingface_hub import HfApi

    return HfApi()


def push_folder(folder: Path, repo: str, message: str, private: bool, api=None) -> list[str]:
    """Create the model repo if needed (``private`` only decides a new repo's visibility), upload ``folder`` to it in one commit, and return the
    files the Hub lists afterwards."""
    api = api or hub_api()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type="model", commit_message=message)
    return sorted(api.list_repo_files(repo, repo_type="model"))
