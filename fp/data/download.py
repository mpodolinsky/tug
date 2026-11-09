"""Fetch a dataset's pooled features, targets, manifest and (LIBERO-X) split files from the Hugging Face Hub into data/<dataset>/.

The Hub repo (``DatasetSpec.hf_repo``) stores the pooled files as ``pooled/features.npz`` and ``pooled/targets.npz``; here they end up as
``data/<dataset>/features.npz`` and ``data/<dataset>/targets.npz``, where ``load_features`` and ``load_targets`` read them. The raw
rollouts are not downloaded.
"""

import os
from pathlib import Path
from typing import Callable

from ..config import DatasetSpec, get

POOLED_FILES = ("features.npz", "targets.npz")


def hub_patterns(spec: DatasetSpec) -> list[str]:
    """Files of the Hub repo that are needed: the pooled files, the manifest, and for LIBERO-X the held-out task lists."""
    patterns = [f"pooled/{name}" for name in POOLED_FILES] + ["manifest.csv"]
    if "task" in spec.split_kinds:
        patterns.append("splits/*")  # fnmatch: also matches the nested leave_*_out folders
    return patterns


def download_dataset(dataset: str, fetch: Callable | None = None) -> Path:
    """Download into data/<dataset>/ and return that folder. An existing features.npz or targets.npz is never overwritten.

    ``fetch`` has the signature of ``huggingface_hub.snapshot_download`` (the default); tests pass a stand-in.
    """
    spec = get(dataset)
    if not spec.hf_repo:
        raise ValueError(f"{dataset} has no Hub repo: its data is kept local (python -m cli.import_real)")
    folder = spec.data_dir
    existing = [folder / name for name in POOLED_FILES if (folder / name).exists()]
    if existing:
        raise FileExistsError(f"{existing[0]} already exists; remove it to download again")
    if fetch is None:
        from huggingface_hub import snapshot_download as fetch
    folder.mkdir(parents=True, exist_ok=True)
    fetch(repo_id=spec.hf_repo, repo_type="dataset", allow_patterns=hub_patterns(spec), local_dir=str(folder))
    for name in POOLED_FILES:
        downloaded = folder / "pooled" / name
        if not downloaded.exists():
            raise FileNotFoundError(f"{spec.hf_repo} has no pooled/{name}")
        os.replace(downloaded, folder / name)
    (folder / "pooled").rmdir()
    return folder
