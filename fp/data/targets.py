"""Targets of the auxiliary heads of the probes, built from the labels of the raw datasets.

``data/<dataset>/targets.npz`` holds

    phrases             (P,)      str      sorted unique phrases of the semantic timelines of the whole dataset
    phrase_embeddings   (P, 512)  float32  L2-normalised CLIP ViT-B/32 text features of the phrases
    mode_names          (10,)     str      the failure-mode classes (MODE_NAMES)
    phrase_ids/<dir>    (T,)      int32    index into ``phrases`` for every policy step of the episode, -1 where there is none
    mode/<dir>          ()        int8     index into ``mode_names``; present for failed episodes only

where <dir> is the manifest ``dir`` of the episode. The phrase of a step is that of the timeline segment covering it (a later segment
overwrites a step shared with the previous one); steps after a segment that no segment covers repeat the previous step's phrase; steps
before the first segment stay -1. Episodes without a ``labels.json`` have an empty timeline.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import get
from .manifest import load_manifest

MODE_NAMES = ("wrong_object", "stuck_or_no_progress", "wrong_target", "object_displacement", "placement_or_insertion_failure",
              "grasp_failure", "open_close_failure", "press_failure", "other", "timeout_or_insufficient_progress")
CLIP_MODEL = "openai/clip-vit-base-patch32"


def phrase_ids(timeline: list[dict], n_steps: int, index: dict[str, int]) -> np.ndarray:
    """Phrase id of every policy step of an episode (see the module docstring for the rules)."""
    ids = np.full(n_steps, -1, dtype=np.int32)
    for segment in timeline:
        ids[segment["policy_step_start"] : segment["policy_step_end"] + 1] = index[segment["phrase"]]
    for t in range(1, n_steps):
        if ids[t] < 0 <= ids[t - 1]:
            ids[t] = ids[t - 1]
    return ids


def read_labels(episode_dir: Path) -> tuple[list[dict], str | None]:
    """(semantic timeline, failure mode) of an episode folder; an empty timeline and None when a file is missing."""
    labels_json, labels_npz = episode_dir / "labels.json", episode_dir / "labels.npz"
    timeline = json.load(open(labels_json))["semantic_timeline"] if labels_json.exists() else []
    mode = None
    if labels_npz.exists():
        with np.load(labels_npz) as z:
            mode = str(z["fail_mode"]) or None
    return timeline, mode


def clip_text_embeddings(phrases: list[str], model_name: str = CLIP_MODEL) -> np.ndarray:
    """L2-normalised CLIP text features (P, 512). Needs `transformers` and the model weights."""
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    tokenizer, model = CLIPTokenizer.from_pretrained(model_name), CLIPModel.from_pretrained(model_name).eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(phrases), 256):
            tokens = tokenizer(phrases[i : i + 256], return_tensors="pt", padding=True, truncation=True)
            chunks.append(torch.nn.functional.normalize(model.get_text_features(**tokens), dim=-1).numpy())
    return np.concatenate(chunks).astype(np.float32)


def build_targets(dataset: str, labels_root: str | Path, embed: Callable[[list[str]], np.ndarray] = clip_text_embeddings) -> Path:
    """Write data/<dataset>/targets.npz from the ``labels.json`` / ``labels.npz`` under ``labels_root/<dir>/``.

    A failed episode must have a failure mode from MODE_NAMES. An existing file is never overwritten.
    """
    out = get(dataset).data_dir / "targets.npz"
    if out.exists():
        raise FileExistsError(f"{out} already exists; remove it to build again")
    episodes = load_manifest(dataset)
    labels = {e.key: read_labels(Path(labels_root) / e.key) for e in episodes}

    problems = []
    for e in episodes:
        mode = labels[e.key][1]
        if not e.success and mode not in MODE_NAMES:
            problems.append((e.key, mode))
    if problems:
        raise ValueError(f"{len(problems)} failed episodes have no valid failure mode, e.g. {problems[:3]}")

    phrases = sorted({s["phrase"] for timeline, _ in labels.values() for s in timeline})
    index = {p: i for i, p in enumerate(phrases)}
    embeddings = embed(phrases)
    if embeddings.shape[0] != len(phrases):
        raise ValueError(f"{embeddings.shape[0]} embeddings for {len(phrases)} phrases")

    arrays = {"phrases": np.array(phrases), "phrase_embeddings": embeddings.astype(np.float32), "mode_names": np.array(MODE_NAMES)}
    for e in episodes:
        timeline, mode = labels[e.key]
        arrays[f"phrase_ids/{e.key}"] = phrase_ids(timeline, e.n_steps, index)
        if not e.success:
            arrays[f"mode/{e.key}"] = np.int8(MODE_NAMES.index(mode))
    tmp = out.with_name("targets.tmp.npz")
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


@dataclass(frozen=True)
class Targets:
    phrases: list[str]
    embeddings: np.ndarray  # (P, 512)
    phrase_ids: dict[str, np.ndarray]  # episode key -> (T,) int32
    modes: dict[str, int]  # failed episode key -> index into MODE_NAMES


def load_targets(dataset: str) -> Targets:
    path = get(dataset).data_dir / "targets.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    with np.load(path) as z:
        if tuple(z["mode_names"]) != MODE_NAMES:
            raise ValueError(f"{path.name} was built with other failure-mode names than MODE_NAMES")
        ids = {k[len("phrase_ids/") :]: z[k] for k in z.files if k.startswith("phrase_ids/")}
        modes = {k[len("mode/") :]: int(z[k]) for k in z.files if k.startswith("mode/")}
        return Targets([str(p) for p in z["phrases"]], z["phrase_embeddings"], ids, modes)
