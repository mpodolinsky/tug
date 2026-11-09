"""The real-robot dataset: collected rollouts and their labels -> data/<dataset>/ (manifest, pooled features, targets).

The source folder (``outputs/gr00t-real8``) holds

    hf_labels_pr/manifest.csv                 one row per episode; ``dir`` is <task>/ep<NNN>
    hf_labels_pr/<dir>/labels.json, labels.npz   the Gemini labels (success flag, onset, failure mode, semantic timeline)
    cache/modality/features_pooled.npz        the pooled features, (n_policy, 6144) per episode, keyed by ``dir``

The token embeddings of the real rollouts were not kept (a ``rollout.npz`` holds only the actions), so the features cannot be pooled again as
the simulator datasets are (fp.data.pooling); they are taken from the cache made when the rollouts were collected, and every array is checked.
Two columns of the source manifest are empty: ``task`` is filled from the first part of ``dir``, and the condition (``scene_variant``) is set to
the task, so that the stratified split puts every task into every part.
"""

import csv
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import FEATURE_DIM, get
from .features import write_arrays
from .manifest import load_manifest
from .targets import MODE_NAMES, build_targets, clip_text_embeddings, read_labels

MANIFEST = "hf_labels_pr/manifest.csv"
LABELS = "hf_labels_pr"
FEATURES = "cache/modality/features_pooled.npz"
OUTPUTS = ("manifest.csv", "features.npz", "targets.npz")


def read_source_manifest(path: Path) -> tuple[list[dict], list[str]]:
    """The rows of the source manifest with ``task`` and ``scene_variant`` filled in, and the column names (both columns present)."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = list(reader)
    for column in ("task", "scene_variant"):
        if column not in fields:
            fields.append(column)
    for r in rows:
        parts = r["dir"].split("/")
        if len(parts) != 2:
            raise ValueError(f"the dir {r['dir']!r} is not <task>/<episode>")
        r["task"], r["scene_variant"] = parts[0], parts[0]
    return rows, fields


def check_labels(rows: list[dict], labels_root: Path) -> list[str]:
    """Problems found in the labels of every episode: the success flag differs from the manifest's; a failure's onset step in labels.json is not
    the manifest's frame divided by its replan steps; a failure has no mode of MODE_NAMES; an episode has no semantic timeline."""
    problems = []
    for r in rows:
        folder = labels_root / r["dir"]
        if not (folder / "labels.json").exists() or not (folder / "labels.npz").exists():
            problems.append(f"{r['dir']}: labels.json or labels.npz is missing")
            continue
        labels = json.load(open(folder / "labels.json"))
        timeline, mode = read_labels(folder)
        success = r["success"] == "True"
        if labels["success"] != success:
            problems.append(f"{r['dir']}: success is {success} in the manifest and {labels['success']} in labels.json")
        if not timeline:
            problems.append(f"{r['dir']}: no semantic timeline")
        if success:
            continue
        if mode not in MODE_NAMES:
            problems.append(f"{r['dir']}: failure mode {mode!r}")
        step = (labels.get("failure_annotation") or {}).get("failure_policy_step")
        frame = r["vlm_failure_onset_frame"].strip()
        if not frame or step != int(float(frame)) // int(r["replan_steps"]):
            problems.append(f"{r['dir']}: onset step {step} in labels.json, frame {frame!r} with {r['replan_steps']} control steps per call in the manifest")
    return problems


def checked_features(rows: list[dict], cache: Path):
    """(dir, array) for every episode in manifest order; raises on a missing episode, a wrong shape or dtype, or a non-finite value."""
    with np.load(cache) as z:
        absent = [r["dir"] for r in rows if r["dir"] not in z.files]
        if absent:
            raise KeyError(f"{len(absent)} episodes are not in {cache.name}, e.g. {absent[:3]}")
        for r in rows:
            x = z[r["dir"]]
            if x.shape != (int(r["n_policy"]), FEATURE_DIM) or x.dtype != np.float32 or not np.isfinite(x).all():
                raise ValueError(f"{r['dir']}: features of shape {x.shape} and dtype {x.dtype} (or non-finite values); expected finite ({r['n_policy']}, {FEATURE_DIM}) float32")
            yield r["dir"], x


def import_real(dataset: str, source: str | Path, embed: Callable[[list[str]], np.ndarray] = clip_text_embeddings) -> Path:
    """Write data/<dataset>/manifest.csv, features.npz and targets.npz from ``source``. Nothing existing is overwritten; if a step fails, what this
    call wrote is removed again. ``embed`` embeds the timeline phrases (CLIP, needs `transformers`)."""
    out, source = get(dataset).data_dir, Path(source)
    existing = [out / name for name in OUTPUTS if (out / name).exists()]
    if existing:
        raise FileExistsError(f"{existing[0]} already exists; remove it to import again")
    rows, fields = read_source_manifest(source / MANIFEST)
    problems = check_labels(rows, source / LABELS)
    if problems:
        raise ValueError(f"{len(problems)} problems in the labels, e.g. {problems[:3]}")
    out.mkdir(parents=True, exist_ok=True)
    written = []
    try:
        tmp = out / "manifest.csv.tmp"
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, out / "manifest.csv")
        written.append(out / "manifest.csv")
        load_manifest(dataset)  # the onset of every failure lies inside its rollout
        write_arrays(out / "features.npz", checked_features(rows, source / FEATURES))
        written.append(out / "features.npz")
        build_targets(dataset, source / LABELS, embed)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return out
