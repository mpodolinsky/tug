"""Push the outputs of the embedding-distance baselines to the Hub dataset repos, next to the pooled features.

    python scripts/push_distance_to_hub.py            # dry run: what would be uploaded, and the text of the README and the card section
    python scripts/push_distance_to_hub.py --push     # upload (needs a Hub login with write access to the repos)

Per dataset repo this adds ``baselines/distance/<split>/<method>/{scores.npz, distances.npz, history.json}`` for the 26 configurations of the
three distance baselines, a README for that folder, and a short section in the dataset card. The code that computes the distances is not
uploaded (it is derived from SAFE's, which has no licence); the README states the definitions and gives a NumPy snippet that rebuilds every
score from the saved distances. Run it with an interpreter that has ``huggingface_hub`` and the repository on its path.
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fp.config import RESULTS, get  # noqa: E402

DATASETS = ("liberox_groot", "libero10_groot", "libero10_pi05")
FAMILIES = ("euclidean", "mahalanobis", "pca_kmeans")
FOLDER = "baselines/distance"
CARD_MARKER = "## Embedding-distance baseline outputs"

README = """# Embedding-distance baselines: per-step distances and scores

Outputs of three embedding-distance failure-detection baselines (Euclidean k-NN, Mahalanobis, PCA-KMeans) with the 26 configurations of SAFE's
sweep (Gu et al., NeurIPS 2025, arXiv 2506.09937), computed on the pooled per-modality features of this dataset (`pooled/features.npz`,
6,144-d: base, wrist and language means, **raw**, not standardised) for every split of the dataset. They serve as baselines for the
failure-detection probes evaluated on this dataset.

## Layout
`baselines/distance/<split>/<method>/` with

| file | content |
|---|---|
| `scores.npz` | the baseline's failure score at every step of every episode of the dataset (key = episode key from `manifest.csv`), float32 |
| `distances.npz` | per episode a `(T, 2)` array: the distance of each step to the **success** set (column 0) and to the **failure** set (column 1) |
| `history.json` | configuration, seed, split hash, seconds to fit and to score, peak memory, size of the fitted state, hardware |

A method is named `<family>__<tag>`, e.g. `euclidean__k10_cumsum`, `mahalanobis__step`, `pca_kmeans__d64_c32_cumsum`. Splits are the ones of the
dataset: `paper_seed{{0,1,2}}` for LIBERO-10, and the task, signature and skill splits for LIBERO-X (see `splits/`). A baseline is fitted on the
training episodes of its split and scores all episodes; calibrate it and evaluate it on the split's calibration, seen and unseen episodes.

## Definitions
Every step of a training episode labelled success is a success step, every step of a failed training episode a failure step. For a step with feature
x, `D_c(x)` is, for class c: **Euclidean** the mean distance to the k nearest stored steps of class c (k in 1, 5, 10); **Mahalanobis** the
Mahalanobis distance to the mean and covariance of class c (covariance plus 1e-4 I, pseudo-inverse); **PCA-KMeans** the distance to the nearest of
`n_clusters` (16, 32, 64) k-means centroids of class c in a `pca_dim` (32, 64, 128) dimensional PCA space. The step score is `D_success(x) - D_failure(x)`
(`step`), or its running sum over the episode (`cumsum`).

## Rebuilding a score from the saved distances
```python
import numpy as np

d = np.load("distances.npz")
episode = d["<episode key>"]            # (T, 2)
step = episode[:, 0] - episode[:, 1]    # the step score of every *_step method
cumulative = np.cumsum(step)            # the score of every *_cumsum method
success_distance_only = episode[:, 0]   # SAFE's `use_success_only` variant (not in the sweep, available from the saved distances)
```
`scores.npz` holds the result for the method's own `step`/`cumsum` setting.

## Notes
- The distance functions are SAFE's (`EmbedModel`); PCA-KMeans projects every step with the **success** PCA for both distances, as SAFE does.
- The raw 6,144-d features are used as they are, as SAFE's embedding baselines use their embeddings; no standardisation.
- Runs were made on one machine (NVIDIA RTX 4090); `history.json` records the times.
"""

CARD_SECTION = """{marker}

`baselines/distance/` holds the per-step scores and distances of three embedding-distance baselines (Euclidean k-NN, Mahalanobis, PCA-KMeans; 26
configurations each split, SAFE's sweep) computed on `pooled/features.npz` for every split of this dataset, with a README that defines them and shows
how to rebuild each score from the saved distances.
"""


def runs(dataset: str) -> list[Path]:
    return sorted(p for p in (RESULTS / dataset).glob("*/*") if p.name.startswith(tuple(f"{f}__" for f in FAMILIES)) and (p / "history.json").exists())


def patterns() -> list[str]:
    return [f"*/{family}__*/{name}" for family in FAMILIES for name in ("scores.npz", "distances.npz", "history.json")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--push", action="store_true", help="upload; without it only list what would be uploaded")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    args = p.parse_args()
    for dataset in args.datasets:
        files = [f for run in runs(dataset) for f in (run / "scores.npz", run / "distances.npz", run / "history.json")]
        size = sum(f.stat().st_size for f in files) / 1e9
        print(f"{dataset} -> {get(dataset).hf_repo}: {len(runs(dataset))} runs, {len(files)} files, {size:.2f} GB, into {FOLDER}/")
    if not args.push:
        print("\n--- README.md of baselines/distance/ ---\n" + README.format() + "\n--- section added to the dataset card ---\n" + CARD_SECTION.format(marker=CARD_MARKER))
        print("dry run: nothing uploaded")
        return
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    for dataset in args.datasets:
        repo = get(dataset).hf_repo
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "README.md").write_text(README.format())
            api.upload_file(path_or_fileobj=str(Path(tmp) / "README.md"), path_in_repo=f"{FOLDER}/README.md", repo_id=repo, repo_type="dataset",
                            commit_message="baselines/distance: README")
        api.upload_folder(folder_path=str(RESULTS / dataset), path_in_repo=FOLDER, repo_id=repo, repo_type="dataset", allow_patterns=patterns(),
                          commit_message="baselines/distance: scores and distances of the embedding-distance baselines")
        card = Path(hf_hub_download(repo, "README.md", repo_type="dataset")).read_text()
        if CARD_MARKER not in card:
            api.upload_file(path_or_fileobj=(card.rstrip() + "\n\n" + CARD_SECTION.format(marker=CARD_MARKER)).encode(), path_in_repo="README.md", repo_id=repo,
                            repo_type="dataset", commit_message="card: embedding-distance baseline outputs")
        print(f"pushed {dataset} -> {repo}")


if __name__ == "__main__":
    main()
