"""t-SNE of the embeddings of the VLA and of the probe, computed once and saved with everything needed to colour the points later.

Two embeddings of a dataset, one point per policy step of every episode:

    vla     the pooled 6144-d embedding the probe reads (agent camera | wrist camera | language), the same for every split
    probe   the 256-d state of the headline probe's LSTM (the latent space its score, mode and CLIP heads read), run over every episode; one probe per split, since a
            probe is trained on one split's training episodes

Each is standardised per dimension over all its points, reduced to ``PCA_DIM`` dimensions by a PCA fitted on a random sample of the points (and applied to all of
them), and embedded in two dimensions by openTSNE (the FFT-accelerated t-SNE, which handles the 235,130 points of LIBERO-X). The result is saved as ``points.npz``:
the 2D coordinates and, for every point, its episode and step, and for every episode its task, skill, outcome, failure step and failure mode, so that every figure
(``fp.eval.tsne_plots``) is drawn from the saved points and a new colouring never repeats the t-SNE. The subsets of a split (all episodes, the seen tasks, the held-out
episodes of the seen tasks, the unseen tasks) are masks over the same points (``subset_masks``): one embedding, so the panels are comparable.
"""

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from ..config import RESULTS
from ..data.features import load_features, standardise
from ..data.splits import Split
from ..results import run_dir

PCA_DIM = 50
PERPLEXITY = 50
PCA_FIT_POINTS = 40_000  # the PCA is fitted on this many random points
CHUNK = 20_000  # rows handled at a time, so that the float64 arithmetic never holds more than a few hundred MB
SUBSETS = ("all", "seen", "seen_held_out", "unseen")
FIT_SETS = ("all", "train_pre")  # the rows the axes of a PCA are fitted on: every point, or the pre-failure steps of the training episodes
LAYERS = ("hidden", "mode_hidden", "score_hidden", "clip")  # the layers of the probe whose activations are embedded


def episode_skills(inputs) -> dict[str, str]:
    """The skill of every episode of a dataset that has skill splits: the fold that holds its task out (empty for the other episodes)."""
    skills = {k: "" for k in inputs.episodes}
    for name, split in inputs.splits.items():
        if name.startswith("skill_"):
            for key in split.unseen:
                skills[key] = name.removeprefix("skill_")
    return skills


def subset_masks(keys: list[str], split: Split) -> dict[str, np.ndarray]:
    """{subset: boolean mask over ``keys``} of a split: ``all`` the episodes, ``seen`` those of the seen tasks (everything but the held-out tasks), ``seen_held_out`` those
    of the seen tasks that no probe was trained or early-stopped on (the calibration and seen test episodes) and ``unseen`` those of the held-out tasks."""
    unseen, held_out = set(split.unseen), set(split.calibration) | set(split.seen)
    return {"all": np.ones(len(keys), bool), "seen": np.array([k not in unseen for k in keys]), "seen_held_out": np.array([k in held_out for k in keys]),
            "unseen": np.array([k in unseen for k in keys])}


def progress(points: dict) -> np.ndarray:
    """(N,) position of every point on its episode's colour ramp: for a success step / (steps - 1); for a failure step / failure step, 1 from the failure step on."""
    ep, step = points["episode"], points["step"].astype(np.float64)
    onset, n_steps, success = points["onset"][ep], points["n_steps"][ep], points["success"][ep]
    to_failure = np.where(onset > 0, np.minimum(step / np.maximum(onset, 1), 1.0), 1.0)
    return np.where(success, step / np.maximum(n_steps - 1, 1), to_failure)


def states(dataset: str, keys: list[str], embedding: str, split: str | None = None, method: str | None = None, root=RESULTS, device: str = "cpu", layer: str = "hidden"):
    """(the states of every step of the episodes ``keys`` as one (N, d) matrix, the episode index and the step of each row): the pooled VLA embeddings (``embedding`` vla) or the
    ``layer`` (``LAYERS``) of the probe ``method`` trained on ``split`` (probe), whose weights are under ``root``."""
    if embedding == "vla":
        return vla_points(dataset, keys)
    raw = load_features(dataset, keys)
    lengths = [len(raw[k]) for k in keys]
    x = probe_states(run_dir(dataset, split, method, root) / "weights.pt", raw, keys, device, layer)
    return x, np.repeat(np.arange(len(keys), dtype=np.int32), lengths), np.concatenate([np.arange(n, dtype=np.int32) for n in lengths])


def every_kth_step(step: np.ndarray, k: int) -> np.ndarray:
    """Mask over the points that keeps the steps 0, k, 2k, ... of every episode: consecutive steps are almost the same point, so a t-SNE of all of them links each episode to
    itself and a perplexity below the length of an episode finds no structure across episodes."""
    if k < 1:
        raise ValueError(f"every must be at least 1, got {k}")
    return step % k == 0


def episode_means(x: np.ndarray, episode: np.ndarray, table: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One point per episode: the mean of its rows up to the failure step (the whole episode for a success), as ((E, d) means, episode index, the last step averaged). The rows
    of an episode are contiguous and in the order of its steps, as ``vla_points`` and ``probe_states`` give them. The last step averaged is where the colour ramps of
    ``fp.eval.tsne_plots`` end: dark blue for a success, red or the colour of the mode for a failure."""
    last = np.where(table["onset"] >= 0, table["onset"], table["n_steps"] - 1).astype(np.int32)
    starts = np.searchsorted(episode, np.arange(len(last)))
    means = np.stack([x[a : a + n + 1].mean(axis=0, dtype=np.float64) for a, n in zip(starts, last)]).astype(np.float32)
    return means, np.arange(len(last), dtype=np.int32), last


def train_mask(keys: list[str], split: Split) -> np.ndarray:
    """Boolean mask over ``keys``: the training episodes of ``split`` (the ones the probe was fitted on and, with ``--fit-set train_pre``, the PCA axes)."""
    train = set(split.train)
    return np.array([k in train for k in keys])


def standardise_all(x: np.ndarray, fit: np.ndarray | None = None) -> np.ndarray:
    """Every column to zero mean and unit standard deviation (plus 1e-6), in place, with float64 sums over chunks of rows. The statistics are those of the rows of the boolean mask
    ``fit`` (default: all rows); every row is standardised with them."""
    total = square = 0.0
    count = 0
    for i in range(0, len(x), CHUNK):
        chunk = x[i : i + CHUNK] if fit is None else x[i : i + CHUNK][fit[i : i + CHUNK]]
        chunk = chunk.astype(np.float64)
        total, square, count = total + chunk.sum(axis=0), square + (chunk**2).sum(axis=0), count + len(chunk)
    if count == 0:
        raise ValueError("no row to fit the standardisation on")
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean**2, 0.0)) + 1e-6
    for i in range(0, len(x), CHUNK):
        x[i : i + CHUNK] = standardise(x[i : i + CHUNK], mean, std)
    return x


def pca_reduce(x: np.ndarray, components: int, seed: int = 0, fit_points: int = PCA_FIT_POINTS, fit: np.ndarray | None = None):
    """(the rows of ``x`` on the first ``components`` principal axes as float32, the fitted PCA): ``x`` is standardised in place and the PCA is fitted on ``fit_points`` random rows;
    with the boolean mask ``fit`` both use only those rows (the axes then do not depend on the others, which are projected onto them like all rows)."""
    from sklearn.decomposition import PCA

    x = standardise_all(x, fit)
    if fit is None:
        sample = x[np.random.default_rng(seed).choice(len(x), size=min(fit_points, len(x)), replace=False)]
    else:
        rows = np.flatnonzero(fit)
        sample = x[rows[np.random.default_rng(seed).choice(len(rows), size=min(fit_points, len(rows)), replace=False)]]
    pca = PCA(n_components=min(components, x.shape[1], len(sample)), random_state=seed).fit(sample)
    return np.concatenate([pca.transform(x[i : i + CHUNK]) for i in range(0, len(x), CHUNK)]).astype(np.float32), pca


def fit_rows(keys: list[str], episode: np.ndarray, step: np.ndarray, onset: np.ndarray, split: Split, which: str) -> np.ndarray | None:
    """The rows the axes of a PCA are fitted on (``FIT_SETS``): None for ``all`` (every point, the first default), for ``train_pre`` a boolean mask over the points of the training
    episodes of ``split``, up to the failure step of a failure (``onset`` is the failure step of each episode, negative for a success) and every step of a success: the episodes and
    steps the probe was trained on, so that the unseen episodes play no part in the axes."""
    if which not in FIT_SETS:
        raise ValueError(f"unknown fit set {which!r}; choose from {FIT_SETS}")
    if which == "all":
        return None
    train = set(split.train)
    own = onset[episode]
    return np.array([k in train for k in keys])[episode] & ((own < 0) | (step <= own))


def pca_axes(x: np.ndarray, n: int = 2, seed: int = 0, fit_points: int = PCA_FIT_POINTS, fit: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """((N, n) the rows of ``x`` on the first ``n`` principal axes, the share of the variance of the (standardised) rows each of them holds, of the ``fit`` rows if given): a
    projection that is linear, so that states that are close stay close and an episode traces a path."""
    reduced, pca = pca_reduce(x, n, seed, fit_points, fit)
    return reduced, pca.explained_variance_ratio_


def embed_2d(x: np.ndarray, seed: int = 0, perplexity: float = PERPLEXITY, pca_dim: int = PCA_DIM, fit_points: int = PCA_FIT_POINTS, n_jobs: int = 8,
             verbose: bool = False) -> np.ndarray:
    """(N, 2) t-SNE of the rows of ``x`` (float32; standardised in place), after reducing to ``pca_dim`` dimensions with a PCA fitted on ``fit_points`` random rows."""
    from openTSNE import TSNE

    reduced, _ = pca_reduce(x, pca_dim, seed, fit_points)
    return np.asarray(TSNE(n_components=2, perplexity=perplexity, n_jobs=n_jobs, random_state=seed, initialization="pca", verbose=verbose).fit(reduced), dtype=np.float32)


def vla_points(dataset: str, keys: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The pooled embeddings of every step of the episodes ``keys`` as one (N, 6144) matrix, with the episode index and the step of each row."""
    raw = load_features(dataset, keys)
    lengths = [len(raw[k]) for k in keys]
    x = np.empty((sum(lengths), raw[keys[0]].shape[1]), np.float32)
    start = 0
    for k, n in zip(keys, lengths):
        x[start : start + n] = raw.pop(k)
        start += n
    return x, np.repeat(np.arange(len(keys), dtype=np.int32), lengths), np.concatenate([np.arange(n, dtype=np.int32) for n in lengths])


def layer_state(model, h, layer: str):
    """The activations of ``layer`` of a probe whose trunk state is ``h`` (B, T, hidden): ``hidden`` the trunk state itself, ``mode_hidden`` and ``score_hidden`` the 64 units of
    the hidden layer of the mode head and of the score head (after their ReLU, before the 10 mode logits and before the failure logit), ``clip`` the 64-d output of the CLIP head."""
    if layer not in LAYERS:
        raise ValueError(f"unknown layer {layer!r}; choose from {LAYERS}")
    if layer == "hidden":
        return h
    if layer == "score_hidden":
        return model.score[:2](h)
    if layer == "mode_hidden":
        if model.mode is None:
            raise ValueError("this probe has no mode head")
        return model.mode[:2](h)
    if model.sem is None:
        raise ValueError("this probe has no CLIP head")
    return model.sem(h)


def probe_states(weights: Path, raw: Mapping[str, np.ndarray], keys: list[str], device: str = "cpu", layer: str = "hidden") -> np.ndarray:
    """The activations of ``layer`` (``layer_state``; the 256-d LSTM state by default) at every step of the episodes ``keys`` as one (N, d) matrix: the saved probe is run over each
    episode (forward only) with the training statistics it was trained with (stored in its checkpoint)."""
    import torch

    from ..models.ours import Probe, ProbeConfig

    saved = torch.load(weights, map_location=device, weights_only=False)
    model = Probe(ProbeConfig(**saved["cfg"])).to(device).eval()
    model.load_state_dict(saved["state"])
    out = []
    with torch.no_grad():
        for k in keys:
            x = torch.from_numpy(standardise(raw[k], saved["mean"], saved["std"])).to(device)
            out.append(layer_state(model, model.hidden(x.unsqueeze(0)), layer)[0].cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def episode_table(inputs, keys: list[str]) -> dict[str, np.ndarray]:
    """The per-episode metadata saved with the points: task, skill, outcome (success), failure step (-1 for a success), failure mode (index into the mode names, -1 for a
    success) and number of steps."""
    skills = episode_skills(inputs)
    episodes = [inputs.episodes[k] for k in keys]
    return {"task": np.array([e.task for e in episodes]), "skill": np.array([skills[k] for k in keys]), "success": np.array([e.success for e in episodes]),
            "onset": np.array([-1 if e.success else e.onset for e in episodes], np.int32),
            "mode": np.array([-1 if e.success else int(inputs.targets.modes[e.key]) for e in episodes], np.int16), "n_steps": np.array([e.n_steps for e in episodes], np.int32)}


def save_points(path: Path, xy: np.ndarray, episode: np.ndarray, step: np.ndarray, keys: list[str], table: dict[str, np.ndarray], config: dict,
                extra: dict[str, np.ndarray] | None = None) -> None:
    """Write ``points.npz``: the 2D points, the episode and step of each, the keys of the episodes, their metadata and the settings (JSON) that made the embedding, and the
    arrays of ``extra`` (the PCA keeps its three axes as ``xyz``, of which ``xy`` is the first two)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, xy=xy, episode=episode, step=step, keys=np.array(keys), config=np.array(json.dumps(config)), **table, **(extra or {}))


def load_points(path: Path) -> dict:
    """The arrays of a ``points.npz`` and its settings under ``config`` (a dict)."""
    with np.load(path) as z:
        out = {name: z[name] for name in z.files if name != "config"}
        out["config"] = json.loads(str(z["config"]))
    return out
