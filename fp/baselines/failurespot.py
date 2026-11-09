"""FailureSpot (Ma, Liu, Zhu; arXiv 2609.04277): label-efficient timestamp-level failure detection. Our own implementation from the paper (no
code is public); everything the paper does not state is marked "ours" below and listed in docs/ACTION_BASELINES.md.

Stage 1, weak supervision. Three signals per policy call from the executed action chunk (H actions, the first K = ``replan_steps`` executed):
consistency c (mean L2 distance between the previous chunk's unexecuted actions and this chunk's first ones, over M = min(K, H - K) actions),
the full-chunk magnitude r (mean L2 norm of the H actions) and the executed-part magnitude r_exec (the same over the first K). Each signal u
gets an exponential moving average E(u) (beta = 0.8, E at the first call = u there), and the combined score is

    s = [c + E(c)] + eta1 [r + E(r)] + eta2 [r_exec + E(r_exec)]

rescaled to [0, 1] with the 1st and 99th percentiles of the training steps and clipped: the soft target. A detector (MLP or LSTM on the pooled,
standardised features) is trained on these soft targets for every training episode.

Stage 2, active learning. The mean binary entropy of the stage-1 probabilities ranks the training episodes; the most uncertain 15% (of the
seen-task episodes: train, validation, calibration and seen parts of the split) get timestamp labels and the detector is fine-tuned on them with
a binary classification loss. Timestamp labels: every step from the failure onset on is 1 in a failed episode, a success is all 0 (the paper
also labels recovered successes between the deviation and the recovery; we have no recovery times).

Ours, because the paper does not state it: eta1 and eta2 (each in {0.25, 1, 4}, nine pairs, each its own method); the width of the detectors
(256); the stage-1 loss (binary cross-entropy against the soft target); Adam, learning rate 1e-3 (MLP) / 3e-4 (LSTM), batches of 32 episodes, at
most 200 epochs, patience 25; the c at the first call (set to c at the second); and what stops each stage and selects its checkpoint: the
validation per-step cross-entropy, against the soft targets in stage 1 (no labels needed) and against the onset-based timestamp labels of the
validation episodes in stage 2. Methods are named ``failurespot_<mlp|lstm>__eta1_<a>_eta2_<b>``, e.g. ``failurespot_lstm__eta1_0p25_eta2_4``.
"""

import itertools
from typing import Mapping

import numpy as np
import torch
from torch import nn

from .. import config
from ..data.features import fit_standardiser
from ..data.manifest import Episode
from ..data.splits import Split
from ..models.base import Scorer
from ..train.items import Item, collate
from ..train.loop import Step, TrainConfig, fit
from .registry import Context, register

KINDS = ("mlp", "lstm")
ETAS = (0.25, 1.0, 4.0)
HIDDEN, BETA, BUDGET, PERCENTILES, EPS = 256, 0.8, 0.15, (1, 99), 1e-6
LEARNING_RATE = {"mlp": 1e-3, "lstm": 3e-4}
BATCH, MAX_EPOCHS, PATIENCE = 32, 200, 25
CLAMP = 1e-6


# ---------------------------------------------------------------- the weak-supervision signals
def ema(u: np.ndarray) -> np.ndarray:
    """E(u)_t = beta E(u)_{t-1} + (1 - beta) u_t, with E(u)_0 = u_0."""
    out = np.empty(len(u), np.float64)
    out[0] = u[0]
    for t in range(1, len(u)):
        out[t] = BETA * out[t - 1] + (1 - BETA) * u[t]
    return out


def action_signals(chunks: np.ndarray, exec_horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(c, r, r_exec), each (T,), from the executed chunk of every call, ``chunks`` (T, H, A). c at the first call is set to c at the second
    (0 for an episode of a single call)."""
    horizon = chunks.shape[1]
    if not 0 < exec_horizon < horizon:
        raise ValueError(f"the execution horizon ({exec_horizon}) must be between 0 and the chunk length ({horizon})")
    overlap = min(exec_horizon, horizon - exec_horizon)
    chunks = chunks.astype(np.float64)
    norms = np.linalg.norm(chunks, axis=-1)  # (T, H)
    c = np.zeros(len(chunks))
    c[1:] = np.linalg.norm(chunks[:-1, exec_horizon : exec_horizon + overlap] - chunks[1:, :overlap], axis=-1).mean(axis=-1)
    if len(c) > 1:
        c[0] = c[1]
    return c, norms.mean(axis=1), norms[:, :exec_horizon].mean(axis=1)


def combined_signal(signals: tuple[np.ndarray, np.ndarray, np.ndarray], eta1: float, eta2: float) -> np.ndarray:
    c, r, r_exec = signals
    return c + ema(c) + eta1 * (r + ema(r)) + eta2 * (r_exec + ema(r_exec))


def soft_targets(scores: Mapping[str, np.ndarray], train_keys: list[str]) -> dict[str, np.ndarray]:
    """Rescale the combined scores to [0, 1] with the 1st and 99th percentiles of the training steps, clipping outside them."""
    low, high = np.percentile(np.concatenate([scores[k] for k in train_keys]), PERCENTILES)
    return {k: np.clip((v - low) / (high - low + EPS), 0, 1).astype(np.float32) for k, v in scores.items()}


def timestamp_labels(episode: Episode) -> np.ndarray:
    """1 from the failure onset on in a failed episode, 0 everywhere else."""
    y = np.zeros(episode.n_steps, np.float32)
    if episode.onset is not None:
        y[episode.onset :] = 1.0
    return y


# ---------------------------------------------------------------- active learning
def mean_entropy(p: np.ndarray) -> float:
    """Mean over the steps of the binary entropy of the probabilities."""
    p = np.clip(p, CLAMP, 1 - CLAMP)
    return float((-p * np.log(p) - (1 - p) * np.log(1 - p)).mean())


def most_uncertain(probabilities: Mapping[str, np.ndarray], budget: int) -> list[str]:
    """The ``budget`` episodes with the highest mean entropy, most uncertain first (ties by key, so the choice is reproducible)."""
    ranked = sorted(probabilities, key=lambda k: (-mean_entropy(probabilities[k]), k))
    return ranked[: min(budget, len(ranked))]


# ---------------------------------------------------------------- the detectors and the loss
class StepMLP(nn.Module):
    def __init__(self, dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, d) -> (B, T)
        return self.net(x).squeeze(-1)


class StepLSTM(nn.Module):
    def __init__(self, dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, d) -> (B, T); causal
        return torch.sigmoid(self.fc(self.lstm(x)[0])).squeeze(-1)


def bce_step(device: str):
    """Step function: binary cross-entropy of the per-step probabilities against the items' ``y`` (soft or hard), pooled over valid steps."""

    def step(model: nn.Module, items) -> Step:
        batch = collate(list(items), device)
        loss = nn.functional.binary_cross_entropy(model(batch.x).clamp(CLAMP, 1 - CLAMP), batch.y, reduction="none")
        total, count = (loss * batch.valid).sum(), batch.valid.sum()
        return Step(total / count, {"bce": (float(total), float(count))})

    return step


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


class FailureSpotScorer(Scorer):
    saves_weights = True
    input_kind = "raw pooled features (standardised) + sampled action chunks (executed chunk) for the weak labels"

    def __init__(self, kind: str, eta1: float, eta2: float, episodes: Mapping[str, Episode], features: Mapping[str, np.ndarray],
                 actions: Mapping[str, np.ndarray], exec_horizon: int, n_seen: int, *, seed: int = 0, device: str = "cpu",
                 max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        self.kind, self.eta1, self.eta2, self.episodes, self.features, self.actions = kind, eta1, eta2, episodes, features, actions
        self.exec_horizon, self.n_seen, self.seed, self.device = exec_horizon, n_seen, seed, device
        self.max_epochs, self.patience = max_epochs, patience
        self.name = f"failurespot_{kind}__eta1_{tag(eta1)}_eta2_{tag(eta2)}"
        self.model: nn.Module | None = None
        self.mean = self.std = None
        self.result: dict | None = None

    @property
    def config(self) -> dict:
        return {"kind": self.kind, "eta1": self.eta1, "eta2": self.eta2, "hidden": HIDDEN, "beta": BETA, "budget": BUDGET, "n_seen": self.n_seen,
                "percentiles": list(PERCENTILES), "lr": LEARNING_RATE[self.kind], "batch": BATCH, "max_epochs": self.max_epochs,
                "patience": self.patience, "exec_horizon": self.exec_horizon, "seed": self.seed}

    def _new_model(self, dim: int) -> nn.Module:
        return (StepMLP if self.kind == "mlp" else StepLSTM)(dim).to(self.device)

    def _standardised(self, key: str) -> torch.Tensor:
        return torch.from_numpy(((self.features[key] - self.mean) / self.std).astype(np.float32)).to(self.device)

    def _items(self, keys: list[str], targets: Mapping[str, np.ndarray]) -> list[Item]:
        return [Item(self._standardised(k), torch.from_numpy(targets[k]), torch.full((len(targets[k]),), -1, dtype=torch.long), -1) for k in keys]

    @torch.no_grad()
    def _predict(self, keys: list[str], batch: int = BATCH) -> dict[str, np.ndarray]:
        self.model.eval()
        out = {}
        for i in range(0, len(keys), batch):
            chunk = keys[i : i + batch]
            n = [self.episodes[k].n_steps for k in chunk]
            items = [Item(self._standardised(k), torch.zeros(m), torch.full((m,), -1, dtype=torch.long), -1) for k, m in zip(chunk, n)]
            probs = self.model(collate(items, self.device).x).cpu().numpy().astype(np.float32)
            out.update({k: probs[j, : n[j]] for j, k in enumerate(chunk)})
        return out

    def fit(self, train_keys: list[str], val_keys: list[str]) -> None:
        self.mean, self.std = fit_standardiser([self.features[k] for k in train_keys])
        both = train_keys + val_keys
        signals = {k: action_signals(self.actions[k][:, 0], self.exec_horizon) for k in both}
        soft = soft_targets({k: combined_signal(signals[k], self.eta1, self.eta2) for k in both}, train_keys)
        torch.manual_seed(self.seed)
        self.model = self._new_model(self.features[train_keys[0]].shape[1])
        cfg = TrainConfig(lr=LEARNING_RATE[self.kind], batch=BATCH, max_epochs=self.max_epochs, patience=self.patience, seed=self.seed)
        step = bce_step(self.device)

        pretrain = fit(self.model, self._items(train_keys, soft), self._items(val_keys, soft), step, cfg, select="bce")
        chosen = most_uncertain(self._predict(train_keys), int(round(BUDGET * self.n_seen)))
        hard = {k: timestamp_labels(self.episodes[k]) for k in chosen + val_keys}
        finetune = fit(self.model, self._items(chosen, hard), self._items(val_keys, hard), step, cfg, select="bce")
        self.result = {"pretrain": pretrain, "finetune": finetune, "annotated": chosen,
                       "annotated_failures": sum(not self.episodes[k].success for k in chosen)}

    def score(self, keys: list[str]) -> dict[str, np.ndarray]:
        if self.model is None:
            raise RuntimeError("fit() has not been called")
        return self._predict(keys)

    def training_details(self) -> dict | None:
        if self.result is None:
            return None
        stage = lambda r: {"best_epoch": r.best_epoch, "epochs": len(r.val["bce"]), "stopped_early": r.stopped_early, "train_bce": r.train["bce"],
                           "val_bce": r.val["bce"]}
        r = self.result
        return {"pretrain": stage(r["pretrain"]), "finetune": stage(r["finetune"]), "n_annotated": len(r["annotated"]),
                "n_annotated_failures": r["annotated_failures"], "annotated": r["annotated"],
                "selection": {"pretrain": "validation per-step cross-entropy against the soft targets (no labels)",
                              "finetune": "validation per-step cross-entropy against the onset-based timestamp labels of the validation episodes",
                              "patience": self.patience, "max_epochs": self.max_epochs, "paper": "not stated in the paper (our choice)"}}

    def state_bytes(self) -> int | None:
        if self.model is None:
            return None
        return sum(p.numel() * p.element_size() for p in self.model.parameters()) + self.mean.nbytes + self.std.nbytes

    def save(self, path: str) -> None:
        torch.save({"config": self.config, "state": self.model.state_dict(), "mean": self.mean, "std": self.std}, path)

    def load(self, path: str) -> None:
        saved = torch.load(path, map_location=self.device, weights_only=False)
        if saved["config"] != self.config:
            raise ValueError(f"{path} holds {saved['config']}, this scorer is {self.config}")
        self.mean, self.std = saved["mean"], saved["std"]
        self.model = self._new_model(len(self.mean))
        self.model.load_state_dict(saved["state"])


# ---------------------------------------------------------------- the grid and the registry
def grid_scorers(kind: str, episodes, features, actions, exec_horizon: int, n_seen: int, seed: int = 0, device: str = "cpu") -> list[FailureSpotScorer]:
    """One unfitted scorer per (eta1, eta2), eta1 first then eta2, both ascending (the order ties are broken in when the best is chosen)."""
    return [FailureSpotScorer(kind, e1, e2, episodes, features, actions, exec_horizon, n_seen, seed=seed, device=device)
            for e1, e2 in itertools.product(ETAS, ETAS)]


def method_names(kind: str) -> list[str]:
    return [s.name for s in grid_scorers(kind, {}, {}, {}, 1, 0)]


def _family(kind: str):
    def build(context: Context, split: Split) -> list[FailureSpotScorer]:
        if context.actions is None:
            raise ValueError("FailureSpot needs the sampled action chunks: python -m cli.prepare_actions <dataset> <rollout_root>")
        n_seen = len(split.train) + len(split.val) + len(split.calibration) + len(split.seen)  # the seen-task episodes
        return grid_scorers(kind, context.inputs.episodes, context.features, context.actions, config.get(context.inputs.dataset).replan_steps, n_seen,
                            split.seed, context.device)

    return build


FAMILIES = {f"failurespot_{kind}": kind for kind in KINDS}
for _name, _kind in FAMILIES.items():
    register(_name, needs_actions=True)(_family(_kind))
