"""ActProbe (arXiv 2606.08508; code: ActProbe repository, Apache-2.0): a small LSTM on two action-space signals, with the task instruction.

At every policy call it reads two numbers from the executed action chunk (sample 0), a time feature, and starts its LSTM state from an embedding
of the task instruction:

    ACM  action chunk magnitude: RMS of the executed part (the first ``replan_steps`` actions), all 7 dimensions
    TCE  temporal consistency error: mean squared difference, over the overlap, between the previous call's unexecuted actions and this
         call's first ones (0 at the first call)
    t    the policy step divided by 100

ACM and TCE are normalised with the training steps' mean and standard deviation. The network (22,945 parameters) outputs a failure probability
per step; training is binary cross-entropy on every step of an episode with the episode's label, so a failed episode is positive from its first
step (the labelling of SAFE, and of our ``onset_traj`` variant). Recipe of the ActProbe repository: AdamW (lr 1e-3, weight decay 1e-4), cosine
schedule over 400 epochs, batch 64, gradient clipping at 1, the checkpoint with the best validation q-AUC (checked every 5 epochs; stop after 50
epochs without improvement). q-AUC (``q_auc_taskmax``) is only the selection rule here; the results are evaluated like every other method, with
the area under the early-detection curve and max-conformal calibration.

The task embedding is whatever ``data/<dataset>/task_embeddings.npz`` holds (``fp/data/task_embeddings.py``), and its source is recorded with
every run. Method name: ``actprobe`` (one configuration, one run per split).
"""

import math
from typing import Mapping

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from .. import config
from ..data.manifest import Episode
from ..data.splits import Split
from ..data.task_embeddings import DIM as LANGUAGE_DIM
from ..data.task_embeddings import load_task_embeddings
from ..models.base import Scorer
from .registry import Context, register

N_FEATURES, HIDDEN, BOTTLENECK, DROPOUT = 2, 32, 16, 0.4
BATCH, LR, WEIGHT_DECAY, EPOCHS, COSINE_EPOCHS, CLIP = 64, 1e-3, 1e-4, 400, 400, 1.0
EVAL_EVERY, PATIENCE, MIN_GAIN, Q = 5, 50, 1e-4, 0.25
TIME_SCALE = 100.0
NORM_EPS = 1e-7


def acm_tce(chunks: np.ndarray, exec_horizon: int) -> np.ndarray:
    """(T, 2) = [ACM, TCE] of an episode; ``chunks`` is the executed plan of every call, (T, H, A)."""
    if chunks.shape[1] <= exec_horizon:
        raise ValueError(f"the chunks (H = {chunks.shape[1]}) must be longer than the execution horizon ({exec_horizon})")
    out = np.zeros((len(chunks), N_FEATURES), np.float32)
    out[:, 0] = np.sqrt(np.mean(chunks[:, :exec_horizon].astype(np.float64) ** 2, axis=(1, 2)))
    if len(chunks) > 1:
        out[1:, 1] = np.mean((chunks[:-1, exec_horizon:].astype(np.float64) - chunks[1:, :-exec_horizon]) ** 2, axis=(1, 2))
    return out


class ActProbeNet(nn.Module):
    """LSTM and MLP with a skip connection from the input; the instruction embedding sets the LSTM's initial state."""

    def __init__(self):
        super().__init__()
        self.lang_proj = nn.Sequential(nn.Linear(LANGUAGE_DIM, BOTTLENECK), nn.ReLU())
        self.h0_proj = nn.Linear(BOTTLENECK, HIDDEN)
        self.c0_proj = nn.Linear(BOTTLENECK, HIDDEN)
        self.lstm = nn.LSTM(N_FEATURES + 1, HIDDEN, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(DROPOUT)
        self.mlp = nn.Sequential(nn.Linear(HIDDEN + N_FEATURES + 1, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x: torch.Tensor, lang: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x (B, T, 3) with the time feature last, lang (B, 1024), lengths (B,): probabilities (B, T); the padded steps hold meaningless values."""
        z = self.lang_proj(lang)
        h0, c0 = self.h0_proj(z).unsqueeze(0).contiguous(), self.c0_proj(z).unsqueeze(0).contiguous()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed, (h0, c0))
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return torch.sigmoid(self.mlp(torch.cat([self.drop(out), x], dim=-1)).squeeze(-1))


def q_auc_taskmax(scores: list[np.ndarray], labels: list[bool], tasks: list[str], q: float = Q) -> float:
    """ActProbe's selection metric: the AUROC of the episodes' maximum score over the first ``ceil(q * longest episode of its task)`` steps
    (the longest among the episodes given); 0.5 when only one outcome is present."""
    if len(set(labels)) < 2:
        return 0.5
    longest: dict[str, int] = {}
    for s, t in zip(scores, tasks):
        longest[t] = max(longest.get(t, 0), len(s))
    cutoff = lambda s, t: min(max(1, math.ceil(q * longest[t])), len(s))
    return float(roc_auc_score(labels, [s[: cutoff(s, t)].max() for s, t in zip(scores, tasks)]))


class ActProbeScorer(Scorer):
    name = "actprobe"
    saves_weights = True
    input_kind = "sampled action chunks (executed chunk: ACM, TCE) + task embedding"

    def __init__(self, episodes: Mapping[str, Episode], actions: Mapping[str, np.ndarray], embeddings: Mapping[str, np.ndarray],
                 embedding_source: str, exec_horizon: int, *, seed: int = 0, device: str = "cpu", epochs: int = EPOCHS, patience: int = PATIENCE):
        self.episodes, self.actions, self.embeddings, self.embedding_source = episodes, actions, embeddings, embedding_source
        self.exec_horizon, self.seed, self.device, self.epochs, self.patience = exec_horizon, seed, device, epochs, patience
        self.model: ActProbeNet | None = None
        self.mean = self.std = None
        self.result: dict | None = None
        self._raw: dict[str, np.ndarray] = {}

    @property
    def config(self) -> dict:
        return {"hidden": HIDDEN, "bottleneck": BOTTLENECK, "dropout": DROPOUT, "batch": BATCH, "lr": LR, "weight_decay": WEIGHT_DECAY,
                "epochs": self.epochs, "cosine_epochs": COSINE_EPOCHS, "patience": self.patience, "eval_every": EVAL_EVERY, "q": Q, "clip": CLIP,
                "time_scale": TIME_SCALE, "exec_horizon": self.exec_horizon, "seed": self.seed, "embedding_source": self.embedding_source,
                "n_params": sum(p.numel() for p in ActProbeNet().parameters())}

    def _features(self, key: str) -> np.ndarray:
        """(T, 2) [ACM, TCE] of an episode, computed once."""
        if key not in self._raw:
            self._raw[key] = acm_tce(self.actions[key][:, 0], self.exec_horizon)
        return self._raw[key]

    def _batch(self, keys: list[str]):
        """x (B, T, 3), lang (B, 1024), lengths (B,) for the episodes, normalised and padded with zeros."""
        lengths = [len(self._features(k)) for k in keys]
        x = np.zeros((len(keys), max(lengths), N_FEATURES + 1), np.float32)
        for i, k in enumerate(keys):
            x[i, : lengths[i], :N_FEATURES] = (self._features(k) - self.mean) / (self.std + NORM_EPS)
            x[i, : lengths[i], N_FEATURES] = np.arange(lengths[i]) / TIME_SCALE
        lang = np.stack([self.embeddings[self.episodes[k].task] for k in keys]).astype(np.float32)
        to = lambda a, dtype=torch.float32: torch.as_tensor(a, dtype=dtype, device=self.device)
        return to(x), to(lang), to(lengths, torch.long)

    @torch.no_grad()
    def _predict(self, keys: list[str], batch: int = BATCH) -> dict[str, np.ndarray]:
        self.model.eval()
        out = {}
        for i in range(0, len(keys), batch):
            chunk = keys[i : i + batch]
            probs = self.model(*self._batch(chunk)).cpu().numpy()
            out.update({k: probs[j, : len(self._features(k))].astype(np.float32) for j, k in enumerate(chunk)})
        return out

    def _val_q_auc(self, keys: list[str]) -> float:
        scores = self._predict(keys)
        return q_auc_taskmax([scores[k] for k in keys], [not self.episodes[k].success for k in keys], [self.episodes[k].task for k in keys])

    def fit(self, train_keys: list[str], val_keys: list[str]) -> None:
        steps = np.concatenate([self._features(k) for k in train_keys])
        self.mean, self.std = steps.mean(0).astype(np.float32), steps.std(0).astype(np.float32)
        torch.manual_seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        self.model = ActProbeNet().to(self.device)
        optimiser = torch.optim.AdamW(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=COSINE_EPOCHS)
        labels = torch.tensor([float(not self.episodes[k].success) for k in train_keys], device=self.device)
        best, best_state, best_epoch, waited, train_loss, val_curve = -1.0, None, 0, 0, [], []
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total, count = 0.0, 0
            for idx in torch.randperm(len(train_keys), generator=generator).split(BATCH):
                x, lang, lengths = self._batch([train_keys[i] for i in idx])
                probs = self.model(x, lang, lengths)
                valid = torch.arange(probs.shape[1], device=self.device)[None] < lengths[:, None]
                target = labels[idx.to(self.device)][:, None].expand_as(probs)
                loss = (nn.functional.binary_cross_entropy(probs.clamp(1e-7, 1 - 1e-7), target, reduction="none") * valid).sum() / valid.sum()
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)
                optimiser.step()
                total, count = total + float(loss) * int(valid.sum()), count + int(valid.sum())
            schedule.step()
            train_loss.append(total / count)
            if epoch % EVAL_EVERY == 0:
                auc = self._val_q_auc(val_keys)
                val_curve.append(auc)
                if auc > best + MIN_GAIN:
                    best, best_epoch, waited = auc, epoch, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    waited += EVAL_EVERY
                    if waited >= self.patience:
                        break
        self.model.load_state_dict(best_state)
        self.result = {"best_epoch": best_epoch, "best_val_q_auc": best, "epochs": epoch, "stopped_early": epoch < self.epochs,
                       "train_loss": train_loss, "val_q_auc": val_curve}

    def score(self, keys: list[str]) -> dict[str, np.ndarray]:
        if self.model is None:
            raise RuntimeError("fit() has not been called")
        return self._predict(keys)

    def training_details(self) -> dict | None:
        if self.result is None:
            return None
        return {**self.result, "label_variant": "onset_traj (every step of a failed episode is positive)",
                "selection": {"criterion": "validation q-AUC (taskmax, q = 0.25), the ActProbe repository's own", "every_epochs": EVAL_EVERY,
                              "patience": self.patience}}

    def state_bytes(self) -> int | None:
        return None if self.model is None else sum(p.numel() * p.element_size() for p in self.model.parameters()) + 2 * N_FEATURES * 4

    def save(self, path: str) -> None:
        torch.save({"config": self.config, "state": self.model.state_dict(), "mean": self.mean, "std": self.std}, path)

    def load(self, path: str) -> None:
        saved = torch.load(path, map_location=self.device, weights_only=False)
        if saved["config"] != self.config:
            raise ValueError(f"{path} holds {saved['config']}, this scorer is {self.config}")
        self.model = ActProbeNet().to(self.device)
        self.model.load_state_dict(saved["state"])
        self.mean, self.std = saved["mean"], saved["std"]


@register("actprobe", needs_actions=True)
def build(context: Context, split: Split) -> list[ActProbeScorer]:
    if context.actions is None:
        raise ValueError("ActProbe needs the sampled action chunks: python -m cli.prepare_actions <dataset> <rollout_root>")
    embeddings, source = load_task_embeddings(context.inputs.dataset)
    return [ActProbeScorer(context.inputs.episodes, context.actions, embeddings, source, config.get(context.inputs.dataset).replan_steps,
                           seed=split.seed, device=context.device)]
