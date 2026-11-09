"""Training of SAFE-MLP and SAFE-LSTM with SAFE's losses, written from SAFE's code (failure_prob/model/{indep,lstm,utils,base}.py and
data/utils.py) in its default configuration.

SAFE's recipe, as its code does it
  * labels: a failed rollout is positive at every step, a successful one negative (episode-level); nothing about the onset;
  * SAFE-MLP loss, per episode: the cumulative score is pushed to 0 on successes (relu(score), the mean over the steps) and up on failures
    (-score, the mean over the steps); SAFE-LSTM loss, per episode: the mean over the steps of the binary cross-entropy of the step
    probability against the label;
  * the episode losses are weighted by class, weight 1 / ((count + 1) / n) with the counts of the training episodes, summed over the
    batch and divided by the batch size; an L2 penalty lambda_reg * (sum of the squared non-bias weights) is added;
  * Adam, batch 512 whole rollouts, 1,000 epochs, the last epoch's model is kept (no validation set is used to stop or select).

DEVIATION FROM SAFE, decided by the user: training stops early. After every epoch the model is scored on the validation episodes with SAFE's
own class-weighted data loss (without the L2 penalty, which is a function of the weights and would pick the epoch by weight shrinkage);
training stops after ``patience`` = 50 epochs without a new best and the weights of the best epoch are kept; 1,000 epochs is the cap.
Everything else (loss, weights, optimiser, batch, learning rates, penalty) is SAFE's. The run's history records this deviation.

Labels: ``labels="safe"`` is SAFE's own (the full failed rollout is positive: our ``onset_traj``). ``labels="pre"`` keeps SAFE's loss and
changes only the data: failed training and validation episodes are cut at the failure onset (our ``onset_pre``). That is not SAFE as
published; it separates what the labels contribute from what the architecture contributes, and it is a separate family in every table.
"""

import torch
import torch.nn.functional as F

from ..data.manifest import Episode
from ..models.base import Scorer
from ..models.safe import SafeLSTM, SafeMLP
from .items import Item, collate, make_item
from .loop import Step, TrainConfig, fit

KINDS = ("mlp", "lstm")
LABEL_VARIANT = {"safe": "onset_traj", "pre": "onset_pre"}  # SAFE's labels are our onset_traj
HIDDEN, BATCH, MAX_EPOCHS, PATIENCE = 256, 512, 1000, 50
# A batch of 512 whole rollouts through the LSTM on 6,144-d inputs needs well over 14 GB of GPU memory. It is run in chunks of 128 with the
# gradients accumulated: SAFE's monitor loss is a mean over the episodes, so the update is the same (``loop.TrainConfig.micro_batch``).
MICRO_BATCH = 128


def tag(value: float) -> str:
    """1e-05 -> '1e-5', 0.001 -> '1e-3', 3e-05 -> '3e-5': the form used in method names."""
    return f"{value:.0e}".replace("e-0", "e-")


def class_weights(failed: list[bool]) -> tuple[float, float]:
    """SAFE's (failure, success) weights from the training episodes: 1 / ((count + 1) / n) for each class."""
    n = len(failed)
    n_fail = sum(failed)
    return 1.0 / ((n_fail + 1) / n), 1.0 / ((n - n_fail + 1) / n)


def step_losses(kind: str, scores: torch.Tensor, y: torch.Tensor, is_failure: torch.Tensor) -> torch.Tensor:
    """(B, T) loss of every step. MLP: relu(cumulative score) on a success, -(cumulative score) on a failure. LSTM: the binary cross-entropy of
    the step probability against the step label (a failure is the positive class)."""
    if kind == "mlp":
        return torch.where(is_failure[:, None], -scores, torch.relu(scores))
    return F.binary_cross_entropy(scores, y, reduction="none")


def monitor_terms(kind: str, scores: torch.Tensor, y: torch.Tensor, valid: torch.Tensor, weights: tuple[float, float]) -> tuple[torch.Tensor, int]:
    """(numerator, denominator) of SAFE's monitor loss of a batch: the class-weighted sum over episodes of each episode's mean step loss, and
    the number of episodes. The ratio is SAFE's monitor loss; the pairs of several batches pool exactly (see ``loop``)."""
    is_failure = (y * valid).amax(dim=1) > 0
    per_episode = (step_losses(kind, scores, y, is_failure) * valid).sum(-1) / valid.sum(-1)
    weight = torch.where(is_failure, per_episode.new_tensor(weights[0]), per_episode.new_tensor(weights[1]))
    return (weight * per_episode).sum(), len(per_episode)


def l2_penalty(model: torch.nn.Module) -> torch.Tensor:
    """SAFE's regulariser without its lambda: the sum of the squares of every parameter that is not a bias."""
    return sum((p ** 2).sum() for name, p in model.named_parameters() if "bias" not in name)


class SafeScorer(Scorer):
    saves_weights = True

    def __init__(self, kind: str, episodes: dict[str, Episode], features, window: int, *, lr: float, lambda_reg: float, labels: str = "safe",
                 batch: int = BATCH, max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE, seed: int = 0, device: str = "cpu",
                 micro_batch: int | None = MICRO_BATCH):
        if kind not in KINDS or labels not in LABEL_VARIANT:
            raise ValueError(f"kind must be one of {KINDS} and labels one of {tuple(LABEL_VARIANT)}")
        self.kind, self.labels, self.lr, self.lambda_reg = kind, labels, lr, lambda_reg
        self.episodes, self.features, self.window, self.device = episodes, features, window, device
        self.batch, self.max_epochs, self.patience, self.seed, self.micro_batch = batch, max_epochs, patience, seed, micro_batch
        self.name = f"safe_{kind}{'_pre' if labels == 'pre' else ''}__lr{tag(lr)}_reg{tag(lambda_reg)}"
        self.model: torch.nn.Module | None = None
        self.result = None

    @property
    def config(self) -> dict:
        return {"kind": self.kind, "labels": self.labels, "lr": self.lr, "lambda_reg": self.lambda_reg, "hidden": HIDDEN, "batch": self.batch,
                "micro_batch": self.micro_batch, "max_epochs": self.max_epochs, "patience": self.patience, "seed": self.seed}

    def _items(self, keys: list[str]) -> list[Item]:
        return [make_item(self.episodes[k], torch.from_numpy(self.features[k]).to(self.device), None, None, LABEL_VARIANT[self.labels], self.window)
                for k in keys]

    def fit(self, train_keys: list[str], val_keys: list[str]) -> None:
        train, val = self._items(train_keys), self._items(val_keys)
        dim = train[0].x.shape[1]
        weights = class_weights([not self.episodes[k].success for k in train_keys])
        torch.manual_seed(self.seed)
        self.model = (SafeMLP if self.kind == "mlp" else SafeLSTM)(dim, HIDDEN).to(self.device)

        def step_fn(model, chunk):
            b = collate(chunk, self.device)
            num, den = monitor_terms(self.kind, model(b.x), b.y, b.valid, weights)
            penalty = self.lambda_reg * l2_penalty(model)
            return Step(num / den + penalty, {"data": (float(num), den), "l2": (float(penalty) * den, den)})

        cfg = TrainConfig(lr=self.lr, batch=self.batch, max_epochs=self.max_epochs, patience=self.patience, seed=self.seed, micro_batch=self.micro_batch)
        self.result = fit(self.model, train, val, step_fn, cfg, select="data")

    @torch.no_grad()
    def score(self, keys: list[str], batch: int = 32) -> dict:
        if self.model is None:
            raise RuntimeError("fit() has not been called")
        self.model.eval()
        out = {}
        for i in range(0, len(keys), batch):
            chunk = keys[i : i + batch]
            x = [torch.from_numpy(self.features[k]).to(self.device) for k in chunk]
            padded = torch.zeros(len(chunk), max(len(a) for a in x), x[0].shape[1], device=self.device)
            for j, a in enumerate(x):
                padded[j, : len(a)] = a
            scores = self.model(padded).cpu().numpy().astype("float32")
            out.update({k: scores[j, : len(x[j])] for j, k in enumerate(chunk)})
        return out

    def training_details(self) -> dict | None:
        if self.result is None:
            return None
        r = self.result
        return {"label_variant": LABEL_VARIANT[self.labels], "best_epoch": r.best_epoch, "epochs": len(r.val["data"]), "stopped_early": r.stopped_early,
                "train": r.train, "val": r.val,
                "early_stopping": {"patience": self.patience, "max_epochs": self.max_epochs, "criterion": "validation data loss, class-weighted, without the L2 penalty",
                                   "safe": "fixed 1000 epochs, the last epoch's model, no validation set (decided deviation: early stopping, patience 50)"}}

    def state_bytes(self) -> int | None:
        return None if self.model is None else sum(p.numel() * p.element_size() for p in self.model.parameters())

    def save(self, path: str) -> None:
        torch.save({"config": self.config, "state": self.model.state_dict()}, path)

    def load(self, path: str) -> None:
        saved = torch.load(path, map_location=self.device, weights_only=False)
        if saved["config"] != self.config:
            raise ValueError(f"{path} holds {saved['config']}, this scorer is {self.config}")
        dim = next(iter(saved["state"].values())).shape[-1]
        self.model = (SafeMLP if self.kind == "mlp" else SafeLSTM)(dim, HIDDEN).to(self.device)
        self.model.load_state_dict(saved["state"])
