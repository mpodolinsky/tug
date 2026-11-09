"""OursScorer: fits one probe (an architecture of ``fp.models.ours`` under one label variant) and scores episodes with it.

The features come standardised (``fp.train.standardised``), once per split and shared by all probes trained on it; ``fit`` checks that they
were standardised with exactly the training episodes it is given.

fit(train_keys, val_keys):
    items under the label variant -> loss context (class weights, and for the semantic head the PCA target space of the training phrases)
    -> train with ``loop.fit`` on the full loss, keep the epoch with the lowest validation score term (the weighted BCE alone, so the
    mode and CLIP heads never choose the epoch; with ``lam_score`` 0 the score is not trained and the validation mode term chooses). Every term is reported per epoch in ``result.train`` / ``result.val``.
outputs(keys): the output of every head at every step of the full rollout (not cut at the variant's cut: the probe is causal, so the steps
before the cut get the same value either way). score(keys): only the failure probability, the sigmoid of the score logit.
save(path) / load(path): weights, feature statistics and configuration in one file.
"""

from dataclasses import asdict
from typing import Mapping

import numpy as np
import torch

from ..data.manifest import Episode
from ..data.targets import Targets
from ..models.base import Scorer
from ..models.ours import Probe, ProbeConfig
from .items import Item, collate, make_item
from .loop import FitResult, Step, TrainConfig, default_lr, fit
from .losses import loss_terms, make_context, total_loss
from .semantic import SemanticSpace, fit_semantic_space
from .standardised import StandardisedFeatures


class OursScorer(Scorer):
    def __init__(self, cfg: ProbeConfig, label_variant: str, episodes: Mapping[str, Episode], features: StandardisedFeatures,
                 targets: Targets, window: int, train_cfg: TrainConfig | None = None, device: str | None = None, lam_mode: float = 1.0,
                 lam_clip: float = 1.0, lam_score: float = 1.0):
        self.cfg, self.label_variant, self.window = cfg, label_variant, window
        self.lam_mode, self.lam_clip, self.lam_score = lam_mode, lam_clip, lam_score  # the weights of the mode, CLIP and score terms in the loss
        if lam_score == 0 and not cfg.modes:
            raise ValueError("a probe trained without the score term needs a mode head to choose its epoch on")
        self.episodes, self.features, self.targets = episodes, features, targets
        self.train_cfg = train_cfg or TrainConfig(lr=default_lr(cfg.trunk))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name = f"{cfg.name}__{label_variant}"
        self.model: Probe | None = None
        self.result: FitResult | None = None
        self.space: SemanticSpace | None = None  # the CLIP head's target space (clip architectures, after fit())

    def _item(self, key: str) -> Item:
        e = self.episodes[key]
        mode = None if e.success else self.targets.modes[key]
        return make_item(e, self.features.arrays[key], self.targets.phrase_ids[key], mode, self.label_variant, self.window)

    def fit(self, train_keys: list[str], val_keys: list[str]) -> None:
        if set(train_keys) != self.features.train_keys:
            raise ValueError("the features were standardised with other training episodes than the ones given to fit()")
        train = [self._item(k) for k in train_keys]
        val = [self._item(k) for k in val_keys]
        space = None
        if self.cfg.clip:  # the target space has the width of the semantic head
            space = fit_semantic_space(self.targets.embeddings, np.concatenate([it.phrase.numpy() for it in train]), dim=self.cfg.clip_dim)
        self.space = space
        ctx = make_context(train, self.cfg, space, self.device, self.lam_mode, self.lam_clip, self.lam_score)
        torch.manual_seed(self.train_cfg.seed)
        self.model = Probe(self.cfg).to(self.device)

        def step_fn(model, chunk):
            batch = collate(chunk, self.device)
            terms = loss_terms(model(batch.x), batch, ctx, self.cfg)
            return Step(total_loss(terms, ctx), {name: (float(num), float(den)) for name, (num, den) in terms.items()})

        select = "score" if self.lam_score > 0 else "mode"  # early stopping follows the score term only; a probe without it follows its mode term
        self.result = fit(self.model, train, val, step_fn, self.train_cfg, select=select)

    @torch.no_grad()
    def outputs(self, keys: list[str], batch: int = 32, heads: bool = True) -> dict[str, dict[str, np.ndarray]]:
        """Per head, per episode, the output at every step of the full rollout (float32): ``score`` is the failure probability; with
        ``heads`` also ``mode`` (T, n_modes) logits and ``sem`` (T, clip_dim) raw semantic vectors, for the heads the architecture has."""
        if self.model is None:
            raise RuntimeError("fit() or load() has not been called")
        self.model.eval()
        out: dict[str, dict[str, np.ndarray]] = {}
        for i in range(0, len(keys), batch):
            chunk = keys[i : i + batch]
            x = [self.features.arrays[k] for k in chunk]
            padded = torch.zeros(len(chunk), max(len(a) for a in x), self.cfg.d_in, device=self.device)
            for j, a in enumerate(x):
                padded[j, : len(a)] = a
            result = self.model(padded)
            result["score"] = torch.sigmoid(result["score"])
            for name, tensor in result.items():
                if name == "score" or heads:
                    array = tensor.cpu().numpy().astype(np.float32)
                    out.setdefault(name, {}).update({k: array[j, : len(x[j])] for j, k in enumerate(chunk)})
        return out

    def score(self, keys: list[str], batch: int = 32) -> dict[str, np.ndarray]:
        return self.outputs(keys, batch, heads=False)["score"]

    def state_bytes(self) -> int | None:
        """The probe's weights; scoring also needs the feature statistics (two vectors of the feature width), which are counted."""
        if self.model is None:
            return None
        return sum(p.numel() * p.element_size() for p in self.model.parameters()) + self.features.mean.nbytes + self.features.std.nbytes

    def save(self, path: str) -> None:
        torch.save({"cfg": asdict(self.cfg), "variant": self.label_variant, "state": self.model.state_dict(), "mean": self.features.mean,
                    "std": self.features.std, "best_epoch": self.result.best_epoch if self.result else None}, path)

    def load(self, path: str) -> None:
        saved = torch.load(path, map_location=self.device, weights_only=False)
        if saved["cfg"] != asdict(self.cfg) or saved["variant"] != self.label_variant:
            raise ValueError(f"{path} holds {saved['cfg']} / {saved['variant']}, this scorer is {asdict(self.cfg)} / {self.label_variant}")
        if not (np.array_equal(saved["mean"], self.features.mean) and np.array_equal(saved["std"], self.features.std)):
            raise ValueError(f"{path} was trained on features standardised with other statistics than the ones of this scorer")
        self.model = Probe(self.cfg).to(self.device)
        self.model.load_state_dict(saved["state"])
