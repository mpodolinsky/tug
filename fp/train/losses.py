"""The loss of a probe.

    loss = BCE(score, y)                                  weighted by n_neg / n_pos of the training steps, averaged over real steps
         + lam_mode * CE(mode head, mode)                 on the steps with y == 1 of failed episodes, class-balanced   (modes)
         + lam_clip * InfoNCE(semantic head, phrase)      on the steps that have a phrase of the training bank          (clip)

InfoNCE: the unit-normalised semantic output of a step is scored against the bank of target vectors (temperature ``tau``) and must pick the
target of the step's phrase. A step whose phrase is not in the bank (a phrase seen only outside the training split) is left out.
Each term is kept as a numerator and a denominator (``loss_terms``), so that its mean over a whole set is the sum of the numerators over
the sum of the denominators, whatever the batches were. Early stopping follows the score term on the validation episodes, so the
auxiliary heads never choose the epoch.
Padding is never counted: it follows the real steps and the probes are causal, so it cannot change the outputs on real steps either.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..models.ours import ProbeConfig
from .items import Batch, Item
from .semantic import SemanticSpace

TAU = 0.07


@dataclass(frozen=True)
class LossContext:
    pos_weight: torch.Tensor  # scalar, n_neg / n_pos of the training steps
    mode_weight: torch.Tensor | None  # (n_modes,) class weights, 0 for a mode absent from the training steps
    bank: torch.Tensor | None  # (B, dim) semantic targets of the training phrases
    bank_index: torch.Tensor | None  # (P,) bank position of each phrase, -1 outside the training split
    lam_mode: float = 1.0
    lam_clip: float = 1.0
    lam_score: float = 1.0  # 0 trains the heads without the score term (a mode-only probe, the ablation of what the score term adds)


def make_context(train: list[Item], cfg: ProbeConfig, space: SemanticSpace | None, device: str | torch.device = "cpu",
                 lam_mode: float = 1.0, lam_clip: float = 1.0, lam_score: float = 1.0) -> LossContext:
    y = torch.cat([it.y for it in train])
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("the training steps need both positive and negative labels")
    mode_weight = None
    if cfg.modes:
        counts = torch.zeros(cfg.n_modes)
        for it in train:
            if it.mode >= 0:
                counts[it.mode] += float(it.y.sum())  # steps with y == 1 of this failed episode
        present = (counts > 0).sum()
        mode_weight = torch.where(counts > 0, counts.sum() / (present * counts.clamp(min=1)), torch.zeros_like(counts)).to(device)
    bank = bank_index = None
    if cfg.clip:
        bank, bank_index = torch.from_numpy(space.bank).to(device), torch.from_numpy(space.bank_index).to(device)
    return LossContext(((1 - y).sum() / y.sum()).to(device), mode_weight, bank, bank_index, lam_mode, lam_clip, lam_score)


Terms = dict[str, tuple[torch.Tensor, torch.Tensor]]  # term name -> (numerator, denominator); the term's mean is numerator / denominator


def loss_terms(out: dict[str, torch.Tensor], batch: Batch, ctx: LossContext, cfg: ProbeConfig) -> Terms:
    """Each term of the loss as a (numerator, denominator) pair, for the heads the architecture has.

    The pairs of several batches can be pooled exactly: add the numerators, add the denominators, divide once. A term whose batch has no
    eligible step has denominator 0 (for the mode term also when its steps all belong to modes absent from the training steps, weight 0).
    """
    bce = F.binary_cross_entropy_with_logits(out["score"], batch.y, pos_weight=ctx.pos_weight, reduction="none")
    terms = {"score": ((bce * batch.valid).sum(), batch.valid.sum())}
    if cfg.modes:
        on = batch.mode >= 0
        target = batch.mode[on]
        terms["mode"] = (F.cross_entropy(out["mode"][on], target, weight=ctx.mode_weight, reduction="sum"), ctx.mode_weight[target].sum())
    if cfg.clip:
        bank_target = torch.where(batch.phrase >= 0, ctx.bank_index[batch.phrase.clamp(min=0)], torch.full_like(batch.phrase, -1))
        on = bank_target >= 0
        logits = F.normalize(out["sem"][on], dim=-1) @ ctx.bank.T / TAU
        terms["clip"] = (F.cross_entropy(logits, bank_target[on], reduction="sum"), on.sum())
    return terms


def total_loss(terms: Terms, ctx: LossContext) -> torch.Tensor:
    """The training loss of one batch: each term's mean over its own eligible steps, weighted by its lambda (score 1 unless ``lam_score`` says otherwise)."""
    weight = {"score": ctx.lam_score, "mode": ctx.lam_mode, "clip": ctx.lam_clip}
    return sum(weight[name] * num / den for name, (num, den) in terms.items() if den > 0)


def probe_loss(out: dict[str, torch.Tensor], batch: Batch, ctx: LossContext, cfg: ProbeConfig) -> torch.Tensor:
    return total_loss(loss_terms(out, batch, ctx, cfg), ctx)
