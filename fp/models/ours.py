"""Our probes: a trunk over the pooled features of the policy steps and a score head, with optional auxiliary heads.

    features (B, T, 3 * D)      [base | wrist | language], D = 2048 columns each
        -> mha off: used as they are
           mha on : per modality Linear(D, P) + ReLU, then one causal self-attention block over the steps
                    (residual + LayerNorm); the three outputs are concatenated (3 * P)
        -> trunk: LSTM(hidden)  |  per-step MLP (-> 2 * hidden -> hidden)
        -> score head                                   (always)   failure logit per step
        -> mode head                                    (modes)    failure-mode logits per step
        -> semantic head                                (clip)     embedding per step, trained against the CLIP embedding of the step's phrase

Every output at step t depends on steps <= t only, so zero padding after the real steps does not change the outputs on real steps.
The label variant (onset_pre, onset, ...) is a training choice and not part of the model. Initialisation follows torch's global seed.

Ported from the earlier experiments' HeadNet (hidden 256, projection 128, 4 attention heads, heads of one 64-unit layer); its other options
(dropout, action head, last-4-steps trunk, projection without attention) are not carried over.
"""

from dataclasses import dataclass

import torch
from torch import nn

from ..config import MODALITY_DIM

HEAD_WIDTH = 64
TRUNKS = ("lstm", "mlp")


@dataclass(frozen=True)
class ProbeConfig:
    trunk: str = "lstm"  # "lstm" | "mlp"
    mha: bool = False  # per-modality projection + causal attention before the trunk
    modes: bool = False  # failure-mode head
    clip: bool = False  # semantic (CLIP) head
    hidden: int = 256
    d_modality: int = MODALITY_DIM  # columns per modality
    n_modalities: int = 3
    proj: int = 128  # width of each modality after the projection (mha only)
    n_attention_heads: int = 4
    n_modes: int = 10
    clip_dim: int = 64

    def __post_init__(self):
        if self.trunk not in TRUNKS:
            raise ValueError(f"unknown trunk {self.trunk!r}; choose from {TRUNKS}")
        if self.mha and self.proj % self.n_attention_heads:
            raise ValueError(f"proj {self.proj} is not divisible by {self.n_attention_heads} attention heads")

    @property
    def d_in(self) -> int:
        return self.n_modalities * self.d_modality

    @property
    def trunk_in(self) -> int:
        return self.n_modalities * self.proj if self.mha else self.d_in

    @property
    def name(self) -> str:
        """e.g. 'lstm_plain', 'mlp_modes_clip', 'lstm_mha_clip'."""
        heads = "_".join(h for h, on in (("modes", self.modes), ("clip", self.clip)) if on) or "plain"
        return f"{self.trunk}{'_mha' if self.mha else ''}_{heads}"


def head(d_in: int, d_out: int) -> nn.Module:
    return nn.Sequential(nn.Linear(d_in, HEAD_WIDTH), nn.ReLU(), nn.Linear(HEAD_WIDTH, d_out))


class ModalityAttention(nn.Module):
    """Per modality: Linear(D, P) + ReLU, then causal self-attention over the steps with a residual connection and LayerNorm."""

    def __init__(self, cfg: ProbeConfig):
        super().__init__()
        self.d = cfg.d_modality
        n = cfg.n_modalities
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(cfg.d_modality, cfg.proj), nn.ReLU()) for _ in range(n)])
        self.attn = nn.ModuleList([nn.MultiheadAttention(cfg.proj, cfg.n_attention_heads, batch_first=True) for _ in range(n)])
        self.norm = nn.ModuleList([nn.LayerNorm(cfg.proj) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        steps = x.shape[1]
        causal = torch.triu(torch.ones(steps, steps, dtype=torch.bool, device=x.device), diagonal=1)  # True = may not attend
        out = []
        for i, (proj, attn, norm) in enumerate(zip(self.proj, self.attn, self.norm)):
            z = proj(x[..., i * self.d : (i + 1) * self.d])
            out.append(norm(z + attn(z, z, z, attn_mask=causal, need_weights=False)[0]))
        return torch.cat(out, dim=-1)


class Probe(nn.Module):
    """forward(x) with x (B, T, 3 * D) -> {"score": (B, T)} plus "mode" (B, T, n_modes) and "sem" (B, T, clip_dim) when configured."""

    def __init__(self, cfg: ProbeConfig):
        super().__init__()
        self.cfg = cfg
        self.attention = ModalityAttention(cfg) if cfg.mha else None
        if cfg.trunk == "lstm":
            self.trunk = nn.LSTM(cfg.trunk_in, cfg.hidden, num_layers=1, batch_first=True)
        else:
            self.trunk = nn.Sequential(nn.Linear(cfg.trunk_in, 2 * cfg.hidden), nn.ReLU(), nn.Linear(2 * cfg.hidden, cfg.hidden), nn.ReLU())
        self.score = head(cfg.hidden, 1)
        self.mode = head(cfg.hidden, cfg.n_modes) if cfg.modes else None
        self.sem = head(cfg.hidden, cfg.clip_dim) if cfg.clip else None

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """The trunk's state at every step, (B, T, hidden): what every head reads."""
        if x.shape[-1] != self.cfg.d_in:
            raise ValueError(f"expected {self.cfg.d_in} feature columns, got {x.shape[-1]}")
        if self.attention is not None:
            x = self.attention(x)
        return self.trunk(x)[0] if self.cfg.trunk == "lstm" else self.trunk(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.hidden(x)
        out = {"score": self.score(h).squeeze(-1)}
        if self.mode is not None:
            out["mode"] = self.mode(h)
        if self.sem is not None:
            out["sem"] = self.sem(h)
        return out


def all_configs() -> list[ProbeConfig]:
    """The 16 architectures of the suite: trunk x attention x (plain, modes, clip, modes + clip); with the 4 label variants, 64 probes."""
    return [ProbeConfig(trunk=t, mha=a, modes=m, clip=c) for t in TRUNKS for a in (False, True) for m, c in ((False, False), (True, False), (False, True), (True, True))]
