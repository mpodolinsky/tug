"""SAFE's two learned failure predictors (Gu et al., NeurIPS 2025, arXiv 2506.09937), written from SAFE's code in its default configuration:
``IndepModel`` in failure_prob/model/indep.py (here ``SafeMLP``) and ``LstmModel`` in failure_prob/model/lstm.py (here ``SafeLSTM``).

SAFE-MLP:  an MLP maps the features of each step to a scalar in (0, 1), independently of the other steps; the failure score at step t is
           the running sum of these scalars up to t.      (SAFE's defaults: 2 layers, hidden 256, sigmoid output, cumsum on)
SAFE-LSTM: an LSTM reads the features of the steps so far; a linear layer and a sigmoid turn its state into a failure probability at each
           step, with no accumulation.                   (SAFE's defaults: 1 layer, hidden 256, full history, no dropout)

Only these default branches are implemented. SAFE's other options (more layers, a different output activation, a history window, dropout,
a cumulative sum for the LSTM) are not used by its default configuration and are not part of this baseline. The parameter names are
SAFE's (``projector``, ``lstm``, ``fc``), so a state dict of SAFE's model loads into these modules (``tests/test_safe_models.py``).
Both are causal, and one class serves every dataset.
"""

import torch
from torch import nn


class SafeMLP(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Sigmoid())

    def step_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, dim) -> (B, T): the scalar in (0, 1) of every step."""
        return self.projector(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, dim) -> (B, T): the cumulative failure score, the running sum of the step probabilities."""
        return torch.cumsum(self.step_probabilities(x), dim=1)


class SafeLSTM(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.lstm = nn.LSTM(dim, hidden, 1, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, dim) -> (B, T): the failure probability at every step."""
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out)).squeeze(-1)
