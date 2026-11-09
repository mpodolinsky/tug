"""The target space of the semantic (CLIP) head: a 64-d PCA of the CLIP phrase embeddings, fitted on the training split only.

The head predicts, at every step, a vector that should be close to the target of the step's phrase and far from the targets of the other
phrases of the training split (the "bank"); see ``losses.py``. A phrase that occurs only outside the training split has a target (it is
projected like the others) but is not in the bank.
"""

from dataclasses import dataclass

import numpy as np

SEMANTIC_DIM = 64


@dataclass(frozen=True)
class SemanticSpace:
    target: np.ndarray  # (P, dim) unit-norm projection of every phrase embedding
    bank: np.ndarray  # (B, dim) targets of the phrases that occur in the training split
    bank_index: np.ndarray  # (P,) position of each phrase in the bank, -1 for phrases outside the training split
    explained: float  # share of the variance of the training phrases that the dim components keep


def fit_semantic_space(embeddings: np.ndarray, train_phrase_ids: np.ndarray, dim: int = SEMANTIC_DIM) -> SemanticSpace:
    """PCA fitted on the embeddings of the phrases that occur in ``train_phrase_ids`` (-1 entries ignored), applied to all phrases."""
    ids = np.unique(train_phrase_ids[train_phrase_ids >= 0])
    if len(ids) < 2:
        raise ValueError("the semantic space needs at least two distinct training phrases")
    mean = embeddings[ids].mean(axis=0)
    _, singular, vt = np.linalg.svd(embeddings[ids] - mean, full_matrices=False)
    projected = (embeddings - mean) @ vt[:dim].T
    target = (projected / np.linalg.norm(projected, axis=1, keepdims=True)).astype(np.float32)
    bank_index = np.full(len(embeddings), -1, dtype=np.int64)
    bank_index[ids] = np.arange(len(ids))
    return SemanticSpace(target, target[ids], bank_index, float((singular[:dim] ** 2).sum() / (singular**2).sum()))
