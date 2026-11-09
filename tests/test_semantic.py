import numpy as np
import pytest

from fp.train.semantic import fit_semantic_space

P, E, DIM = 12, 16, 4


def embeddings(seed=0):
    x = np.random.default_rng(seed).normal(size=(P, E)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


IDS = np.array([0, 1, 2, 3, 5, 5, 7, 7, 8, 9])  # phrases 4, 6, 10, 11 never occur in the training split


def test_every_phrase_gets_a_unit_norm_target_of_the_requested_width():
    s = fit_semantic_space(embeddings(), IDS, dim=DIM)
    assert s.target.shape == (P, DIM) and s.target.dtype == np.float32
    assert np.allclose(np.linalg.norm(s.target, axis=1), 1.0, atol=1e-5)


def test_the_bank_holds_exactly_the_training_phrases():
    s = fit_semantic_space(embeddings(), IDS, dim=DIM)
    train = np.unique(IDS)
    assert s.bank.shape == (len(train), DIM) and np.array_equal(s.bank, s.target[train])
    assert np.array_equal(np.flatnonzero(s.bank_index >= 0), train)  # the other phrases have no place in the bank
    assert np.array_equal(s.bank_index[train], np.arange(len(train)))


def test_minus_one_entries_mean_no_phrase_and_are_ignored():
    with_gaps = np.concatenate([IDS, np.full(7, -1)])
    a, b = fit_semantic_space(embeddings(), IDS, dim=DIM), fit_semantic_space(embeddings(), with_gaps, dim=DIM)
    assert np.array_equal(a.target, b.target) and np.array_equal(a.bank_index, b.bank_index)


def test_the_space_is_fitted_on_the_training_phrases_only():
    e = embeddings()
    changed = e.copy()
    changed[[4, 6, 10, 11]] = np.random.default_rng(9).normal(size=(4, E))  # phrases absent from the training split
    a, b = fit_semantic_space(e, IDS, dim=DIM), fit_semantic_space(changed, IDS, dim=DIM)
    assert np.array_equal(a.bank, b.bank)  # the PCA did not see them
    assert not np.array_equal(a.target[[4, 6, 10, 11]], b.target[[4, 6, 10, 11]])  # but they are projected like all others


def test_explained_variance_is_a_share_and_grows_with_the_width():
    e = embeddings()
    narrow, wide = fit_semantic_space(e, IDS, dim=2), fit_semantic_space(e, IDS, dim=6)
    assert 0 < narrow.explained < wide.explained <= 1.0


def test_fewer_than_two_distinct_phrases_is_an_error():
    with pytest.raises(ValueError, match="at least two"):
        fit_semantic_space(embeddings(), np.array([3, 3, 3, -1]), dim=DIM)
