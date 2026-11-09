import numpy as np
import torch

from fp.data.features import fit_standardiser, standardise
from fp.train.standardised import standardise_features


def raw_features():
    rng = np.random.default_rng(0)
    return {f"e{i}": rng.normal(2.0, 3.0, (n, 5)).astype(np.float32) for i, n in enumerate((4, 9, 6, 7))}


def test_every_episode_equals_the_cpu_standardisation_with_the_training_statistics():
    raw = raw_features()
    train = ["e0", "e2"]
    mean, std = fit_standardiser([raw[k] for k in train])
    expected = {k: standardise(v, mean, std) for k, v in raw.items()}
    s = standardise_features(raw, train)
    assert np.array_equal(s.mean, mean) and np.array_equal(s.std, std) and s.train_keys == frozenset(train)
    for key, x in s.arrays.items():
        assert x.dtype == torch.float32 and np.array_equal(x.numpy(), expected[key])  # bit-identical, not just close


def test_the_statistics_use_the_training_episodes_only():
    raw = raw_features()
    a = standardise_features({k: v.copy() for k, v in raw.items()}, ["e0", "e1"])
    b = standardise_features({k: v.copy() for k, v in raw.items()}, ["e2", "e3"])
    assert not np.allclose(a.mean, b.mean)


def test_the_raw_features_are_consumed_so_they_are_never_held_twice():
    raw = raw_features()
    standardise_features(raw, ["e0"])
    assert raw == {}
