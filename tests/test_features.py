import numpy as np
import pytest

from fp import config
from fp.data.features import fit_standardiser, load_features, standardise


def save(tmp_path, monkeypatch, arrays):
    monkeypatch.setattr(config, "DATA", tmp_path)
    folder = tmp_path / "libero10_groot"
    folder.mkdir()
    np.savez(folder / "features.npz", **arrays)


def good(t=4):
    return np.zeros((t, config.FEATURE_DIM), dtype=np.float32)


def test_returns_only_the_requested_episodes(tmp_path, monkeypatch):
    save(tmp_path, monkeypatch, {"normal/t0/ep000": good(3), "normal/t0/ep001": good(5), "occluded/t1/ep000": good(2)})
    out = load_features("libero10_groot", ["occluded/t1/ep000", "normal/t0/ep000"])
    assert list(out) == ["occluded/t1/ep000", "normal/t0/ep000"] and out["normal/t0/ep000"].shape == (3, config.FEATURE_DIM)


def test_a_missing_file_is_a_file_not_found_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_features("libero10_groot", ["a"])


def test_an_absent_episode_is_a_key_error(tmp_path, monkeypatch):
    save(tmp_path, monkeypatch, {"a": good()})
    with pytest.raises(KeyError, match="1 episodes are not in features.npz"):
        load_features("libero10_groot", ["a", "b"])


@pytest.mark.parametrize("bad", [np.zeros((4, 2048), dtype=np.float32), np.zeros((4, config.FEATURE_DIM), dtype=np.float16),
                                 np.zeros(config.FEATURE_DIM, dtype=np.float32)])
def test_wrong_shape_or_dtype_is_an_error(tmp_path, monkeypatch, bad):
    save(tmp_path, monkeypatch, {"a": bad})
    with pytest.raises(ValueError, match="expected"):
        load_features("libero10_groot", ["a"])
