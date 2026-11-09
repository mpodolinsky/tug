import csv

import numpy as np
import pytest

from fp import config
from fp.data.features import load_features
from fp.data.pooling import pool_modalities, pool_rollout, prepare_dataset

D = config.MODALITY_DIM


def tiny(t=2, n_img=3, n_lang=4, d=4, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(t, n_img, d)).astype(np.float16), rng.normal(size=(t, n_img, d)).astype(np.float16),
            rng.normal(size=(t, n_lang, d)).astype(np.float16), np.array([[1, 1, 0, 0]] * t, dtype=bool))


# ---------------------------------------------------------------- pool_modalities
def test_columns_are_base_then_wrist_then_language():
    base, wrist, lang, mask = tiny()
    x = pool_modalities(base, wrist, lang, mask)
    assert x.shape == (2, 12) and x.dtype == np.float32
    assert np.allclose(x[:, 0:4], base.astype(np.float32).mean(axis=1))
    assert np.allclose(x[:, 4:8], wrist.astype(np.float32).mean(axis=1))
    assert np.allclose(x[:, 8:12], lang.astype(np.float32)[:, :2].mean(axis=1))  # the first two tokens are the real ones


def test_padded_language_tokens_do_not_change_the_result():
    base, wrist, lang, mask = tiny()
    changed = lang.copy()
    changed[:, 2:] = 1000.0  # padding positions
    assert np.array_equal(pool_modalities(base, wrist, lang, mask), pool_modalities(base, wrist, changed, mask))


def test_a_step_without_valid_language_tokens_gives_zeros_not_nan():
    base, wrist, lang, _ = tiny()
    x = pool_modalities(base, wrist, lang, np.zeros((2, 4), dtype=bool))
    assert np.isfinite(x).all() and np.all(x[:, 8:] == 0)


def test_image_tokens_all_count_and_the_number_of_image_tokens_may_differ_by_policy():
    for n_img in (64, 256):  # GR00T and pi0.5
        base, wrist, lang, mask = tiny(n_img=n_img)
        assert pool_modalities(base, wrist, lang, mask).shape == (2, 12)


# ---------------------------------------------------------------- pool_rollout and prepare_dataset
def write_rollout(folder, steps, seed, has_features=None):
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    arrays = dict(base_image=rng.normal(size=(steps, 2, D)).astype(np.float16), wrist_image=rng.normal(size=(steps, 2, D)).astype(np.float16),
                  language=rng.normal(size=(steps, 3, D)).astype(np.float16), language_mask=np.array([[1, 1, 0]] * steps, dtype=bool))
    if has_features is not None:
        arrays["has_features"] = np.array(has_features)
    np.savez(folder / "rollout.npz", **arrays)
    return arrays


def test_pool_rollout_matches_pool_modalities(tmp_path):
    a = write_rollout(tmp_path / "ep", 3, 0)
    x = pool_rollout(tmp_path / "ep" / "rollout.npz")
    assert x.shape == (3, config.FEATURE_DIM)
    assert np.array_equal(x, pool_modalities(a["base_image"], a["wrist_image"], a["language"], a["language_mask"]))


def test_a_rollout_marked_as_having_no_features_is_an_error(tmp_path):
    write_rollout(tmp_path / "ep", 3, 0, has_features=False)
    with pytest.raises(ValueError, match="no backbone features"):
        pool_rollout(tmp_path / "ep" / "rollout.npz")
    write_rollout(tmp_path / "ok", 3, 0, has_features=True)
    assert pool_rollout(tmp_path / "ok" / "rollout.npz").shape[0] == 3


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """A 3-episode dataset 'libero10_groot' with a manifest in the data folder and rollouts in a separate raw folder."""
    data, raw = tmp_path / "data", tmp_path / "raw"
    monkeypatch.setattr(config, "DATA", data)
    (data / "libero10_groot").mkdir(parents=True)
    steps = {"normal/t0/ep000": 3, "normal/t0/ep001": 5, "occluded/t1/ep000": 2}
    with open(data / "libero10_groot" / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dir", "success", "task", "n_policy", "replan_steps", "vlm_failure_onset_frame", "scene_variant"])
        w.writeheader()
        for key, n in steps.items():
            w.writerow(dict(dir=key, success="True", task=key.split("/")[1], n_policy=n, replan_steps=8, vlm_failure_onset_frame="", scene_variant="normal"))
    arrays = {key: write_rollout(raw / key, n, i) for i, (key, n) in enumerate(steps.items())}
    return raw, steps, arrays


@pytest.mark.parametrize("workers", [1, 2])
def test_prepare_writes_every_episode_in_the_manifest_with_the_pooled_values(dataset, workers):
    raw, steps, arrays = dataset
    out = prepare_dataset("libero10_groot", raw, workers=workers)
    assert out.name == "features.npz" and not out.with_name("features.npz.tmp").exists()
    got = load_features("libero10_groot", list(steps))  # also validates width and dtype
    for key, n in steps.items():
        a = arrays[key]
        assert got[key].shape == (n, config.FEATURE_DIM)
        assert np.array_equal(got[key], pool_modalities(a["base_image"], a["wrist_image"], a["language"], a["language_mask"]))


def test_prepare_never_overwrites_an_existing_file(dataset):
    raw, _, _ = dataset
    prepare_dataset("libero10_groot", raw, workers=1)
    with pytest.raises(FileExistsError):
        prepare_dataset("libero10_groot", raw, workers=1)


def test_prepare_reports_missing_rollouts_before_writing_anything(dataset):
    raw, _, _ = dataset
    (raw / "normal/t0/ep001/rollout.npz").unlink()
    with pytest.raises(FileNotFoundError, match="1 rollouts not found"):
        prepare_dataset("libero10_groot", raw, workers=1)
    assert not (config.DATA / "libero10_groot" / "features.npz").exists()


def test_a_rollout_with_the_wrong_number_of_steps_leaves_no_partial_file(dataset):
    raw, _, _ = dataset
    write_rollout(raw / "normal/t0/ep001", 7, 9)  # the manifest says 5 steps
    with pytest.raises(ValueError, match="the manifest says 5 steps"):
        prepare_dataset("libero10_groot", raw, workers=1)
    folder = config.DATA / "libero10_groot"
    assert not (folder / "features.npz").exists() and not (folder / "features.npz.tmp").exists()
