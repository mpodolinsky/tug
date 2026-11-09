"""train_unit on a small synthetic dataset: full-width features, two architectures, two epochs, CPU."""

import json

import numpy as np
import pytest

from fp.config import FEATURE_DIM
from fp.data.manifest import Episode
from fp.data.splits import Split
from fp.data.targets import MODE_NAMES, Targets
from fp.grid import Unit
from fp.results import is_done, read_history, read_scores, run_dir, split_hash
from fp.train import sweep
from fp.train.sweep import Inputs, Settings, train_unit

N_PHRASES, WINDOW, ARCHS = 100, 3, ["lstm_plain", "mlp_modes_clip"]


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    episodes, ids, modes, raw = {}, {}, {}, {}
    for i in range(40):
        fail = i % 2 == 0
        n = int(rng.integers(12, 20))
        key = f"{'f' if fail else 's'}{i:02d}"
        episodes[key] = Episode(key, not fail, f"t{i % 4}", n, int(rng.integers(5, n - 1)) if fail else None, "a")
        ids[key] = rng.integers(0, N_PHRASES, n).astype(np.int32)
        raw[key] = rng.normal(size=(n, FEATURE_DIM)).astype(np.float32) + (1.0 if fail else 0.0)
        if fail:
            modes[key] = int(rng.integers(0, len(MODE_NAMES)))
    keys = list(episodes)
    split = Split("task_seed4", keys[:24], keys[24:30], keys[30:34], keys[34:37], keys[37:], seed=4)
    emb = rng.normal(size=(N_PHRASES, 512)).astype(np.float32)  # CLIP width: the PCA must be able to keep the semantic head's 64 components
    targets = Targets([f"p{j}" for j in range(N_PHRASES)], emb / np.linalg.norm(emb, axis=1, keepdims=True), ids, modes)
    return Inputs("synthetic", episodes, targets, {split.name: split}, WINDOW), raw, split


@pytest.fixture
def patched(world, monkeypatch):
    inputs, raw, _ = world
    monkeypatch.setattr(sweep, "load_features", lambda dataset, keys: {k: raw[k].copy() for k in keys})
    return inputs


def settings(tmp_path, **kw):
    return Settings(tmp_path, max_epochs=2, patience=2, **kw)


UNIT = Unit("synthetic", "task_seed4", "onset_pre")


def test_every_requested_architecture_gets_a_complete_run(world, patched, tmp_path):
    inputs, raw, split = world
    messages = []
    assert train_unit(UNIT, patched, settings(tmp_path), "cpu", messages.append, ARCHS) == 2
    for arch in ARCHS:
        path = run_dir("synthetic", "task_seed4", f"{arch}__onset_pre", tmp_path)
        assert is_done(path) and (path / "weights.pt").exists()
        scores = read_score_checked(path, inputs)
        info = read_history(path)
        assert info["seed"] == 4 and info["split_sha256"] == split_hash(split) and info["epochs"] <= 2 and info["variant"] == "onset_pre"
        assert info["train_cfg"]["seed"] == 4 and info["weights_saved"] and info["code"]["fp_sha256"] and 0 <= info["best_epoch"] < info["epochs"]
        assert set(info["val"]) >= {"score"} and len(info["val"]["score"]) == info["epochs"]
    assert any("features ready" in m for m in messages) and sum(arch in m for m in messages for arch in ARCHS) >= 2


def read_score_checked(path, inputs):
    scores = read_scores(path)
    assert set(scores) == set(inputs.episodes)  # every episode of the dataset, not only the held-out ones
    for key, p in scores.items():
        assert p.shape == (inputs.episodes[key].n_steps,) and p.dtype == np.float32 and np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
    return scores


def test_runs_that_exist_are_skipped_so_a_sweep_can_resume(world, patched, tmp_path, monkeypatch):
    train_unit(UNIT, patched, settings(tmp_path), "cpu", lambda m: None, ARCHS[:1])
    loaded = []
    monkeypatch.setattr(sweep, "load_features", lambda dataset, keys: loaded.append(1) or {})
    assert train_unit(UNIT, patched, settings(tmp_path), "cpu", lambda m: None, ARCHS[:1]) == 0 and loaded == []  # nothing to do: no data loaded
    scores_before = read_scores(run_dir("synthetic", "task_seed4", "lstm_plain__onset_pre", tmp_path))
    monkeypatch.setattr(sweep, "load_features", lambda dataset, keys: {k: world[1][k].copy() for k in keys})
    assert train_unit(UNIT, patched, settings(tmp_path), "cpu", lambda m: None, ARCHS) == 1  # only the missing architecture is trained
    after = read_scores(run_dir("synthetic", "task_seed4", "lstm_plain__onset_pre", tmp_path))
    assert all(np.array_equal(scores_before[k], after[k]) for k in scores_before)


def test_a_probe_that_fails_is_logged_and_skipped_and_the_others_still_run(world, patched, tmp_path, monkeypatch):
    real = sweep.train_probe

    def flaky(cfg, *args):
        if cfg.name == ARCHS[0]:
            raise RuntimeError("CUDA out of memory")
        return real(cfg, *args)

    monkeypatch.setattr(sweep, "train_probe", flaky)
    messages = []
    assert train_unit(UNIT, patched, settings(tmp_path), "cpu", messages.append, ARCHS) == 1
    assert not run_dir("synthetic", "task_seed4", f"{ARCHS[0]}__onset_pre", tmp_path).exists()
    assert is_done(run_dir("synthetic", "task_seed4", f"{ARCHS[1]}__onset_pre", tmp_path))
    assert any("FAILED" in m and "CUDA out of memory" in m for m in messages)


def test_head_outputs_are_saved_only_for_the_architectures_that_have_the_heads(world, patched, tmp_path):
    inputs, _, _ = world
    train_unit(UNIT, patched, settings(tmp_path), "cpu", lambda m: None, ARCHS)
    plain = run_dir("synthetic", "task_seed4", f"{ARCHS[0]}__onset_pre", tmp_path)
    full = run_dir("synthetic", "task_seed4", f"{ARCHS[1]}__onset_pre", tmp_path)
    assert sorted(p.name for p in plain.iterdir()) == ["history.json", "scores.npz", "weights.pt"]
    assert sorted(p.name for p in full.iterdir()) == ["history.json", "mode_logits.npz", "scores.npz", "semantic.npz", "semantic_space.npz", "weights.pt"]
    with np.load(full / "mode_logits.npz") as modes, np.load(full / "semantic.npz") as sem, np.load(full / "semantic_space.npz") as space:
        assert set(modes.files) == set(sem.files) == set(inputs.episodes)  # every episode of the dataset
        for key, e in inputs.episodes.items():
            assert modes[key].shape == (e.n_steps, len(MODE_NAMES)) and sem[key].shape == (e.n_steps, 64) and np.isfinite(sem[key]).all()
        assert space["target"].shape == (N_PHRASES, 64) and space["bank_index"].shape == (N_PHRASES,)
        assert (space["bank_index"] >= 0).sum() > 64  # the bank holds the training phrases
    assert read_history(full)["heads_saved"] is True


def test_head_outputs_can_be_switched_off(world, patched, tmp_path):
    train_unit(UNIT, patched, settings(tmp_path, save_heads=False), "cpu", lambda m: None, ARCHS[1:])
    path = run_dir("synthetic", "task_seed4", f"{ARCHS[1]}__onset_pre", tmp_path)
    assert sorted(p.name for p in path.iterdir()) == ["history.json", "scores.npz", "weights.pt"] and read_history(path)["heads_saved"] is False


def test_weights_and_heads_are_skipped_when_the_disk_is_nearly_full_but_scores_are_kept(world, patched, tmp_path):
    train_unit(UNIT, patched, settings(tmp_path, min_free_gb=1e9), "cpu", lambda m: None, ARCHS[1:])
    path = run_dir("synthetic", "task_seed4", f"{ARCHS[1]}__onset_pre", tmp_path)
    assert sorted(p.name for p in path.iterdir()) == ["history.json", "scores.npz"]
    assert read_history(path)["weights_saved"] is False and read_history(path)["heads_saved"] is False
    read_score_checked(path, world[0])


def test_no_weights_when_switched_off(world, patched, tmp_path):
    train_unit(UNIT, patched, settings(tmp_path, save_weights=False), "cpu", lambda m: None, ARCHS[:1])
    assert not (run_dir("synthetic", "task_seed4", f"{ARCHS[0]}__onset_pre", tmp_path) / "weights.pt").exists()


def test_the_history_is_plain_json(world, patched, tmp_path):
    train_unit(UNIT, patched, settings(tmp_path), "cpu", lambda m: None, ARCHS[1:])
    text = (run_dir("synthetic", "task_seed4", f"{ARCHS[1]}__onset_pre", tmp_path) / "history.json").read_text()
    assert set(json.loads(text)["val"]) == {"score", "mode", "clip"}


def test_the_loss_weights_of_a_run_are_recorded_in_its_history(world, patched, tmp_path):
    train_unit(UNIT, patched, settings(tmp_path, lam_mode=0.3, lam_clip=0.1, lam_score=0.5), "cpu", lambda m: None, ARCHS[1:])
    info = read_history(run_dir("synthetic", "task_seed4", f"{ARCHS[1]}__onset_pre", tmp_path))
    assert (info["lam_mode"], info["lam_clip"], info["lam_score"]) == (0.3, 0.1, 0.5)
    train_unit(UNIT, patched, settings(tmp_path / "default"), "cpu", lambda m: None, ARCHS[1:])
    default = read_history(run_dir("synthetic", "task_seed4", f"{ARCHS[1]}__onset_pre", tmp_path / "default"))
    assert (default["lam_mode"], default["lam_clip"], default["lam_score"]) == (1.0, 1.0, 1.0)


def test_a_run_without_the_score_term_keeps_its_mode_logits_and_chooses_its_epoch_on_the_mode_term(world, patched, tmp_path):
    train_unit(UNIT, patched, settings(tmp_path, lam_score=0.0), "cpu", lambda m: None, ARCHS[1:])
    folder = run_dir("synthetic", "task_seed4", f"{ARCHS[1]}__onset_pre", tmp_path)
    info = read_history(folder)
    assert info["lam_score"] == 0.0 and info["best_epoch"] == int(np.argmin(info["val"]["mode"]))
    assert (folder / "mode_logits.npz").exists() and (folder / "scores.npz").exists()  # the score head still outputs something; the mode logits are what the run is for
