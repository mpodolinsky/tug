import csv
import dataclasses

import numpy as np
import pytest

from fp import config
from fp.data.actions import load_actions, prepare_actions, read_rollout_actions

K, H, A, REPLAN = 4, 6, 7, 2


def write_rollout(folder, steps, seed, executed=0, replan=REPLAN, predicted_differs=False, nan=False):
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    sampled = rng.normal(size=(steps, K, H, A)).astype(np.float32)
    predicted = sampled[:, executed].copy() if not predicted_differs else sampled[:, 1].copy()
    if nan:
        sampled[0, 0, 0, 0] = np.nan
    np.savez(folder / "rollout.npz", sampled_action_chunks=sampled, predicted_action_chunks=predicted, executed_sample_index=np.int32(executed),
             replan_steps=np.int32(replan), base_image=np.zeros((steps, 2, 3), np.float16))  # (the token embeddings are never read)
    return sampled


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """A 3-episode dataset 'libero10_pi05' (replan 5 in the registry; patched to REPLAN) with a manifest in the data folder."""
    data, raw = tmp_path / "data", tmp_path / "raw"
    monkeypatch.setattr(config, "DATA", data)
    monkeypatch.setitem(config.REGISTRY, "libero10_pi05", dataclasses.replace(config.REGISTRY["libero10_pi05"], replan_steps=REPLAN))
    (data / "libero10_pi05").mkdir(parents=True)
    steps = {"normal/t0/ep000": 3, "normal/t0/ep001": 5, "occluded/t1/ep000": 2}
    with open(data / "libero10_pi05" / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dir", "success", "task", "n_policy", "replan_steps", "vlm_failure_onset_frame", "scene_variant"])
        w.writeheader()
        for key, n in steps.items():
            w.writerow(dict(dir=key, success="True", task=key.split("/")[1], n_policy=n, replan_steps=REPLAN, vlm_failure_onset_frame="", scene_variant="normal"))
    arrays = {key: write_rollout(raw / key, n, i) for i, (key, n) in enumerate(steps.items())}
    return raw, steps, arrays


def test_one_rollout_gives_its_sampled_chunks_without_reading_anything_else(tmp_path):
    a = write_rollout(tmp_path / "ep", 4, 0)
    x = read_rollout_actions(tmp_path / "ep" / "rollout.npz", REPLAN)
    assert x.shape == (4, K, H, A) and x.dtype == np.float32 and np.array_equal(x, a)


@pytest.mark.parametrize("kwargs, message", [({"executed": 2}, "not the executed chunk"), ({"predicted_differs": True}, "not the executed chunk"),
                                             ({"replan": 3}, "replan_steps 3"), ({"nan": True}, "finite")])
def test_a_rollout_that_breaks_what_the_baselines_rely_on_is_an_error(tmp_path, kwargs, message):
    write_rollout(tmp_path / "ep", 3, 0, **kwargs)
    with pytest.raises(ValueError, match=message):
        read_rollout_actions(tmp_path / "ep" / "rollout.npz", REPLAN)


def test_the_chunks_must_be_longer_than_the_replan_step(tmp_path):
    write_rollout(tmp_path / "ep", 3, 0, replan=H)  # nothing would overlap between consecutive calls
    with pytest.raises(ValueError, match="finite"):
        read_rollout_actions(tmp_path / "ep" / "rollout.npz", H)


@pytest.mark.parametrize("workers", [1, 2])
def test_prepare_writes_every_episode_of_the_manifest_and_load_reads_it_back(dataset, workers):
    raw, steps, arrays = dataset
    out = prepare_actions("libero10_pi05", raw, workers=workers)
    assert out.name == "actions.npz" and not out.with_name("actions.npz.tmp").exists()
    got = load_actions("libero10_pi05", list(steps))
    for key, n in steps.items():
        assert got[key].shape == (n, K, H, A) and np.array_equal(got[key], arrays[key])
    assert list(load_actions("libero10_pi05", ["normal/t0/ep001"])) == ["normal/t0/ep001"]  # only the keys asked for


def test_prepare_never_overwrites_an_existing_file(dataset):
    raw, _, _ = dataset
    prepare_actions("libero10_pi05", raw, workers=1)
    with pytest.raises(FileExistsError):
        prepare_actions("libero10_pi05", raw, workers=1)


def test_prepare_reports_missing_rollouts_before_writing_anything(dataset):
    raw, _, _ = dataset
    (raw / "normal/t0/ep001/rollout.npz").unlink()
    with pytest.raises(FileNotFoundError, match="1 rollouts not found"):
        prepare_actions("libero10_pi05", raw, workers=1)
    assert not (config.DATA / "libero10_pi05" / "actions.npz").exists()


def test_a_wrong_number_of_policy_calls_or_a_bad_episode_leaves_no_partial_file(dataset):
    raw, _, _ = dataset
    write_rollout(raw / "normal/t0/ep001", 7, 9)  # the manifest says 5 calls
    with pytest.raises(ValueError, match="the manifest says 5"):
        prepare_actions("libero10_pi05", raw, workers=1)
    write_rollout(raw / "normal/t0/ep001", 5, 9, executed=3)
    with pytest.raises(ValueError, match="not the executed chunk"):
        prepare_actions("libero10_pi05", raw, workers=1)
    folder = config.DATA / "libero10_pi05"
    assert not (folder / "actions.npz").exists() and not (folder / "actions.npz.tmp").exists()


def test_load_reports_a_missing_file_a_missing_episode_and_a_wrong_shape(dataset, tmp_path):
    raw, steps, _ = dataset
    with pytest.raises(FileNotFoundError, match="cli.prepare_actions"):
        load_actions("libero10_pi05", list(steps))
    prepare_actions("libero10_pi05", raw, workers=1)
    with pytest.raises(KeyError, match="1 episodes are not in"):
        load_actions("libero10_pi05", list(steps) + ["nope"])
    np.savez(config.DATA / "libero10_pi05" / "actions.npz", **{"normal/t0/ep000": np.zeros((3, K, H, 5), np.float32)})
    with pytest.raises(ValueError, match="expected"):
        load_actions("libero10_pi05", ["normal/t0/ep000"])
