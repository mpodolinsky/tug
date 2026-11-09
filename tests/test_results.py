import json

import numpy as np
import pytest

from fp.data.splits import Split
from fp.results import code_version, is_done, read_history, read_scores, run_dir, split_hash, write_run


def scores():
    return {"a/ep0": np.array([0.1, 0.9], dtype=np.float32), "b/ep1": np.array([0.5], dtype=np.float32)}


def test_a_run_is_written_and_read_back(tmp_path):
    path = run_dir("ds", "task_seed0", "lstm_plain__onset_pre", tmp_path)
    assert path == tmp_path / "ds" / "task_seed0" / "lstm_plain__onset_pre" and not is_done(path)
    write_run(path, scores(), {"best_epoch": 3}, lambda p: open(p, "w").write("weights"))
    assert is_done(path) and read_history(path) == {"best_epoch": 3} and (path / "weights.pt").read_text() == "weights"
    back = read_scores(path)
    assert set(back) == set(scores()) and all(np.array_equal(back[k], v) and back[k].dtype == np.float32 for k, v in scores().items())


def test_weights_are_optional(tmp_path):
    path = run_dir("ds", "s", "m", tmp_path)
    write_run(path, scores(), {})
    assert is_done(path) and not (path / "weights.pt").exists()


def test_extra_arrays_files_are_written_next_to_the_scores_and_only_when_given(tmp_path):
    extras = {"mode_logits": {"a/ep0": np.ones((2, 10), np.float32)}, "semantic_space": {"target": np.eye(3, dtype=np.float32), "bank_index": np.arange(3)}}
    path = run_dir("ds", "s", "with", tmp_path)
    write_run(path, scores(), {}, extras=extras)
    with np.load(path / "mode_logits.npz") as z:
        assert z.files == ["a/ep0"] and z["a/ep0"].shape == (2, 10)
    with np.load(path / "semantic_space.npz") as z:
        assert np.array_equal(z["target"], np.eye(3)) and np.array_equal(z["bank_index"], np.arange(3))
    bare = run_dir("ds", "s", "without", tmp_path)
    write_run(bare, scores(), {})
    assert sorted(p.name for p in bare.iterdir()) == ["history.json", "scores.npz"]


def test_a_run_is_never_overwritten(tmp_path):
    path = run_dir("ds", "s", "m", tmp_path)
    write_run(path, scores(), {"v": 1})
    with pytest.raises(FileExistsError):
        write_run(path, scores(), {"v": 2})
    assert read_history(path) == {"v": 1}


def test_a_write_that_fails_halfway_leaves_nothing_that_looks_finished(tmp_path):
    path = run_dir("ds", "s", "m", tmp_path)

    def broken(_):
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        write_run(path, scores(), {}, broken)
    assert not path.exists() and not is_done(path)
    write_run(path, scores(), {"ok": True})  # a stale partial directory does not block the retry
    assert is_done(path)


def test_the_split_hash_identifies_the_exact_episodes():
    base = Split("s", ["a", "b"], ["c"], ["d"], ["e"], ["f"], 0)
    same = Split("other name", ["a", "b"], ["c"], ["d"], ["e"], ["f"], 7)
    moved = Split("s", ["a"], ["b", "c"], ["d"], ["e"], ["f"], 0)  # one episode moved from train to val
    assert split_hash(base) == split_hash(same) != split_hash(moved)


def test_the_code_version_changes_with_the_source_not_with_the_time():
    a, b = code_version(), code_version()
    assert a == b and len(a["fp_sha256"]) == 64 and set(a) == {"git_head", "fp_sha256"}
    json.dumps(a)
