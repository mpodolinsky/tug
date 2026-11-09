import csv

import pytest

from fp import config
from fp.data.manifest import Episode, load_manifest, read_manifest

COLUMNS = ["dir", "success", "task", "n_policy", "replan_steps", "vlm_failure_onset_frame"]


def write(path, rows, condition_column="scene_variant"):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS + [condition_column])
        w.writeheader()
        for r in rows:
            w.writerow({**dict(zip(COLUMNS, r[:6])), condition_column: r[6]})


def test_failure_onset_is_the_labelled_frame_in_policy_steps(tmp_path):
    p = tmp_path / "manifest.csv"
    write(p, [("normal/t0/ep000", "True", "t0", 30, 8, "", "normal"), ("occluded/t0/ep001", "False", "t0", 40, 8, "100", "occluded")])
    ok, fail = read_manifest(p)
    assert ok == Episode("normal/t0/ep000", True, "t0", 30, None, "normal")
    assert fail == Episode("occluded/t0/ep001", False, "t0", 40, 12, "occluded")  # 100 // 8


def test_the_frame_may_be_written_as_a_float(tmp_path):
    p = tmp_path / "manifest.csv"
    write(p, [("a/ep0", "False", "t", 40, 5, "62.0", "x")])
    assert read_manifest(p)[0].onset == 12


def test_libero_x_manifests_use_the_level_as_condition(tmp_path):
    p = tmp_path / "manifest.csv"
    write(p, [("LEVEL2/t/ep0", "True", "t", 10, 8, "", "LEVEL2")], condition_column="level")
    assert read_manifest(p)[0].condition == "LEVEL2"


def test_a_failure_without_a_labelled_frame_is_an_error(tmp_path):
    p = tmp_path / "manifest.csv"
    write(p, [("a/ep0", "False", "t", 40, 5, "", "x"), ("a/ep1", "False", "t", 40, 5, " ", "x")])
    with pytest.raises(ValueError, match="2 failed episodes have no labelled failure frame"):
        read_manifest(p)


@pytest.mark.parametrize("frame", ["400", "-8"])
def test_an_onset_outside_the_rollout_is_an_error(tmp_path, frame):
    p = tmp_path / "manifest.csv"
    write(p, [("a/ep0", "False", "t", 40, 8, frame, "x")])  # onset 50 or -1 with 40 policy steps
    with pytest.raises(ValueError, match="outside the rollout"):
        read_manifest(p)


def test_the_last_policy_step_is_a_valid_onset(tmp_path):
    p = tmp_path / "manifest.csv"
    write(p, [("a/ep0", "False", "t", 40, 8, "319", "x")])  # 319 // 8 = 39 = n_policy - 1
    assert read_manifest(p)[0].onset == 39


def test_load_manifest_reads_the_datasets_data_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA", tmp_path)
    folder = tmp_path / "libero10_groot"
    folder.mkdir()
    write(folder / "manifest.csv", [("a/ep0", "True", "t", 5, 8, "", "normal")])
    assert [e.key for e in load_manifest("libero10_groot")] == ["a/ep0"]


def test_a_missing_manifest_is_a_file_not_found_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_manifest("libero10_groot")
