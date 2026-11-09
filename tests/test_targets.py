import csv
import json

import numpy as np
import pytest

from fp import config
from fp.data.targets import MODE_NAMES, build_targets, load_targets, phrase_ids, read_labels

INDEX = {"reach": 0, "grasp": 1, "lift": 2}


def seg(start, end, phrase):
    return {"policy_step_start": start, "policy_step_end": end, "phrase": phrase}


# ---------------------------------------------------------------- phrase_ids
def test_each_step_gets_the_phrase_of_its_segment():
    ids = phrase_ids([seg(0, 2, "reach"), seg(3, 5, "grasp")], 6, INDEX)
    assert list(ids) == [0, 0, 0, 1, 1, 1] and ids.dtype == np.int32


def test_a_step_shared_by_two_segments_takes_the_later_phrase():
    ids = phrase_ids([seg(0, 3, "reach"), seg(3, 5, "grasp")], 6, INDEX)  # step 3 is in both
    assert list(ids) == [0, 0, 0, 1, 1, 1]


def test_steps_after_a_segment_repeat_its_phrase_until_the_next_one():
    ids = phrase_ids([seg(0, 1, "reach"), seg(4, 5, "grasp")], 7, INDEX)
    assert list(ids) == [0, 0, 0, 0, 1, 1, 1]  # steps 2, 3 repeat reach; step 6 repeats grasp


def test_steps_before_the_first_segment_stay_without_a_phrase():
    ids = phrase_ids([seg(2, 3, "grasp")], 5, INDEX)
    assert list(ids) == [-1, -1, 1, 1, 1]


def test_an_empty_timeline_gives_no_phrase_anywhere():
    assert list(phrase_ids([], 4, INDEX)) == [-1, -1, -1, -1]


def test_a_segment_that_runs_past_the_end_is_clipped_and_one_past_it_is_ignored():
    assert list(phrase_ids([seg(0, 99, "reach")], 3, INDEX)) == [0, 0, 0]
    assert list(phrase_ids([seg(10, 12, "reach")], 3, INDEX)) == [-1, -1, -1]


# ---------------------------------------------------------------- read_labels
def write_labels(folder, timeline=None, mode=None):
    folder.mkdir(parents=True, exist_ok=True)
    if timeline is not None:
        (folder / "labels.json").write_text(json.dumps({"semantic_timeline": timeline}))
    if mode is not None:
        np.savez(folder / "labels.npz", fail_mode=np.array(mode))


def test_read_labels_returns_timeline_and_mode(tmp_path):
    write_labels(tmp_path / "ep", [seg(0, 1, "reach")], "grasp_failure")
    assert read_labels(tmp_path / "ep") == ([seg(0, 1, "reach")], "grasp_failure")


def test_missing_files_give_an_empty_timeline_and_no_mode(tmp_path):
    (tmp_path / "ep").mkdir()
    assert read_labels(tmp_path / "ep") == ([], None)


def test_an_empty_mode_string_means_no_mode(tmp_path):
    write_labels(tmp_path / "ep", [], "")
    assert read_labels(tmp_path / "ep")[1] is None


# ---------------------------------------------------------------- build_targets / load_targets
def fake_embed(phrases):
    return np.eye(len(phrases), 512, dtype=np.float32)


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    data, labels = tmp_path / "data", tmp_path / "labels"
    monkeypatch.setattr(config, "DATA", data)
    (data / "libero10_groot").mkdir(parents=True)
    rows = [("normal/t0/ep000", "True", 4, ""), ("normal/t0/ep001", "False", 6, "16"), ("occluded/t1/ep000", "True", 3, "")]
    with open(data / "libero10_groot" / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dir", "success", "task", "n_policy", "replan_steps", "vlm_failure_onset_frame", "scene_variant"])
        w.writeheader()
        for key, ok, n, frame in rows:
            w.writerow(dict(dir=key, success=ok, task=key.split("/")[1], n_policy=n, replan_steps=8, vlm_failure_onset_frame=frame, scene_variant="normal"))
    write_labels(labels / "normal/t0/ep000", [seg(0, 3, "reach")], "")
    write_labels(labels / "normal/t0/ep001", [seg(0, 2, "reach"), seg(3, 5, "grasp")], "grasp_failure")
    (labels / "occluded/t1/ep000").mkdir(parents=True)  # a success without any label file
    return labels


def test_targets_hold_phrases_ids_and_modes(dataset):
    out = build_targets("libero10_groot", dataset, embed=fake_embed)
    assert out.name == "targets.npz" and not out.with_name("targets.tmp.npz").exists()
    t = load_targets("libero10_groot")
    assert t.phrases == ["grasp", "reach"]  # sorted
    assert t.embeddings.shape == (2, 512) and t.embeddings.dtype == np.float32
    assert list(t.phrase_ids["normal/t0/ep000"]) == [1, 1, 1, 1]
    assert list(t.phrase_ids["normal/t0/ep001"]) == [1, 1, 1, 0, 0, 0]
    assert list(t.phrase_ids["occluded/t1/ep000"]) == [-1, -1, -1]  # no labels: no phrase
    assert t.modes == {"normal/t0/ep001": MODE_NAMES.index("grasp_failure")}  # failures only


def test_the_embedding_function_gets_the_sorted_phrases_once(dataset):
    seen = []

    def spy(phrases):
        seen.append(list(phrases))
        return fake_embed(phrases)

    build_targets("libero10_groot", dataset, embed=spy)
    assert seen == [["grasp", "reach"]]


def test_a_failure_without_a_valid_mode_is_an_error(dataset):
    write_labels(dataset / "normal/t0/ep001", [seg(0, 2, "reach")], "not_a_mode")
    with pytest.raises(ValueError, match="no valid failure mode"):
        build_targets("libero10_groot", dataset, embed=fake_embed)
    assert not (config.DATA / "libero10_groot" / "targets.npz").exists()


def test_a_failure_without_label_files_is_an_error(dataset):
    (dataset / "normal/t0/ep001/labels.npz").unlink()
    with pytest.raises(ValueError, match="no valid failure mode"):
        build_targets("libero10_groot", dataset, embed=fake_embed)


def test_an_existing_file_is_never_overwritten(dataset):
    build_targets("libero10_groot", dataset, embed=fake_embed)
    with pytest.raises(FileExistsError):
        build_targets("libero10_groot", dataset, embed=fake_embed)


def test_a_wrong_number_of_embeddings_is_an_error(dataset):
    with pytest.raises(ValueError, match="embeddings for"):
        build_targets("libero10_groot", dataset, embed=lambda p: fake_embed(p)[:1])


def test_a_missing_file_is_a_file_not_found_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_targets("libero10_groot")


def test_the_ten_failure_modes_are_those_of_the_earlier_experiments():
    assert len(MODE_NAMES) == 10 and MODE_NAMES[0] == "wrong_object" and MODE_NAMES[-1] == "timeout_or_insufficient_progress"
