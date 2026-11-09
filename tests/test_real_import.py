"""fp.data.real_import and the real-robot dataset's entries (config, leave-one-task-out splits, skipped action baselines), on a synthetic source folder."""

import csv
import json

import numpy as np
import pytest

from fp import config
from fp.baselines.registry import Context, build
from fp.baselines.sweep import BaselineUnit, Settings, run_unit
from fp.data.download import download_dataset
from fp.data.features import load_features
from fp.data.inputs import Inputs
from fp.data.manifest import load_manifest
from fp.data.real_import import check_labels, import_real, read_source_manifest
from fp.data.splits import make_splits, split_plan
from fp.data.targets import MODE_NAMES, load_targets
from fp.eval.select_config import split_group
from fp.grid import split_priority

TASKS = ["basket", "closer_drawer", "tomato"]  # a task name with an underscore on purpose
FIELDS = ["scene_variant", "suite", "task_id", "task", "prompt", "episode", "rollout_id", "success", "n_policy", "n_control", "control_hz", "replan_steps",
          "vlm_failure_onset_frame", "dir"]


def embed(phrases):
    return np.eye(len(phrases), 4, dtype=np.float32)


def episode_plan():
    """(task, index, success, n_policy, replan, onset_frame): 12 per task, every third a success, one failure with 5 control steps per call."""
    plan = []
    for t in TASKS:
        for i in range(12):
            ok = i % 3 == 0
            replan = 5 if (t == "tomato" and i == 1) else 6
            n = 20 + i
            plan.append((t, i, ok, n, replan, None if ok else replan * (8 + i % 5) + 2))  # frame // replan = 8 + i % 5 (+ 0: the +2 stays below one call)
    return plan


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    (root / "cache" / "modality").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows, features = [], {}
    for t, i, ok, n, replan, frame in episode_plan():
        key = f"{t}/ep{i:03d}"
        rows.append({"scene_variant": "", "suite": "", "task_id": "", "task": "", "prompt": f"do {t}", "episode": i, "rollout_id": f"r_{t}_{i}", "success": str(ok), "n_policy": n,
                     "n_control": n * replan, "control_hz": 50.0, "replan_steps": replan, "vlm_failure_onset_frame": "" if ok else frame, "dir": key})
        folder = root / "hf_labels_pr" / key
        folder.mkdir(parents=True)
        timeline = [{"policy_step_start": 0, "policy_step_end": n // 2, "phrase": f"reach {t}"}, {"policy_step_start": n // 2 + 1, "policy_step_end": n - 1, "phrase": f"hold {t}"}]
        labels = {"success": ok, "semantic_timeline": timeline}
        if not ok:
            labels["failure_annotation"] = {"failure_policy_step": frame // replan}
        (folder / "labels.json").write_text(json.dumps(labels))
        np.savez(folder / "labels.npz", fail_mode=np.array("" if ok else MODE_NAMES[5]))
        features[key] = rng.normal(size=(n, config.FEATURE_DIM)).astype(np.float32)
    with open(root / "hf_labels_pr" / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    np.savez(root / "cache" / "modality" / "features_pooled.npz", **features)
    return root, features


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA", tmp_path / "data")
    return tmp_path / "data" / "real8_groot"


# ---------------------------------------------------------------- the import
def test_the_import_writes_manifest_features_and_targets_for_every_episode(source, data):
    root, features = source
    import_real("real8_groot", root, embed)
    episodes = load_manifest("real8_groot")
    assert len(episodes) == 36 and {e.task for e in episodes} == set(TASKS)
    assert {e.condition for e in episodes if e.task == "basket"} == {"basket"}  # the task stands in for the condition
    loaded = load_features("real8_groot", [e.key for e in episodes])
    assert all(np.array_equal(loaded[k], features[k]) for k in features)
    targets = load_targets("real8_groot")
    assert sorted(targets.modes) == sorted(e.key for e in episodes if not e.success) and set(targets.modes.values()) == {5}
    assert len(targets.phrases) == 6 and set(targets.phrase_ids) == set(features)


def test_the_onset_is_the_labelled_frame_divided_by_that_episodes_replan_steps(source, data):
    import_real("real8_groot", source[0], embed)
    by_key = {e.key: e for e in load_manifest("real8_groot")}
    assert by_key["tomato/ep001"].onset == 9 and by_key["basket/ep001"].onset == 9 and by_key["basket/ep000"].onset is None  # ep001 has 5 control steps per call
    first, last = by_key["basket/ep002"], by_key["basket/ep000"]
    assert first.onset == 10 and last.success


def test_the_source_manifest_gets_its_empty_task_column_from_the_directory(source):
    rows, fields = read_source_manifest(source[0] / "hf_labels_pr" / "manifest.csv")
    assert {r["task"] for r in rows} == set(TASKS) and all(r["scene_variant"] == r["task"] for r in rows) and "task" in fields


def test_nothing_existing_is_overwritten(source, data):
    data.mkdir(parents=True)
    (data / "features.npz").write_bytes(b"x")
    with pytest.raises(FileExistsError, match="features.npz"):
        import_real("real8_groot", source[0], embed)
    assert (data / "features.npz").read_bytes() == b"x" and not (data / "manifest.csv").exists()


@pytest.mark.parametrize("damage, message", [("short", "expected finite"), ("nan", "expected finite"), ("missing", "are not in")])
def test_bad_features_stop_the_import_and_leave_no_files(source, data, damage, message):
    root, features = source
    key = "basket/ep002"
    if damage == "short":
        features[key] = features[key][:-1]
    if damage == "nan":
        features[key][3, 5] = np.nan
    if damage == "missing":
        del features[key]
    np.savez(root / "cache" / "modality" / "features_pooled.npz", **features)
    with pytest.raises((ValueError, KeyError), match=message):
        import_real("real8_groot", root, embed)
    assert not list(data.glob("*"))  # the manifest written first was removed again


def test_a_failure_in_the_target_step_removes_what_the_call_wrote(source, data):
    def broken(phrases):
        raise RuntimeError("no CLIP weights")

    with pytest.raises(RuntimeError, match="no CLIP"):
        import_real("real8_groot", source[0], broken)
    assert not list(data.glob("*"))


def test_label_problems_are_found_before_anything_is_written(source, data):
    root, _ = source
    folder = root / "hf_labels_pr" / "basket" / "ep002"
    labels = json.load(open(folder / "labels.json"))
    labels["failure_annotation"]["failure_policy_step"] += 1  # onset step differs from the manifest's frame // replan
    json.dump(labels, open(folder / "labels.json", "w"))
    np.savez(root / "hf_labels_pr" / "tomato" / "ep002" / "labels.npz", fail_mode=np.array("not_a_mode"))
    (root / "hf_labels_pr" / "tomato" / "ep004" / "labels.json").write_text(json.dumps({"success": True, "semantic_timeline": []}))  # wrong flag, no timeline
    rows, _ = read_source_manifest(root / "hf_labels_pr" / "manifest.csv")
    problems = check_labels(rows, root / "hf_labels_pr")
    assert any("basket/ep002" in p and "onset step" in p for p in problems)
    assert any("tomato/ep002" in p and "not_a_mode" in p for p in problems)
    assert any("tomato/ep004" in p and "success is False" in p for p in problems) and any("tomato/ep004" in p and "no semantic timeline" in p for p in problems)
    with pytest.raises(ValueError, match="problems in the labels"):
        import_real("real8_groot", root, embed)
    assert not data.exists() or not list(data.glob("*"))


# ---------------------------------------------------------------- the dataset's entries
def test_the_real_dataset_matches_the_two_point_four_second_window_of_the_other_groot_datasets():
    spec = config.get("real8_groot")
    assert spec.window_seconds == pytest.approx(2.4) and spec.window_steps == 20 and spec.replan_steps == 6 and not spec.has_actions
    assert spec.split_kinds == ("loto", "full", "lotoseeds") and spec.hf_repo == "" and spec.policy == "groot"
    assert config.get("liberox_groot").has_actions and config.get("libero10_pi05").has_actions


def test_leave_one_task_out_has_one_split_per_task_that_holds_that_task_out(source, data):
    import_real("real8_groot", source[0], embed)
    splits = [s for s in make_splits("real8_groot") if s.name.startswith("loto_")]
    assert [s.name for s in splits] == ["loto_basket", "loto_closer_drawer", "loto_tomato"] and {s.seed for s in splits} == {0}
    for s in splits:
        task = s.name.removeprefix("loto_")
        assert {k.split("/")[0] for k in s.unseen} == {task} and len(s.unseen) == 12
        assert task not in {k.split("/")[0] for k in s.train + s.val + s.calibration + s.seen}
        assert len({k.split("/")[0] for k in s.train}) == 2
        assert split_group(s) == "all"  # the runs are pooled when a configuration is selected, like LIBERO-10's
    assert [split_priority(s.name) for s in splits] == [(0, 0, "loto_basket"), (0, 0, "loto_closer_drawer"), (0, 0, "loto_tomato")]
    plan = [n for n, _, _ in split_plan("real8_groot", TASKS)]
    assert plan[:6] == ["loto_basket", "loto_closer_drawer", "loto_tomato", "full_seed0", "full_seed1", "full_seed2"] and len(plan) == 6 + 4 * 3
    assert plan[6:9] == ["lotoseed1_basket", "lotoseed1_closer_drawer", "lotoseed1_tomato"] and plan[-1] == "lotoseed4_tomato"


def test_the_extra_seeds_repeat_the_leave_one_task_out_runs_with_other_shuffles_and_stay_out_of_the_pooled_group(source, data):
    import_real("real8_groot", source[0], embed)
    splits = {s.name: s for s in make_splits("real8_groot")}
    for seed in (1, 2, 3, 4):
        for task in TASKS:
            s, base = splits[f"lotoseed{seed}_{task}"], splits[f"loto_{task}"]
            assert s.seed == seed and s.unseen == base.unseen  # the same held-out task and episodes
            assert {k.split("/")[0] for k in s.train + s.val + s.calibration + s.seen} == set(TASKS) - {task}
            assert split_group(s) == "loto_seeds" and split_priority(s.name) == (seed, 0, s.name)
    assert splits["lotoseed1_basket"].train != splits["loto_basket"].train and splits["lotoseed1_basket"].train != splits["lotoseed2_basket"].train  # other shuffles
    assert split_group(splits["loto_basket"]) == "all" and split_group(splits["full_seed1"]) == "full"  # the seed-0 runs and the full splits keep their groups


def test_the_full_splits_hold_nothing_out_use_every_task_in_training_and_are_their_own_group(source, data):
    import_real("real8_groot", source[0], embed)
    full = [s for s in make_splits("real8_groot") if s.name.startswith("full_")]
    assert [s.name for s in full] == ["full_seed0", "full_seed1", "full_seed2"] and [s.seed for s in full] == [0, 1, 2]
    for s in full:
        assert s.unseen == [] and len(s.train + s.val + s.calibration + s.seen) == 36
        assert {k.split("/")[0] for k in s.train} == set(TASKS) and split_group(s) == "full"  # not pooled into the leave-one-task-out setting
    assert full[0].train != full[1].train  # the seed changes the shuffle
    assert split_priority("full_seed2") == (2, 0, "full_seed2")


def test_the_action_baselines_are_skipped_with_the_reason_and_nothing_is_written(source, data, tmp_path):
    import_real("real8_groot", source[0], embed)
    from fp.data.inputs import load_inputs

    inputs = load_inputs("real8_groot")
    context = Context(inputs, {}, "cpu", {})
    for family in ("stac", "stac_single", "actprobe", "failurespot_mlp", "failurespot_lstm"):
        with pytest.raises(NotImplementedError, match="real8_groot does not have"):
            build(family, context, inputs.splits["loto_basket"])
    messages = []
    out = tmp_path / "results"
    assert run_unit(BaselineUnit("real8_groot", "loto_basket", "stac"), context, Settings(out), messages.append) == 0
    assert len(messages) == 1 and "skipped" in messages[0] and "stac" in messages[0] and not out.exists()


def test_other_baseline_families_are_not_affected_by_the_missing_actions(source, data):
    import_real("real8_groot", source[0], embed)
    from fp.data.inputs import load_inputs

    inputs = load_inputs("real8_groot")
    scorers = build("mahalanobis", Context(inputs, {}), inputs.splits["loto_tomato"])
    assert len(scorers) == 2  # SAFE's two Mahalanobis configurations


def test_a_dataset_that_is_kept_local_cannot_be_downloaded():
    with pytest.raises(ValueError, match="kept local"):
        download_dataset("real8_groot", fetch=lambda **kw: pytest.fail("must not reach the Hub"))
