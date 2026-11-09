"""The runs of each dataset: which tasks every run holds out (fp.data.splits.split_plan / make_splits), on small synthetic datasets."""

import csv

import pytest

from fp import config
from fp.data.splits import make_splits, pick_unseen_tasks, read_unseen_tasks, split_plan
from fp.eval.select_config import split_group

LIBERO10_TASKS = [f"T{i:02d}_task" for i in range(10)]
LIBEROX_TASKS = [f"EXT_SCENE{i:02d}_task" for i in range(12)]


def write_manifest(folder, tasks, condition_column, conditions, per_task=20):
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dir", "success", "task", "n_policy", "replan_steps", "vlm_failure_onset_frame", condition_column])
        w.writeheader()
        for t in tasks:
            for i in range(per_task):
                ok = i % 3 != 0
                w.writerow({"dir": f"{conditions[i % len(conditions)]}/{t}/ep{i:03d}", "success": str(ok), "task": t, "n_policy": 30, "replan_steps": 8,
                            "vlm_failure_onset_frame": "" if ok else "80", condition_column: conditions[i % len(conditions)]})


@pytest.fixture
def libero10(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA", tmp_path)
    write_manifest(tmp_path / "libero10_groot", LIBERO10_TASKS, "scene_variant", ["normal", "occluded"])


@pytest.fixture
def liberox(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA", tmp_path)
    folder = tmp_path / "liberox_groot"
    write_manifest(folder, LIBEROX_TASKS, "level", ["LEVEL1", "LEVEL2", "LEVEL3"])
    splits = folder / "splits"
    held_out = {"leave_task_out": LIBEROX_TASKS[:3], "leave_signature_out": LIBEROX_TASKS[3:6],
                "leave_skill_out/pick": LIBEROX_TASKS[6:8], "leave_skill_out/stack": LIBEROX_TASKS[8:12]}
    for rel, tasks in held_out.items():
        (splits / rel).mkdir(parents=True)
        (splits / rel / "test.txt").write_text("".join(f"{t}.bddl\n" for t in tasks))
        (splits / rel / "train.txt").write_text("unused\n")


def test_libero10_has_three_paper_seeds(libero10):
    splits = make_splits("libero10_groot")
    assert [s.name for s in splits] == ["paper_seed0", "paper_seed1", "paper_seed2"]
    assert [s.seed for s in splits] == [0, 1, 2]
    assert {split_group(s) for s in splits} == {"all"}


def test_each_libero10_run_holds_out_three_tasks_with_all_their_episodes(libero10):
    for s in make_splits("libero10_groot"):
        assert len({k.split("/")[1] for k in s.unseen}) == 3 and len(s.unseen) == 60
        held_out = {k.split("/")[1] for k in s.unseen}
        assert not held_out & {k.split("/")[1] for k in s.train + s.val + s.calibration + s.seen}


def test_paper_seeds_hold_out_the_seeded_draw_of_three_tasks():
    held_out = {name: unseen for name, _, unseen in split_plan("libero10_groot", LIBERO10_TASKS)}
    assert held_out["paper_seed0"] == pick_unseen_tasks(LIBERO10_TASKS, 0)
    assert all(len(u) == 3 for u in held_out.values())
    assert len({frozenset(u) for u in held_out.values()}) > 1  # the seeds do not all hold out the same tasks


def test_liberox_has_task_and_signature_seeds_and_one_fold_per_skill(liberox):
    splits = make_splits("liberox_groot")
    assert [s.name for s in splits] == ["task_seed0", "task_seed1", "task_seed2", "signature_seed0", "signature_seed1", "signature_seed2",
                                       "skill_pick", "skill_stack"]
    assert [s.seed for s in splits] == [0, 1, 2, 0, 1, 2, 0, 0]
    assert [split_group(s) for s in splits] == ["task"] * 3 + ["signature"] * 3 + ["skill"] * 2


def test_liberox_seeds_hold_out_the_same_episodes_and_differ_in_the_seen_shuffle(liberox):
    a, b, c = (s for s in make_splits("liberox_groot") if s.name.startswith("task_seed"))
    assert a.unseen == b.unseen == c.unseen and len(a.unseen) == 60  # 3 tasks x 20 episodes
    assert a.train != b.train and a.val != b.val


def test_liberox_held_out_tasks_come_from_the_split_files(liberox):
    splits = {s.name: s for s in make_splits("liberox_groot")}
    assert {k.split("/")[1] for k in splits["skill_stack"].unseen} == set(LIBEROX_TASKS[8:12])
    assert {k.split("/")[1] for k in splits["signature_seed1"].unseen} == set(LIBEROX_TASKS[3:6])


def test_read_unseen_tasks_strips_the_bddl_extension(tmp_path):
    p = tmp_path / "test.txt"
    p.write_text("A_task.bddl\n\nB_task.bddl\n")
    assert read_unseen_tasks(p) == {"A_task", "B_task"}


def test_a_held_out_task_that_is_not_in_the_manifest_is_an_error(liberox, tmp_path):
    (tmp_path / "liberox_groot/splits/leave_task_out/test.txt").write_text("not_a_task.bddl\n")
    with pytest.raises(ValueError, match="not in the manifest"):
        make_splits("liberox_groot")
