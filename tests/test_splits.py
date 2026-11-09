import pytest

from fp.data.manifest import Episode
from fp.data.splits import build_split, pick_unseen_tasks, stratified_partition

TASKS = [f"task{i}" for i in range(10)]


def episodes():
    out = []
    for t_i, t in enumerate(TASKS):
        for k in range(20):
            out.append(Episode(f"{t}/ep{k:02d}", success=(k % 3 != 0), task=t, n_steps=30, onset=None if k % 3 else 10,
                               condition="a" if k % 2 else "b"))
    return out


def group(e):
    return e.task


def test_parts_are_disjoint_and_cover_every_seen_episode():
    eps = episodes()
    s = build_split(eps, "x", 0, group, {"task0", "task1", "task2"})
    parts = [s.train, s.val, s.calibration, s.seen]
    all_keys = [k for p in parts for k in p]
    assert len(all_keys) == len(set(all_keys))
    assert set(all_keys) | set(s.unseen) == {e.key for e in eps}
    assert not set(all_keys) & set(s.unseen)


def test_held_out_groups_never_reach_training():
    s = build_split(episodes(), "x", 1, group, {"task4"})
    assert all(not k.startswith("task4") for k in s.train + s.val + s.calibration + s.seen)
    assert all(k.startswith("task4") for k in s.unseen) and len(s.unseen) == 20


def test_fractions_are_about_65_15_20():
    s = build_split(episodes(), "x", 0, group, {"task0"})
    n = len(s.train) + len(s.val) + len(s.calibration) + len(s.seen)
    assert abs(len(s.train) / n - 0.65) < 0.03 and abs(len(s.val) / n - 0.15) < 0.03
    assert abs((len(s.calibration) + len(s.seen)) / n - 0.20) < 0.03


def test_every_part_keeps_both_outcomes_and_both_conditions():
    eps = {e.key: e for e in episodes()}
    s = build_split(list(eps.values()), "x", 0, group, {"task0"})
    for part in (s.train, s.val, s.calibration, s.seen):
        assert {eps[k].success for k in part} == {True, False}
        assert {eps[k].condition for k in part} == {"a", "b"}


def test_calibration_and_seen_halves_are_balanced():
    s = build_split(episodes(), "x", 0, group, {"task0"})
    assert abs(len(s.calibration) - len(s.seen)) <= 4  # one rounding per stratum


def test_same_seed_same_split_different_seed_different_split():
    a = build_split(episodes(), "x", 0, group, {"task0"})
    b = build_split(episodes(), "x", 0, group, {"task0"})
    c = build_split(episodes(), "x", 1, group, {"task0"})
    assert a == b and a.train != c.train


def test_stratified_partition_remainder_goes_to_last_part():
    eps = [Episode(f"e{i}", True, "t", 10, None, "a") for i in range(7)]
    out = stratified_partition(eps, (("train", 0.65), ("val", 0.15), ("test", 0.20)), 0)
    assert (len(out["train"]), len(out["val"]), len(out["test"])) == (5, 1, 1)


def test_unseen_tasks_depend_only_on_the_seed():
    a = pick_unseen_tasks(TASKS, 0)
    assert a == pick_unseen_tasks(list(reversed(TASKS)), 0) and len(a) == 3
    assert any(pick_unseen_tasks(TASKS, s) != a for s in range(1, 6))


def test_empty_held_out_group_is_an_error():
    with pytest.raises(ValueError):
        build_split(episodes(), "x", 0, group, {"no_such_task"})
