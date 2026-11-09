import csv

import numpy as np
import pytest

from cli import figures as cli_figures
from fp.data.targets import MODE_NAMES
from fp.eval import figures, mode_balanced, modes, tables
from fp.eval.mode_balanced import balanced, balanced_results, class_index, class_probs, dataset_groups, group_rows, merge_groups
from fp.eval.modes import LIBEROX
from tests.test_modes import FAILURES, SUCCESSES, TRUE_MODE, synthetic_inputs

ARCH = figures.OURS_ARCHITECTURE
TASK = next(s for s in LIBEROX if s.group == "task")
LABEL = "Ours (onset_pre)"
VARIANTS = ("onset_pre", "onset_traj", "onset", "onset_post")


# ---------------------------------------------------------------- the classes
def test_the_two_smallest_classes_are_merged_until_every_class_holds_the_floor():
    counts = {0: 50, 1: 30, 2: 10, 3: 6, 4: 4}  # 100 failures
    assert merge_groups(counts, floor=0.04) == [(0,), (1,), (2,), (3,), (4,)]  # nothing is below 4%
    assert merge_groups(counts, floor=0.05) == [(0,), (1,), (2,), (3, 4)]  # 4 (4%) is below 5%: it joins 3 (6%)
    assert merge_groups(counts, floor=0.08) == [(0,), (1,), (2,), (3, 4)]  # 4 (4%) joins 3 (6%): 10%
    assert merge_groups(counts, floor=0.15) == [(0,), (1,), (2, 3, 4)]  # 10 and 10 merge into 20%, and that is not below 15%
    assert merge_groups(counts, floor=0.9) == [(0, 1, 2, 3, 4)]  # no class can hold 90%: everything merges, and one class is the last


def test_a_mode_that_never_occurs_is_not_a_class_and_ties_go_to_the_lower_mode_index():
    assert merge_groups({0: 10, 1: 0, 2: 10}, floor=0.05) == [(0,), (2,)]
    assert merge_groups({0: 5, 1: 5, 2: 5, 3: 85}, floor=0.1) == [(0, 1, 2), (3,)]  # (0), (1) merge first (a tie goes to the lower mode); the 10% class holds the floor, (2) is below it and joins
    assert merge_groups({0: 3, 1: 3, 2: 4}, floor=0.0) == [(0,), (1,), (2,)] and merge_groups({}, floor=0.05) == []


def test_a_merged_class_gets_the_sum_of_the_probabilities_of_its_modes():
    groups = [(0,), (1, 3), (2,)]
    assert class_index(groups) == {0: 0, 1: 1, 3: 1, 2: 2}
    assert np.allclose(class_probs(np.array([0.1, 0.2, 0.3, 0.4]), groups), [0.1, 0.6, 0.3])


# ---------------------------------------------------------------- the score
def test_the_balanced_accuracy_is_the_mean_over_the_classes_of_their_hit_rates_and_not_the_overall_rate():
    pairs = [(0, 1.0), (0, 1.0), (0, 1.0), (1, 0.0)]  # the common class is always right, the rare one never: 75% overall accuracy
    assert balanced(pairs) == pytest.approx(50.0)
    assert balanced([(0, 0.5), (1, 1.0)]) == pytest.approx(75.0) and balanced([(2, 1.0)]) == 100.0 and balanced([]) is None


# ---------------------------------------------------------------- the whole computation on a synthetic results folder
def write_runs(root, early_for_split, dataset="liberox_groot"):
    """Every variant of our one architecture. Failures alarm at step 2 and their onset is step 4 (successes top out at 0.1 ... 0.5 on the calibration episodes).
    The head says ``early(mode)`` in steps 0..2 and the true mode from step 3 on."""
    for split, early in early_for_split.items():
        for variant in VARIANTS:
            folder = root / dataset / split / f"{ARCH}__{variant}"
            folder.mkdir(parents=True)
            scores = {f"S{i}": np.full(8, 0.1 * (i - 1)) for i in range(len(SUCCESSES))} | {k: np.array([0, 0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]) for k in FAILURES}
            np.savez(folder / "scores.npz", **scores)
            logits = {}
            for k, mode in TRUE_MODE.items():
                out = np.zeros((8, 4))
                out[:3, early(mode)] = 5.0
                out[3:, mode] = 5.0
                logits[k] = out
            np.savez(folder / "mode_logits.npz", **logits)


RIGHT, ALWAYS_0, SWAP_1_2 = (lambda m: m), (lambda m: 0), (lambda m: {1: 2, 2: 1}.get(m, m))


def key(rule, label, metric, eval_set):
    return (rule, label, metric, "LIBERO-X task", eval_set)


def test_the_dataset_classes_come_from_all_its_failures():
    inputs = synthetic_inputs()
    assert dataset_groups(inputs, floor=0.05) == [(0,), (1,), (2,)]  # the synthetic failures have modes 0 (5 of 10), 1 (3) and 2 (2)
    assert dataset_groups(inputs, floor=0.25) == [(0,), (1, 2)]  # mode 2 (20%) joins mode 1 (30%)


def test_pooled_over_the_splits_a_class_the_probe_misses_in_one_split_counts_half(tmp_path):
    write_runs(tmp_path, {"task_seed0": RIGHT, "task_seed1": ALWAYS_0})
    r = balanced_results(synthetic_inputs(), tmp_path, settings=[TASK])
    # seen failures F4, F5, F6 have modes 0, 1, 0. Alarm at step 2: split 0 is right everywhere; split 1 says mode 0 everywhere, so it misses F5 (the only mode 1).
    # Pooled: class 0 hit 4 of 4, class 1 hit 1 of 2: balanced 75%, where the plain accuracy over the 6 predictions would be 83.3%
    assert r[key("detected", LABEL, "top1", "seen")] == pytest.approx(75.0)
    assert r[key("oracle", LABEL, "top1", "seen")] == pytest.approx(100.0) and r[key("detected", LABEL, "top3", "seen")] == pytest.approx(100.0)  # 3 classes: top-3 is always a hit
    assert r[key("detected", LABEL, "share", "seen")] == pytest.approx(100.0) and r[key("oracle_on_detected", LABEL, "top1", "seen")] == pytest.approx(100.0)


def test_the_baselines_are_scored_in_the_same_classes_and_the_constant_prediction_is_the_chance_level(tmp_path):
    write_runs(tmp_path, {"task_seed0": RIGHT, "task_seed1": RIGHT})
    r = balanced_results(synthetic_inputs(), tmp_path, settings=[TASK])
    # the most common training mode is 0 (train failures F0..F3 have modes 0, 0, 1, 2). Seen F4, F5, F6 (0, 1, 0): class 0 right, class 1 wrong: 50% balanced (67% plain).
    # Unseen F7, F8, F9 (0, 2, 1): class 0 right, classes 1 and 2 wrong: 33.3%. The random guess is 1/3 for each class present in training.
    assert r[("majority", None, "top1", "LIBERO-X task", "seen")] == pytest.approx(50.0)
    assert r[("majority", None, "top1", "LIBERO-X task", "unseen")] == pytest.approx(100 / 3)
    assert r[("random", None, "top1", "LIBERO-X task", "seen")] == pytest.approx(100 / 3) and r[("random", None, "top3", "LIBERO-X task", "unseen")] == pytest.approx(100.0)


def test_a_confusion_inside_a_merged_class_is_a_hit_and_one_between_classes_is_not(tmp_path):
    write_runs(tmp_path, {"task_seed0": SWAP_1_2})  # at the alarm the head says mode 2 for mode 1 and mode 1 for mode 2
    inputs = synthetic_inputs(names=("task_seed0",))
    separate = balanced_results(inputs, tmp_path, settings=[TASK], floor=0.05)
    merged = balanced_results(inputs, tmp_path, settings=[TASK], floor=0.25)  # classes (0) and (1, 2)
    assert separate[key("detected", LABEL, "top1", "seen")] == pytest.approx(50.0)  # F5 (mode 1) is read as mode 2: class 1 is missed
    assert merged[key("detected", LABEL, "top1", "seen")] == pytest.approx(100.0)  # the same prediction is right in the merged class


def test_the_classes_csv_lists_each_class_with_its_modes_and_its_share_of_the_failures():
    rows = group_rows(synthetic_inputs(), floor=0.25)
    assert [(r["class"], r["failures"], r["share %"]) for r in rows] == [(1, 5, 50.0), (2, 5, 50.0)]
    assert rows[0]["modes"] == MODE_NAMES[0] and rows[1]["modes"] == f"{MODE_NAMES[1]} + {MODE_NAMES[2]}" and rows[0]["dataset"] == "liberox_groot"


# ---------------------------------------------------------------- the table
def test_the_balanced_caption_says_what_is_balanced_and_what_is_merged():
    rows = [tables.ModeRow("onset_pre", "detected", [50.0], [80.0], share=[70.0])]
    caption = tables.mode_caption(rows, ["LIBERO-X task"], balanced=True)
    assert "balanced top-1 and top-3 accuracy" in caption and r"at least 5\%" in caption and "summing their probabilities" in caption
    assert r"70 to 70\%" in caption and "COVERAGE" not in caption and "REAL" not in caption and "WINDOW" not in caption
    assert "balanced" not in tables.mode_caption(rows, ["LIBERO-X task"])  # the plain caption is unchanged
    real = tables.mode_caption(rows, ["LIBERO-X task", tables.REAL_ROBOT], balanced=True)
    assert "40 splits" in real and "seven of the ten" not in real


def test_the_command_writes_the_balanced_table_next_to_the_plain_one_and_the_classes(tmp_path, monkeypatch):
    from types import SimpleNamespace

    root = tmp_path / "results"
    write_runs(root, {"task_seed0": RIGHT, "task_seed1": ALWAYS_0})
    inputs = synthetic_inputs()
    monkeypatch.setattr(cli_figures, "load_inputs", lambda dataset: inputs)
    monkeypatch.setattr(modes, "LIBEROX", (TASK,))
    monkeypatch.setattr(modes, "REAL", ())
    out = tmp_path / "out"
    out.mkdir()
    args = SimpleNamespace(root=root, out=out)
    cli_figures.write_mode_table(args)
    cli_figures.write_mode_table(args, balanced=True)
    assert (out / "table_modes.tex").exists() and (out / "table_modes_balanced.tex").exists() and (out / "mode_classes.csv").exists()
    tex = (out / "table_modes_balanced.tex").read_text()
    assert r"\label{tab:modes_balanced}" in tex and "balanced top-1" in tex and r"\label{tab:modes}" in (out / "table_modes.tex").read_text()
    plain = {r["method"] + r["group"]: r for r in csv.DictReader(open(out / "table_modes.csv"))}
    bal = {r["method"] + r["group"]: r for r in csv.DictReader(open(out / "table_modes_balanced.csv"))}
    assert float(bal["onset_predetected"]["top-1 LIBERO-X task seen"]) == pytest.approx(75.0)  # balanced: class 0 right 4 of 4, class 1 right 1 of 2
    assert float(plain["onset_predetected"]["top-1 LIBERO-X task seen"]) == pytest.approx(83.333, abs=1e-3)  # the plain table is unchanged: right on 3 of 3, then 2 of 3
    assert float(bal["Most commonbaseline"]["top-1 LIBERO-X task seen"]) == pytest.approx(50.0)
