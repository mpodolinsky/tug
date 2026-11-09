import csv
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cli import figures as cli_figures
from fp.calibration.max_conformal import MaxConformal
from fp.data.manifest import Episode
from fp.data.splits import Split
from fp.eval import figures, modes, tables
from fp.eval.modes import LIBEROX, REAL, baseline_accuracy, evaluate_split, hit, mode_results, mode_rows, softmax, window_probs

ARCH = figures.OURS_ARCHITECTURE
VARIANTS = ("onset_pre", "onset_traj", "onset", "onset_post")
TASK = next(s for s in LIBEROX if s.group == "task")
ROBOT = REAL[0]


# ---------------------------------------------------------------- the pieces
def test_softmax_rows_sum_to_one_and_do_not_overflow():
    p = softmax(np.array([[1000.0, 1000.0, 0.0], [0.0, 1.0, 2.0]]))
    assert np.allclose(p.sum(axis=1), 1) and np.allclose(p[0], [0.5, 0.5, 0.0])


def test_the_window_is_the_steps_before_the_anchor_and_the_anchor_clipped_at_the_first_step():
    logits = np.zeros((10, 3))
    logits[3:6, 0] = 20.0  # steps 3, 4, 5 confidently mode 0
    logits[:3, 1] = 20.0  # steps 0, 1, 2 confidently mode 1
    assert np.allclose(window_probs(logits, anchor=5, window=2), [1, 0, 0], atol=1e-6)  # steps 3, 4, 5
    assert np.allclose(window_probs(logits, anchor=5, window=3), [0.75, 0.25, 0], atol=1e-6)  # steps 2..5: one step of mode 1, three of mode 0
    assert np.allclose(window_probs(logits, anchor=1, window=6), [0, 1, 0], atol=1e-6)  # clipped at step 0: steps 0 and 1
    assert len(range(max(7 - 6, 0), 7 + 1)) == 7  # the detection metric's window: 6 steps before the anchor and the anchor


def test_top_k_hit_with_ties_going_to_the_lower_mode():
    p = np.array([0.1, 0.4, 0.3, 0.2])
    assert [hit(p, t, 1) for t in range(4)] == [False, True, False, False]
    assert [hit(p, t, 3) for t in range(4)] == [False, True, True, True]  # the three most probable are modes 1, 2 and 3
    assert hit(np.array([0.25] * 4), 0, 1) and not hit(np.array([0.25] * 4), 1, 1)  # a tie: mode 0 comes first


def test_the_baselines_by_hand():
    train = [0, 0, 0, 1, 1, 2]  # modes 0, 1, 2 occur; 0 is the most common, then 1, then 2
    true = [0, 1, 3]  # 3 never occurs in training
    assert baseline_accuracy(train, true, 1) == (pytest.approx(1 / 3), pytest.approx(2 / 9))  # always 0 hits one; a guess hits 1/3 of those present
    assert baseline_accuracy(train, true, 3) == (pytest.approx(2 / 3), pytest.approx(2 / 3))  # the three modes cover 0 and 1 but not 3


# ---------------------------------------------------------------- one split
def make_episodes():
    return {"S2": Episode("S2", True, "t", 8, None, "a"), "S3": Episode("S3", True, "t", 8, None, "a"),
            "A": Episode("A", False, "t", 8, 5, "a"), "B": Episode("B", False, "t", 8, 5, "a")}


def logits_for(early, late, steps=8):
    out = np.zeros((steps, 4))
    out[:3, early] = 5.0  # what the head says in the first steps
    out[3:, late] = 5.0  # and from step 3 on
    return out


def test_oracle_and_own_detection_use_different_windows_and_only_alarmed_failures_have_a_detection():
    episodes = make_episodes()
    flat = lambda v: np.full(8, v)
    alarm_at_2 = np.array([0.1, 0.1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    scores = {"S2": flat(0.5), "S3": flat(0.7), "A": alarm_at_2, "B": flat(0.6)}  # the threshold is the larger success maximum, 0.7: A alarms at step 2, B never
    calibrator = MaxConformal().fit([scores["S2"], scores["S3"]])
    # mode 0 is the truth. Steps 1..2 (the detection window with window = 1) say mode 1 with mode 2 second and mode 0 third; steps 4..5 (the oracle window) say 0
    wrong_early = np.zeros((8, 4))
    wrong_early[:3] = [1, 5, 3, 0]
    wrong_early[3:, 0] = 5.0
    logits = {"A": wrong_early, "B": logits_for(early=0, late=0)}
    result = evaluate_split(logits, scores, episodes, {"A": 0, "B": 0}, ["S2", "S3", "A", "B"], calibrator, alpha=0.5, window=1)
    assert result["coverage"] == 0.5  # B has no alarm
    assert (result["oracle1"], result["oracle3"]) == (1.0, 1.0)  # at the onset both are right
    assert (result["detected1"], result["detected3"]) == (0.0, 1.0)  # A at its alarm: top-1 is mode 1, but mode 0 is in the top 3
    assert (result["oracle_on_detected1"], result["oracle_on_detected3"]) == (1.0, 1.0)  # the oracle on the same episode (A) only


def test_no_alarm_at_all_gives_nan_for_the_detection_rule_and_zero_coverage():
    episodes = make_episodes()
    scores = {"S2": np.full(8, 0.5), "S3": np.full(8, 0.5), "A": np.full(8, 0.4), "B": np.full(8, 0.4)}  # no failure reaches the threshold (0.5)
    calibrator = MaxConformal().fit([scores["S2"], scores["S3"]])
    logits = {"A": logits_for(0, 0), "B": logits_for(0, 0)}
    result = evaluate_split(logits, scores, episodes, {"A": 0, "B": 0}, list(episodes), calibrator, alpha=0.9, window=1)
    assert result["coverage"] == 0.0 and np.isnan(result["detected1"]) and np.isnan(result["oracle_on_detected3"]) and result["oracle1"] == 1.0


# ---------------------------------------------------------------- the whole computation on a synthetic results folder
SUCCESSES = [f"S{i}" for i in range(10)]
FAILURES = [f"F{i}" for i in range(10)]
TRUE_MODE = {"F0": 0, "F1": 0, "F2": 1, "F3": 2, "F4": 0, "F5": 1, "F6": 0, "F7": 0, "F8": 2, "F9": 1}


def synthetic_inputs(dataset="liberox_groot", names=("task_seed0", "task_seed1")):
    episodes = {k: Episode(k, True, "t", 8, None, "a") for k in SUCCESSES} | {k: Episode(k, False, "t", 8, 4, "a") for k in FAILURES}
    train, val = ["S0", "S1", "F0", "F1", "F2", "F3"], ["S9", "F7"]
    calibration, seen, unseen = [f"S{i}" for i in range(2, 7)], ["S7", "F4", "F5", "F6"], ["S8", "F7", "F8", "F9"]
    splits = {name: Split(name, train, val, calibration, seen, unseen, seed=i) for i, name in enumerate(names)}
    return SimpleNamespace(dataset=dataset, episodes=episodes, targets=SimpleNamespace(modes=TRUE_MODE), splits=splits, window=1)


def write_runs(root, early_for_split, dataset="liberox_groot"):
    """Every variant of our one architecture on both splits. Successes top out at 0.1 ... 0.5 on the calibration episodes (threshold 0.5 at alpha 0.2);
    failures alarm at step 2 and their onset is step 4. The head says ``early_for_split[split]`` in steps 0..2 and the true mode from step 3 on."""
    for split, early in early_for_split.items():
        for variant in ("onset_pre", "onset_traj", "onset", "onset_post"):
            folder = root / dataset / split / f"{ARCH}__{variant}"
            folder.mkdir(parents=True)
            scores = {f"S{i}": np.full(8, 0.1 * (i - 1)) for i in range(10)} | {k: np.array([0, 0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]) for k in FAILURES}
            np.savez(folder / "scores.npz", **scores)
            logits = {}
            for k, mode in TRUE_MODE.items():
                out = np.zeros((8, 4))
                out[:3, mode if early == "true" else 3] = 5.0
                out[3:, mode] = 5.0
                logits[k] = out
            np.savez(folder / "mode_logits.npz", **logits)


@pytest.fixture
def results(tmp_path):
    write_runs(tmp_path, {"task_seed0": "true", "task_seed1": "wrong"})  # in the second split the head is wrong until step 3
    return mode_results(synthetic_inputs(), tmp_path, settings=[TASK]), tmp_path


def test_the_results_are_means_over_the_splits_in_percent_with_the_baselines_from_each_splits_training_failures(results):
    r, _ = results
    key = lambda rule, label, metric, s: (rule, label, metric, "LIBERO-X task", s)
    assert r[key("majority", None, "top1", "seen")] == pytest.approx(100 * 2 / 3)  # always mode 0; seen failures have modes 0, 1, 0
    assert r[key("majority", None, "top1", "unseen")] == pytest.approx(100 / 3) and r[key("majority", None, "top3", "seen")] == pytest.approx(100.0)
    assert r[key("random", None, "top1", "seen")] == pytest.approx(100 / 3) and r[key("random", None, "top3", "unseen")] == pytest.approx(100.0)
    label = "Ours (onset_pre)"
    assert r[key("oracle", label, "top1", "seen")] == r[key("oracle", label, "top1", "unseen")] == pytest.approx(100.0)  # at the onset the head is right
    assert r[key("detected", label, "top1", "seen")] == pytest.approx(50.0)  # right at the alarm in one split, wrong in the other
    assert r[key("detected", label, "share", "seen")] == pytest.approx(100.0)  # every failure alarms at step 2
    assert r[key("oracle_on_detected", label, "top1", "unseen")] == pytest.approx(100.0)


def test_a_missing_run_is_an_error_not_a_silent_gap(tmp_path):
    write_runs(tmp_path, {"task_seed0": "true"})  # task_seed1 has no runs
    with pytest.raises(FileNotFoundError):
        mode_results(synthetic_inputs(), tmp_path, settings=[TASK])


def test_a_set_without_a_failure_is_left_out(tmp_path):
    write_runs(tmp_path, {"task_seed0": "true", "task_seed1": "true"})
    inputs = synthetic_inputs()
    inputs.splits = {n: Split(n, s.train, s.val, s.calibration, ["S7"], s.unseen, s.seed) for n, s in inputs.splits.items()}  # the seen set has only a success
    r = mode_results(inputs, tmp_path, settings=[TASK])
    assert not any(key[4] == "seen" for key in r) and ("majority", None, "top1", "LIBERO-X task", "unseen") in r


def test_the_rows_have_the_baselines_then_own_detection_then_the_oracle_and_a_hidden_control(results):
    r, _ = results
    rows = mode_rows(r, settings=[TASK])
    assert [(x.label, x.group) for x in rows] == ([("Most common", "baseline"), ("Random", "baseline")] + [(v, "detected") for v in VARIANTS]
                                                  + [(v, "oracle") for v in VARIANTS] + [(v, "oracle_on_detected") for v in VARIANTS])
    assert [x.in_latex for x in rows] == [True] * 10 + [False] * 4  # the oracle on the detected episodes is for the CSV only
    own = rows[2]
    assert own.top1[0] == pytest.approx(50.0) and own.share == [pytest.approx(100.0)] * 2 and len(own.top1) == len(own.top3) == 2  # seen and unseen of the one setting
    assert rows[0].share is None and rows[6].share is None  # only the own-detection rows have a share of failures with an alarm


# ---------------------------------------------------------------- the formatting
def toy_rows():
    return [tables.ModeRow("Most common", "baseline", [60.0, 30.0], [70.0, 50.0]), tables.ModeRow("Random", "baseline", [10.0, 10.0], [30.0, 30.0]),
            tables.ModeRow("onset_pre", "detected", [55.0, 40.0], [80.0, 60.0], share=[70.0, 65.0]),
            tables.ModeRow("onset_pre", "oracle", [90.0, 80.0], [95.0, 90.0]),
            tables.ModeRow("hidden_row", "oracle_on_detected", [1.0, 2.0], [3.0, 4.0], in_latex=False)]


def test_best_and_second_best_are_marked_among_the_baselines_and_own_detection_only_and_never_on_the_oracle():
    tex = tables.mode_latex_table(toy_rows(), ["LIBERO-X task"])
    lines = tex.splitlines()
    common = next(line for line in lines if line.startswith("Most common"))
    own = next(line for line in lines if line.startswith(r"\rowcolor{gray!15} onset\_pre") and "55.0" in line)
    oracle = next(line for line in lines if line.startswith(r"\rowcolor{gray!15} onset\_pre") and "90.0" in line)
    # top-1 seen 60 > 55 > 10, top-1 unseen 40 > 30 > 10, top-3 seen 80 > 70 > 30, top-3 unseen 60 > 50 > 30
    assert r"\textit{55.0}" in own and r"\textbf{40.0}" in own and r"\textbf{80.0}" in own and r"\textbf{60.0}" in own
    assert r"\textbf{60.0}" in common and r"\textit{30.0}" in common and r"\textit{70.0}" in common and r"\textit{50.0}" in common
    assert "textbf" not in oracle and "textit" not in oracle  # the oracle's 90, 80, 95, 90 are above the others and not marked
    assert not common.startswith(r"\rowcolor") and "hidden" not in tex  # baselines unshaded; the hidden row is not in the LaTeX, escaped or not


def test_the_layout_is_compact_top_1_and_top_3_side_by_side_with_two_group_headings_and_booktabs_rules():
    tex = tables.mode_latex_table(toy_rows(), ["LIBERO-X task"])
    assert r"\begin{tabular}{lrrrr}" in tex  # a label column and 2 metrics x (seen, unseen)
    assert r"\multicolumn{2}{c}{Top-1 accuracy (\%)}" in tex and r"\multicolumn{2}{c}{Top-3 accuracy (\%)}" in tex and r"\multicolumn{2}{c}{task}" in tex
    assert "LIBERO-X task" not in tex.split(r"\begin{tabular}")[1]  # the header says task, the caption says LIBERO-X
    assert r"\multicolumn{5}{l}{Ours, own detection (first alarm)}" in tex and r"\multicolumn{5}{l}{Ours, oracle onset (upper bound)}" in tex
    assert tex.count(r"\toprule") == tex.count(r"\bottomrule") == 1 and tex.count(r"\midrule") == 3 and tex.count(r"\cmidrule") >= 4
    assert tex.index("Most common") < tex.index("own detection") < tex.index("oracle onset (upper bound)")  # baselines, own detection, oracle
    order = [tex.index(s) for s in (r"\begin{table*}", r"\caption{", r"\label{tab:modes}", rf"\resizebox{{{tables.mode_table_width(1)}\textwidth}}{{!}}{{%", r"\begin{tabular}", r"\end{tabular}%", r"\end{table*}")]
    assert order == sorted(order) and tex.count("{") == tex.count("}")
    assert len([line for line in tex.splitlines() if line.startswith(("Most common", "Random", r"\rowcolor"))]) == 4  # a compact table: four rows here


def test_the_caption_gets_the_range_of_the_share_of_failures_with_an_alarm_from_the_rows():
    assert r"65 to 70\%" in tables.mode_caption(toy_rows()) and "COVERAGE" not in tables.mode_caption(toy_rows())
    assert "see the CSV" in tables.mode_caption([tables.ModeRow("Random", "baseline", [10.0], [30.0])])
    assert "10 modes" in tables.MODE_CAPTION and "upper bound" in tables.MODE_CAPTION


def test_the_csv_has_every_row_including_the_hidden_one_with_a_column_per_metric_setting_and_set():
    records = tables.mode_csv_rows(toy_rows(), ["LIBERO-X task"])
    assert [r["method"] for r in records] == ["Most common", "Random", "onset_pre", "onset_pre", "hidden_row"] and records[4]["group"] == "oracle_on_detected"
    assert records[0] == {"method": "Most common", "group": "baseline", "top-1 LIBERO-X task seen": 60.0, "top-1 LIBERO-X task unseen": 30.0,
                          "top-3 LIBERO-X task seen": 70.0, "top-3 LIBERO-X task unseen": 50.0}  # no share column for a baseline
    assert records[2]["alarm share LIBERO-X task seen"] == 70.0 and records[2]["top-3 LIBERO-X task unseen"] == 60.0
    assert tables.mode_csv_rows([tables.ModeRow("r", "baseline", [None, 1.0], [2.0, None])], ["LIBERO-X task"])[0]["top-1 LIBERO-X task seen"] is None


def test_write_csv_uses_the_columns_of_all_the_records_and_leaves_a_missing_one_empty(tmp_path):
    cli_figures.write_csv(tmp_path / "x.csv", [{"a": 1, "b": 2}, {"a": 3, "c": 4}])
    rows = list(csv.DictReader(open(tmp_path / "x.csv")))
    assert list(rows[0]) == ["a", "b", "c"] and rows[0] == {"a": "1", "b": "2", "c": ""} and rows[1] == {"a": "3", "b": "", "c": "4"}


# ---------------------------------------------------------------- the command
def test_the_command_writes_the_mode_table_when_the_runs_exist_and_says_so_when_they_do_not(tmp_path, monkeypatch, capsys):
    root = tmp_path / "results"
    write_runs(root, {"task_seed0": "true", "task_seed1": "wrong"})
    inputs = synthetic_inputs()
    monkeypatch.setattr(cli_figures, "load_inputs", lambda dataset: inputs)
    monkeypatch.setattr(modes, "LIBEROX", (TASK,))
    monkeypatch.setattr(modes, "REAL", ())
    monkeypatch.setattr(sys, "argv", ["figures", "--root", str(root), "--out", str(tmp_path / "out"), "--costs", str(tmp_path / "none")])
    cli_figures.main()
    out = tmp_path / "out"
    assert "wrote" in capsys.readouterr().out and (out / "table_modes.tex").exists()
    rows = list(csv.DictReader(open(out / "table_modes.csv")))
    assert any(r["method"] == "onset_pre" and r["group"] == "detected" and float(r["top-1 LIBERO-X task seen"]) == 50.0 for r in rows)
    # no runs: the table is skipped with a message and the rest is still written
    monkeypatch.setattr(sys, "argv", ["figures", "--root", str(tmp_path / "empty"), "--out", str(tmp_path / "out2"), "--costs", str(tmp_path / "none")])
    (tmp_path / "empty").mkdir()
    cli_figures.main()
    printed = capsys.readouterr().out
    assert "no failure-mode table" in printed and "no evaluated runs" in printed and not (tmp_path / "out2" / "table_modes.tex").exists()
    assert not (tmp_path / "out2" / "table_main.tex").exists()  # nothing evaluated: no main table and no curves, and no crash


def test_the_command_adds_the_real_robot_columns_when_its_runs_exist_and_leaves_them_out_with_a_message_when_they_do_not(tmp_path, monkeypatch, capsys):
    root = tmp_path / "results"
    write_runs(root, {"task_seed0": "true", "task_seed1": "wrong"})
    write_runs(root, {"loto_a": "true", "lotoseed1_a": "wrong"}, dataset="real8_groot")
    inputs = {"liberox_groot": synthetic_inputs(), "real8_groot": synthetic_inputs("real8_groot", ("loto_a", "lotoseed1_a"))}
    monkeypatch.setattr(cli_figures, "load_inputs", lambda dataset: inputs[dataset])
    monkeypatch.setattr(modes, "LIBEROX", (TASK,))
    monkeypatch.setattr(modes, "REAL", (ROBOT,))
    argv = lambda out: ["figures", "--root", str(root), "--out", str(tmp_path / out), "--costs", str(tmp_path / "none")]
    monkeypatch.setattr(sys, "argv", argv("with_real"))
    cli_figures.main()
    tex = (tmp_path / "with_real" / "table_modes.tex").read_text()
    rows = list(csv.DictReader(open(tmp_path / "with_real" / "table_modes.csv")))
    own = next(r for r in rows if r["method"] == "onset_pre" and r["group"] == "detected")
    assert r"\multicolumn{2}{c}{Real robot}" in tex and "and on the real robot (GR00T on eight tasks" in tex  # the columns and the caption sentence
    assert float(own["top-1 LIBERO-X task seen"]) == 50.0 and float(own["top-1 Real robot seen"]) == 50.0  # right in one split, wrong in the other, in both datasets
    # the real robot's runs are missing: the LIBERO-X table is still written, without its columns
    (root / "real8_groot" / "lotoseed1_a" / f"{ARCH}__onset_pre" / "scores.npz").unlink()
    monkeypatch.setattr(sys, "argv", argv("without_real"))
    capsys.readouterr()
    cli_figures.main()
    assert "no failure-mode column for real8_groot" in capsys.readouterr().out
    assert "Real robot" not in (tmp_path / "without_real" / "table_modes.tex").read_text()


def test_the_extra_seed_splits_of_a_setting_are_averaged_in_with_the_pooled_ones_and_only_if_the_setting_names_their_group(tmp_path):
    write_runs(tmp_path, {"loto_a": "true", "lotoseed1_a": "wrong"}, dataset="real8_groot")
    inputs = synthetic_inputs("real8_groot", ("loto_a", "lotoseed1_a"))
    key = ("detected", "Ours (onset_pre)", "top1", "Real robot", "seen")
    assert mode_results(inputs, tmp_path, settings=[ROBOT])[key] == pytest.approx(50.0)  # both splits count once: right in one, wrong in the other
    pooled_only = figures.Setting("Real robot", "real8_groot", "all")
    assert mode_results(inputs, tmp_path, settings=[pooled_only])[key] == pytest.approx(100.0)  # the seed split is not read


def test_the_real_robot_setting_is_registered_and_the_caption_and_the_width_follow_the_columns():
    assert (ROBOT.dataset, ROBOT.group, ROBOT.also) == ("real8_groot", "all", ("loto_seeds",)) and len(REAL) == 1
    rows = toy_rows()
    plain, real = tables.mode_caption(rows, ["LIBERO-X task"]), tables.mode_caption(rows, ["LIBERO-X task", "Real robot"])
    assert "real robot" not in plain and "over the six steps before an anchor step" in plain  # unchanged without the real robot
    assert "and on the real robot" in real and "2.4 s (six steps on LIBERO-X, 20 on the real robot)" in real and "REAL" not in real and "WINDOW" not in real
    assert tables.mode_caption(rows) == plain  # no settings given: the LIBERO-X text
    assert [tables.mode_table_width(m) for m in (3, 4)] == [0.61, 0.78]  # measured: 480.6 pt and 616.4 pt of the main table's 785.7 pt
