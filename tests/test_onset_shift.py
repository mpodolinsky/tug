import csv
import sys

import numpy as np
import pytest

from cli import onset_shift as cli_shift
from fp.data.manifest import Episode
from fp.eval import figures, modes, onset_shift, tables
from fp.eval.run_eval import evaluate_run
from fp.results import run_dir
from tests.test_modes import FAILURES, SUCCESSES, synthetic_inputs

METHOD = "lstm_modes_clip__onset_pre"
TASK = next(s for s in modes.LIBEROX if s.group == "task")


def write_scores(root, split, alarm_step, dataset="liberox_groot"):
    """Successes top out at 0.1 ... 0.5 on the calibration episodes; every failure (onset: step 4 of 8) crosses 0.9 at ``alarm_step``."""
    folder = run_dir(dataset, split, METHOD, root)
    folder.mkdir(parents=True)
    failure = np.where(np.arange(8) >= alarm_step, 0.9, 0.0)
    np.savez(folder / "scores.npz", **({f"S{i}": np.full(8, 0.1 * (i - 1)) for i in range(len(SUCCESSES))} | {k: failure for k in FAILURES}))


def test_a_shift_in_seconds_is_the_whole_policy_steps_that_fit_in_it():
    # LIBERO-X (GR00T: 8 actions at 20 Hz, a step of 0.4 s), LIBERO-10 pi0.5 (5 at 20 Hz, 0.25 s), the real robot (6 at 50 Hz, 0.12 s)
    assert [onset_shift.shift_steps(s, "liberox_groot") for s in (0, 1, 2)] == [0, 2, 5]  # 2.5 steps: an alarm step is an integer, so 2
    assert [onset_shift.shift_steps(s, "libero10_pi05") for s in (0, 1, 2)] == [0, 4, 8]
    assert [onset_shift.shift_steps(s, "real8_groot") for s in (0, 1, 2)] == [0, 8, 16]


def test_the_failure_time_of_failed_episodes_moves_later_and_successes_and_the_original_do_not_change():
    episodes = {"s": Episode("s", True, "t", 10, None), "f": Episode("f", False, "t", 10, 4)}
    shifted = onset_shift.shift_episodes(episodes, 3)
    assert shifted["f"].onset == 7 and shifted["s"] is episodes["s"] and episodes["f"].onset == 4
    assert onset_shift.shift_episodes(episodes, 0)["f"].onset == 4


def test_a_later_failure_time_turns_late_alarms_into_early_ones_and_raises_the_area(tmp_path):
    write_scores(tmp_path, "task_seed0", alarm_step=6)  # the alarm comes 2 steps after the onset: late, until the failure time moves 2 steps
    inputs = synthetic_inputs(names=("task_seed0",))
    areas = onset_shift.shifted_areas(inputs, tmp_path, METHOD, TASK, (0.0, 1.0, 2.0))
    seen = [areas[(s, "seen")][0] for s in (0.0, 1.0, 2.0)]
    assert seen[0] == 0.0 and seen[1] > seen[0] and seen[2] >= seen[1]  # 0 s: every alarm is late; 1 s = 2 steps: at the failure time; 2 s = 5 steps: before it
    assert all(len(v) == 1 for v in areas.values())  # one split


def test_a_shift_of_zero_is_the_evaluation_of_the_saved_run_and_an_early_alarm_gains_nothing(tmp_path):
    write_scores(tmp_path, "task_seed0", alarm_step=2)  # already before the failure time
    inputs = synthetic_inputs(names=("task_seed0",))
    areas = onset_shift.shifted_areas(inputs, tmp_path, METHOD, TASK, (0.0, 2.0))
    direct = evaluate_run(run_dir("liberox_groot", "task_seed0", METHOD, tmp_path), METHOD, inputs.splits["task_seed0"], inputs.episodes, inputs.window)["max_conformal"]
    assert areas[(0.0, "seen")] == [direct["sets"]["seen"].area] and areas[(0.0, "unseen")] == [direct["sets"]["unseen"].area]
    assert areas[(2.0, "seen")] == areas[(0.0, "seen")]  # the alarm was early anyway


def test_a_split_without_a_run_is_left_out(tmp_path):
    write_scores(tmp_path, "task_seed0", alarm_step=6)
    inputs = synthetic_inputs(names=("task_seed0", "task_seed1"))  # task_seed1 has no run
    assert all(len(v) == 1 for v in onset_shift.shifted_areas(inputs, tmp_path, METHOD, TASK, (0.0,)).values())


def test_the_command_writes_the_mean_and_the_spread_of_each_shift_and_the_steps_it_means(tmp_path, monkeypatch):
    write_scores(tmp_path, "task_seed0", alarm_step=6)
    write_scores(tmp_path, "task_seed1", alarm_step=2)
    (tmp_path / "chosen.csv").write_text(f"setting,entry,method,calibrator\n{TASK.label},Ours (onset_pre),{METHOD},max_conformal\n")
    monkeypatch.setattr(cli_shift, "load_inputs", lambda dataset: synthetic_inputs())
    monkeypatch.setattr(figures, "SETTINGS", (TASK,))
    monkeypatch.setattr(figures, "TABLE_LABELS", ("Ours (onset_pre)",))
    monkeypatch.setattr(sys, "argv", ["onset_shift", "--root", str(tmp_path), "--chosen", str(tmp_path / "chosen.csv"), "--out", str(tmp_path / "out"), "--seconds", "0", "1"])
    cli_shift.main()
    rows = list(csv.DictReader(open(tmp_path / "out" / "table_onset_shift.csv")))
    assert len(rows) == 4 and {(r["set"], r["shift (s)"]) for r in rows} == {("seen", "0.0"), ("seen", "1.0"), ("unseen", "0.0"), ("unseen", "1.0")}
    seen = {r["shift (s)"]: r for r in rows if r["set"] == "seen"}
    assert seen["1.0"]["steps"] == "2" and seen["0.0"]["steps"] == "0" and seen["0.0"]["splits"] == "2"
    assert float(seen["1.0"]["area mean"]) > float(seen["0.0"]["area mean"])
    assert float(seen["0.0"]["area std"]) > 0 and float(seen["1.0"]["area std"]) == 0  # at 0 s one split is early and the other late; at 1 s both are early
    unseen_tex, seen_tex = [(tmp_path / "out" / f"{name}.tex").read_text() for name in ("table_onset_shift", "table_onset_shift_seen")]
    assert r"\label{tab:onset_shift}" in unseen_tex and r"\label{tab:onset_shift_seen}" in seen_tex and r"\multicolumn{2}{c}{LIBERO-X task}" in unseen_tex


# ---------------------------------------------------------------- the LaTeX table
RECORDS = [{"setting": s, "method": m, "set": e, "shift (s)": str(x), "area mean": str(v)}
           for s, m, e, x, v in [("A", "Ours (onset_pre)", "unseen", 0.0, 40.0), ("A", "Ours (onset_pre)", "unseen", 1.0, 42.0), ("A", "SAFE-LSTM", "unseen", 0.0, 41.0),
                                 ("A", "SAFE-LSTM", "unseen", 1.0, 41.5), ("B", "Ours (onset_pre)", "unseen", 0.0, 30.0), ("A", "Ours (onset_pre)", "seen", 0.0, 99.0)]]


def test_the_rows_have_a_value_per_setting_and_shift_none_where_a_method_has_no_record_and_only_the_rows_with_a_record():
    rows = onset_shift.shift_rows(RECORDS, ["A", "B"], ["SAFE-LSTM", "STAC", "Ours (onset_pre)"], [0.0, 1.0], "unseen")
    assert [(r.label, r.ours) for r in rows] == [("SAFE-LSTM", False), ("Ours (onset_pre)", True)]  # STAC has no record at all
    assert rows[0].values == [41.0, 41.5, None, None] and rows[1].values == [40.0, 42.0, 30.0, None]  # setting A at 0 and 1 s, then B
    assert onset_shift.shift_rows(RECORDS, ["A"], ["Ours (onset_pre)"], [0.0], "seen")[0].values == [99.0]  # the seen episodes are their own table


def test_the_latex_table_has_a_column_group_per_setting_a_column_per_shift_ours_shaded_and_the_best_and_second_best_marked_in_each_column():
    rows = onset_shift.shift_rows(RECORDS, ["A", "B"], ["SAFE-LSTM", "Ours (onset_pre)"], [0.0, 1.0], "unseen")
    tex = tables.onset_shift_latex_table(rows, ["A", "B"], [0.0, 1.0], "unseen")
    lines = tex.splitlines()
    assert r"\begin{tabular}{lrrrr}" in tex and r"\multicolumn{2}{c}{A}" in tex and r"\multicolumn{2}{c}{B}" in tex and r"Method & 0\,s & +1\,s & 0\,s & +1\,s \\" in tex
    ours = next(line for line in lines if line.startswith(r"\rowcolor{gray!15} Ours"))
    safe = next(line for line in lines if line.startswith("SAFE-LSTM"))
    assert r"\textbf{42.0}" in ours and r"\textit{41.5}" in safe and r"\textbf{41.0}" in safe and r"\textit{40.0}" in ours  # each column judged on its own
    assert safe.count("--") == 2 and tex.count(r"\midrule") == 2  # B has no SAFE-LSTM value; a rule under the header and one before ours
    assert r"\label{tab:onset_shift}" in tex and "unseen episodes" in tex and "Cal." not in tex and tex.count("{") == tex.count("}")
    assert "seen episodes" in tables.onset_shift_latex_table(rows, ["A", "B"], [0.0, 1.0], "seen") and "unseen" not in tables.onset_shift_latex_table(rows, ["A", "B"], [0.0, 1.0], "seen")
