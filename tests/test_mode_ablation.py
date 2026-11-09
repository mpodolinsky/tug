import numpy as np
import pytest

from fp.eval import mode_ablation, modes, tables
from fp.eval.figures import OURS_ARCHITECTURE as ARCH
from fp.eval.mode_ablation import MODEL_NAMES, mode_ablation_blocks, mode_ablation_results
from fp.eval.modes import LIBEROX
from tests.test_modes import FAILURES, SUCCESSES, TRUE_MODE, synthetic_inputs

TASK = next(s for s in LIBEROX if s.group == "task")
ROBOT = mode_ablation.REAL_SEED0
OURS, NO_CLIP, MODES_CLIP, MODES_ONLY = MODEL_NAMES


def write_logits(folder, early):
    """The head says ``early`` in steps 0..2 ('true': the true mode, 'wrong': mode 3) and the true mode from step 3 on."""
    folder.mkdir(parents=True)
    logits = {}
    for k, mode in TRUE_MODE.items():
        out = np.zeros((8, 4))
        out[:3, mode if early == "true" else 3] = 5.0
        out[3:, mode] = 5.0
        logits[k] = out
    np.savez(folder / "mode_logits.npz", **logits)


def write_runs(root, split, ours="true", no_clip="true", modes_clip="true", modes_only="true", dataset="liberox_groot"):
    """Our probe has the scores (successes top out at 0.1 ... 0.5 on the calibration episodes; failures alarm at step 2, their onset is step 4); the probes
    trained without the score term have only mode logits, as in the sweep's ``--lam-score 0`` runs (their score head is never read)."""
    full = root / dataset / split
    write_logits(full / f"{ARCH}__onset_pre", ours)
    scores = {f"S{i}": np.full(8, 0.1 * (i - 1)) for i in SUCCESSES_RANGE} | {k: np.array([0, 0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]) for k in FAILURES}
    np.savez(full / f"{ARCH}__onset_pre" / "scores.npz", **scores)
    write_logits(full / "lstm_modes__onset_pre", no_clip)
    only = root / mode_ablation.MODE_ONLY_DIR / dataset / split
    write_logits(only / f"{ARCH}__onset_pre", modes_clip)
    write_logits(only / "lstm_modes__onset_pre", modes_only)


SUCCESSES_RANGE = range(len(SUCCESSES))


def key(label, anchor, metric, setting="LIBERO-X task", eval_set="seen"):
    return (label, anchor, metric, setting, eval_set)


def test_the_four_models_are_ours_without_clip_without_the_score_term_and_the_mode_head_alone():
    assert MODEL_NAMES == ("Score + modes + CLIP (ours)", "Score + modes", "Modes + CLIP", "Modes")
    assert [with_score for _, with_score, _ in mode_ablation.MODELS] == [True, True, False, False]
    assert [a for _, _, a in mode_ablation.MODELS] == [ARCH, "lstm_modes", ARCH, "lstm_modes"]


def test_every_model_is_read_at_the_alarm_of_our_probe_with_its_own_mode_logits(tmp_path):
    write_runs(tmp_path, "task_seed0", ours="true", no_clip="wrong", modes_clip="true", modes_only="wrong")
    r = mode_ablation_results(synthetic_inputs(names=("task_seed0",)), tmp_path, settings=[TASK])
    # the alarm is at step 2 for every failure (window 1: steps 1 and 2): a head that says the true mode early is right there, one that says mode 3 is wrong
    assert r[key(OURS, "alarm", "top1")] == [100.0] and r[key(MODES_CLIP, "alarm", "top1")] == [100.0]
    assert r[key(NO_CLIP, "alarm", "top1")] == [0.0] and r[key(MODES_ONLY, "alarm", "top1")] == [0.0]
    # at the onset (step 4) every head is right
    assert all(r[key(m, "oracle", "top1")] == [100.0] for m in MODEL_NAMES)
    assert all(len(v) == 1 for v in r.values())  # one split, one value per cell


def test_a_probe_without_the_score_term_needs_no_scores_and_a_missing_run_is_an_error(tmp_path):
    write_runs(tmp_path, "task_seed0")
    assert not list((tmp_path / mode_ablation.MODE_ONLY_DIR).rglob("scores.npz"))  # the alarms come from our probe
    mode_ablation_results(synthetic_inputs(names=("task_seed0",)), tmp_path, settings=[TASK])
    with pytest.raises(FileNotFoundError):
        mode_ablation_results(synthetic_inputs(names=("task_seed0", "task_seed1")), tmp_path, settings=[TASK])  # task_seed1 has no runs


def test_the_blocks_have_the_mean_and_the_spread_over_the_splits_in_percent(tmp_path):
    write_runs(tmp_path, "task_seed0", ours="true", modes_only="true")
    write_runs(tmp_path, "task_seed1", ours="wrong", modes_only="wrong")
    r = mode_ablation_results(synthetic_inputs(), tmp_path, settings=[TASK])
    alarm, oracle = mode_ablation_blocks(r, settings=[TASK])
    assert [b.title for b in (alarm, oracle)] == [title for _, title in mode_ablation.ANCHORS]
    assert [row.label for row in alarm.rows] == list(MODEL_NAMES) and [row.ours for row in alarm.rows] == [True, False, False, False]
    assert alarm.rows[0].top1 == [pytest.approx(50.0)] * 2 and alarm.rows[0].top1_std == [pytest.approx(50.0)] * 2  # right in one split, wrong in the other
    assert alarm.rows[3].top1 == [pytest.approx(50.0)] * 2
    assert oracle.rows[0].top1 == [pytest.approx(100.0)] * 2 and oracle.rows[0].top1_std == [pytest.approx(0.0)] * 2  # one value per set: seen and unseen


def test_a_split_without_an_alarm_leaves_the_alarm_cell_empty_but_not_the_oracle_cell(tmp_path):
    write_runs(tmp_path, "task_seed0")
    folder = tmp_path / "liberox_groot" / "task_seed0" / f"{ARCH}__onset_pre"
    np.savez(folder / "scores.npz", **{f"S{i}": np.full(8, 0.1 * (i - 1)) for i in SUCCESSES_RANGE}, **{k: np.zeros(8) for k in FAILURES})  # no failure alarms
    r = mode_ablation_results(synthetic_inputs(names=("task_seed0",)), tmp_path, settings=[TASK])
    assert not any(k[1] == "alarm" for k in r) and r[key(OURS, "oracle", "top1")] == [100.0]
    alarm, oracle = mode_ablation_blocks(r, settings=[TASK])
    assert alarm.rows[0].top1 == [None, None] and oracle.rows[0].top1 == [100.0, 100.0]


def test_the_real_robot_uses_the_seed_zero_splits_only(tmp_path):
    names = ("loto_a", "lotoseed1_a")  # the extra seeds were run for our probe only
    write_runs(tmp_path, "loto_a", dataset="real8_groot")
    r = mode_ablation_results(synthetic_inputs("real8_groot", names), tmp_path, settings=[ROBOT])
    assert r[key(OURS, "oracle", "top1", "Real robot")] == [100.0]  # one value: the other seed's split (without runs) was not read


# ---------------------------------------------------------------- the table
def toy_blocks():
    row = lambda label, ours, a, b, c, d: tables.ModeAblationRow(label, ours, [a, b], [c, d], [1.0, 2.0], [3.0, 4.0])
    return [tables.ModeAblationBlock("At the first alarm", [row("Ours", True, 50.0, 40.0, 80.0, 70.0), row("Other", False, 45.0, 42.0, 80.0, 60.0)]),
            tables.ModeAblationBlock("At the true onset", [row("Ours", True, 90.0, 80.0, 95.0, 90.0), row("Other", False, 92.0, 70.0, 94.0, 91.0)])]


def test_best_and_second_best_are_marked_within_each_block_and_only_ours_is_shaded():
    tex = tables.mode_ablation_latex_table(toy_blocks(), ["LIBERO-X task"])
    rows = [line for line in tex.splitlines() if line.startswith((r"\rowcolor", "Ours", "Other"))]
    assert [r.startswith(r"\rowcolor") for r in rows] == [True, False, True, False]
    first, second = rows[0], rows[1]
    assert r"\textbf{50.0}" in first and r"\textit{45.0}" in second and r"\textit{40.0}" in first and r"\textbf{42.0}" in second  # top-1 seen, unseen
    assert r"\textbf{80.0}" in first and r"\textbf{80.0}" in second  # a tie: both best
    assert r"\textbf{92.0}" in rows[3] and r"\textit{90.0}" in rows[2]  # the second block is judged on its own rows, not against the first block


def test_the_table_has_the_layout_of_the_mode_table_with_a_heading_per_block():
    tex = tables.mode_ablation_latex_table(toy_blocks(), ["LIBERO-X task"])
    assert r"\begin{tabular}{lrrrr}" in tex and r"\multicolumn{2}{c}{Top-3 accuracy (\%)}" in tex and r"\label{tab:mode_ablation}" in tex
    assert r"\multicolumn{5}{l}{At the first alarm}" in tex and r"\multicolumn{5}{l}{At the true onset}" in tex
    assert tex.count(r"\toprule") == tex.count(r"\bottomrule") == 1 and tex.count("{") == tex.count("}")
    assert tables.MODE_ABLATION_CAPTION in tex and "upper bound" in tables.MODE_ABLATION_CAPTION


def test_the_std_version_adds_the_spread_to_every_cell_and_sets_the_table_at_the_full_text_width():
    plain, with_std = (tables.mode_ablation_latex_table(toy_blocks(), ["LIBERO-X task"], with_std=w) for w in (False, True))
    assert r"\pm" not in plain.split(r"\begin{tabular}")[1] and r"\resizebox{0.27\textwidth}" in plain
    first = next(line for line in with_std.splitlines() if line.startswith(r"\rowcolor"))
    assert r"\textbf{50.0{\scriptsize$\pm$1.0}}" in first and r"\textbf{80.0{\scriptsize$\pm$3.0}}" in first  # top-1 seen and top-3 seen: mean, then the std
    assert first.count(r"\pm") == 4 and r"\resizebox{1.0\textwidth}" in with_std and tables.STD_CAPTION in with_std and tables.STD_CAPTION not in plain


def test_the_mode_table_still_has_the_same_header_and_float_after_sharing_them():
    rows = [tables.ModeRow("Random", "baseline", [10.0, 10.0], [30.0, 30.0])]
    tex = tables.mode_latex_table(rows, ["LIBERO-X task"])
    assert "\n".join(tables.mode_header(["LIBERO-X task"])) in tex and r"\label{tab:modes}" in tex


def test_a_missing_cell_is_a_dash_and_the_csv_has_the_spread_of_every_cell():
    blocks = toy_blocks()
    blocks[0].rows[1].top1[0] = None
    assert "-- &" in tables.mode_ablation_latex_table(blocks, ["LIBERO-X task"])
    records = tables.mode_ablation_csv_rows(blocks, ["LIBERO-X task"])
    assert [(r["anchor"], r["method"]) for r in records] == [("At the first alarm", "Ours"), ("At the first alarm", "Other"), ("At the true onset", "Ours"), ("At the true onset", "Other")]
    assert records[0]["top-1 LIBERO-X task seen"] == 50.0 and records[0]["top-1 LIBERO-X task seen std"] == 1.0 and records[0]["top-3 LIBERO-X task unseen std"] == 4.0
    assert records[1]["top-1 LIBERO-X task seen"] is None and records[1]["top-1 LIBERO-X task seen std"] == 1.0


# ---------------------------------------------------------------- the command
def command_args(tmp_path):
    from types import SimpleNamespace

    out = tmp_path / "out"
    out.mkdir()
    return SimpleNamespace(root=tmp_path, out=out)


def test_the_command_writes_the_table_and_the_csv_from_the_runs(tmp_path, monkeypatch):
    from cli import figures as cli_figures

    names = {"liberox_groot": ("task_seed0", "signature_seed0", "skill_a"), "real8_groot": ("loto_a",)}
    for dataset, splits in names.items():
        for split in splits:
            write_runs(tmp_path, split, dataset=dataset)
    monkeypatch.setattr(cli_figures, "load_inputs", lambda dataset: synthetic_inputs(dataset, names[dataset]))
    args = command_args(tmp_path)
    cli_figures.write_mode_ablation_table(args)
    tex = (args.out / "table_mode_ablation.tex").read_text()
    assert r"\multicolumn{2}{c}{Real robot}" in tex and r"\multicolumn{2}{c}{skill}" in tex and "Score + modes + CLIP (ours)" in tex
    assert "top-1 Real robot seen std" in (args.out / "table_mode_ablation.csv").read_text()
    assert r"\pm" in (args.out / "table_mode_ablation_std.tex").read_text()


def test_the_command_skips_the_table_with_a_message_without_the_runs_of_the_probes_trained_without_the_score_term(tmp_path, monkeypatch, capsys):
    from cli import figures as cli_figures

    write_runs(tmp_path, "task_seed0")
    for folder in (tmp_path / mode_ablation.MODE_ONLY_DIR).rglob("lstm_modes__onset_pre"):
        for f in folder.iterdir():
            f.unlink()
        folder.rmdir()
    monkeypatch.setattr(cli_figures, "load_inputs", lambda dataset: synthetic_inputs(dataset, ("task_seed0",)))
    args = command_args(tmp_path)
    cli_figures.write_mode_ablation_table(args)
    assert "no mode-ablation table" in capsys.readouterr().out and not list(args.out.iterdir())
