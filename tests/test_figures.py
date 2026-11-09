import json
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from cli import figures as cli_figures  # noqa: E402
from fp.baselines.safe_grid import method_names as safe_method_names  # noqa: E402
from fp.eval import figures, plots, tables  # noqa: E402
from fp.eval.collect import collect_rows  # noqa: E402
from fp.eval.compute import collect_costs  # noqa: E402
from fp.metrics.curves import area_to_fpr, curve_on_grid, curve_points  # noqa: E402
from fp.models.ours import all_configs  # noqa: E402

LX_TASK = figures.Setting("LIBERO-X task", "liberox_groot", "task")
L10 = figures.Setting("LIBERO-10 GR00T", "libero10_groot", "all")


# ---------------------------------------------------------------- the curve on a grid (shared with the headline area)
def test_a_curve_starts_at_the_origin_is_made_monotone_and_keeps_the_best_point_at_one_fpr():
    xs, ys = curve_points(np.array([10, 10, 20, 30]), np.array([30, 50, 40, 60.0]))
    assert xs.tolist() == [0, 10, 20, 30] and ys.tolist() == [0, 50, 50, 60]  # 40 at FPR 20 is lifted to the running max; the better of two at 10 wins


def test_without_the_running_max_a_decrease_stays_and_points_at_one_fpr_are_averaged():
    xs, ys = curve_points(np.array([10, 10, 20.0]), np.array([30, 50, 10.0]), running_max=False)
    assert xs.tolist() == [0, 10, 20] and ys.tolist() == [0, 40, 10]


def test_the_grid_values_are_linear_between_points_and_flat_past_the_last_one():
    grid = np.array([0.0, 5.0, 10.0, 25.0, 40.0])
    assert curve_on_grid(np.array([10, 20.0]), np.array([20, 40.0]), grid).tolist() == [0, 10, 20, 40, 40]  # flat at 40 from FPR 20 on
    assert np.isnan(curve_on_grid(np.array([np.nan]), np.array([np.nan]), grid)).sum() == 0  # NaN points are dropped, the origin remains


def test_the_headline_area_is_unchanged_by_sharing_the_envelope_code():
    fpr, y = np.array([0.0, 20.0, 40.0]), np.array([0.0, 50.0, 100.0])
    assert area_to_fpr(fpr, y).value == pytest.approx(50.0) and not area_to_fpr(fpr, y).extended
    assert area_to_fpr(np.array([20.0]), np.array([50.0])).extended


# ---------------------------------------------------------------- the table formatting
def test_numbers_are_fixed_decimals_with_two_significant_digits_below_a_tenth_and_none_from_a_thousand():
    assert [tables.number(v, 1) for v in (0, 5.66, 40.04, 3056.4)] == ["0.0", "5.7", "40.0", "3056"]
    assert [tables.number(v, 2) for v in (0.0008, 0.0026, 0.013, 0.25, 0.0)] == ["0.0008", "0.0026", "0.013", "0.25", "0.00"]


def test_best_and_second_best_are_judged_on_the_displayed_numbers_and_ties_are_marked_alike():
    assert tables.marks([50.04, 50.01, 40.0, None], 1, True) == ["best", "best", "second", None]  # 50.0 and 50.0 as displayed
    assert tables.marks([1.0, 0.5, 0.5, 2.0], 1, False) == ["second", "best", "best", None]  # lower is better: 0.5 best, 1.0 second
    assert tables.marks([0.0008, 0.0004], 2, False) == ["second", "best"]  # not both rounded to 0.00
    assert tables.marks([None, None], 1, True) == [None, None] and tables.marks([7.0], 1, True) == ["best"]


def make_rows():
    return [tables.Row("Ours (onset_pre)", True, "max_conformal", [(50.0, 1.0), (40.0, 2.0), None, (30.0, 0.5)], [5.0, 26.4, 0.12, 0.92]),
            tables.Row("SAFE-LSTM", False, "safe_band", [(60.0, 1.5), (35.0, 2.5), (20.0, 1.0), (30.04, 0.0)], [39.1, 26.2, 0.067, 1.9]),
            tables.Row("Euclidean", False, "safe_band", [(55.0, 1.0), (10.0, 1.0), (15.0, 1.0), (12.0, 1.0)], [0.4, 3056.0, 13.0, 374.0])]


SETTINGS = ["LIBERO-X task", "LIBERO-10 π0.5"]


def test_the_latex_table_shades_our_rows_marks_the_best_and_second_best_and_escapes():
    tex = tables.latex_table(make_rows(), SETTINGS)
    lines = tex.splitlines()
    ours = next(line for line in lines if "Ours" in line)
    assert ours.startswith(r"\rowcolor{gray!15} Ours (onset\_pre) & MC & ")
    assert next(line for line in lines if line.startswith("SAFE-LSTM")).startswith("SAFE-LSTM & FC & ")  # the SAFE band is labelled FC
    assert not any(line.startswith(r"\rowcolor") for line in lines if "Ours" not in line)
    assert r"$\pi_{0.5}$" in tex and r"\multicolumn{2}{c}{LIBERO-X task}" in tex and "Cost (LIBERO-X)" in tex
    assert tex.count(r"\midrule") == 2  # after the header and between our rows and the baselines
    safe = next(line for line in lines if line.startswith("SAFE-LSTM"))
    assert r"\textbf{60.0}" in safe  # best seen area of the first setting
    assert r"\textit{55.0}" in next(line for line in lines if line.startswith("Euclidean"))  # the second
    assert r"\textbf{30.0}" in ours and r"\textbf{30.0}" in safe  # 30.0 and 30.04 display alike: both best
    assert ours.count("--") == 1  # the one missing area
    assert r"\textbf{0.4}" in next(line for line in lines if line.startswith("Euclidean"))  # the cheapest fit is the best in a cost column
    assert r"\textit{0.12}" in ours and r"\textbf{0.067}" in safe and r"\textbf{0.92}" in ours and r"\textit{1.90}" in safe  # the GPU latency: SAFE-LSTM best, ours second; the CPU latency: ours best


def test_the_std_version_adds_the_spread_without_changing_what_is_marked():
    plain, with_std = tables.latex_table(make_rows(), SETTINGS), tables.latex_table(make_rows(), SETTINGS, with_std=True)
    assert r"\scriptsize$\pm$1.5" in with_std and r"\pm" not in plain
    assert plain.count(r"\textbf") == with_std.count(r"\textbf") and plain.count(r"\textit") == with_std.count(r"\textit")


def test_the_latex_is_a_full_width_table_float_with_the_caption_above_a_label_and_a_scaled_tabular():
    tex = tables.latex_table(make_rows(), SETTINGS)
    lines = tex.splitlines()
    assert lines[0].startswith("% python -m cli.figures") and "booktabs, graphicx and xcolor" in lines[0]
    assert lines[1] == r"\begin{table*}[t]" and lines[2] == r"\centering" and tex.rstrip().endswith(r"\end{table*}")
    order = [tex.index(s) for s in (r"\begin{table*}", r"\caption{", r"\label{tab:main}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}", r"\end{tabular}%", r"\end{table*}")]
    assert order == sorted(order) and len(set(order)) == 7  # caption above the table, then the label, then the scaled tabular
    assert tex.count(r"\begin{table*}") == tex.count(r"\end{table*}") == tex.count(r"\begin{tabular}") == tex.count(r"\end{tabular}") == 1
    assert tex.count("{") == tex.count("}")  # balanced braces, so the caption and the \resizebox close properly
    caption = tex[tex.index(r"\caption{") : tex.index(r"\label{")]
    assert all(s in caption for s in ("area under the curve of TPR early", "40\\%", "MC max-conformal", "FC functional conformal", "seen", "unseen", "Best in bold"))
    assert r"\pm" not in tex and "standard deviation" not in caption


def test_the_std_table_has_its_own_label_and_says_what_the_plus_minus_is_and_a_caption_and_label_can_be_given():
    std = tables.latex_table(make_rows(), SETTINGS, with_std=True)
    assert r"\label{tab:main_std}" in std and r"\label{tab:main}" not in std and "Values after $\\pm$ are the standard deviation over the splits." in std
    custom = tables.latex_table(make_rows(), SETTINGS, caption="My caption.", label="tab:mine")
    assert r"\caption{My caption.}" in custom and r"\label{tab:mine}" in custom and "Early failure detection" not in custom


def test_the_table_lists_the_baselines_first_and_ours_at_the_bottom_set_off_by_a_rule():
    entries = {e.label: e for e in figures.ENTRIES}
    kinds = [entries[label].kind == "ours" for label in figures.TABLE_LABELS]
    assert kinds == [False] * 8 + [True] and figures.TABLE_LABELS[-1] == "Ours (onset_pre)"  # only the headline variant, as the last row (the others: the ablation table)
    assert "Ours (onset_traj)" not in figures.TABLE_LABELS
    rows = [tables.Row(label, entries[label].kind == "ours", "max_conformal", [(50.0, 1.0)] * 4, [1.0, 2.0, 0.5, 0.25]) for label in figures.TABLE_LABELS]
    tex = tables.latex_table(rows, SETTINGS)
    lines = [line for line in tex.splitlines() if line.startswith(("SAFE-LSTM", r"\rowcolor", r"\midrule"))]
    assert lines[0] == r"\midrule" and lines[1].startswith("SAFE-LSTM") and lines[2] == r"\midrule" and all(line.startswith(r"\rowcolor") for line in lines[3:])
    assert tex.index("PCA-KMeans") < tex.index(r"\rowcolor") < tex.index(r"Ours (onset\_pre)") and r"onset\_traj" not in tex.split(r"\begin{tabular}")[1]


def test_the_csv_has_every_mean_and_std_and_the_costs():
    records = tables.csv_rows(make_rows(), SETTINGS)
    assert records[0]["method"] == "Ours (onset_pre)" and records[0]["LIBERO-X task seen mean"] == 50.0 and records[0]["LIBERO-X task unseen std"] == 2.0
    assert records[0]["LIBERO-10 π0.5 seen mean"] is None and records[1]["State (MB)"] == 26.2 and records[2]["calibrator"] == "safe_band"


# ---------------------------------------------------------------- the data layer on a synthetic results folder
CAL = {"stac": "max_conformal", "actprobe": "max_conformal", "safe_lstm": "safe_band", "ours": "max_conformal"}


def curve_rows(scale, kind="early"):
    fpr = [0.0, 10.0, 20.0, 30.0, 40.0]
    early = [scale * f / 40 * 100 for f in fpr]
    on_time = [5.0, 8.0, 4.0, 6.0, 3.0]  # not monotone on purpose
    return [dict(alpha=0.1 * i, fpr=f, tpr=min(100.0, e + 10), tpr_early=e, tpr_on_time=o, ldr=10.0) for i, (f, e, o) in enumerate(zip(fpr, early, on_time))]


def write_run(root, dataset, split, method, area, scale=1.0, costs=None, calibrator=None):
    calibrator = calibrator or CAL.get(method.split("__")[0], "max_conformal")
    path = root / dataset / split / method
    path.mkdir(parents=True)
    (path / "history.json").write_text(json.dumps({"method": method, "dataset": dataset, "split": split, **(costs or {})}))
    sets = {s: {"area": area + (1.0 if s == "seen" else 0.0), "extended": False, "n_fail": 5, "n_succ": 5} for s in ("val", "seen", "unseen")}
    (path / "summary.json").write_text(json.dumps({"calibrators": {calibrator: {"sets": sets}}}))
    (path / f"curve_{calibrator}.json").write_text(
        json.dumps({"calibrator": calibrator, "sets": {s: curve_rows(scale) for s in ("val", "seen", "unseen")}}))


@pytest.fixture
def results(tmp_path):
    root = tmp_path / "results"
    for split in ("paper_seed0", "paper_seed1"):
        write_run(root, "libero10_groot", split, "stac__mmd3_step", 20.0, 0.4, costs=dict(fit_seconds=0.0, state_mb=0.0, score_ms_per_step=0.013))
        write_run(root, "libero10_groot", split, "stac__mmd7_step", 10.0, 0.2)
        write_run(root, "libero10_groot", split, "actprobe", 30.0, 0.6, costs=dict(fit_seconds=7.0, state_mb=0.09, score_ms_per_step=0.0005))
        write_run(root, "libero10_groot", split, "safe_lstm__lr1e-3_reg1e-2", 50.0, 1.0, calibrator="safe_band")
    # one split of one dataset has an extra, worse-on-validation configuration
    return root


def test_the_chosen_method_is_the_best_on_the_mean_validation_area_and_missing_runs_are_reported_not_hidden(results):
    rows = collect_rows(results)
    stac = next(e for e in figures.ENTRIES if e.label == "STAC")
    assert figures.choose(rows, L10, stac) == "stac__mmd3_step"  # validation area 20 against 10
    with pytest.raises(ValueError, match="no evaluated runs"):
        figures.choose(rows, LX_TASK, stac)
    chosen, skipped = figures.choose_all(rows)
    assert chosen[("LIBERO-10 GR00T", "STAC")] == "stac__mmd3_step" and chosen[("LIBERO-10 GR00T", "ActProbe")] == "actprobe"
    assert not any(label == "LIBERO-X task" for label, _ in chosen) and any("LIBERO-X task / STAC" in line for line in skipped)
    assert ("LIBERO-10 GR00T", "Ours (onset_pre)") not in chosen  # no probe runs here


def test_an_entry_of_two_families_takes_the_one_with_the_better_validation_area(results):
    rows = collect_rows(results)
    entry = figures.Entry("Both", "action", "solid", families=("stac", "actprobe"))
    assert figures.choose(rows, L10, entry) == "actprobe"  # validation 30 against 20
    assert figures.choose(rows, L10, figures.Entry("Both", "action", "solid", families=("actprobe", "stac"))) == "actprobe"  # the order of the families does not matter


def test_one_of_our_label_variants_picks_the_best_architecture_on_validation(tmp_path):
    root = tmp_path / "results"
    configs = [c.name for c in all_configs()]
    for split in ("paper_seed0", "paper_seed1"):
        for i, name in enumerate(configs):
            write_run(root, "libero10_groot", split, f"{name}__onset_pre", 10.0 + i)  # the last architecture is the best
    assert figures.choose(collect_rows(root), L10, figures.Entry("Ours", "ours", "solid", variant="onset_pre")) == f"{configs[-1]}__onset_pre"


def test_all_our_entries_use_the_one_decided_architecture_which_has_the_failure_mode_head():
    ours = [e for e in figures.ENTRIES if e.kind == "ours"]
    assert {e.architecture for e in ours} == {figures.OURS_ARCHITECTURE} and len(ours) == 4
    assert figures.OURS_ARCHITECTURE in {c.name for c in all_configs()} and "modes" in figures.OURS_ARCHITECTURE


def test_with_a_fixed_architecture_every_setting_gets_that_architecture_and_a_missing_run_is_reported(tmp_path):
    root = tmp_path / "results"
    arch = figures.OURS_ARCHITECTURE
    for split in ("paper_seed0", "paper_seed1"):
        write_run(root, "libero10_groot", split, f"{arch}__onset_pre", 10.0)
        write_run(root, "libero10_groot", split, "mlp_plain__onset_pre", 90.0)  # a much better architecture on validation: not chosen, the architecture is fixed
    write_run(root, "liberox_groot", "task_seed0", f"{arch}__onset_pre", 10.0)
    write_run(root, "liberox_groot", "task_seed1", "mlp_plain__onset_pre", 10.0)  # the fixed architecture is missing on this split
    rows = collect_rows(root)
    entry = next(e for e in figures.ENTRIES if e.label == "Ours (onset_pre)")
    assert figures.choose(rows, L10, entry) == f"{arch}__onset_pre"
    with pytest.raises(ValueError, match="task_seed1"):
        figures.choose(rows, LX_TASK, entry)
    with pytest.raises(ValueError, match="any split"):
        figures.choose(rows, figures.SETTINGS[0], entry)  # LIBERO-10 pi0.5 has no runs at all


def write_safe_runs(root, dataset, splits, areas):
    """One run of each of the 12 SAFE-LSTM configurations on each split; ``areas[i]`` is the validation area of configuration i."""
    for split in splits:
        for name, area in zip(safe_method_names("safe_lstm"), areas):
            write_run(root, dataset, split, name, area)


def test_safe_shares_one_configuration_across_the_three_libero_x_splits_but_libero_10_keeps_its_own(tmp_path):
    root = tmp_path / "results"
    names = safe_method_names("safe_lstm")
    for group, winner in (("task", 2), ("signature", 5), ("skill", 8)):  # each group has its own best configuration...
        areas = [5.0] * 12
        areas[winner], areas[11] = 70.0, 60.0  # ...but configuration 11 is second best everywhere and the best on average
        write_safe_runs(root, "liberox_groot", [f"{group}_seed0", f"{group}_seed1"], areas)
    areas = [5.0] * 12
    areas[4] = 80.0
    write_safe_runs(root, "libero10_groot", ["paper_seed0", "paper_seed1"], areas)
    rows = collect_rows(root)
    entry = next(e for e in figures.ENTRIES if e.label == "SAFE-LSTM")
    settings = {s.label: s for s in figures.SETTINGS}
    assert [figures.choose(rows, settings[f"LIBERO-X {g}"], entry) for g in ("task", "signature", "skill")] == [names[11]] * 3
    assert figures.choose(rows, settings["LIBERO-10 GR00T"], entry) == names[4]  # its own per-setting best
    # a baseline without the shared rule still picks per setting
    unshared = figures.Entry("SAFE-LSTM", "safe", "solid", families=("safe_lstm",))
    assert [figures.choose(rows, settings[f"LIBERO-X {g}"], unshared) for g in ("task", "signature", "skill")] == [names[2], names[5], names[8]]


def test_in_the_shared_choice_each_group_counts_once_whatever_its_number_of_splits(tmp_path):
    root = tmp_path / "results"
    names = safe_method_names("safe_lstm")
    # configuration 3 is great on task and signature (1 split each) and poor on skill (4 splits); configuration 7 is average everywhere
    for group, splits, (va, vb) in (("task", ["task_seed0"], (90.0, 50.0)), ("signature", ["signature_seed0"], (90.0, 50.0)),
                                    ("skill", [f"skill_{c}" for c in "abcd"], (10.0, 50.0))):
        areas = [5.0] * 12
        areas[3], areas[7] = va, vb
        write_safe_runs(root, "liberox_groot", splits, areas)
    # per group: (90 + 90 + 10) / 3 = 63.3 for configuration 3 against 50; per split it would be (90 + 90 + 4 * 10) / 6 = 36.7 against 50
    entry = next(e for e in figures.ENTRIES if e.label == "SAFE-LSTM")
    assert figures.choose(collect_rows(root), figures.SETTINGS[2], entry) == names[3]


def test_a_shared_choice_needs_every_configuration_evaluated_in_every_setting(tmp_path):
    root = tmp_path / "results"
    write_safe_runs(root, "liberox_groot", ["task_seed0"], [10.0] * 12)  # signature and skill have no runs
    entry = next(e for e in figures.ENTRIES if e.label == "SAFE-LSTM")
    with pytest.raises(ValueError, match="no evaluated runs"):
        figures.choose(collect_rows(root), figures.SETTINGS[2], entry)


def test_the_area_cell_is_the_mean_and_the_population_std_over_the_splits_with_the_protocols_calibrator(results):
    rows = collect_rows(results)
    write_run(results, "libero10_groot", "paper_seed2", "actprobe", 40.0)
    rows = collect_rows(results)
    cell = figures.area_cell(rows, L10, "actprobe", "unseen")
    assert (cell.n, cell.mean) == (3, pytest.approx(100 / 3)) and cell.std == pytest.approx(np.std([30.0, 30.0, 40.0]))
    assert figures.area_cell(rows, L10, "actprobe", "seen").mean == pytest.approx(100 / 3 + 1)
    with pytest.raises(ValueError, match="no area"):
        figures.area_cell(rows, L10, "stac__mmd3_step", "nope")


def test_the_mean_curve_averages_the_splits_on_the_grid_and_the_on_time_curve_is_not_flattened(results):
    rows = collect_rows(results)
    early = figures.mean_curve(rows, L10, "actprobe", "unseen", "tpr_early", results)
    assert early.shape == figures.GRID.shape and early[0] == 0 and early[-1] == pytest.approx(60.0)  # scale 0.6 at FPR 40
    assert early[np.searchsorted(figures.GRID, 20.0)] == pytest.approx(30.0) and (np.diff(early) >= 0).all()
    on_time = figures.mean_curve(rows, L10, "actprobe", "unseen", "tpr_on_time", results)
    assert on_time[np.searchsorted(figures.GRID, 20.0)] == pytest.approx(4.0) and on_time[np.searchsorted(figures.GRID, 10.0)] == pytest.approx(8.0)  # it goes down
    with pytest.raises(FileNotFoundError):
        figures.mean_curve(rows, L10, "stac__mmd3_step", "unseen", "tpr_early", results / "elsewhere")


def test_costs_are_the_median_over_the_chosen_runs_of_each_kind_of_split_on_liberox(tmp_path):
    root = tmp_path / "results"
    for split, fit in (("task_seed0", 10.0), ("task_seed1", 30.0), ("signature_seed0", 20.0), ("skill_stack_objects", 100.0)):
        write_run(root, "liberox_groot", split, "m_task" if split.startswith("task") else "m_other", 1.0, costs=dict(fit_seconds=fit, state_mb=2.0, score_ms_per_step=None))
    write_run(root, "libero10_groot", "paper_seed0", "m_task", 1.0, costs=dict(fit_seconds=999.0, state_mb=9.0, score_ms_per_step=1.0))  # another dataset: ignored
    chosen = {("LIBERO-X task", "E"): "m_task", ("LIBERO-X signature", "E"): "m_other", ("LIBERO-X skill", "E"): "m_other"}
    costs = figures.entry_costs(figures.Entry("E", "safe", "solid"), chosen, collect_costs(root))
    assert costs == {"fit_seconds": 25.0, "state_mb": 2.0, "score_ms_per_step": None}  # median of 10, 30 (task), 20 (signature), 100 (skill)
    assert figures.entry_costs(figures.Entry("Z", "safe", "solid"), chosen, collect_costs(root)) == {f: None for f in figures.COST_FIELDS}


def test_the_benchmark_plan_lists_the_chosen_architecture_and_the_splits_of_each_kind_for_our_entries_only(results):
    rows = collect_rows(results)
    for split in ("task_seed0", "task_seed1"):
        write_run(results, "liberox_groot", split, "lstm_plain__onset_pre", 1.0)
    rows = collect_rows(results)
    chosen = {("LIBERO-X task", "Ours (onset_pre)"): "lstm_plain__onset_pre", ("LIBERO-X task", "STAC"): "stac__mmd3_step"}
    assert figures.benchmark_plan(chosen, rows) == [("lstm_plain", "onset_pre", ["task_seed0", "task_seed1"])]


# ---------------------------------------------------------------- the plots
def synthetic_curves(settings, entries):
    return {(s.label, e.label): np.linspace(0, 20 + 10 * i, len(figures.GRID)) for s in settings for i, e in enumerate(entries)}


def test_the_entries_leave_out_the_safe_pre_onset_ablations_and_every_label_has_a_colour():
    labels = [e.label for e in figures.ENTRIES]
    assert not any("pre-onset" in label for label in labels) and len(set(labels)) == len(labels) == 13
    assert all(e.variant in plots.VIRIDIS_AT for e in figures.ENTRIES if e.kind == "ours") and {e.kind for e in figures.ENTRIES} <= {"ours", "safe", "action", "distance"}


def one_row(settings, entries):
    return [("Unseen tasks", synthetic_curves(settings, entries))]


def test_seen_is_the_top_row_unseen_the_bottom_one_and_one_legend_below_has_no_category_names_or_chance_entry():
    entries = list(figures.ENTRIES)
    unseen = synthetic_curves(figures.SETTINGS, entries)
    seen = {k: np.asarray(v) + 5.0 for k, v in unseen.items()}  # distinguishable from the unseen curves
    fig = plots.curves_figure([("Seen tasks", seen), ("Unseen tasks", unseen)], list(figures.SETTINGS), entries, "TPR early (%)")
    n = len(figures.SETTINGS)
    top, bottom = fig.axes[:n], fig.axes[n:2 * n]
    assert len(fig.axes) == 2 * n and len({round(ax.get_position().y0, 3) for ax in top}) == len({round(ax.get_position().y0, 3) for ax in bottom}) == 1
    assert top[0].get_position().y0 > bottom[0].get_position().y0  # seen above unseen
    assert [ax.get_title() for ax in top] == [plots.PANEL_TITLES.get(s.label, s.label).split("\n")[0] for s in figures.SETTINGS] and not any(ax.get_title() or ax.texts for ax in bottom)  # names above the top row only
    assert all(ax.get_xlabel() == "FPR (%)" for ax in bottom) and not any(ax.get_xlabel() for ax in top)
    assert top[0].get_ylabel() == "Seen tasks\nTPR early (%)" and bottom[0].get_ylabel() == "Unseen tasks\nTPR early (%)"
    assert not any(ax.get_ylabel() for ax in top[1:] + bottom[1:])
    assert all(len(ax.lines) == len(entries) for ax in fig.axes)  # one line per method and nothing else (no chance line)
    top_max, bottom_max = (max(max(line.get_ydata()) for line in ax.lines) for ax in (top[0], bottom[0]))
    assert top_max == pytest.approx(bottom_max + 5.0)  # the top row draws the seen curves
    (legend,) = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == [e.label for e in entries]  # no headings such as "SAFE" or "Embedding distance"
    fig.canvas.draw()
    assert legend.get_window_extent().y1 <= min(ax.get_window_extent().y0 for ax in bottom)  # below both rows
    assert fig.get_size_inches()[1] == pytest.approx(2 * plots.PANEL_HEIGHT)
    matplotlib.pyplot.close(fig)


def test_ours_are_thick_viridis_lines_and_every_other_method_is_a_thin_black_or_gray_one():
    entries = list(figures.ENTRIES)
    fig = plots.curves_figure(one_row(figures.SETTINGS[:1], entries), list(figures.SETTINGS[:1]), entries, "y")
    drawn_order = sorted(entries, key=lambda e: e.kind == "ours")  # ours are drawn last, on top
    lines = {e.label: line for e, line in zip(drawn_order, fig.axes[0].lines)}
    viridis = matplotlib.colormaps["viridis"]
    for e in (e for e in entries if e.kind == "ours"):
        line = lines[e.label]
        assert matplotlib.colors.to_rgba(line.get_color()) == pytest.approx(viridis(plots.VIRIDIS_AT[e.variant])) and line.get_linewidth() == plots.WIDTH["ours"]
        assert line.get_zorder() == 3
    assert len({str(lines[e.label].get_color()) for e in entries if e.kind == "ours"}) == 4  # four different colours
    for e in (e for e in entries if e.kind != "ours"):
        r, g, b = matplotlib.colors.to_rgb(lines[e.label].get_color())
        assert r == g == b and lines[e.label].get_linewidth() == plots.DEFAULT_WIDTH < plots.WIDTH["ours"]  # black or gray, thin
    for kind in ("safe", "action", "distance"):  # methods of one shade differ by line style
        styles = [e.style for e in entries if e.kind == kind]
        assert len(set(styles)) == len(styles)
    ours_entries = [e for e in entries if e.kind == "ours"]
    assert len({str(plots.line_kwargs(e)["linestyle"]) for e in ours_entries}) == 4 and len({e.style for e in ours_entries}) == 4  # line style, not hue
    matplotlib.pyplot.close(fig)


def test_axis_text_and_spines_are_black_and_only_horizontal_grid_lines_are_drawn():
    entries = list(figures.ENTRIES[:2])
    rows = [("Seen tasks", synthetic_curves(figures.SETTINGS[:2], entries)), ("Unseen tasks", synthetic_curves(figures.SETTINGS[:2], entries))]
    fig = plots.curves_figure(rows, list(figures.SETTINGS[:2]), entries, "TPR early (%)")
    fig.canvas.draw()
    for ax in fig.axes:
        assert matplotlib.colors.to_hex(ax.xaxis.label.get_color()) == matplotlib.colors.to_hex(ax.yaxis.label.get_color()) == "#000000"
        assert all(matplotlib.colors.to_hex(t.get_color()) == "#000000" for t in ax.get_xticklabels() + ax.get_yticklabels())
        assert matplotlib.colors.to_hex(ax.spines["left"].get_edgecolor()) == matplotlib.colors.to_hex(ax.spines["bottom"].get_edgecolor()) == "#000000"
        assert any(g.get_visible() for g in ax.yaxis.get_gridlines()) and not any(g.get_visible() for g in ax.xaxis.get_gridlines())
        assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
    matplotlib.pyplot.close(fig)


def test_methods_or_settings_without_a_curve_are_left_out():
    entries = [e for e in figures.ENTRIES if e.label in ("Ours (onset_pre)", "SAFE-LSTM")]
    curves = {(figures.SETTINGS[0].label, "SAFE-LSTM"): np.linspace(0, 10, len(figures.GRID))}
    fig = plots.curves_figure([("Unseen tasks", curves)], list(figures.SETTINGS[:2]), entries, "TPR on-time (%)")
    first, second = fig.axes
    assert len(first.lines) == 1 and len(second.lines) == 0
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["SAFE-LSTM"]  # only what was drawn
    matplotlib.pyplot.close(fig)


def test_the_figure_is_as_wide_as_its_panels_and_save_writes_png_and_pdf(tmp_path):
    entries = list(figures.ENTRIES[:2])
    narrow = plots.curves_figure(one_row(figures.SETTINGS[:2], entries), list(figures.SETTINGS[:2]), entries, "y")
    wide = plots.curves_figure(one_row(figures.SETTINGS, entries), list(figures.SETTINGS), entries, "y")
    assert wide.get_size_inches()[0] / narrow.get_size_inches()[0] == pytest.approx(len(figures.SETTINGS) / 2)
    paths = plots.save(wide, "demo", tmp_path / "out")
    assert [p.name for p in paths] == ["demo.png", "demo.pdf"] and all(p.stat().st_size > 1000 for p in paths)
    matplotlib.pyplot.close(narrow)


def test_the_text_uses_matplotlibs_default_font_and_the_pdf_embeds_it_as_truetype(tmp_path):
    entries = list(figures.ENTRIES[:2])
    fig = plots.curves_figure(one_row(figures.SETTINGS[:2], entries), list(figures.SETTINGS[:2]), entries, "TPR early (%)")
    texts = [fig.axes[0].title, fig.axes[0].xaxis.label, fig.axes[0].yaxis.label, *fig.legends[0].get_texts(), *fig.axes[0].get_xticklabels()]
    assert {t.get_fontname() for t in texts} == {matplotlib.rcParamsDefault["font.sans-serif"][0]}  # DejaVu Sans, nothing is overridden
    pdf = plots.save(fig, "font", tmp_path)[1].read_bytes()
    assert b"DejaVuSans" in pdf.replace(b" ", b"") and b"/Type3" not in pdf  # embedded as TrueType, not as Type 3 glyph outlines


# ---------------------------------------------------------------- the command
def test_the_command_writes_the_tables_the_chosen_methods_and_the_two_curve_figures(results, monkeypatch, capsys):
    out = results / "figures"
    monkeypatch.setattr(sys, "argv", ["figures", "--root", str(results), "--out", str(out), "--costs", str(results / "no_benchmark")])
    cli_figures.main()
    tex = (out / "table_main.tex").read_text()
    assert "STAC & MC" in tex and "ActProbe & MC" in tex and "Ours" not in tex  # only entries with runs get a row
    assert not any(line.startswith("SAFE-LSTM") for line in tex.splitlines())  # its grid has 12 configurations and only one was run: no row
    assert "skipped: LIBERO-10 GR00T / SAFE-LSTM" in capsys.readouterr().out
    stac = next(line for line in tex.splitlines() if line.startswith("STAC"))
    assert "20.0" in stac and "21.0" in stac and stac.count("--") == 2 * len(figures.SETTINGS) - 2 + 4  # one setting has runs (2 area cells); the other area cells and, with no LIBERO-X runs and no latency file, 4 costs are missing
    assert (out / "table_main_std.tex").exists()
    chosen = (out / "chosen.csv").read_text()
    assert "LIBERO-10 GR00T,STAC,stac__mmd3_step,max_conformal" in chosen
    for stem in ("curves_tpr_early", "curves_tpr_on_time"):
        assert (out / f"{stem}.png").exists() and (out / f"{stem}.pdf").exists()
    header = (out / "table_main.csv").read_text().splitlines()[0]
    assert header.startswith(f"method,calibrator,{figures.SETTINGS[0].label} seen mean") and header.endswith("CPU (ms)")
    assert [s.label for s in figures.SETTINGS] == ["LIBERO-10 π0.5", "LIBERO-10 GR00T", "LIBERO-X task", "LIBERO-X signature", "LIBERO-X skill", "Real robot"]  # the requested order, the real robot last


def test_the_command_prints_the_benchmark_commands_instead_of_writing_anything(results, monkeypatch, capsys):
    for split in ("task_seed0", "task_seed1"):
        for cfg in all_configs():
            write_run(results, "liberox_groot", split, f"{cfg.name}__onset_pre", 10.0 + len(cfg.name))
    out = results / "figures"
    monkeypatch.setattr(sys, "argv", ["figures", "--root", str(results), "--out", str(out), "--costs", "/tmp/bench", "--print-benchmark"])
    cli_figures.main()
    printed = capsys.readouterr().out
    assert "python -m cli.train --out /tmp/bench --datasets liberox_groot --variants onset_pre --probes" in printed and "--splits task_seed0 task_seed1" in printed
    assert not out.exists()


def test_the_real_robot_setting_is_last_pools_its_leave_one_task_out_splits_and_has_no_action_baselines(tmp_path):
    real = figures.SETTINGS[-1]
    assert (real.label, real.dataset, real.group, real.also) == ("Real robot", "real8_groot", "all", ("loto_seeds",))
    root = tmp_path / "results"
    arch = figures.OURS_ARCHITECTURE
    safe = safe_method_names("safe_lstm")
    for split in ("loto_basket", "loto_tomato"):
        write_run(root, "real8_groot", split, f"{arch}__onset_pre", 40.0)
        write_run(root, "real8_groot", split, "mahalanobis__d32_c16_step", 25.0, calibrator="safe_band")
        for i, name in enumerate(safe):
            write_run(root, "real8_groot", split, name, 80.0 if i == 7 else 5.0, calibrator="safe_band")
    rows = collect_rows(root)
    chosen, skipped = figures.choose_all(rows, settings=[real])
    assert chosen[("Real robot", "Ours (onset_pre)")] == f"{arch}__onset_pre"
    assert chosen[("Real robot", "SAFE-LSTM")] == safe[7]  # its own best configuration: the shared rule belongs to LIBERO-X only
    for entry in ("STAC", "STAC-Single", "ActProbe", "FailureSpot"):
        assert ("Real robot", entry) not in chosen and any(line.startswith(f"Real robot / {entry}:") for line in skipped)
    assert figures.splits_of(rows, real) == ["loto_basket", "loto_tomato"] and figures.area_cell(rows, real, chosen[("Real robot", "SAFE-LSTM")], "unseen").n == 2


def test_the_real_robot_cells_average_over_the_extra_seeds_while_the_configuration_is_chosen_on_seed_zero_only(tmp_path):
    real = figures.SETTINGS[-1]
    root = tmp_path / "results"
    names = safe_method_names("safe_lstm")
    for task in ("basket", "tomato"):
        for i, name in enumerate(names):  # seed 0: configuration 7 is the best on validation
            write_run(root, "real8_groot", f"loto_{task}", name, 80.0 if i == 7 else 5.0, calibrator="safe_band")
        for seed in (1, 2):  # the extra seeds: only the chosen configuration was run (and configuration 3 would have been better there)
            write_run(root, "real8_groot", f"lotoseed{seed}_{task}", names[7], 40.0 + seed, calibrator="safe_band")
            write_run(root, "real8_groot", f"lotoseed{seed}_{task}", names[3], 90.0, calibrator="safe_band")
        write_run(root, "real8_groot", f"loto_{task}", f"{figures.OURS_ARCHITECTURE}__onset_pre", 30.0)
        for seed in (1, 2):
            write_run(root, "real8_groot", f"lotoseed{seed}_{task}", f"{figures.OURS_ARCHITECTURE}__onset_pre", 30.0)
    rows = collect_rows(root)
    assert figures.splits_of(rows, real) == sorted([f"loto_{t}" for t in ("basket", "tomato")] + [f"lotoseed{s}_{t}" for s in (1, 2) for t in ("basket", "tomato")])
    chosen, skipped = figures.choose_all(rows, settings=[real])
    assert chosen[("Real robot", "SAFE-LSTM")] == names[7]  # chosen on the seed-0 splits only
    cell = figures.area_cell(rows, real, names[7], "val")
    assert cell.n == 6 and cell.mean == pytest.approx((2 * 80.0 + 2 * 41.0 + 2 * 42.0) / 6)  # the mean over the 6 splits that have it (2 tasks x seeds 0, 1, 2)
    assert chosen[("Real robot", "Ours (onset_pre)")] == f"{figures.OURS_ARCHITECTURE}__onset_pre"
    missing = tmp_path / "missing"
    write_run(missing, "real8_groot", "loto_basket", f"{figures.OURS_ARCHITECTURE}__onset_pre", 30.0)
    write_run(missing, "real8_groot", "lotoseed1_basket", "mlp_plain__onset_pre", 30.0)  # the fixed architecture is absent on an extra seed: reported
    with pytest.raises(ValueError, match="lotoseed1_basket"):
        figures.choose(collect_rows(missing), real, next(e for e in figures.ENTRIES if e.label == "Ours (onset_pre)"))


def test_a_split_without_a_curve_file_is_left_out_of_the_mean_curve_like_it_is_left_out_of_the_cell(tmp_path):
    real = figures.SETTINGS[-1]
    root = tmp_path / "results"
    method = "mahalanobis__cumsum"
    write_run(root, "real8_groot", "loto_basket", method, 30.0, 1.0, calibrator="safe_band")
    write_run(root, "real8_groot", "loto_tomato", method, 50.0, 0.5, calibrator="safe_band")
    write_run(root, "real8_groot", "lotoseed1_basket", "lstm_plain__onset_pre", 10.0)  # this split exists, but this method has no run on it
    rows = collect_rows(root)
    assert "lotoseed1_basket" in figures.splits_of(rows, real)
    both = figures.mean_curve(rows, real, method, "unseen", root=root)
    one = figures.mean_curve(collect_rows(tmp_path / "results"), real, method, "unseen", root=root)
    assert np.array_equal(both, one) and figures.area_cell(rows, real, method, "unseen").n == 2


# ---------------------------------------------------------------- the compact curve figures
def test_a_panel_can_be_made_wider_so_that_fewer_panels_keep_the_width_of_a_full_figure():
    entries = list(figures.ENTRIES[:2])
    default = plots.curves_figure(one_row(figures.SETTINGS[:3], entries), list(figures.SETTINGS[:3]), entries, "y")
    wide = plots.curves_figure(one_row(figures.SETTINGS[:3], entries), list(figures.SETTINGS[:3]), entries, "y", panel_width=2 * plots.PANEL_WIDTH)
    assert wide.get_size_inches()[0] == pytest.approx(2 * default.get_size_inches()[0]) and wide.get_size_inches()[1] == pytest.approx(default.get_size_inches()[1])
    matplotlib.pyplot.close("all")


def test_the_height_scale_shrinks_the_height_and_not_the_width_or_the_font_sizes():
    entries = list(figures.ENTRIES[:5])
    args = (one_row(figures.SETTINGS[:2], entries), list(figures.SETTINGS[:2]), entries, "y")
    full, flat = plots.curves_figure(*args), plots.curves_figure(*args, height_scale=0.5)
    (fw, fh), (lw, lh) = full.get_size_inches(), flat.get_size_inches()
    assert lw == pytest.approx(fw) and lh == pytest.approx(fh / 2)  # the same width at half the height
    assert flat.axes[0].title.get_fontsize() == full.axes[0].title.get_fontsize() and flat.axes[0].xaxis.label.get_fontsize() == full.axes[0].xaxis.label.get_fontsize()
    matplotlib.pyplot.close("all")


def test_the_legend_is_set_closer_together_only_in_a_figure_too_narrow_for_it():
    entries = list(figures.ENTRIES[:5])
    narrow = plots.curves_figure(one_row(figures.SETTINGS[:2], entries), list(figures.SETTINGS[:2]), entries, "y", legend_rows=1)  # 5.0 in
    wide = plots.curves_figure(one_row(figures.SETTINGS, entries), list(figures.SETTINGS), entries, "y", legend_rows=1)  # 15.0 in
    assert narrow.get_size_inches()[0] < plots.NARROW_FIGURE <= wide.get_size_inches()[0]
    assert narrow.legends[0].get_texts()[0].get_fontsize() == 8.5 and wide.legends[0].get_texts()[0].get_fontsize() == 9
    matplotlib.pyplot.close("all")


def test_the_legend_has_two_rows_unless_asked_for_one():
    entries = list(figures.ENTRIES[:5])
    two = plots.curves_figure(one_row(figures.SETTINGS[:2], entries), list(figures.SETTINGS[:2]), entries, "y")
    one = plots.curves_figure(one_row(figures.SETTINGS[:2], entries), list(figures.SETTINGS[:2]), entries, "y", legend_rows=1)
    assert (two.legends[0]._ncols, one.legends[0]._ncols) == (3, 5)  # five entries: two rows of up to three columns, or one row of five
    matplotlib.pyplot.close("all")


def test_the_compact_curve_figures_are_the_simulated_settings_with_the_compact_methods_at_the_width_of_the_full_figures(tmp_path, monkeypatch, capsys):
    root = tmp_path / "results"
    for split in ("task_seed0", "task_seed1"):
        write_run(root, "liberox_groot", split, "stac__mmd3_step", 20.0, 0.4)
        write_run(root, "liberox_groot", split, "actprobe", 30.0, 0.6)
        write_run(root, "liberox_groot", split, f"{figures.OURS_ARCHITECTURE}__onset_traj", 40.0, 0.8)  # a method that is not in the compact figure
    rows = collect_rows(root)
    chosen, _ = figures.choose_all(rows)
    assert ("LIBERO-X task", "Ours (onset_traj)") in chosen
    calls, original = [], plots.curves_figure

    def spy(panels, settings, entries, ylabel, panel_width=plots.PANEL_WIDTH, legend_rows=2, height_scale=1.0, font_scale=1.0):
        fig = original(panels, settings, entries, ylabel, panel_width, legend_rows, height_scale, font_scale)
        calls.append((panels, settings, entries, panel_width, tuple(fig.get_size_inches()), fig.legends[0]._ncols, fig.legends[0].get_texts()[0].get_fontsize()))
        return fig

    monkeypatch.setattr(plots, "curves_figure", spy)
    args = type("Args", (), {"out": tmp_path / "out", "root": root})
    cli_figures.write_compact_curves(args, rows, chosen)
    assert len(calls) == 2  # TPR early and on-time TPR
    panels, settings, entries, panel_width, (width, height), legend_columns, legend_font = calls[0]
    assert [s.label for s in settings] == ["LIBERO-10 π0.5", "LIBERO-10 GR00T", "LIBERO-X task", "LIBERO-X signature", "LIBERO-X skill"]  # not the real robot
    assert [r[0] for r in panels] == ["Seen tasks", "Unseen tasks"] and settings == list(figures.SIMULATED_SETTINGS)
    assert [e.label for e in entries] == ["Ours (onset_pre)", "SAFE-LSTM", "SAFE-MLP", "STAC", "STAC-Single", "ActProbe", "FailureSpot"]
    assert panel_width == pytest.approx(1.2 * plots.PANEL_WIDTH)  # wider panels, five instead of six
    assert width == pytest.approx(plots.PANEL_WIDTH * len(figures.SETTINGS)) and width == pytest.approx(15.0)  # the width of the six-panel figure
    assert figures.COMPACT_SCALE == 1.0 and height == pytest.approx(figures.COMPACT_SCALE * 2 * plots.PANEL_HEIGHT) and height == pytest.approx(5.8)  # two rows of panels of 2.9 in
    assert legend_font == pytest.approx(9 * figures.COMPACT_FONT_SCALE) and figures.COMPACT_FONT_SCALE == 1.25  # the figure is wide enough for the usual legend spacing, at a larger font
    assert legend_columns == 2  # a single legend row: one column per method drawn (STAC and ActProbe here; two rows would make it one column)
    assert {entry for _, entry in panels[0][1]} == {"STAC", "ActProbe"}  # our onset_traj probe has a curve in the data but is not drawn
    for stem in ("curves_tpr_early_compact", "curves_tpr_on_time_compact"):
        assert (tmp_path / "out" / f"{stem}.png").exists() and (tmp_path / "out" / f"{stem}.pdf").exists()


def test_the_compact_curve_figures_are_skipped_with_a_message_when_there_is_no_simulated_curve(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    rows = collect_rows(tmp_path / "empty")
    chosen, _ = figures.choose_all(rows)
    cli_figures.write_compact_curves(type("Args", (), {"out": tmp_path / "out", "root": tmp_path / "empty"}), rows, chosen)
    assert "no curves of the simulated settings" in capsys.readouterr().out and not (tmp_path / "out").exists()


def test_the_compact_curve_figures_also_cover_libero10_and_not_the_real_robot(results, tmp_path):
    """The fixture has LIBERO-10 runs only (STAC, ActProbe, SAFE-LSTM): the compact figures are written with curves for those panels."""
    rows = collect_rows(results)
    chosen, _ = figures.choose_all(rows)
    assert ("LIBERO-10 GR00T", "STAC") in chosen
    cli_figures.write_compact_curves(type("Args", (), {"out": tmp_path / "out", "root": results}), rows, chosen)
    assert (tmp_path / "out" / "curves_tpr_early_compact.png").exists()


# ---------------------------------------------------------------- the online latency columns
def latency_file(vla=True):
    gpu, cpu = (lambda ms: {"median_ms": ms, "p99_ms": 2 * ms}), (lambda ms: {"median_ms": ms, "p99_ms": 2 * ms})
    out = {"methods": {"m_task": {"gpu": gpu(0.1), "cpu4": cpu(1.0), "cpu1": cpu(2.0)}, "m_sig": {"gpu": gpu(0.3), "cpu4": cpu(3.0), "cpu1": cpu(5.0)},
                       "stac__mmd3_step": {"gpu": None, "cpu4": cpu(0.03), "cpu1": cpu(0.03)}, "stac_single__cumsum": {"gpu": None, "cpu4": cpu(0.004), "cpu1": cpu(0.004)}}}
    if vla:
        out["vla_sampling"] = {"extra_ms": 21.0}
    return out


def test_the_latency_of_a_method_is_its_median_on_the_gpu_and_on_the_cpu_with_four_threads():
    latency = latency_file()
    assert figures.method_latency(latency, "m_task", "gpu") == 0.1 and figures.method_latency(latency, "m_task", "cpu") == 1.0
    assert figures.method_latency(latency, "unknown", "gpu") is None and figures.method_latency(None, "m_task", "gpu") is None


def test_stac_pays_for_sampling_its_plans_on_the_gpu_and_has_no_cpu_latency_while_stac_single_samples_nothing():
    latency = latency_file()
    assert figures.method_latency(latency, "stac__mmd3_step", "gpu") == pytest.approx(21.03) and figures.method_latency(latency, "stac__mmd3_step", "cpu") is None
    assert figures.method_latency(latency_file(vla=False), "stac__mmd3_step", "gpu") is None  # without the measured sampling time, STAC has no number rather than a too low one
    assert figures.method_latency(latency, "stac_single__cumsum", "gpu") == figures.method_latency(latency, "stac_single__cumsum", "cpu") == 0.004


def test_an_entry_has_the_median_latency_of_the_methods_chosen_for_the_three_kinds_of_split():
    chosen = {("LIBERO-X task", "E"): "m_task", ("LIBERO-X signature", "E"): "m_sig", ("LIBERO-X skill", "E"): "m_task"}
    entry = figures.Entry("E", "safe", "solid")
    assert figures.entry_latency(entry, chosen, latency_file()) == {"latency_gpu_ms": 0.1, "latency_cpu_ms": 1.0}  # median of 0.1, 0.3, 0.1 and of 1.0, 3.0, 1.0
    assert figures.entry_latency(entry, chosen, None) == {"latency_gpu_ms": None, "latency_cpu_ms": None}
    assert figures.entry_latency(figures.Entry("Z", "safe", "solid"), chosen, latency_file()) == {"latency_gpu_ms": None, "latency_cpu_ms": None}


def test_the_main_table_has_the_gpu_and_cpu_latency_columns_in_place_of_the_batched_score_time():
    assert [name for _, name, _ in tables.COST_COLUMNS] == ["Fit (s)", "State (MB)", "GPU (ms)", "CPU (ms)"]
    assert "score_ms_per_step" not in [field for field, _, _ in tables.COST_COLUMNS] and "online latency" in tables.CAPTION and "ten plans" in tables.CAPTION


def test_the_panels_of_the_libero_x_splits_are_titled_with_the_names_of_the_splits_and_the_other_settings_keep_their_labels():
    settings = [s for s in figures.SETTINGS if s.dataset in ("libero10_groot", "liberox_groot")]
    entries = list(figures.ENTRIES[:2])
    fig = plots.curves_figure(one_row(settings, entries), settings, entries, "y")
    titles = [(ax.get_title(), [t.get_text() for t in ax.texts]) for ax in fig.axes if ax.get_title()]
    assert titles == [("LIBERO-10 GR00T", ["leave-three-out"]), ("LIBERO-X GR00T", ["new-task"]), ("LIBERO-X GR00T", ["new-signature"]), ("LIBERO-X GR00T", ["new-skill"])]
    assert [s.label for s in settings][1:] == ["LIBERO-X task", "LIBERO-X signature", "LIBERO-X skill"]  # the labels of the tables and of the cache keys are unchanged
    matplotlib.pyplot.close("all")


def test_every_title_says_the_benchmark_the_policy_and_the_split_in_the_same_way():
    assert set(plots.PANEL_TITLES) == {s.label for s in figures.SETTINGS}
    for setting in figures.SETTINGS:
        first, split = plots.PANEL_TITLES[setting.label].split("\n")  # two lines: benchmark and policy, then the split
        assert first.startswith(("LIBERO-10", "LIBERO-X", "Real robot")) and ("π0.5" in first or "GR00T" in first) and split in ("leave-three-out", "leave-one-out", "new-task", "new-signature", "new-skill")


def test_the_split_names_of_libero_x_are_coloured_as_in_the_paper_and_the_other_split_names_stay_black():
    settings = [s for s in figures.SETTINGS if s.dataset in ("libero10_groot", "liberox_groot")]
    entries = list(figures.ENTRIES[:2])
    fig = plots.curves_figure(one_row(settings, entries), settings, entries, "y")
    colours = {ax.texts[0].get_text(): matplotlib.colors.to_hex(ax.texts[0].get_color()) for ax in fig.axes if ax.texts}
    assert colours == {"leave-three-out": "#000000", "new-task": "#de8e46", "new-signature": "#4a8c64", "new-skill": "#3e5496"}  # taskOrange, sigGreen, skillNavy
    assert all(ax.title.get_color() == plots.INK for ax in fig.axes if ax.get_title())  # the first line of every title is black
    assert tuple(round(255 * c) for c in matplotlib.colors.to_rgb(plots.SPLIT_COLORS["LIBERO-X task"])) == (222, 142, 70)
    assert tuple(round(255 * c) for c in matplotlib.colors.to_rgb(plots.SPLIT_COLORS["LIBERO-X signature"])) == (74, 140, 100)
    assert tuple(round(255 * c) for c in matplotlib.colors.to_rgb(plots.SPLIT_COLORS["LIBERO-X skill"])) == (62, 84, 150)
    matplotlib.pyplot.close("all")


def test_the_font_scale_enlarges_every_text_of_a_curve_figure_and_leaves_its_size_alone():
    entries = list(figures.ENTRIES[:5])
    args = (one_row(figures.SETTINGS[:2], entries), list(figures.SETTINGS[:2]), entries, "y")
    normal, larger = plots.curves_figure(*args), plots.curves_figure(*args, font_scale=1.25)
    assert tuple(larger.get_size_inches()) == tuple(normal.get_size_inches())
    pairs = [(normal.axes[0].title.get_fontsize(), larger.axes[0].title.get_fontsize()), (normal.axes[0].xaxis.label.get_fontsize(), larger.axes[0].xaxis.label.get_fontsize()),
             (normal.axes[0].get_xticklabels()[0].get_fontsize(), larger.axes[0].get_xticklabels()[0].get_fontsize()),
             (normal.legends[0].get_texts()[0].get_fontsize(), larger.legends[0].get_texts()[0].get_fontsize()), (normal.axes[0].texts[0].get_fontsize(), larger.axes[0].texts[0].get_fontsize())]
    assert all(b == pytest.approx(1.25 * a) for a, b in pairs)  # titles, axis labels, tick labels, the legend and the split names
    matplotlib.pyplot.close("all")
    plots.style_axes()  # the default style again for the other tests
