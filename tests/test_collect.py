import csv

import numpy as np
import pytest

from fp.baselines.distance_grid import method_names
from fp.eval.collect import (COLUMNS, arch_name, collect_rows, family_of, mean_area, per_split_areas, probe_rows, select_baseline_from_saved,
                             select_from_saved, write_csv)
from fp.eval.report import methods_compared, report
from fp.eval.select_config import group_of
from fp.eval.run_eval import EVAL_SETS, scan_ltt
from fp.models.ours import all_configs
from fp.results import run_dir, write_json, write_run

DATASET, SPLITS, VARIANT = "liberox_groot", ("task_seed0", "task_seed1", "signature_seed0"), "onset_pre"
ARCHS = [c.name for c in all_configs()]


def area_of(arch_index, split, calibrator, eval_set):
    """Deterministic areas: higher index = better on val; the unseen areas are the other way round (a bait for a selection that peeks)."""
    base = 20.0 + arch_index + SPLITS.index(split)
    return {"val": base, "seen": base - 1.0, "unseen": 60.0 - arch_index}[eval_set] - (3.0 if calibrator == "safe_band" else 0.0)


def make_tree(root, calibrators=("max_conformal", "safe_band"), skip=()):
    for i, arch in enumerate(ARCHS):
        for split in SPLITS:
            if (arch, split) in skip:
                continue
            path = run_dir(DATASET, split, f"{arch}__{VARIANT}", root)
            write_run(path, {}, {"best_epoch": 1})
            write_json(path / "summary.json", {"calibrators": {
                c: {"sets": {s: {"area": area_of(i, split, c, s), "extended": False, "n_fail": 10, "n_succ": 12} for s in EVAL_SETS}} for c in calibrators}})


def test_one_row_per_run_calibrator_and_set_with_the_factors_parsed(tmp_path):
    make_tree(tmp_path)
    rows = collect_rows(tmp_path)
    assert len(rows) == 16 * 3 * 2 * 3
    row = next(r for r in rows if r["method"] == "mlp_mha_modes_clip__onset_pre" and r["split"] == "signature_seed0" and r["eval_set"] == "unseen"
               and r["calibrator"] == "safe_band")
    assert (row["trunk"], row["mha"], row["heads"], row["variant"], row["group"], row["dataset"]) == ("mlp", True, "modes_clip", VARIANT, "signature", DATASET)
    assert arch_name(row) == "mlp_mha_modes_clip" and row["area"] == area_of(ARCHS.index("mlp_mha_modes_clip"), "signature_seed0", "safe_band", "unseen")
    assert {r["group"] for r in rows} == {"task", "signature"}


def test_runs_without_an_evaluation_and_calibrators_that_failed_have_no_rows(tmp_path):
    make_tree(tmp_path)
    write_run(run_dir(DATASET, "task_seed0", "extra_run", tmp_path), {}, {})  # a finished run that was never evaluated
    path = run_dir(DATASET, "task_seed0", f"{ARCHS[0]}__{VARIANT}", tmp_path)
    write_json(path / "summary.json", {"calibrators": {"max_conformal": {"sets": {s: {"area": 1.0, "extended": False, "n_fail": 1, "n_succ": 1} for s in EVAL_SETS}},
                                                       "safe_band": {"error": "too few calibration successes"}}})
    rows = collect_rows(tmp_path)
    assert not [r for r in rows if r["method"] == "extra_run"]
    assert not [r for r in rows if r["method"] == f"{ARCHS[0]}__{VARIANT}" and r["split"] == "task_seed0" and r["calibrator"] == "safe_band"]
    assert len(rows) == 16 * 3 * 2 * 3 - 3


def test_the_best_architecture_is_chosen_on_validation_and_never_on_unseen(tmp_path):
    make_tree(tmp_path)
    rows = collect_rows(tmp_path)
    sel = select_from_saved(rows, DATASET, "task", VARIANT)
    assert sel.best == ARCHS[-1]  # the best on val ...
    assert max(range(16), key=lambda i: area_of(i, "task_seed0", "max_conformal", "unseen")) == 0  # ... although arch 0 is best on unseen
    assert list(sel.mean_val_area) == ARCHS and len(sel.per_run[ARCHS[0]]) == 2  # grid order; one area per split of the group
    changed = [{**r, "area": 99.0} if r["eval_set"] != "val" else r for r in rows]  # whatever the seen and unseen areas are, the choice stays
    assert select_from_saved(changed, DATASET, "task", VARIANT).best == sel.best


def test_selection_uses_the_requested_calibrators_validation_areas(tmp_path):
    make_tree(tmp_path)
    rows = [{**r, "area": (50.0 - ARCHS.index(arch_name(r))) if r["calibrator"] == "safe_band" and r["eval_set"] == "val" else r["area"]} for r in collect_rows(tmp_path)]
    assert select_from_saved(rows, DATASET, "task", VARIANT, "safe_band").best == ARCHS[0]
    assert select_from_saved(rows, DATASET, "task", VARIANT, "max_conformal").best == ARCHS[-1]


def test_an_architecture_without_a_validation_area_on_every_split_is_an_error(tmp_path):
    make_tree(tmp_path, skip={(ARCHS[3], "task_seed1")})
    with pytest.raises(ValueError, match="no validation area"):
        select_from_saved(collect_rows(tmp_path), DATASET, "task", VARIANT)
    with pytest.raises(ValueError, match="no evaluated runs"):
        select_from_saved(collect_rows(tmp_path), DATASET, "skill", VARIANT)


def test_mean_area_filters_and_skips_nan():
    rows = [{"trunk": "lstm", "area": 10.0}, {"trunk": "lstm", "area": float("nan")}, {"trunk": "lstm", "area": 30.0}, {"trunk": "mlp", "area": 99.0}]
    assert mean_area(rows, trunk="lstm") == (20.0, 2) and np.isnan(mean_area(rows, trunk="none")[0])


def test_per_split_areas_returns_one_value_per_split(tmp_path):
    make_tree(tmp_path)
    got = per_split_areas(collect_rows(tmp_path), DATASET, "task", f"{ARCHS[2]}__{VARIANT}", "max_conformal", "val")
    assert got == {s: area_of(2, s, "max_conformal", "val") for s in ("task_seed0", "task_seed1")}


def test_the_long_table_is_written_with_one_line_per_row(tmp_path):
    make_tree(tmp_path)
    rows = collect_rows(tmp_path)
    write_csv(rows, tmp_path / "areas.csv")
    with open(tmp_path / "areas.csv") as f:
        back = list(csv.DictReader(f))
    assert list(back[0]) == list(COLUMNS) and len(back) == len(rows) and float(back[0]["area"]) == rows[0]["area"]


def test_the_report_covers_both_calibrators_and_compares_them_on_the_same_runs(tmp_path):
    make_tree(tmp_path)
    text = report(collect_rows(tmp_path))
    assert "OUR PROBES, CALIBRATOR max_conformal" in text and "OUR PROBES, CALIBRATOR safe_band" in text and "OUR PROBES UNDER BOTH CALIBRATORS" in text
    assert "BASELINES AND OUR PROBES" not in text  # no baseline runs in this tree
    assert "onset_pm_cut" not in text and "nan" not in text  # only the label variants the sweep trained, and no missing cells
    block = text.split("OUR PROBES UNDER BOTH CALIBRATORS")[1].split("liberox_groot / signature")[1].splitlines()
    line = next(l for l in block if l.split()[:2] == ["onset_pre", "unseen"])
    assert float(line.split()[-1]) == pytest.approx(3.0)  # safe_band is 3 points lower than max-conformal everywhere in this tree
    assert text.count("best architecture per label variant") == 4  # 2 groups x 2 calibrators


# ---------------------------------------------------------------- baselines next to the probes
def baseline_rows(areas, splits=("task_seed0", "task_seed1"), unseen=50.0):
    """Rows of one baseline family: {method: [val area per split]}; the protocol's calibrator for the distance baselines is the SAFE band."""
    return [{"dataset": DATASET, "split": s, "group": group_of(s), "method": m, "family": family_of(m), "calibrator": "safe_band", "eval_set": e,
             "area": (v if e == "val" else unseen), "extended": False, "n_fail": 10, "n_succ": 12}
            for m, vals in areas.items() for s, v in zip(splits, vals) for e in EVAL_SETS]


def test_family_names_separate_the_baselines_from_our_probes():
    assert family_of("lstm_mha_modes_clip__onset_pre") == "ours" and family_of("euclidean__k10_cumsum") == "euclidean" and family_of("safe_lstm") == "safe_lstm"


def test_a_baselines_best_configuration_is_chosen_on_validation_in_grid_order(tmp_path):
    first, second, third = "euclidean__k1_step", "euclidean__k1_cumsum", "euclidean__k5_step"
    order = [first, second, third]
    rows = baseline_rows({first: [40.0, 40.0], second: [40.0, 40.0], third: [10.0, 60.0]})  # first and second tie; third averages 35
    sel = select_baseline_from_saved(rows, DATASET, "task", "euclidean", order)
    assert sel.best == first and list(sel.mean_val_area) == order  # a tie goes to the earlier configuration of the grid
    assert select_baseline_from_saved(rows, DATASET, "task", "euclidean", [second, first, third]).best == second  # so the order is part of the rule
    bait = [{**r, "area": 99.0 if (r["method"] == third and r["eval_set"] != "val") else r["area"]} for r in rows]
    assert select_baseline_from_saved(bait, DATASET, "task", "euclidean", order).best == first  # unseen and seen areas never decide


def test_a_baseline_with_a_missing_validation_area_or_no_runs_is_an_error():
    first, second = "euclidean__k1_step", "euclidean__k1_cumsum"
    rows = baseline_rows({first: [40.0, 40.0], second: [40.0, 40.0]})
    with pytest.raises(ValueError, match="no validation area"):
        select_baseline_from_saved([r for r in rows if not (r["method"] == second and r["split"] == "task_seed1")], DATASET, "task", "euclidean", [first, second])
    with pytest.raises(ValueError, match="no evaluated runs"):
        select_baseline_from_saved(rows, DATASET, "skill", "euclidean", [first])


def test_a_tie_between_safe_configurations_goes_to_the_earlier_one_in_grid_order_not_alphabetical_order():
    from fp.baselines.safe_grid import method_names as safe_names
    from fp.eval.report import method_order

    names = safe_names("safe_lstm")
    rows = baseline_rows({m: [30.0, 30.0] for m in names})  # every configuration ties on validation
    rows = [{**r, "calibrator": "safe_band"} for r in rows]
    order = method_order(rows, "safe_lstm")
    assert order == names and order != sorted(order)
    best = select_baseline_from_saved(rows, DATASET, "task", "safe_lstm", order).best
    assert best == "safe_lstm__lr3e-5_reg1e-3" and best != sorted(names)[0]  # the first configuration of the grid, not the alphabetical first


def test_baseline_rows_never_enter_the_probe_tables_or_the_probe_selection(tmp_path):
    make_tree(tmp_path)
    ours = collect_rows(tmp_path)
    mixed = ours + baseline_rows({m: [30.0, 30.0] for m in method_names("euclidean")})
    assert probe_rows(mixed) == ours
    assert select_from_saved(mixed, DATASET, "task", VARIANT).best == select_from_saved(ours, DATASET, "task", VARIANT).best  # no KeyError either
    sections_ours = report(ours).split("\n", 1)[1].rsplit("\nNotes\n", 1)[0].rstrip()  # after the one-line header, up to the notes
    sections_mixed = report(mixed).split("\n", 1)[1].split("\n" + "=" * 100 + "\nBASELINES AND OUR PROBES")[0].rstrip()
    assert sections_mixed == sections_ours  # the probe tables are identical with and without baseline rows
    assert "euclidean" not in sections_mixed


def test_every_baseline_is_listed_next_to_our_probes_with_the_calibrator_the_protocol_gives_it(tmp_path):
    make_tree(tmp_path)
    areas = {m: [20.0 + i, 20.0 + i] for i, m in enumerate(method_names("euclidean"))}  # the last configuration is best on validation
    rows = collect_rows(tmp_path) + baseline_rows(areas)
    text = "\n".join(methods_compared(rows))
    task = text.split("liberox_groot / task")[1].split("liberox_groot / signature")[0] if "liberox_groot / task" in text else text
    chosen = method_names("euclidean")[-1]
    line = next(l for l in task.splitlines() if l.strip().startswith(chosen))
    assert "safe_band" in line and float(line.split()[2]) == 25.0  # val area of the chosen configuration, calibrated with the band
    ours_line = next(l for l in task.splitlines() if "__onset_pre" in l)
    assert "max_conformal" in ours_line  # our probe in the same table, calibrated with max-conformal
    assert "BASELINES AND OUR PROBES" in report(rows) and "BASELINES AND OUR PROBES" not in report(collect_rows(tmp_path))


def test_the_report_counts_the_curves_that_are_held_flat_per_group_on_the_unseen_set_only():
    from fp.eval.report import extended_table

    def row(group, extended, eval_set="unseen", calibrator="max_conformal"):
        return {"dataset": "d", "group": group, "calibrator": calibrator, "eval_set": eval_set, "extended": extended}

    rows = [row("task", True), row("task", False), row("task", True), row("skill", False), row("skill", False),
            row("task", True, eval_set="val"), row("task", True, calibrator="safe_band")]  # the last two are other sets and must not be counted
    text = "\n".join(extended_table(rows, "max_conformal"))
    assert "d / task: 2 of 3" in text and "d / skill: 0 of 2" in text


# ---------------------------------------------------------------- split sizes
def test_split_sizes_count_failures_and_successes_in_every_part():
    from tests.test_run_eval import world
    from fp.eval.report import PARTS, split_size_table, split_sizes

    episodes, split = world()
    (row,) = split_sizes("synthetic", {split.name: split}, episodes)
    assert (row["dataset"], row["split"], row["group"]) == ("synthetic", "task_seed0", "task")
    # world(): train 6 successes + 4 failures, calibration 20 + 10, val 10 + 8, seen 10 + 8, unseen 12 + 8
    assert [(row[p], row[p + "_fail"], row[p + "_succ"]) for p in PARTS] == [(10, 4, 6), (18, 8, 10), (30, 10, 20), (18, 8, 10), (20, 8, 12)]
    assert all(row[p] == row[p + "_fail"] + row[p + "_succ"] for p in PARTS) and sum(row[p] for p in PARTS) == len(episodes)  # a partition
    text = "\n".join(split_size_table([row]))
    assert "30 ( 10F/ 20S)" in text and "20 (  8F/ 12S)" in text and "task_seed0" in text


def test_the_split_size_table_widens_for_long_split_names_and_follows_run_order():
    from tests.test_run_eval import world
    from fp.data.splits import Split
    from fp.eval.report import split_size_table, split_sizes

    episodes, split = world()
    named = {name: Split(name, split.train, split.val, split.calibration, split.seen, split.unseen, 0)
             for name in ("signature_seed1", "skill_place_relative_to_landmark", "task_seed1", "task_seed0")}
    sizes = split_sizes("synthetic", named, episodes)
    assert [r["split"] for r in sizes] == ["task_seed0", "skill_place_relative_to_landmark", "task_seed1", "signature_seed1"]  # seed 0 first; task, signature, skill
    rows = [l for l in split_size_table(sizes) if "F/" in l and not l.lstrip().startswith("split")]
    assert len({len(l) for l in rows}) == 1  # every row has the same width: the long name did not push its columns out of line


# ---------------------------------------------------------------- the LTT target scan
def test_scan_ltt_gives_one_result_per_target_pair_equal_to_a_single_evaluation():
    from tests.test_run_eval import WINDOW, separable_scores, world
    from fp.eval.run_eval import evaluate_ltt

    episodes, split = world(n_cal_succ=60)
    scores = separable_scores(episodes, split)
    pairs = [(0.5, 0.2), (0.001, 0.001)]
    scan = scan_ltt(scores, split, episodes, WINDOW, pairs, delta=0.1)
    assert [s["certified"] for s in scan] == [True, False]
    assert scan[0] == evaluate_ltt(scores, split, episodes, WINDOW, alpha_tpr_early=0.5, alpha_fpr=0.2, delta=0.1)
