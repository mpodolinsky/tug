"""python -m cli.figures [--root results] [--out results/figures] [--costs results/cost_benchmark] [--print-benchmark]

The paper's main table and curve figures, from the evaluated runs under results/ (python -m cli.evaluate first; no model is run):

    table_main.tex / .csv    one row per method: area under TPR early vs FPR (seen and unseen) for each setting, the calibrator, and the cost
                             on LIBERO-X; table_main_std.tex adds the spread over splits; chosen.csv names the configuration behind every cell
    table_modes.tex / .csv   failure-mode prediction on LIBERO-X and the real robot: top-1 and top-3 accuracy of the mode head at the probe's own
                             alarm and at the true onset (an upper bound), against the most-common-mode baselines (fp/eval/modes.py)
    table_ablations.tex/.csv the ablations of our probe on the three LIBERO-X split kinds: its heads and its training strategy, the area with the
                             spread over the splits and the change against our probe (fp/eval/ablations.py)
    table_modes_balanced.tex the same with the balanced accuracy over merged classes (rare modes merged until every class holds 5% of the failures; pooled
                             over the splits of each kind; mode_classes.csv lists the classes): fp/eval/mode_balanced.py
    table_mode_ablation.tex  (and _std.tex with the spread over splits) what the CLIP head and the score term add to the failure-mode prediction: four LSTM probes (with and without the CLIP head,
                             with and without the score term) read at the same anchors, on LIBERO-X and the real robot's seed 0, the mean and
                             the spread over the splits (fp/eval/mode_ablation.py; needs the probes of results/mode_only, see docs/FIGURES.md)
    curves_tpr_early         TPR early against FPR for every setting and method (.png and .pdf): a row for the seen tasks on top, one for
                             the unseen tasks below
    curves_tpr_on_time       the same with the on-time TPR
    curves_tpr_early_compact smaller versions of the two curve figures (also curves_tpr_on_time_compact): the five simulated settings (LIBERO-10 with pi0.5 and
                             with GR00T, the three LIBERO-X kinds; every panel is titled with the benchmark, the policy and the split: LIBERO-10 pi0.5 / leave-three-out, LIBERO-X GR00T / new-task, ...) at the width of the full figures, 15 in by 5.8 in, and only ours (onset_pre), the two
                             SAFE models, the two STAC variants, ActProbe and FailureSpot; the full figures are written too

Each method is, in each setting, the configuration with the best mean validation area (one architecture for all of our rows and one LIBERO-X
configuration for each SAFE model; docs/FIGURES.md), with the calibrator the protocol gives it. Our probes carry no per-run costs (they were trained before the sweeps recorded them); --print-benchmark prints the commands that
re-train the probes the table uses into --costs, which is then read for their cost columns.
"""

import argparse
import csv
import json
from pathlib import Path

from fp.calibration.protocol import calibrator_name
from fp.config import RESULTS
from fp.data.inputs import load_inputs
from fp.eval import ablations, figures, mode_ablation, mode_balanced, modes, plots, tables
from fp.eval.collect import collect_rows
from fp.eval.compute import collect_costs


def table_rows(rows: list[dict], chosen: dict, costs: dict) -> list[tables.Row]:
    """The rows of the main table: the entries in ``figures.TABLE_LABELS`` that have at least one chosen method."""
    out = []
    for label in figures.TABLE_LABELS:
        entry = next(e for e in figures.ENTRIES if e.label == label)
        methods = [chosen[(s.label, label)] for s in figures.SETTINGS if (s.label, label) in chosen]
        if not methods:
            print(f"no row for {label}: no setting has complete runs")
            continue
        areas = []
        for setting in figures.SETTINGS:
            method = chosen.get((setting.label, label))
            for eval_set in ("seen", "unseen"):
                cell = figures.area_cell(rows, setting, method, eval_set) if method else None
                areas.append(None if cell is None else (cell.mean, cell.std))
        entry_cost = figures.entry_costs(entry, chosen, costs["ours" if entry.kind == "ours" else "baselines"]) | figures.entry_latency(entry, chosen, costs["latency"])
        out.append(tables.Row(label, entry.kind == "ours", calibrator_name(methods[0]), areas, [entry_cost[name] for name, _, _ in tables.COST_COLUMNS]))
    return out


def write_csv(path: Path, records: list[dict]) -> None:
    """One CSV row per record; the columns are those of all records in order of first appearance (a record without a column leaves it empty)."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(key for record in records for key in record)))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=RESULTS)
    p.add_argument("--out", type=Path, default=RESULTS / "figures")
    p.add_argument("--costs", type=Path, default=RESULTS / "cost_benchmark", help="runs of our probes re-trained to measure their costs")
    p.add_argument("--latency", type=Path, default=RESULTS / "latency" / "latency.json", help="online latencies of the methods (python -m cli.latency)")
    p.add_argument("--print-benchmark", action="store_true", help="print the commands that re-train the probes the table uses, and stop")
    args = p.parse_args()

    rows = collect_rows(args.root)
    chosen, skipped = figures.choose_all(rows)
    for line in skipped:
        print("skipped:", line)

    if args.print_benchmark:
        table_entries = [e for e in figures.ENTRIES if e.label in figures.TABLE_LABELS]
        for arch, variant, splits in figures.benchmark_plan(chosen, rows, table_entries):
            print(f"python -m cli.train --out {args.costs} --datasets liberox_groot --variants {variant} --probes {arch} --splits {' '.join(splits)}")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    if rows:
        write_main_table(args, rows, chosen)
        write_curves(args, rows, chosen)
        write_compact_curves(args, rows, chosen)
        write_ablation_table(args, rows)
    else:
        print(f"no evaluated runs under {args.root} (python -m cli.evaluate first): the main table and the curves are skipped")
    write_mode_table(args)
    write_mode_table(args, balanced=True)
    write_mode_ablation_table(args)


def write_main_table(args, rows: list[dict], chosen: dict) -> None:
    costs = {"baselines": collect_costs(args.root, ("liberox_groot",)),
             "ours": collect_costs(args.costs, ("liberox_groot",)) if args.costs.exists() else [],
             "latency": json.loads(args.latency.read_text()) if args.latency.exists() else None}
    if costs["latency"] is None:
        print(f"no latency file {args.latency} (python -m cli.latency): the latency columns are left empty")
    settings = [s.label for s in figures.SETTINGS]
    table = table_rows(rows, chosen, costs)
    (args.out / "table_main.tex").write_text(tables.latex_table(table, settings))
    (args.out / "table_main_std.tex").write_text(tables.latex_table(table, settings, with_std=True))
    write_csv(args.out / "table_main.csv", tables.csv_rows(table, settings))
    write_csv(args.out / "chosen.csv", [{"setting": s, "entry": e, "method": m, "calibrator": calibrator_name(m)} for (s, e), m in chosen.items()])
    print("wrote", args.out / "table_main.tex", "(+ _std, .csv) and chosen.csv")


def write_ablation_table(args, rows: list[dict]) -> None:
    """The ablation table of LIBERO-X (fp/eval/ablations.py); skipped with a message when one of its probes has no evaluated runs."""
    try:
        blocks = ablations.ablation_blocks(rows)
    except ValueError as error:
        print(f"no ablation table: {error}")
        return
    labels = [s.label for s in ablations.ABLATION_SETTINGS]
    (args.out / "table_ablations.tex").write_text(tables.ablation_latex_table(blocks, labels))
    write_csv(args.out / "table_ablations.csv", tables.ablation_csv_rows(blocks, labels))
    print("wrote", args.out / "table_ablations.tex", "and table_ablations.csv")


def write_mode_table(args, balanced: bool = False) -> None:
    """The failure-mode table: the LIBERO-X columns, then the real robot's. The plain accuracy of fp/eval/modes.py, or with ``balanced`` the balanced accuracy over
    merged classes of fp/eval/mode_balanced.py (table_modes_balanced, and mode_classes.csv with the classes). The table is skipped with a message when the
    LIBERO-X probe runs or data are not there; the real robot's data are not published, so its columns are left out, with a message, when they are not there."""
    compute = mode_balanced.balanced_results if balanced else modes.mode_results
    stem = "table_modes_balanced" if balanced else "table_modes"
    results, settings, classes = {}, [], []
    for dataset, group in (("liberox_groot", modes.LIBEROX), ("real8_groot", modes.REAL)):
        if not group:
            continue
        try:
            inputs = load_inputs(dataset)
            results |= compute(inputs, args.root, settings=group)
        except (FileNotFoundError, KeyError) as error:
            print(f"no failure-mode table: {error}" if not settings else f"no failure-mode column for {dataset}: {error}")
            if not settings:
                return
            continue
        settings += group
        classes += mode_balanced.group_rows(inputs) if balanced else []
    rows = modes.mode_rows(results, settings=settings)
    labels = [s.label for s in settings]
    (args.out / f"{stem}.tex").write_text(tables.mode_latex_table(rows, labels, caption=tables.mode_caption(rows, labels, balanced), label="tab:modes_balanced" if balanced else "tab:modes"))
    write_csv(args.out / f"{stem}.csv", tables.mode_csv_rows(rows, labels))
    print("wrote", args.out / f"{stem}.tex", f"and {stem}.csv")
    if balanced:
        write_csv(args.out / "mode_classes.csv", classes)
        print("wrote", args.out / "mode_classes.csv")


def write_mode_ablation_table(args) -> None:
    """The mode-ablation table (fp/eval/mode_ablation.py): LIBERO-X, then the real robot's seed 0. It needs the probes trained without the score term
    (python -m cli.train --out results/mode_only --lam-score 0 ..., docs/FIGURES.md); skipped with a message when they or the data are not there."""
    results = {}
    for dataset, settings in (("liberox_groot", mode_ablation.SETTINGS[:-1]), ("real8_groot", mode_ablation.SETTINGS[-1:])):
        try:
            results |= mode_ablation.mode_ablation_results(load_inputs(dataset), args.root, settings)
        except (FileNotFoundError, KeyError) as error:
            print(f"no mode-ablation table: {error}")
            return
    blocks = mode_ablation.mode_ablation_blocks(results)
    labels = [s.label for s in mode_ablation.SETTINGS]
    (args.out / "table_mode_ablation.tex").write_text(tables.mode_ablation_latex_table(blocks, labels))
    (args.out / "table_mode_ablation_std.tex").write_text(tables.mode_ablation_latex_table(blocks, labels, with_std=True))
    write_csv(args.out / "table_mode_ablation.csv", tables.mode_ablation_csv_rows(blocks, labels))
    print("wrote", args.out / "table_mode_ablation.tex", "(+ _std, .csv)")


def write_curves(args, rows: list[dict], chosen: dict, settings=figures.SETTINGS, entries=figures.ENTRIES, suffix: str = "",
                 panel_width: float = plots.PANEL_WIDTH, legend_rows: int = 2, height_scale: float = 1.0, font_scale: float = 1.0) -> None:
    """The two curve figures (TPR early and on-time TPR) for ``settings`` and ``entries``, written as curves_<metric><suffix>."""
    for metric, ylabel in (("tpr_early", "TPR early (%)"), ("tpr_on_time", "TPR on-time (%)")):
        panels = []
        for eval_set, row_label in (("seen", "Seen tasks"), ("unseen", "Unseen tasks")):  # seen on top
            curves = {(s.label, e.label): figures.mean_curve(rows, s, chosen[(s.label, e.label)], eval_set, metric, args.root)
                      for s in settings for e in entries if (s.label, e.label) in chosen}
            panels.append((row_label, curves))
        fig = plots.curves_figure(panels, list(settings), list(entries), ylabel, panel_width, legend_rows, height_scale, font_scale)
        print("wrote", *plots.save(fig, f"curves_{metric}{suffix}", args.out))


def write_compact_curves(args, rows: list[dict], chosen: dict) -> None:
    """The curve figures again, smaller: the simulated settings (five panels per row; the real robot is left out) and the methods of ``figures.COMPACT_LABELS``,
    with the legend in a single row. The figure has the width of the full figures (each panel is as many times wider as there are fewer panels) and
    ``figures.COMPACT_SCALE`` times their height; the font sizes are ``figures.COMPACT_FONT_SCALE`` times larger. Skipped with a message when there is no curve of these settings."""
    settings = figures.SIMULATED_SETTINGS
    entries = [e for e in figures.ENTRIES if e.label in figures.COMPACT_LABELS]
    if not any((s.label, e.label) in chosen for s in settings for e in entries):
        print("no curves of the simulated settings: the compact curve figures are skipped")
        return
    write_curves(args, rows, chosen, settings, entries, "_compact", plots.PANEL_WIDTH * len(figures.SETTINGS) / len(settings), legend_rows=1,
                 height_scale=figures.COMPACT_SCALE, font_scale=figures.COMPACT_FONT_SCALE)


if __name__ == "__main__":
    main()
