"""Text report of the evaluation of our 64 probes: marginal means per factor, and the configuration chosen on validation.

All areas are the area under TPR early vs FPR up to 40, divided by 40 (a perfect detector scores 100). Two calibrators: max-conformal, which
the protocol gives our probes, and the SAFE band as the secondary table; the comparison section sets them side by side on the same runs.
Three evaluation sets: ``val`` (used for early stopping and for choosing configurations; optimistic), ``seen`` and ``unseen``.
"""

from typing import Mapping

import numpy as np

from ..baselines.distance import DistanceScorer
from ..baselines.distance_grid import method_names
from ..baselines.failurespot import FAMILIES as FAILURESPOT_FAMILIES
from ..baselines.failurespot import method_names as failurespot_method_names
from ..baselines.safe_grid import FAMILIES as SAFE_FAMILIES
from ..baselines.safe_grid import method_names as safe_method_names
from ..calibration.protocol import MAX_CONFORMAL, SAFE_BAND, calibrator_name
from ..data.manifest import Episode
from ..data.splits import Split
from ..grid import VARIANTS, split_priority
from .collect import FACTORS, OURS, mean_area, per_split_areas, probe_rows, select_baseline_from_saved, select_from_saved
from .compute import compute_table
from .select_config import group_of

LEVELS = {"trunk": ("lstm", "mlp"), "mha": (False, True), "heads": ("plain", "modes", "clip", "modes_clip"),
          "variant": VARIANTS}  # the label variants the sweep trained (fp.grid), not every variant the label builder knows
SETS = ("val", "seen", "unseen")
NOTES = (
    "Notes",
    "- The area uses the alphas 0.005 ... 0.40 only: an alpha promises FPR <= alpha, so only these can stand for operating points at FPR <= 40%.",
    "  An operating point from a larger alpha is never used, however low its FPR is on the evaluation episodes.",
    "- Choose on val, report seen and unseen: the probes' early stopping saw the val episodes, so val is optimistic.",
    "- LIBERO-X task and signature splits (seeds 0-2) hold out the SAME unseen episodes; their spread is not spread over tasks.",
    "  Only the 8 skill folds and the 3 LIBERO-10 seeds per policy hold out different tasks.",
    "- Earlier noise levels of one configuration across seeds: about +-3 (task), +-7.6 (signature), +-5.8 (skill).",
)


PARTS = ("train", "val", "calibration", "seen", "unseen")


def split_sizes(dataset: str, splits: Mapping[str, Split], episodes: Mapping[str, Episode]) -> list[dict]:
    """Per split, the number of episodes of each part (train, val, calibration, seen, unseen) with its failures and successes, in run order.
    Columns: dataset, split, group, then <part>, <part>_fail, <part>_succ for each part."""
    rows = []
    for name in sorted(splits, key=split_priority):
        row = {"dataset": dataset, "split": name, "group": group_of(name)}
        for part in PARTS:
            keys = getattr(splits[name], part)
            fail = sum(not episodes[k].success for k in keys)
            row.update({part: len(keys), f"{part}_fail": fail, f"{part}_succ": len(keys) - fail})
        rows.append(row)
    return rows


def split_size_table(sizes: list[dict]) -> list[str]:
    """One block per dataset; each cell reads 'episodes (failures F / successes S)'. The calibration successes are what the calibrators
    are fitted on; seen and unseen are what the results are reported on."""
    lines = []
    width = max(len(r["split"]) for r in sizes) + 2
    for dataset in dict.fromkeys(r["dataset"] for r in sizes):
        lines += ["", f"{dataset}: episodes per split, 'n (failures F / successes S)'",
                  f"  {'split':{width}s}" + "".join(f"{p:>20s}" for p in PARTS)]
        for r in (r for r in sizes if r["dataset"] == dataset):
            lines.append(f"  {r['split']:{width}s}" + "".join(f"{r[p]:>8d} ({r[p + '_fail']:>3d}F/{r[p + '_succ']:>3d}S)" for p in PARTS))
    return lines


def cell(value: float) -> str:
    return f"{value:6.1f}" if not np.isnan(value) else "   n/a"


def groups(rows: list[dict]) -> list[tuple[str, str]]:
    return sorted({(r["dataset"], r["group"]) for r in rows})


def n_splits(rows: list[dict], dataset: str, group: str) -> int:
    return len({r["split"] for r in rows if r["dataset"] == dataset and r["group"] == group})


def factor_tables(rows: list[dict], calibrator: str = MAX_CONFORMAL) -> list[str]:
    """Mean area per level of each factor, over all other factors and all splits of the group."""
    lines = []
    for dataset, group in groups(rows):
        sub = [r for r in rows if r["dataset"] == dataset and r["group"] == group and r["calibrator"] == calibrator]
        if not sub:
            continue
        lines += ["", f"{dataset} / {group}  ({n_splits(rows, dataset, group)} splits; mean over the other factors and splits)",
                  f"  {'':22s}" + "".join(f"{s:>7s}" for s in SETS)]
        for factor in FACTORS:
            for level in LEVELS[factor]:
                lines.append(f"  {factor + ' = ' + str(level):22s}" + "".join(cell(mean_area(sub, eval_set=s, **{factor: level})[0]) + " " for s in SETS))
    return lines


def selection_table(rows: list[dict], calibrator: str = MAX_CONFORMAL) -> list[str]:
    """Per dataset and group, per label variant: the architecture with the best mean validation area and its val / seen / unseen areas
    (mean and standard deviation over the group's splits)."""
    lines = []
    for dataset, group in groups(rows):
        lines += ["", f"{dataset} / {group}: best architecture per label variant, chosen on the mean validation area",
                  f"  {'variant':11s}{'architecture':22s}{'val':>8s}{'seen':>8s}{'unseen':>8s}{'  unseen std':>14s}"]
        for variant in LEVELS["variant"]:
            try:
                best = select_from_saved(rows, dataset, group, variant, calibrator).best
            except ValueError as error:
                lines.append(f"  {variant:11s}no selection: {error}")
                continue
            method = f"{best}__{variant}"
            by_set = {s: list(per_split_areas(rows, dataset, group, method, calibrator, s).values()) for s in SETS}
            lines.append(f"  {variant:11s}{best:22s}" + "".join(f"{np.mean(by_set[s]):8.1f}" for s in SETS) + f"{np.std(by_set['unseen']):14.1f}")
    return lines


def extended_table(rows: list[dict], calibrator: str = MAX_CONFORMAL) -> list[str]:
    """How many curves never reach 40% FPR within the headline alphas (<= 0.40) and are held flat from their last point to 40. A held-flat
    area assumes TPR early does not fall between the last point and 40; it is the more cautious reading of a curve that stops short."""
    lines = ["", "curves that stay below 40% FPR for every alpha <= 0.40 (held flat to 40), unseen set:"]
    for dataset, group in groups(rows):
        sub = [r for r in rows if (r["dataset"], r["group"], r["calibrator"], r["eval_set"]) == (dataset, group, calibrator, "unseen")]
        if sub:
            lines.append(f"  {dataset} / {group}: {sum(bool(r['extended']) for r in sub)} of {len(sub)}")
    return lines


def calibrator_comparison(rows: list[dict], calibrators: tuple[str, ...] = (MAX_CONFORMAL, SAFE_BAND)) -> list[str]:
    """Mean area per label variant under each calibrator on the same runs (all 16 architectures, all splits of the group), and the
    difference between the first and the last calibrator listed."""
    lines = []
    for dataset, group in groups(rows):
        lines += ["", f"{dataset} / {group}: the same runs under each calibrator (mean over the 16 architectures and the group's splits)",
                  f"  {'variant':11s}{'set':>8s}" + "".join(f"{c:>15s}" for c in calibrators) + f"{'difference':>13s}"]
        for variant in LEVELS["variant"]:
            for eval_set in SETS:
                means = [mean_area(rows, dataset=dataset, group=group, variant=variant, calibrator=c, eval_set=eval_set)[0] for c in calibrators]
                if np.isnan(means).all():  # this label variant was not trained
                    continue
                lines.append(f"  {variant:11s}{eval_set:>8s}" + "".join(f"{m:15.1f}" for m in means) + f"{means[0] - means[-1]:13.1f}")
    return lines


def method_order(rows: list[dict], family: str) -> list[str]:
    """The methods of a baseline family in the order ties are broken in: SAFE's grid order for the distance families, else sorted names."""
    if family in DistanceScorer.NAMES.values():
        return method_names(family)
    if family in SAFE_FAMILIES:
        return safe_method_names(family)
    if family in FAILURESPOT_FAMILIES:
        return failurespot_method_names(FAILURESPOT_FAMILIES[family])
    return sorted({r["method"] for r in rows if r["family"] == family})


def methods_compared(rows: list[dict]) -> list[str]:
    """Per dataset and group: each baseline family's configuration and each label variant's architecture of our probes, chosen on the mean
    validation area, with val / seen / unseen areas (mean over the group's splits) and the spread of the unseen area over the splits. Every
    method is calibrated as the protocol says (``calibrator`` column), so this table mixes calibrators."""
    lines = []
    for dataset, group in groups(rows):
        lines += ["", f"{dataset} / {group}: best configuration chosen on validation, calibrated as the protocol says",
                  f"  {'method':32s}{'calibrator':>15s}{'val':>8s}{'seen':>8s}{'unseen':>8s}{'  unseen std':>14s}"]
        chosen = []
        for family in sorted({r["family"] for r in rows if (r["dataset"], r["group"]) == (dataset, group) and r["family"] != OURS}):
            try:
                chosen.append(select_baseline_from_saved(rows, dataset, group, family, method_order(rows, family)).best)
            except ValueError as error:
                lines.append(f"  {family:32s}no selection: {error}")
        for variant in LEVELS["variant"]:
            try:
                chosen.append(f"{select_from_saved(rows, dataset, group, variant).best}__{variant}")
            except ValueError:
                pass  # this label variant was not trained
        for method in chosen:
            calibrator = calibrator_name(method)
            by_set = {s: list(per_split_areas(rows, dataset, group, method, calibrator, s).values()) for s in SETS}
            lines.append(f"  {method:32s}{calibrator:>15s}" + "".join(f"{np.mean(by_set[s]):8.1f}" for s in SETS) + f"{np.std(by_set['unseen']):14.1f}")
    return lines


def report(rows: list[dict], calibrators: tuple[str, ...] = (MAX_CONFORMAL, SAFE_BAND), sizes: list[dict] | None = None,
           costs: list[dict] | None = None) -> str:
    """The split sizes (when given); the factor tables and validation-chosen architectures of our probes under each calibrator; the
    calibrators side by side; then every baseline against our probes."""
    ours = probe_rows(rows)
    n_runs = len({(r["dataset"], r["split"], r["method"]) for r in rows})
    lines = [f"{n_runs} evaluated runs ({len({r['method'] for r in ours})} probes, {len({r['method'] for r in rows} - {r['method'] for r in ours})} baseline "
             f"methods); areas in [0, 100]; calibrators: {', '.join(calibrators)}"]
    if sizes:
        lines += ["", "=" * 100, "SPLIT SIZES"] + split_size_table(sizes)
    for calibrator in calibrators:
        lines += ["", "=" * 100, f"OUR PROBES, CALIBRATOR {calibrator}" + ("  (the protocol's for our probes)" if calibrator == MAX_CONFORMAL else "  (secondary table)")]
        lines += factor_tables(ours, calibrator) + selection_table(ours, calibrator) + extended_table(ours, calibrator)
    if len(calibrators) > 1:
        lines += ["", "=" * 100, "OUR PROBES UNDER BOTH CALIBRATORS"] + calibrator_comparison(ours, calibrators)
    if len(ours) < len(rows):
        lines += ["", "=" * 100, "BASELINES AND OUR PROBES"] + methods_compared(rows)
    if costs:
        lines += ["", "=" * 100, "WHAT THE RUNS COST"] + compute_table(costs)
    return "\n".join(lines + [""] + list(NOTES))
