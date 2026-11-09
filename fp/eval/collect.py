"""Read the evaluation summaries of all runs into one long table, and aggregate it.

One row per run x calibrator x evaluation set: dataset, split, group, method, trunk, mha, heads, variant, calibrator, eval_set, area,
extended, n_fail, n_succ. ``group`` is the held-out set a split belongs to (task, signature, skill for LIBERO-X; all for LIBERO-10).
The configuration of our probes is chosen on the VALIDATION area by the same rule as every other method's (``select_config.choose_best``).
"""

import csv
from pathlib import Path

import numpy as np

from ..calibration.protocol import MAX_CONFORMAL, calibrator_name, parse_probe_name
from ..config import RESULTS
from ..models.ours import all_configs
from ..results import iter_runs, read_json
from .select_config import Selection, choose_best, group_of

FACTORS = ("trunk", "mha", "heads", "variant")
COLUMNS = ("dataset", "split", "group", "method", "family", "trunk", "mha", "heads", "variant", "calibrator", "eval_set", "area", "extended", "n_fail", "n_succ")
OURS = "ours"


def family_of(method: str) -> str:
    """'ours' for our 64 probes; for a baseline the name before '__' (euclidean__k10_cumsum -> euclidean, safe_lstm -> safe_lstm)."""
    return OURS if parse_probe_name(method) is not None else method.split("__")[0]


def probe_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["family"] == OURS]


def collect_rows(root: Path = RESULTS, datasets: tuple[str, ...] | None = None) -> list[dict]:
    """Every evaluated run under ``root``; runs without a summary.json are not in the table. A calibrator that could not be fitted
    for a run has no rows."""
    rows = []
    for dataset, split, method, path in iter_runs(root, datasets):
        if not (path / "summary.json").exists():
            continue
        probe = parse_probe_name(method) or {}
        for calibrator, result in read_json(path / "summary.json")["calibrators"].items():
            for eval_set, s in result.get("sets", {}).items():
                rows.append({"dataset": dataset, "split": split, "group": group_of(split), "method": method, "family": family_of(method), **probe,
                             "calibrator": calibrator,
                             "eval_set": eval_set, "area": s["area"], "extended": s["extended"], "n_fail": s["n_fail"], "n_succ": s["n_succ"]})
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def arch_name(row: dict) -> str:
    return f"{row['trunk']}{'_mha' if row['mha'] else ''}_{row['heads']}"


def mean_area(rows: list[dict], **where) -> tuple[float, int]:
    """(mean area, number of rows) of the rows matching ``where``, e.g. mean_area(rows, trunk='lstm', eval_set='unseen'). NaN areas are skipped."""
    values = [r["area"] for r in rows if all(r.get(k) == v for k, v in where.items()) and not np.isnan(r["area"])]
    return (float(np.mean(values)) if values else float("nan")), len(values)


def select_from_saved(rows: list[dict], dataset: str, group: str, variant: str, calibrator: str = MAX_CONFORMAL) -> Selection:
    """The best of the 16 architectures of ``variant`` on a dataset's group of splits, chosen on the validation area by ``choose_best``.
    Every architecture must have a validation area on every split of the group."""
    cells = [r for r in probe_rows(rows) if (r["dataset"], r["group"], r["variant"], r["calibrator"], r["eval_set"]) == (dataset, group, variant, calibrator, "val")]
    splits = sorted({r["split"] for r in cells})
    area = {(arch_name(r), r["split"]): r["area"] for r in cells}
    per_run = {}
    for cfg in all_configs():
        missing = [s for s in splits if (cfg.name, s) not in area]
        if missing:
            raise ValueError(f"{cfg.name}__{variant} has no validation area on {missing} of {dataset}/{group}")
        per_run[cfg.name] = [area[(cfg.name, s)] for s in splits]
    if not splits:
        raise ValueError(f"no evaluated runs for {dataset}/{group}/{variant}")
    return choose_best(per_run)


def select_baseline_from_saved(rows: list[dict], dataset: str, group: str, family: str, order: list[str]) -> Selection:
    """The best configuration of a baseline family on a dataset's group of splits, chosen on the validation area by the same ``choose_best``
    as our probes, with the protocol's calibrator for the family. ``order`` lists the family's methods in grid order (ties go to the earlier)."""
    if not order:  # a family without a fixed grid is ordered by the names of its evaluated runs, so no runs means no methods
        raise ValueError(f"no evaluated runs of {family} for {dataset}/{group}")
    calibrator = calibrator_name(order[0])
    cells = [r for r in rows if (r["dataset"], r["group"], r["family"], r["calibrator"], r["eval_set"]) == (dataset, group, family, calibrator, "val")]
    splits = sorted({r["split"] for r in cells})
    if not splits:
        raise ValueError(f"no evaluated runs of {family} for {dataset}/{group}")
    area = {(r["method"], r["split"]): r["area"] for r in cells}
    per_run = {}
    for method in order:
        missing = [s for s in splits if (method, s) not in area]
        if missing:
            raise ValueError(f"{method} has no validation area on {missing} of {dataset}/{group}")
        per_run[method] = [area[(method, s)] for s in splits]
    return choose_best(per_run)


def per_split_areas(rows: list[dict], dataset: str, group: str, method: str, calibrator: str, eval_set: str) -> dict[str, float]:
    """{split: area} of one method, to show the spread over the runs of a group."""
    return {r["split"]: r["area"] for r in rows
            if (r["dataset"], r["group"], r["method"], r["calibrator"], r["eval_set"]) == (dataset, group, method, calibrator, eval_set)}
