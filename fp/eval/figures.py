"""The data behind the paper's main table and curve figures, read from the saved runs under results/ (no model is run).

A figure or table row is an ``Entry``: one of our label variants, or a baseline (one family, or the better of two). For each setting (a dataset
and, for LIBERO-X, a kind of held-out split) the entry stands for ONE method (``choose``):

    our probes        one architecture in every setting and for every label variant (``OURS_ARCHITECTURE``), so the variants differ in their
                      labels only
    SAFE-LSTM, -MLP   the configuration with the best validation area averaged over the three kinds of LIBERO-X split, the same for all three
                      (``shared_choice``); each LIBERO-10 setting keeps its own best configuration
    every other row   the configuration with the best mean validation area over the setting's splits, the rule every other result uses

Its areas come from the run summaries, its curves from the runs' curve files (``mean_curve``: the curves of the setting's splits put on a
common FPR grid and averaged), and its costs from the run histories.

Costs are measured on LIBERO-X, the largest dataset, with the configuration chosen for each of its three kinds of split; our probes were trained
before the sweeps recorded costs, so their costs come from a separate benchmark of the chosen probes (``benchmark_plan``), kept in its own folder.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..calibration.protocol import calibrator_name
from ..config import RESULTS
from ..metrics.curves import FPR_MAX, curve_on_grid
from ..results import read_json, run_dir
from .collect import per_split_areas, select_baseline_from_saved, select_from_saved
from .report import method_order
from .select_config import LIBEROX_KINDS, group_of

GRID = np.arange(0.0, FPR_MAX + 0.25, 0.5)  # the FPR values (%) the mean curves are drawn at
COST_FIELDS = ("fit_seconds", "state_mb", "score_ms_per_step")


@dataclass(frozen=True)
class Setting:
    label: str
    dataset: str
    group: str  # 'all' (LIBERO-10, real robot: every split of the dataset) or the kind of held-out split of LIBERO-X
    also: tuple[str, ...] = ()  # more groups whose splits are averaged into the cells and curves; the choice of a configuration uses ``group`` only


SETTINGS = (  # the order of the panels and of the table's columns
    Setting("LIBERO-10 π0.5", "libero10_pi05", "all"),
    Setting("LIBERO-10 GR00T", "libero10_groot", "all"),
    Setting("LIBERO-X task", "liberox_groot", "task"),
    Setting("LIBERO-X signature", "liberox_groot", "signature"),
    Setting("LIBERO-X skill", "liberox_groot", "skill"),
    Setting("Real robot", "real8_groot", "all", also=("loto_seeds",)),  # leave one task out: 8 tasks x 5 seeds (seed 0 is the group "all", seeds 1-4 "loto_seeds");
    # the configurations are chosen on seed 0 and averaged over the 40 splits; no action-based baselines (no planned chunks)
)

SIMULATED_SETTINGS = tuple(s for s in SETTINGS if s.dataset != "real8_groot")  # LIBERO-10 with pi0.5 and with GR00T, the three kinds of LIBERO-X split


@dataclass(frozen=True)
class Entry:
    label: str
    kind: str  # colour family: ours (viridis), safe (black), action (dark gray), distance (light gray)
    style: str  # line style inside the family: solid, dashed, dotted, dashdot
    variant: str | None = None  # one of our label variants, or
    families: tuple[str, ...] = ()  # the baseline families the best configuration is chosen among
    architecture: str | None = None  # our probe: this one architecture in every setting (None: the best of the 16 in each setting)
    shared: str | None = None  # a baseline: the dataset whose settings share one configuration (None: each setting chooses its own)


# Decided: one architecture for all of our rows and settings, with the failure-mode head, with or without the CLIP head. Among the 8 architectures
# with a `modes` or `modes_clip` head it has the best validation area of `onset_pre`, averaged over the five settings (69.3; the best `modes`-only
# one, lstm_modes, has 68.3).
OURS_ARCHITECTURE = "lstm_modes_clip"

ENTRIES = (
    Entry("Ours (onset_pre)", "ours", "solid", variant="onset_pre", architecture=OURS_ARCHITECTURE),
    Entry("Ours (onset_traj)", "ours", "dashed", variant="onset_traj", architecture=OURS_ARCHITECTURE),
    Entry("Ours (onset)", "ours", "dotted", variant="onset", architecture=OURS_ARCHITECTURE),
    Entry("Ours (onset_post)", "ours", "dashdot", variant="onset_post", architecture=OURS_ARCHITECTURE),
    Entry("SAFE-LSTM", "safe", "solid", families=("safe_lstm",), shared="liberox_groot"),
    Entry("SAFE-MLP", "safe", "dashed", families=("safe_mlp",), shared="liberox_groot"),
    Entry("STAC", "action", "solid", families=("stac",)),
    Entry("STAC-Single", "action", "dashed", families=("stac_single",)),
    Entry("ActProbe", "action", "dotted", families=("actprobe",)),
    Entry("FailureSpot", "action", "dashdot", families=("failurespot_mlp", "failurespot_lstm")),
    Entry("Mahalanobis", "distance", "solid", families=("mahalanobis",)),
    Entry("Euclidean", "distance", "dashed", families=("euclidean",)),
    Entry("PCA-KMeans", "distance", "dotted", families=("pca_kmeans",)),
)
TABLE_LABELS = ("SAFE-LSTM", "SAFE-MLP", "STAC", "ActProbe", "FailureSpot", "Mahalanobis", "Euclidean", "PCA-KMeans",  # the table's rows, top to bottom:
                "Ours (onset_pre)")  # the baselines, then ours at the bottom (the other label variants are in the ablation table)
COMPACT_LABELS = ("Ours (onset_pre)", "SAFE-LSTM", "SAFE-MLP", "STAC", "STAC-Single", "ActProbe", "FailureSpot")  # the methods of the compact curve figures
COMPACT_FONT_SCALE = 1.25  # their font sizes relative to the full figures' (9 pt text and legend, 10 pt titles): 11.25 and 12.5 pt
COMPACT_SCALE = 1.0  # their height relative to a figure of two rows of panels (2 x 2.9 in): 5.8 in (it was 0.5 and 2.9 in; their width is 15 in, as the full figures')


def require_runs(rows: list[dict], setting: Setting, method: str) -> str:
    """``method`` itself, after checking that it was evaluated on every split of the setting (ValueError otherwise)."""
    splits = splits_of(rows, setting)
    groups = (setting.group, *setting.also)
    have = {r["split"] for r in rows if r["dataset"] == setting.dataset and r["group"] in groups and r["method"] == method and r["eval_set"] == "val"}
    missing = [s for s in splits if s not in have]
    if not splits or missing:
        raise ValueError(f"{method} has no evaluated run on {missing or 'any split'} of {setting.dataset}/{setting.group}")
    return method


def shared_choice(rows: list[dict], settings: list[Setting], family: str) -> str:
    """The configuration of ``family`` with the best validation area averaged over ``settings`` (each setting's mean over its own splits
    counts once, whatever its number of splits; ties: grid order). Raises ValueError when a setting lacks runs of a configuration."""
    order = method_order(rows, family)
    selections = [select_baseline_from_saved(rows, s.dataset, s.group, family, order) for s in settings]
    mean = {m: float(np.mean([sel.mean_val_area[m] for sel in selections])) for m in order}
    return max(order, key=lambda m: (mean[m], -order.index(m)))


def choose(rows: list[dict], setting: Setting, entry: Entry) -> str:
    """The method the entry stands for in the setting (see the module docstring). Otherwise the best configuration on the mean validation area
    of the setting (ties: grid order); for an entry of two families, the family whose best configuration has the higher one. Raises
    ValueError when runs needed for the choice are missing."""
    if entry.variant:
        if entry.architecture:
            return require_runs(rows, setting, f"{entry.architecture}__{entry.variant}")
        return f"{select_from_saved(rows, setting.dataset, setting.group, entry.variant).best}__{entry.variant}"
    if entry.shared == setting.dataset:
        if len(entry.families) != 1:
            raise ValueError(f"{entry.label}: a shared configuration is chosen within one family")
        return shared_choice(rows, [s for s in SETTINGS if s.dataset == setting.dataset], entry.families[0])
    best = None
    for family in entry.families:
        selection = select_baseline_from_saved(rows, setting.dataset, setting.group, family, method_order(rows, family))
        if best is None or selection.mean_val_area[selection.best] > best[0]:
            best = (selection.mean_val_area[selection.best], selection.best)
    return best[1]


def choose_all(rows: list[dict], entries=ENTRIES, settings=SETTINGS) -> tuple[dict[tuple[str, str], str], list[str]]:
    """{(setting label, entry label): method} for every entry that can be chosen, and a line for each that cannot (missing runs)."""
    chosen, skipped = {}, []
    for setting in settings:
        for entry in entries:
            try:
                chosen[(setting.label, entry.label)] = choose(rows, setting, entry)
            except ValueError as error:
                skipped.append(f"{setting.label} / {entry.label}: {error}")
    return chosen, skipped


def splits_of(rows: list[dict], setting: Setting) -> list[str]:
    groups = (setting.group, *setting.also)
    return sorted({r["split"] for r in rows if r["dataset"] == setting.dataset and r["group"] in groups})


@dataclass(frozen=True)
class Cell:
    mean: float
    std: float  # population std over the setting's splits, as in the report
    n: int


def area_cell(rows: list[dict], setting: Setting, method: str, eval_set: str) -> Cell:
    """Mean and spread of the area of ``method`` over the setting's splits, with the calibrator the protocol gives it."""
    areas = {}
    for group in (setting.group, *setting.also):
        areas.update(per_split_areas(rows, setting.dataset, group, method, calibrator_name(method), eval_set))
    values = np.array([a for a in areas.values() if not np.isnan(a)])
    if not len(values):
        raise ValueError(f"{method} has no area on {setting.label} ({eval_set})")
    return Cell(float(values.mean()), float(values.std()), len(values))


def mean_curve(rows: list[dict], setting: Setting, method: str, eval_set: str, metric: str = "tpr_early", root: Path = RESULTS) -> np.ndarray:
    """``metric`` (a percentage) of ``method`` at the FPR values of ``GRID``, averaged over the setting's splits. Each split's curve is put on the
    grid like the headline area does it: TPR early is made monotone (a looser threshold can only raise it) and every curve is held flat past
    its last point; on-time TPR, which a looser threshold can lower, is neither flattened nor made monotone."""
    calibrator = calibrator_name(method)
    curves = []
    evaluated = {r["split"] for r in rows if r["dataset"] == setting.dataset and r["method"] == method}
    for split in splits_of(rows, setting):
        if split not in evaluated:  # a run that failed has no evaluation (Mahalanobis: a singular covariance): the mean is over the splits that have one, as in the cells
            continue
        points = read_json(run_dir(setting.dataset, split, method, root) / f"curve_{calibrator}.json")["sets"][eval_set]  # (a missing file of an evaluated run is an error)
        curves.append(curve_on_grid(np.array([p["fpr"] for p in points]), np.array([p[metric] for p in points]), GRID, running_max=metric == "tpr_early"))
    if not curves:
        raise ValueError(f"no runs for {setting.label}")
    return np.mean(curves, axis=0)


def entry_costs(entry: Entry, chosen: dict[tuple[str, str], str], costs: list[dict]) -> dict[str, float | None]:
    """Median of each cost field over the LIBERO-X runs of the configurations chosen for the three kinds of split (each on its own splits).
    None where no run recorded the field. ``costs`` comes from ``fp.eval.compute.collect_costs`` (for our probes, of the benchmark folder)."""
    runs = []
    for kind in LIBEROX_KINDS:
        method = chosen.get((f"LIBERO-X {kind}", entry.label))
        runs += [r for r in costs if r["method"] == method and r["dataset"] == "liberox_groot" and group_of(r["split"]) == kind]
    return {f: (float(np.median([r[f] for r in runs if r[f] is not None])) if any(r[f] is not None for r in runs) else None) for f in COST_FIELDS}


def method_latency(latency: dict | None, method: str, device: str) -> float | None:
    """The median online latency (ms) of ``method`` on the ``device`` ('gpu', or 'cpu' with 4 threads) from the file of ``cli.latency``; None where it has none.
    STAC is scored with numpy on the CPU and also samples K plans from the policy, which runs on the GPU: its GPU latency is its scoring time plus the recorded
    sampling time (None without it) and it has no CPU latency; STAC-Single samples nothing."""
    record = (latency or {}).get("methods", {}).get(method)
    if record is None:
        return None
    family = method.split("__")[0]
    if family in ("stac", "stac_single"):
        score = record["cpu4"]["median_ms"]
        if family == "stac_single":
            return score
        extra = (latency.get("vla_sampling") or {}).get("extra_ms")
        return score + extra if device == "gpu" and extra is not None else None
    measured = record["gpu" if device == "gpu" else "cpu4"]
    return None if measured is None else measured["median_ms"]


def entry_latency(entry: Entry, chosen: dict[tuple[str, str], str], latency: dict | None) -> dict[str, float | None]:
    """Median over the methods chosen for the three kinds of LIBERO-X split of the online latency on the GPU and on the CPU (``method_latency``)."""
    methods = [chosen[(f"LIBERO-X {kind}", entry.label)] for kind in LIBEROX_KINDS if (f"LIBERO-X {kind}", entry.label) in chosen]
    out = {}
    for field, device in (("latency_gpu_ms", "gpu"), ("latency_cpu_ms", "cpu")):
        values = [v for v in (method_latency(latency, m, device) for m in methods) if v is not None]
        out[field] = float(np.median(values)) if values else None
    return out


def benchmark_plan(chosen: dict[tuple[str, str], str], rows: list[dict], entries=ENTRIES) -> list[tuple[str, str, list[str]]]:
    """What to re-train to measure the costs of our probes: (architecture, label variant, LIBERO-X splits) for each of our entries and kind of
    split, with the architecture chosen for it."""
    plan = []
    for entry in (e for e in entries if e.variant):
        for kind in LIBEROX_KINDS:
            setting = next(s for s in SETTINGS if s.dataset == "liberox_groot" and s.group == kind)
            method, splits = chosen.get((setting.label, entry.label)), splits_of(rows, setting)
            if method and splits:
                plan.append((method.split("__")[0], entry.variant, splits))
    return plan
