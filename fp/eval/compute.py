"""What the runs cost: time, memory and the size of what each method keeps, read from the histories of the runs.

One row per run (``collect_costs``), summarised per dataset and method family (``compute_table``). The baselines' histories record, per run,
the seconds to fit and to score all episodes, the milliseconds per scored step, peak GPU memory, peak resident memory of the process,
``state_mb`` (what a fitted scorer must keep to score a new episode) and the hardware; see ``baselines/sweep.py`` for exactly what each
number includes. The first sweep of our probes recorded only a total time and the epochs, so those columns read n/a for it; probes trained
since then record the same fields as the baselines.
"""

from pathlib import Path

import numpy as np

from ..config import RESULTS
from ..results import iter_runs, read_history
from .collect import family_of

FIELDS = ("seconds", "fit_seconds", "score_seconds", "score_ms_per_step", "scored_steps", "peak_gpu_gb", "peak_rss_gb", "state_mb", "n_params", "epochs")


def collect_costs(root: Path = RESULTS, datasets: tuple[str, ...] | None = None) -> list[dict]:
    """One row per finished run with the cost fields its history has (None where it has none)."""
    rows = []
    for dataset, split, method, path in iter_runs(root, datasets):
        history = read_history(path)
        rows.append({"dataset": dataset, "split": split, "method": method, "family": family_of(method), "hardware": history.get("hardware"),
                     **{field: history.get(field) for field in FIELDS}})
    return rows


def _stat(values: list, width: int, decimals: int = 1, how=np.mean) -> str:
    """The mean (or ``how``) of the recorded values, right-aligned in ``width`` columns; n/a if none was recorded."""
    values = [v for v in values if v is not None]
    return f"{float(how(values)):{width}.{decimals}f}" if values else f"{'n/a':>{width}}"


def compute_table(costs: list[dict]) -> list[str]:
    """Per dataset and family: runs, total hours, seconds to fit and to score all of the dataset's episodes (mean and max over runs),
    milliseconds per scored step, the largest GPU and RAM peaks, and the mean size of the fitted state."""
    lines = []
    head = f"  {'family':14s}{'runs':>6s}{'total h':>9s}{'fit s mean':>12s}{'max':>8s}{'score s mean':>14s}{'max':>8s}{'ms/step':>9s}{'GPU GB':>8s}{'RAM GB':>8s}{'state MB':>10s}"
    for dataset in dict.fromkeys(r["dataset"] for r in costs):
        lines += ["", f"{dataset}: cost per run (max = largest over the runs; n/a = not recorded)", head]
        for family in sorted({r["family"] for r in costs if r["dataset"] == dataset}):
            rows = [r for r in costs if r["dataset"] == dataset and r["family"] == family]
            total = sum((r["fit_seconds"] or 0) + (r["score_seconds"] or 0) if r["fit_seconds"] is not None else (r["seconds"] or 0) for r in rows)
            col = lambda field: [r[field] for r in rows]  # noqa: E731
            lines.append(f"  {family:14s}{len(rows):6d}{total / 3600:9.2f}{_stat(col('fit_seconds'), 12)}{_stat(col('fit_seconds'), 8, how=max)}"
                         f"{_stat(col('score_seconds'), 14)}{_stat(col('score_seconds'), 8, how=max)}{_stat(col('score_ms_per_step'), 9, 3)}"
                         f"{_stat(col('peak_gpu_gb'), 8, how=max)}{_stat(col('peak_rss_gb'), 8, how=max)}{_stat(col('state_mb'), 10)}")
    hardware = next((r["hardware"] for r in costs if r["hardware"]), None)
    if hardware:
        lines += ["", f"measured on: {hardware['gpu']} ({hardware['gpu_memory_gb']} GB), {hardware['cpu_threads']} CPU threads, {hardware['ram_gb']} GB RAM, "
                      f"torch {hardware['torch']}, Python {hardware['python']}"]
    return lines + ["",
                    "Notes: 'fit' is training for the probes and the fit of the stored statistics for the baselines; 'score' is one forward pass over every episode",
                    "of the dataset. GPU and RAM peaks include the features a sweep keeps loaded for its other runs, so they overstate a single run; several",
                    "workers shared the GPU while the sweeps ran, which lengthens the times. 'total h' sums the runs of the family on that dataset."]
