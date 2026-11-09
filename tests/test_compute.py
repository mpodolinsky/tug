import numpy as np

from fp.eval.compute import collect_costs, compute_table
from fp.results import run_dir, write_run


def row(dataset, family, **kw):
    base = {"dataset": dataset, "split": "s", "method": f"{family}__x", "family": family, "hardware": None, "seconds": None, "fit_seconds": None,
            "score_seconds": None, "score_ms_per_step": None, "scored_steps": None, "peak_gpu_gb": None, "peak_rss_gb": None, "state_mb": None,
            "n_params": None, "epochs": None}
    return {**base, **kw}


def test_the_table_summarises_recorded_values_per_dataset_and_family():
    rows = [row("d", "euclidean", fit_seconds=6.0, score_seconds=60.0, score_ms_per_step=0.25, peak_gpu_gb=4.0, peak_rss_gb=6.7, state_mb=3000.0),
            row("d", "euclidean", fit_seconds=8.0, score_seconds=62.0, score_ms_per_step=0.27, peak_gpu_gb=5.0, peak_rss_gb=7.0, state_mb=3200.0),
            row("d", "mahalanobis", fit_seconds=21.0, score_seconds=2.0, score_ms_per_step=0.01, peak_gpu_gb=3.0, peak_rss_gb=6.7, state_mb=300.0)]
    text = "\n".join(compute_table(rows))
    line = next(l for l in text.splitlines() if l.strip().startswith("euclidean"))
    assert line.split() == ["euclidean", "2", "0.04", "7.0", "8.0", "61.0", "62.0", "0.260", "5.0", "7.0", "3100.0"]  # means, and the maxima of the GPU/RAM peaks
    assert "total h" in text and "d: cost per run" in text


def test_a_run_without_recorded_costs_reads_n_a_and_its_total_time_still_counts():
    """The first sweep of the probes recorded only one total time."""
    text = "\n".join(compute_table([row("d", "ours", seconds=36.0), row("d", "ours", seconds=36.0)]))
    line = next(l for l in text.splitlines() if l.strip().startswith("ours"))
    assert line.split()[:3] == ["ours", "2", "0.02"] and line.split()[3:] == ["n/a"] * 8  # 72 s in total; nothing else recorded


def test_the_hardware_line_comes_from_the_first_run_that_recorded_it():
    hw = {"gpu": "RTX 4090", "gpu_memory_gb": 24.0, "cpu_threads": 32, "ram_gb": 31.0, "torch": "2.7.1", "python": "3.11"}
    text = "\n".join(compute_table([row("d", "a"), row("d", "b", hardware=hw)]))
    assert "RTX 4090 (24.0 GB), 32 CPU threads, 31.0 GB RAM" in text


def test_costs_are_read_from_the_histories_of_finished_runs(tmp_path):
    write_run(run_dir("ds", "sp", "euclidean__k5_cumsum", tmp_path), {"a": np.zeros(2, np.float32)},
              {"fit_seconds": 5.0, "score_seconds": 50.0, "peak_gpu_gb": 3.2, "state_mb": 100.0, "hardware": {"gpu": "g"}})
    write_run(run_dir("ds", "sp", "lstm_plain__onset_pre", tmp_path), {"a": np.zeros(2, np.float32)}, {"seconds": 40.0, "epochs": 19})
    rows = {r["method"]: r for r in collect_costs(tmp_path)}
    assert rows["euclidean__k5_cumsum"]["family"] == "euclidean" and rows["euclidean__k5_cumsum"]["fit_seconds"] == 5.0
    assert rows["lstm_plain__onset_pre"]["family"] == "ours" and rows["lstm_plain__onset_pre"]["fit_seconds"] is None and rows["lstm_plain__onset_pre"]["epochs"] == 19
