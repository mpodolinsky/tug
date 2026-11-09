"""The baseline driver on a small synthetic dataset (feature width 8, CPU)."""

import gc
import json
import weakref

import numpy as np
import pytest

from fp.baselines import sweep
from fp.baselines.distance import DistanceScorer
from fp.baselines.distance_grid import method_names
from fp.baselines.registry import FAMILIES, Context, register
from fp.baselines.sweep import BaselineUnit, LazyFeatures, Settings, baseline_units, run_scorer, run_unit
from fp.data.inputs import Inputs
from fp.data.manifest import Episode
from fp.data.splits import Split
from fp.eval.collect import collect_rows
from fp.eval.run_eval import evaluate_saved_run
from fp.models.base import Scorer
from fp.results import hardware_info, is_done, peak_rss_gb, read_history, read_json, read_scores, reset_peak_rss, run_dir

D, WINDOW = 8, 3


@pytest.fixture(scope="module")
def world():
    """100 episodes: failures drift away from the success distribution; every part of the split holds both outcomes, and the calibration
    part has enough successes for the SAFE band (it needs at least two in its 30% fitting part)."""
    rng = np.random.default_rng(0)
    episodes, features = {}, {}
    for i in range(100):
        fail = i % 2 == 0
        n = int(rng.integers(12, 20))
        key = f"{'f' if fail else 's'}{i:02d}"
        episodes[key] = Episode(key, not fail, f"t{i % 4}", n, int(rng.integers(6, n - 1)) if fail else None, "a")
        x = rng.normal(size=(n, D)).astype(np.float32)
        features[key] = x + (np.linspace(0, 3, n)[:, None] if fail else 0)
    keys = list(episodes)
    split = Split("task_seed3", keys[:40], keys[40:48], keys[48:78], keys[78:88], keys[88:], seed=3)
    return Inputs("synthetic", episodes, None, {split.name: split}, WINDOW), features, split


def context(world):
    inputs, features, _ = world
    return Context(inputs, features, "cpu")


UNIT = BaselineUnit("synthetic", "task_seed3", "euclidean")
TWO = ["euclidean__k1_step", "euclidean__k5_cumsum"]


def test_each_run_has_its_scores_and_a_history_that_records_what_it_cost(world, tmp_path):
    inputs, _, split = world
    assert run_unit(UNIT, context(world), Settings(tmp_path), lambda m: None, TWO) == 2
    for method in TWO:
        path = run_dir("synthetic", "task_seed3", method, tmp_path)
        assert sorted(p.name for p in path.iterdir()) == ["distances.npz", "history.json", "scores.npz"]  # no weights: nothing was learned
        scores = read_scores(path)
        assert set(scores) == set(inputs.episodes) and all(s.shape == (inputs.episodes[k].n_steps,) and np.isfinite(s).all() for k, s in scores.items())
        h = read_history(path)
        assert (h["method"], h["family"], h["features"], h["seed"], h["weights_saved"]) == (method, "euclidean", "raw", 3, False)
        assert h["config"]["topk"] in (1, 5) and h["config"]["seed"] == 3 and h["code"]["fp_sha256"] and h["split"] == "task_seed3"
        assert h["fit_seconds"] >= 0 and h["score_seconds"] >= 0 and h["scored_steps"] == sum(e.n_steps for e in inputs.episodes.values())
        assert h["n_train_steps"] == sum(inputs.episodes[k].n_steps for k in split.train) and h["n_train_episodes"] == 40
        assert h["score_ms_per_step"] >= 0 and h["peak_rss_gb"] > 0 and h["score_min"] <= h["score_max"] and h["hardware"]["cpu_threads"] >= 1
        assert h["state_mb"] == pytest.approx(h["n_train_steps"] * D * 4 / 1e6, rel=1e-6, abs=0.01)  # a k-NN keeps every training step


@pytest.mark.parametrize("method", ["euclidean__k1_step", "euclidean__k5_cumsum"])
def test_the_saved_distances_rebuild_the_saved_score(world, tmp_path, method):
    """distances.npz holds, per episode, the (T, 2) distances to the success and failure sets; score = column 0 - column 1, accumulated for cumsum."""
    inputs = world[0]
    run_unit(UNIT, context(world), Settings(tmp_path), lambda m: None, [method])
    path = run_dir("synthetic", "task_seed3", method, tmp_path)
    scores = read_scores(path)
    with np.load(path / "distances.npz") as z:
        assert set(z.files) == set(inputs.episodes)
        for key, d in ((k, z[k]) for k in list(inputs.episodes)[::7]):
            assert d.shape == (inputs.episodes[key].n_steps, 2) and np.isfinite(d).all() and (d >= 0).all()
            step = d[:, 0] - d[:, 1]
            rebuilt = np.cumsum(step) if method.endswith("cumsum") else step
            np.testing.assert_allclose(scores[key], rebuilt, rtol=1e-4, atol=1e-3)
    assert read_history(path)["extras_saved"] == ["distances"]


def test_the_failure_column_is_nan_when_only_the_success_distance_is_used(world):
    inputs, features, split = world
    scorer = DistanceScorer("euclid", inputs.episodes, features, topk=3, use_success_only=True, cumsum=False)
    scorer.fit(split.train, split.val)
    scores, extras = scorer.score_with_extras(split.val)
    key = split.val[0]
    d = extras["distances"][key]
    assert np.isnan(d[:, 1]).all() and np.allclose(scores[key], d[:, 0])  # the score is the distance to the success set alone


def test_scoring_without_extras_gives_the_same_scores(world):
    inputs, features, split = world
    scorer = DistanceScorer("mahala", inputs.episodes, features)
    scorer.fit(split.train, split.val)
    plain, (with_extras, extras) = scorer.score(split.val), (scorer.score_with_extras(split.val)[0], scorer.score_with_extras(split.val)[1])
    assert all(np.array_equal(plain[k], with_extras[k]) for k in split.val) and set(extras["distances"]) == set(split.val)


def test_runs_that_exist_are_skipped_without_loading_any_data(world, tmp_path, monkeypatch):
    run_unit(UNIT, context(world), Settings(tmp_path), lambda m: None, TWO)
    loaded = []
    monkeypatch.setattr(sweep, "load_features", lambda dataset, keys: loaded.append(1) or {})
    lazy = LazyFeatures("synthetic", list(world[0].episodes))
    assert run_unit(UNIT, Context(world[0], lazy, "cpu"), Settings(tmp_path), lambda m: None, TWO) == 0 and loaded == []


def test_a_scorer_that_fails_is_logged_and_skipped_and_the_others_still_run(world, tmp_path, monkeypatch):
    real = sweep.run_scorer

    def flaky(scorer, *args):
        if scorer.name == TWO[0]:
            raise RuntimeError("CUDA out of memory")
        return real(scorer, *args)

    monkeypatch.setattr(sweep, "run_scorer", flaky)
    messages = []
    assert run_unit(UNIT, context(world), Settings(tmp_path), messages.append, TWO) == 1
    assert not run_dir("synthetic", "task_seed3", TWO[0], tmp_path).exists() and is_done(run_dir("synthetic", "task_seed3", TWO[1], tmp_path))
    assert any("FAILED" in m and "CUDA out of memory" in m for m in messages)


def test_non_finite_scores_are_a_failure_not_a_result(world, tmp_path):
    class Broken(Scorer):
        name = "broken__x"

        def fit(self, train_keys, val_keys):
            pass

        def score(self, keys):
            return {k: np.array([0.0, np.inf]) for k in keys}

    with pytest.raises(ValueError, match="non-finite"):
        run_scorer(Broken(), "broken", world[2], world[0], "cpu")


class Tracked(Scorer):
    """Records whether the scorer built before it has been released by the time it scores."""

    instances: list = []
    released: list = []

    def __init__(self, name):
        self.name = name
        self.index = len(Tracked.instances)
        Tracked.instances.append(weakref.ref(self))

    def fit(self, train_keys, val_keys):
        pass

    def score(self, keys):
        gc.collect()
        if self.index > 0:
            Tracked.released.append(Tracked.instances[self.index - 1]() is None)
        return {k: np.zeros(3) for k in keys}


def test_a_fitted_scorer_is_released_before_the_next_one_runs(world, tmp_path):
    """A fitted distance scorer holds its training steps (about 3 GB on LIBERO-X): never two at a time."""
    Tracked.instances, Tracked.released = [], []
    register("tracked")(lambda ctx, split: [Tracked(f"tracked__{i}") for i in range(4)])
    try:
        assert run_unit(BaselineUnit("synthetic", "task_seed3", "tracked"), context(world), Settings(tmp_path), lambda m: None) == 4
    finally:
        del FAMILIES["tracked"]
    assert Tracked.released == [True, True, True]


def test_lazy_features_are_read_once_on_first_use_and_can_be_released(world, monkeypatch):
    calls = []
    monkeypatch.setattr(sweep, "load_features", lambda dataset, keys: calls.append(dataset) or {k: world[1][k] for k in keys})
    lazy = LazyFeatures("synthetic", list(world[0].episodes))
    assert calls == [] and len(lazy) == 100 and list(lazy) == list(world[0].episodes)  # asking for the keys does not load
    first = lazy[list(lazy)[0]]
    assert calls == ["synthetic"] and np.array_equal(first, world[1][list(lazy)[0]])
    lazy[list(lazy)[1]]
    assert calls == ["synthetic"]  # still one read
    lazy.release()
    lazy[list(lazy)[0]]
    assert calls == ["synthetic", "synthetic"]


def test_units_follow_the_run_order_and_a_datasets_units_are_adjacent():
    names = {"libero10_groot": ["paper_seed1", "paper_seed0"], "liberox_groot": ["signature_seed0", "task_seed0", "skill_a"]}
    units = baseline_units(names, ["mahalanobis", "euclidean"])
    assert [(u.dataset, u.split, u.family) for u in units[:4]] == [("liberox_groot", "task_seed0", "mahalanobis"), ("liberox_groot", "task_seed0", "euclidean"),
                                                                    ("liberox_groot", "signature_seed0", "mahalanobis"), ("liberox_groot", "signature_seed0", "euclidean")]
    assert [u.dataset for u in units] == ["liberox_groot"] * 6 + ["libero10_groot"] * 4 and len(units) == 10


def test_state_bytes_count_what_scoring_a_new_episode_needs(world):
    inputs, features, split = world
    n_succ = sum(inputs.episodes[k].n_steps for k in split.train if inputs.episodes[k].success)
    n_fail = sum(inputs.episodes[k].n_steps for k in split.train if not inputs.episodes[k].success)
    scorers = {d: DistanceScorer(d, inputs.episodes, features, pca_dim=4, n_clusters=3) for d in ("euclid", "mahala", "pca_kmeans")}
    assert all(s.state_bytes() is None for s in scorers.values())  # not fitted
    for s in scorers.values():
        s.fit(split.train, split.val)
    assert scorers["euclid"].state_bytes() == (n_succ + n_fail) * D * 4  # every training step is kept
    assert scorers["mahala"].state_bytes() == 2 * D * 4 + 2 * D * D * 4  # two means and two pseudo-inverse covariances, independent of the data size
    assert 0 < scorers["pca_kmeans"].state_bytes() < scorers["euclid"].state_bytes()  # centroids and a basis are smaller than the stored steps


def test_the_resource_helpers_report_plausible_numbers():
    h = hardware_info()
    assert {"gpu", "gpu_memory_gb", "cpu_threads", "ram_gb", "torch", "python", "platform"} <= set(h) and h["ram_gb"] > 0
    blob = np.ones(50_000_000, dtype=np.float64)  # 400 MB, so the peak is clearly above what comes after the reset
    before = peak_rss_gb()
    del blob
    reset_peak_rss()
    assert 0 < peak_rss_gb() <= before


def test_a_baseline_run_is_evaluated_with_the_protocols_calibrator_and_collected_as_a_baseline(world, tmp_path):
    inputs, _, split = world
    run_unit(UNIT, context(world), Settings(tmp_path), lambda m: None, ["euclidean__k5_cumsum"])
    path = run_dir("synthetic", "task_seed3", "euclidean__k5_cumsum", tmp_path)
    assert evaluate_saved_run("synthetic", "task_seed3", "euclidean__k5_cumsum", path, inputs) == "done"
    summary = read_json(path / "summary.json")
    assert list(summary["calibrators"]) == ["safe_band"]  # the SAFE band: the protocol's calibrator for the distance baselines
    assert (path / "curve_safe_band.json").exists() and not (path / "curve_max_conformal.json").exists()
    rows = collect_rows(tmp_path)
    assert {(r["family"], r["calibrator"]) for r in rows} == {("euclidean", "safe_band")} and len(rows) == 3  # val, seen, unseen
    assert all("variant" not in r and "trunk" not in r for r in rows)  # baseline rows carry no probe factors


def test_the_grid_order_of_a_family_is_stable():
    assert method_names("mahalanobis") == ["mahalanobis__step", "mahalanobis__cumsum"]
    assert len(method_names("euclidean")) == 6 and len(method_names("pca_kmeans")) == 18
    assert json.dumps(method_names("pca_kmeans")[:2]) == json.dumps(["pca_kmeans__d32_c16_step", "pca_kmeans__d32_c16_cumsum"])
