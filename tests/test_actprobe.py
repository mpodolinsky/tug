"""ActProbe on a synthetic dataset, and against the ActProbe repository's own network, feature and metric code (ACTPROBE_ROOT, the folder holding
its lib_shared/ and benchmarks/; the comparisons are skipped when it is absent)."""

import dataclasses
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from fp import config
from fp.baselines import actprobe
from fp.baselines.actprobe import ActProbeNet, ActProbeScorer, acm_tce, q_auc_taskmax
from fp.baselines.registry import Context, build
from fp.baselines.sweep import BaselineUnit, Settings, run_unit
from fp.calibration.protocol import MAX_CONFORMAL, calibrator_name
from fp.data.inputs import Inputs
from fp.data.manifest import Episode
from fp.data.splits import Split
from fp.eval.run_eval import evaluate_saved_run
from fp.results import read_history, read_scores, run_dir

K, H, A, E = 3, 6, 7, 2


# ---------------------------------------------------------------- the two features
def test_acm_is_the_rms_of_the_executed_part_and_tce_the_mean_squared_change_over_the_overlap():
    chunks = np.zeros((3, H, A), np.float32)
    chunks[0, :E] = 2.0  # executed part of call 0: all 2 -> RMS 2
    chunks[0, E:] = 1.0  # call 0's unexecuted rest
    chunks[1] = 1.0  # call 1 repeats that rest exactly on the overlap -> TCE 0, and executes values of 1
    chunks[2, : H - E] = 3.0  # call 2 starts 2 away from call 1's unexecuted rest everywhere -> TCE 4, and executes values of 3
    f = acm_tce(chunks, E)
    assert f.shape == (3, 2) and f.dtype == np.float32
    assert f[:, 0].tolist() == pytest.approx([2.0, 1.0, 3.0])
    assert f[:, 1].tolist() == pytest.approx([0.0, 0.0, 4.0])  # the first call has no previous one


def test_only_the_executed_part_counts_for_acm_and_a_single_call_scores_zero_tce():
    chunks = np.random.default_rng(0).normal(size=(4, H, A)).astype(np.float32)
    other = chunks.copy()
    other[:, E:] += 5.0  # not executed
    assert np.array_equal(acm_tce(chunks, E)[:, 0], acm_tce(other, E)[:, 0])
    assert acm_tce(chunks[:1], E)[0, 1] == 0
    with pytest.raises(ValueError, match="must be longer"):
        acm_tce(chunks[:, :E], E)


# ---------------------------------------------------------------- q-AUC, the selection metric
def test_q_auc_uses_the_maximum_over_the_first_quarter_of_the_longest_episode_of_the_task():
    # task a: longest 8 -> cutoff ceil(0.25 * 8) = 2 steps; task b: longest 4 -> cutoff 1 step
    scores = [np.array([0.1, 0.9, 0.0, 0.0, 0, 0, 0, 0.99]), np.array([0.2, 0.3]), np.array([0.9, 0.0, 0.0, 0.0]), np.array([0.1, 0.0])]
    labels, tasks = [True, False, True, False], ["a", "a", "b", "b"]
    # episode maxima over the cutoff: 0.9, 0.3, 0.9, 0.1 -> both failures above both successes
    assert q_auc_taskmax(scores, labels, tasks) == 1.0
    assert q_auc_taskmax(scores, [False, True, False, True], tasks) == 0.0
    # the late peak (0.99 at step 7) is outside the cutoff and cannot count: a successful episode with it would still rank below
    late = [np.array([0.1, 0.2, 0, 0, 0, 0, 0, 0.99]), np.array([0.2, 0.3])]
    assert q_auc_taskmax(late, [True, False], ["a", "a"]) == 0.0


def test_q_auc_is_one_half_when_only_one_outcome_is_present_and_the_cutoff_never_exceeds_the_episode():
    assert q_auc_taskmax([np.array([0.5, 0.5])], [True], ["a"]) == 0.5
    assert q_auc_taskmax([np.array([0.9]), np.array([0.1])], [True, False], ["a", "a"], q=1.0) == 1.0


# ---------------------------------------------------------------- the network
def test_the_network_is_causal_ignores_padding_and_has_the_size_of_its_layers():
    torch.manual_seed(0)
    net = ActProbeNet().eval()
    x, lang = torch.randn(2, 9, 3), torch.randn(2, 1024)
    full = net(x, lang, torch.tensor([9, 9]))
    assert full.shape == (2, 9) and ((full > 0) & (full < 1)).all()
    short = net(x[:, :5], lang, torch.tensor([5, 5]))
    assert torch.allclose(full[:, :5], short, atol=1e-6)  # later steps cannot change earlier scores
    padded = x.clone()
    padded[1, 5:] = 99.0
    ragged = net(padded, lang, torch.tensor([9, 5]))
    assert torch.allclose(ragged[1, :5], full[1, :5], atol=1e-6)  # what is in the padded steps cannot change the real ones
    assert sum(p.numel() for p in net.parameters()) == 22945  # 16400 (instruction) + 1088 (h0, c0) + 4736 (LSTM) + 721 (MLP)


# ---------------------------------------------------------------- against the ActProbe repository
ROOT = Path(os.environ["ACTPROBE_ROOT"]) if os.environ.get("ACTPROBE_ROOT") else Path(__file__).parent / "missing"
needs_actprobe = pytest.mark.skipif(not (ROOT / "lib_shared/metrics.py").exists(), reason="ActProbe checkout not found (set ACTPROBE_ROOT)")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_actprobe
def test_the_network_equals_the_repositorys_when_given_its_weights():
    bench = ROOT / "benchmarks/pi0_libero/code"
    sys.path.insert(0, str(bench))
    try:
        theirs_module = load_module("actprobe_reference_net", bench / "lib/methods/actprobe.py")
    finally:
        sys.path.remove(str(bench))
        for name in [n for n in sys.modules if n == "lib" or n.startswith("lib.")]:
            del sys.modules[name]
    torch.manual_seed(1)
    theirs, mine = theirs_module.ActProbeNet().eval(), ActProbeNet().eval()
    assert [(k, tuple(v.shape)) for k, v in theirs.state_dict().items()] == [(k, tuple(v.shape)) for k, v in mine.state_dict().items()]
    mine.load_state_dict(theirs.state_dict())
    x, lang, lengths = torch.randn(3, 8, 3), torch.randn(3, 1024), torch.tensor([8, 6, 3])
    assert torch.allclose(theirs(x, lang, lengths), mine(x, lang, lengths), atol=1e-6)


@needs_actprobe
def test_the_features_and_q_auc_equal_the_repositorys():
    try:
        extract = load_module("actprobe_reference_features", ROOT / "benchmarks/pi0_libero/scripts/collect/extract_features.py")
    except ImportError as error:  # (the script imports tqdm at the top)
        pytest.skip(f"the reference feature script cannot be imported here: {error}")
    metrics = load_module("actprobe_reference_metrics", ROOT / "lib_shared/metrics.py")
    rng = np.random.default_rng(2)
    chunks = rng.normal(size=(7, H, A)).astype(np.float32)
    ours = acm_tce(chunks, E)
    for t in range(7):
        assert ours[t, 0] == pytest.approx(extract.compute_action_norm(chunks[t], E), rel=1e-5)
        assert ours[t, 1] == pytest.approx(extract.compute_chunk_mse(chunks[t - 1] if t else None, chunks[t], E), rel=1e-5, abs=1e-9)
    lengths, tasks, labels = rng.integers(6, 14, size=24), rng.integers(0, 3, size=24), rng.integers(0, 2, size=24)
    scores = [rng.random(n) for n in lengths]
    theirs = metrics.q_auc([dict(scores=s, length=len(s), label=int(y), task_id=int(t)) for s, y, t in zip(scores, labels, tasks)], mode="taskmax", q=0.25)
    assert q_auc_taskmax(scores, [bool(y) for y in labels], [str(t) for t in tasks]) == pytest.approx(theirs, abs=1e-12)


# ---------------------------------------------------------------- the scorer on a synthetic dataset
def make_world(n=100, seed=0):
    """Failures change their plans between calls (large TCE) and move more (large ACM); successes are smooth. Four tasks, one instruction each."""
    rng = np.random.default_rng(seed)
    episodes, actions = {}, {}
    for i in range(n):
        fail = i % 2 == 0
        steps = int(rng.integers(10, 16))
        key = f"{'f' if fail else 's'}{i:03d}"
        episodes[key] = Episode(key, not fail, f"t{i % 4}", steps, int(rng.integers(3, steps - 1)) if fail else None, "a")
        base = 0.1 * rng.normal(size=(1, 1, H, A))
        chunks = np.repeat(base, steps, axis=0) * np.ones((steps, K, H, A))
        chunks = chunks + 0.05 * rng.normal(size=chunks.shape)
        if fail:
            chunks = chunks + (np.linspace(0.0, 0.5, steps)[:, None, None, None]) * rng.normal(size=chunks.shape)
        actions[key] = chunks.astype(np.float32)
    keys = list(episodes)
    split = Split("task_seed3", keys[:40], keys[40:60], keys[60:78], keys[78:88], keys[88:], seed=3)
    embeddings = {f"t{j}": (lambda v: (v / np.linalg.norm(v)).astype(np.float32))(rng.normal(size=1024)) for j in range(4)}
    return episodes, actions, split, embeddings


@pytest.fixture(scope="module")
def world():
    return make_world()


def scorer(world, **kw):
    episodes, actions, split, embeddings = world
    return ActProbeScorer(episodes, actions, embeddings, "shipped", E, seed=split.seed, **kw)


def test_it_learns_to_rank_failures_above_successes_and_records_how_it_was_selected(world):
    episodes, _, split, _ = world
    s = scorer(world, epochs=300, patience=100)
    s.fit(split.train, split.val)
    scores = s.score(list(episodes))
    assert all(v.shape == (episodes[k].n_steps,) and v.dtype == np.float32 and ((v > 0) & (v < 1)).all() for k, v in scores.items())
    unused = split.calibration + split.seen + split.unseen  # never trained on and never used to pick the checkpoint
    auroc = roc_auc_score([not episodes[k].success for k in unused], [scores[k].max() for k in unused])
    assert auroc > 0.8
    d = s.training_details()
    assert d["epochs"] == len(d["train_loss"]) and d["val_q_auc"] and len(d["val_q_auc"]) == d["epochs"] // 5
    assert d["best_val_q_auc"] <= max(d["val_q_auc"]) and max(d["val_q_auc"]) - d["best_val_q_auc"] < 1e-4 + 1e-9
    assert d["best_val_q_auc"] > 0.8 and d["val_q_auc"][0] < d["best_val_q_auc"]  # it improved over the first check
    assert d["best_epoch"] == 5 * (int(np.argmax(d["val_q_auc"])) + 1)  # a later epoch with the same q-AUC does not replace the checkpoint
    assert d["train_loss"][-1] < d["train_loss"][0] and "q-AUC" in d["selection"]["criterion"]


def test_a_validation_set_that_stops_improving_stops_the_training(world):
    _, _, split, _ = world
    s = scorer(world, epochs=400, patience=10)
    s.fit(split.train, split.val)
    d = s.training_details()
    assert d["stopped_early"] and d["epochs"] < 400 and d["epochs"] - d["best_epoch"] >= 10


def test_the_same_seed_gives_the_same_scores_and_the_weights_survive_a_round_trip(world, tmp_path):
    episodes, _, split, _ = world
    a, b = scorer(world, epochs=10, patience=10), scorer(world, epochs=10, patience=10)
    a.fit(split.train, split.val)
    b.fit(split.train, split.val)
    keys = split.seen[:6]
    assert all(np.array_equal(a.score(keys)[k], b.score(keys)[k]) for k in keys)
    a.save(str(tmp_path / "w.pt"))
    c = scorer(world, epochs=10, patience=10)
    c.load(str(tmp_path / "w.pt"))
    assert all(np.allclose(a.score(keys)[k], c.score(keys)[k], atol=1e-6) for k in keys)
    other = scorer(world, epochs=10, patience=11)  # a different configuration must not load these weights
    with pytest.raises(ValueError, match="this scorer is"):
        other.load(str(tmp_path / "w.pt"))
    assert a.state_bytes() > 22945 * 4 and a.saves_weights and a.name == "actprobe"


def test_scoring_before_fitting_is_an_error_and_the_config_records_the_embedding_source(world):
    s = scorer(world)
    with pytest.raises(RuntimeError, match="fit"):
        s.score(["f000"])
    assert s.config["embedding_source"] == "shipped" and s.config["n_params"] == 22945 and s.config["exec_horizon"] == E


# ---------------------------------------------------------------- registry, protocol and the sweep
@pytest.fixture
def registered(world, monkeypatch):
    monkeypatch.setitem(config.REGISTRY, "synthetic", dataclasses.replace(config.REGISTRY["libero10_pi05"], name="synthetic", replan_steps=E))
    episodes, actions, split, embeddings = world
    monkeypatch.setattr(actprobe, "load_task_embeddings", lambda dataset: (embeddings, "qwen3"))
    return Inputs("synthetic", episodes, None, {split.name: split}, 3), actions, split


def test_the_family_builds_one_scorer_with_the_splits_seed_and_needs_the_action_chunks(registered):
    inputs, actions, split = registered
    (s,) = build("actprobe", Context(inputs, {}, "cpu", actions), split)
    assert s.name == "actprobe" and s.seed == split.seed and s.exec_horizon == E and s.embedding_source == "qwen3"
    with pytest.raises(ValueError, match="cli.prepare_actions"):
        build("actprobe", Context(inputs, {}, "cpu"), split)
    assert calibrator_name("actprobe") == MAX_CONFORMAL


def test_a_sweep_run_stores_the_weights_and_the_history_and_can_be_evaluated_with_the_protocol(registered, tmp_path, monkeypatch):
    inputs, actions, split = registered
    original = actprobe.ActProbeScorer.__init__
    monkeypatch.setattr(actprobe.ActProbeScorer, "__init__", lambda self, *a, **k: original(self, *a, **{**k, "epochs": 10, "patience": 10}))
    unit = BaselineUnit("synthetic", "task_seed3", "actprobe")
    assert run_unit(unit, Context(inputs, {}, "cpu", actions), Settings(tmp_path), lambda m: None) == 1
    path = run_dir("synthetic", "task_seed3", "actprobe", tmp_path)
    assert sorted(p.name for p in path.iterdir()) == ["history.json", "scores.npz", "weights.pt"]
    h = read_history(path)
    assert h["family"] == "actprobe" and h["features"] == actprobe.ActProbeScorer.input_kind and h["weights_saved"] is True
    assert h["config"]["embedding_source"] == "qwen3" and h["best_epoch"] >= 5 and h["state_mb"] > 0 and h["seed"] == 3
    assert set(read_scores(path)) == set(inputs.episodes)
    assert evaluate_saved_run("synthetic", "task_seed3", "actprobe", path, inputs) == "done"
