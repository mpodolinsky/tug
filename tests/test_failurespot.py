import dataclasses

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from fp import config
from fp.baselines import failurespot as fs
from fp.baselines.failurespot import (FAMILIES, FailureSpotScorer, StepLSTM, StepMLP, action_signals, bce_step, combined_signal, ema, grid_scorers,
                                      mean_entropy, method_names, most_uncertain, soft_targets, tag, timestamp_labels)
from fp.baselines.registry import Context, build
from fp.baselines.sweep import BaselineUnit, Settings, run_unit
from fp.calibration.protocol import SAFE_BAND, calibrator_name
from fp.data.inputs import Inputs
from fp.data.manifest import Episode
from fp.data.splits import Split
from fp.eval.run_eval import evaluate_saved_run
from fp.results import read_history, read_scores, run_dir
from fp.train.items import Item, collate

E, H, A, K, D = 2, 6, 7, 3, 8


# ---------------------------------------------------------------- the signals
def test_ema_follows_the_recursion_and_starts_at_the_first_value():
    assert ema(np.array([1.0, 0.0, 0.0])).tolist() == pytest.approx([1.0, 0.8, 0.64])
    assert ema(np.array([2.0])).tolist() == [2.0]


def test_the_three_signals_by_hand():
    # one action dimension, H = 4, K = 2, so M = min(2, 2) = 2: the previous chunk's actions 2..3 meet this chunk's actions 0..1
    chunks = np.array([[[1], [2], [3], [4]], [[3], [5], [0], [0]], [[1], [1], [2], [2]]], np.float32)
    c, r, r_exec = action_signals(chunks, 2)
    assert c[1] == pytest.approx((abs(3 - 3) + abs(4 - 5)) / 2)  # 0.5
    assert c[2] == pytest.approx((abs(0 - 1) + abs(0 - 1)) / 2)  # 1.0
    assert c[0] == c[1]  # the first call takes the second's value
    assert r.tolist() == pytest.approx([2.5, 2.0, 1.5]) and r_exec.tolist() == pytest.approx([1.5, 4.0, 1.0])


def test_the_overlap_is_min_of_k_and_h_minus_k_and_norms_are_l2_over_the_dimensions():
    # H = 5, K = 3: only M = 2 actions overlap (previous 3..4 against this 0..1); two dimensions, so the distance is an L2 norm
    prev, cur = np.zeros((5, 2), np.float32), np.zeros((5, 2), np.float32)
    prev[3], prev[4] = [3, 4], [0, 5]
    cur[0], cur[1] = [0, 0], [0, 0]
    c, r, r_exec = action_signals(np.stack([prev, cur]), 3)
    assert c[1] == pytest.approx((5 + 5) / 2)
    assert r[0] == pytest.approx((0 + 0 + 0 + 5 + 5) / 5) and r_exec[0] == 0


def test_a_single_call_has_zero_consistency_and_a_bad_execution_horizon_is_an_error():
    chunks = np.random.default_rng(0).normal(size=(1, H, A)).astype(np.float32)
    assert action_signals(chunks, E)[0].tolist() == [0.0]
    for bad in (0, H):
        with pytest.raises(ValueError, match="between 0 and the chunk length"):
            action_signals(chunks, bad)


def test_the_combined_score_is_the_papers_formula():
    c, r, x = (np.array([1.0, 2.0, 0.0]), np.array([0.5, 0.5, 1.0]), np.array([3.0, 0.0, 0.0]))
    want = c + ema(c) + 2.0 * (r + ema(r)) + 4.0 * (x + ema(x))
    assert np.allclose(combined_signal((c, r, x), 2.0, 4.0), want)


def test_soft_targets_use_the_training_percentiles_only_and_clip_to_zero_one():
    train = {"a": np.linspace(0, 100, 101), "b": np.linspace(0, 100, 101)}
    other = {"v": np.array([-50.0, 50.0, 500.0])}  # not a training episode: it cannot move the percentiles
    out = soft_targets({**train, **other}, ["a", "b"])
    low, high = np.percentile(np.concatenate(list(train.values())), [1, 99])
    assert out["v"].tolist() == pytest.approx([0.0, (50 - low) / (high - low), 1.0], abs=1e-5)
    assert all(v.dtype == np.float32 and v.min() >= 0 and v.max() <= 1 for v in out.values())


def test_timestamp_labels_are_one_from_the_onset_on_in_a_failure_and_zero_in_a_success():
    assert timestamp_labels(Episode("f", False, "t", 6, 2, "a")).tolist() == [0, 0, 1, 1, 1, 1]
    assert timestamp_labels(Episode("f", False, "t", 4, 0, "a")).tolist() == [1, 1, 1, 1]
    assert timestamp_labels(Episode("s", True, "t", 5, None, "a")).tolist() == [0] * 5


# ---------------------------------------------------------------- active learning
def test_entropy_is_the_mean_binary_entropy_and_certain_probabilities_have_none():
    assert mean_entropy(np.array([0.5, 0.5])) == pytest.approx(np.log(2))
    assert mean_entropy(np.array([0.0, 1.0])) < 1e-4
    assert mean_entropy(np.array([0.5, 0.0])) == pytest.approx(np.log(2) / 2, abs=1e-4)


def test_the_most_uncertain_episodes_are_chosen_ties_by_key_and_a_large_budget_takes_all():
    p = {"c": np.array([0.5, 0.5]), "a": np.array([0.5, 0.5]), "b": np.array([0.9, 0.9]), "d": np.array([0.0, 1.0])}
    assert most_uncertain(p, 2) == ["a", "c"]  # equal entropy: by key
    assert most_uncertain(p, 3) == ["a", "c", "b"] and most_uncertain(p, 10) == ["a", "c", "b", "d"] and most_uncertain(p, 0) == []


# ---------------------------------------------------------------- the detectors and the loss
def test_the_detectors_give_one_probability_per_step_and_the_lstm_is_causal():
    torch.manual_seed(0)
    x = torch.randn(3, 7, D)
    for model in (StepMLP(D).eval(), StepLSTM(D).eval()):
        p = model(x)
        assert p.shape == (3, 7) and ((p > 0) & (p < 1)).all()
    lstm = StepLSTM(D).eval()
    assert torch.allclose(lstm(x)[:, :4], lstm(x[:, :4]), atol=1e-6)  # later steps cannot change earlier ones


def test_the_loss_is_binary_cross_entropy_pooled_over_valid_steps_with_soft_targets():
    torch.manual_seed(1)
    model = StepMLP(D).eval()
    x1, x2 = torch.randn(3, D), torch.randn(5, D)
    y1, y2 = torch.tensor([0.0, 0.25, 1.0]), torch.tensor([1.0, 0.5, 0.0, 0.0, 0.75])
    items = [Item(x1, y1, torch.full((3,), -1), -1), Item(x2, y2, torch.full((5,), -1), -1)]  # different lengths: the shorter is padded
    step = bce_step("cpu")(model, items)
    with torch.no_grad():
        p = [model(x[None])[0].clamp(1e-6, 1 - 1e-6) for x in (x1, x2)]
        per_step = torch.cat([torch.nn.functional.binary_cross_entropy(pi, yi, reduction="none") for pi, yi in zip(p, (y1, y2))])
    assert float(step.loss) == pytest.approx(float(per_step.mean()), rel=1e-5)
    total, count = step.parts["bce"]
    assert count == 8 and total == pytest.approx(float(per_step.sum()), rel=1e-5)


# ---------------------------------------------------------------- the scorer on a synthetic dataset
def make_world(n=100, seed=0):
    """Failures drift in feature space and in their action signals (plans jitter and grow); the split has 40 train, 20 val, 18 calibration, 10 seen,
    12 unseen episodes."""
    rng = np.random.default_rng(seed)
    episodes, features, actions = {}, {}, {}
    for i in range(n):
        fail = i % 2 == 0
        steps = int(rng.integers(10, 16))
        key = f"{'f' if fail else 's'}{i:03d}"
        episodes[key] = Episode(key, not fail, f"t{i % 4}", steps, int(rng.integers(3, steps - 1)) if fail else None, "a")
        drift = np.linspace(0, 3, steps)[:, None] if fail else 0
        features[key] = (rng.normal(size=(steps, D)) + drift).astype(np.float32)
        base = 0.1 * rng.normal(size=(1, 1, H, A))
        chunks = base * np.ones((steps, K, H, A)) + 0.02 * rng.normal(size=(steps, K, H, A))
        if fail:
            chunks = chunks + np.linspace(0.0, 0.6, steps)[:, None, None, None] * rng.normal(size=chunks.shape)
        actions[key] = chunks.astype(np.float32)
    keys = list(episodes)
    return episodes, features, actions, Split("task_seed3", keys[:40], keys[40:60], keys[60:78], keys[78:88], keys[88:], seed=3)


@pytest.fixture(scope="module")
def world():
    return make_world()


def scorer(world, kind="mlp", eta1=1.0, eta2=1.0, **kw):
    episodes, features, actions, split = world
    n_seen = len(split.train) + len(split.val) + len(split.calibration) + len(split.seen)
    return FailureSpotScorer(kind, eta1, eta2, episodes, features, actions, E, n_seen, seed=split.seed, **kw)


@pytest.fixture
def fits(monkeypatch):
    """Records every call of the training loop: (number of training items, number of validation items, the targets of both)."""
    calls = []
    original = fs.fit

    def recording(model, train, val, step_fn, cfg, select="score"):
        calls.append(dict(train=[it.y.clone() for it in train], val=[it.y.clone() for it in val], select=select))
        return original(model, train, val, step_fn, cfg, select)

    monkeypatch.setattr(fs, "fit", recording)
    return calls


@pytest.mark.parametrize("kind", ["mlp", "lstm"])
def test_the_two_stages_train_on_what_the_paper_says(world, fits, kind):
    episodes, _, _, split = world
    s = scorer(world, kind, 0.25, 4.0, max_epochs=4, patience=2)
    s.fit(split.train, split.val)
    pretrain, finetune = fits
    n_seen = len(split.train) + len(split.val) + len(split.calibration) + len(split.seen)
    # stage 1: every training episode, with soft targets in [0, 1] (not all 0/1), selected on the validation soft loss
    assert len(pretrain["train"]) == len(split.train) and len(pretrain["val"]) == len(split.val) and pretrain["select"] == "bce"
    soft = torch.cat(pretrain["train"])
    assert soft.min() >= 0 and soft.max() <= 1 and ((soft > 0) & (soft < 1)).any()
    # stage 2: the 15% most uncertain training episodes, with 0/1 onset labels; the validation episodes carry hard labels too
    chosen = s.result["annotated"]
    assert len(chosen) == round(0.15 * n_seen) == len(finetune["train"]) and set(chosen) <= set(split.train) and len(set(chosen)) == len(chosen)
    for key, y in zip(chosen, finetune["train"]):
        assert np.array_equal(y.numpy(), timestamp_labels(episodes[key]))
    for key, y in zip(split.val, finetune["val"]):
        assert np.array_equal(y.numpy(), timestamp_labels(episodes[key]))
    assert s.result["annotated_failures"] == sum(not episodes[k].success for k in chosen)


def test_the_chosen_episodes_are_the_ones_the_pretrained_detector_is_least_sure_about(world, fits):
    _, _, _, split = world
    s = scorer(world, max_epochs=4, patience=2)
    probabilities = {}
    original = s._predict
    s._predict = lambda keys, batch=fs.BATCH: probabilities.setdefault("p", original(keys, batch))  # the stage-1 probabilities of the train episodes
    s.fit(split.train, split.val)
    expected = fs.most_uncertain(probabilities["p"], len(s.result["annotated"]))
    assert s.result["annotated"] == expected


def test_it_learns_to_rank_failures_above_successes_on_episodes_it_never_saw(world):
    episodes, _, _, split = world
    s = scorer(world, "lstm", 1.0, 1.0, max_epochs=60, patience=15)
    s.fit(split.train, split.val)
    unused = split.calibration + split.seen + split.unseen
    scores = s.score(unused)
    assert all(scores[k].shape == (episodes[k].n_steps,) and scores[k].dtype == np.float32 and ((scores[k] > 0) & (scores[k] < 1)).all() for k in unused)
    assert roc_auc_score([not episodes[k].success for k in unused], [scores[k].max() for k in unused]) > 0.75
    d = s.training_details()
    assert d["pretrain"]["epochs"] == len(d["pretrain"]["val_bce"]) and d["finetune"]["best_epoch"] < d["finetune"]["epochs"]
    assert d["n_annotated"] == len(d["annotated"]) and "not stated in the paper" in d["selection"]["paper"]
    assert min(d["pretrain"]["val_bce"]) == d["pretrain"]["val_bce"][d["pretrain"]["best_epoch"]]  # the checkpoint is the best validation epoch


def test_the_same_seed_gives_the_same_scores_and_weights_round_trip(world, tmp_path):
    _, _, _, split = world
    a, b = scorer(world, max_epochs=5, patience=3), scorer(world, max_epochs=5, patience=3)
    a.fit(split.train, split.val)
    b.fit(split.train, split.val)
    keys = split.seen[:5]
    assert all(np.array_equal(a.score(keys)[k], b.score(keys)[k]) for k in keys)
    a.save(str(tmp_path / "w.pt"))
    c = scorer(world, max_epochs=5, patience=3)
    c.load(str(tmp_path / "w.pt"))
    assert all(np.allclose(a.score(keys)[k], c.score(keys)[k], atol=1e-6) for k in keys)
    with pytest.raises(ValueError, match="this scorer is"):
        scorer(world, eta1=4.0, max_epochs=5, patience=3).load(str(tmp_path / "w.pt"))
    assert a.state_bytes() > 0 and a.saves_weights


def test_scoring_before_fitting_is_an_error_and_bad_arguments_are_rejected(world):
    with pytest.raises(RuntimeError, match="fit"):
        scorer(world).score(["f000"])
    with pytest.raises(ValueError, match="kind must be"):
        scorer(world, kind="cnn")


# ---------------------------------------------------------------- the grid, the protocol and the sweep
def test_each_detector_has_nine_uniquely_named_methods_in_grid_order_and_the_protocol_gives_them_the_safe_band():
    names = {kind: method_names(kind) for kind in fs.KINDS}
    assert names["mlp"][:4] == ["failurespot_mlp__eta1_0p25_eta2_0p25", "failurespot_mlp__eta1_0p25_eta2_1", "failurespot_mlp__eta1_0p25_eta2_4",
                                "failurespot_mlp__eta1_1_eta2_0p25"]
    assert all(len(v) == 9 for v in names.values()) and len(set(names["mlp"] + names["lstm"])) == 18
    assert all(calibrator_name(n) == SAFE_BAND for v in names.values() for n in v)
    for wrong in ("failurespot_mlp__eta1_2_eta2_1", "failurespot_cnn__eta1_1_eta2_1", "failurespot__eta1_1_eta2_1", "failurespot_mlp__eta1_1"):
        with pytest.raises(KeyError):
            calibrator_name(wrong)
    assert tag(0.25) == "0p25" and tag(1.0) == "1" and tag(4.0) == "4"


@pytest.fixture
def registered(world, monkeypatch):
    monkeypatch.setitem(config.REGISTRY, "synthetic", dataclasses.replace(config.REGISTRY["libero10_pi05"], name="synthetic", replan_steps=E))
    episodes, features, actions, split = world
    return Inputs("synthetic", episodes, None, {split.name: split}, 3), features, actions, split


def test_the_families_build_nine_scorers_with_the_seen_task_count_and_need_the_action_chunks(registered):
    inputs, features, actions, split = registered
    assert set(FAMILIES) == {"failurespot_mlp", "failurespot_lstm"}
    scorers = build("failurespot_lstm", Context(inputs, features, "cpu", actions), split)
    assert [s.name for s in scorers] == method_names("lstm") and all(s.n_seen == 88 and s.seed == split.seed and s.exec_horizon == E for s in scorers)
    with pytest.raises(ValueError, match="cli.prepare_actions"):
        build("failurespot_mlp", Context(inputs, features, "cpu"), split)


def test_a_sweep_run_stores_weights_and_a_history_and_can_be_evaluated_with_the_protocol(registered, tmp_path, monkeypatch):
    inputs, features, actions, split = registered
    original = fs.FailureSpotScorer.__init__
    monkeypatch.setattr(fs.FailureSpotScorer, "__init__", lambda self, *a, **k: original(self, *a, **{**k, "max_epochs": 4, "patience": 2}))
    unit = BaselineUnit("synthetic", "task_seed3", "failurespot_mlp")
    method = "failurespot_mlp__eta1_1_eta2_1"
    assert run_unit(unit, Context(inputs, features, "cpu", actions), Settings(tmp_path), lambda m: None, [method]) == 1
    path = run_dir("synthetic", "task_seed3", method, tmp_path)
    assert sorted(p.name for p in path.iterdir()) == ["history.json", "scores.npz", "weights.pt"]
    h = read_history(path)
    assert h["family"] == "failurespot_mlp" and h["weights_saved"] is True and h["config"]["eta1"] == 1.0 and h["n_annotated"] == 13 and h["seed"] == 3
    assert h["finetune"]["val_bce"] and h["pretrain"]["val_bce"] and h["state_mb"] > 0
    assert set(read_scores(path)) == set(inputs.episodes)
    assert evaluate_saved_run("synthetic", "task_seed3", method, path, inputs) == "done"
