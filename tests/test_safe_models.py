"""SAFE-MLP and SAFE-LSTM: the networks, SAFE's losses, the scorer, the grid, and a comparison with SAFE's own code."""

import importlib.util
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from fp.baselines import sweep
from fp.baselines.registry import FAMILIES, Context, build, register
from fp.baselines.safe_grid import FAMILIES as SAFE_FAMILIES
from fp.baselines.safe_grid import LAMBDA_REGS, LEARNING_RATES, grid_configs, method_names
from fp.baselines.sweep import BaselineUnit, Settings, run_unit
from fp.calibration.protocol import SAFE_BAND, calibrator_name
from fp.data.inputs import Inputs
from fp.data.manifest import Episode
from fp.data.splits import Split
from fp.eval.collect import family_of
from fp.eval.run_eval import evaluate_saved_run
from fp.models.safe import SafeLSTM, SafeMLP
from fp.results import read_history, run_dir
from fp.train.safe import class_weights, l2_penalty, monitor_terms, step_losses, tag, SafeScorer

D = 8


# ---------------------------------------------------------------- the networks
@pytest.mark.parametrize("net", [SafeMLP, SafeLSTM])
def test_the_networks_are_causal_and_ignore_padding(net):
    torch.manual_seed(0)
    m = net(D, 16).eval()
    x = torch.randn(3, 12, D)
    out = m(x)
    assert out.shape == (3, 12)
    changed = x.clone()
    changed[:, 7:] += 5.0  # change the future
    assert torch.allclose(m(changed)[:, :7], out[:, :7], atol=1e-6) and not torch.allclose(m(changed)[:, 7:], out[:, 7:])
    padded = torch.cat([x, torch.zeros(3, 5, D)], dim=1)  # zeros appended after the real steps change nothing before them
    assert torch.allclose(m(padded)[:, :12], out, atol=1e-6)


def test_the_mlp_score_is_the_running_sum_of_its_step_probabilities_and_the_lstm_score_a_probability():
    torch.manual_seed(1)
    mlp, lstm = SafeMLP(D, 16).eval(), SafeLSTM(D, 16).eval()
    x = torch.randn(2, 9, D)
    p = mlp.step_probabilities(x)
    assert ((p > 0) & (p < 1)).all() and torch.allclose(mlp(x), torch.cumsum(p, dim=1)) and (mlp(x).diff(dim=1) > 0).all()  # a sum of positive terms rises
    assert ((lstm(x) > 0) & (lstm(x) < 1)).all()


def test_the_parameter_names_are_safes():
    assert [n for n, _ in SafeMLP(D, 16).named_parameters()] == ["projector.0.weight", "projector.0.bias", "projector.2.weight", "projector.2.bias"]
    assert {n for n, _ in SafeLSTM(D, 16).named_parameters()} == {"lstm.weight_ih_l0", "lstm.weight_hh_l0", "lstm.bias_ih_l0", "lstm.bias_hh_l0", "fc.weight", "fc.bias"}
    assert sum(p.numel() for p in SafeMLP(6144, 256).parameters()) == 6144 * 256 + 256 + 256 + 1  # the sizes of the real runs


# ---------------------------------------------------------------- SAFE's losses, by hand
def test_class_weights_are_one_over_the_smoothed_class_frequency():
    w_fail, w_succ = class_weights([True, True] + [False] * 6)  # 2 failures, 6 successes, n = 8
    assert w_fail == pytest.approx(1 / (3 / 8)) and w_succ == pytest.approx(1 / (7 / 8))


def test_mlp_step_losses_push_successes_down_and_failures_up():
    scores = torch.tensor([[0.5, 1.0], [0.5, 1.0]])
    loss = step_losses("mlp", scores, torch.zeros(2, 2), torch.tensor([False, True]))
    assert loss.tolist() == [[0.5, 1.0], [-0.5, -1.0]]  # success: relu(score); failure: -score


def test_lstm_step_losses_are_the_binary_cross_entropy_of_the_step_probability():
    p, y = torch.tensor([[0.8, 0.8]]), torch.tensor([[1.0, 0.0]])
    loss = step_losses("lstm", p, y, torch.tensor([True]))
    assert loss[0, 0].item() == pytest.approx(-math.log(0.8)) and loss[0, 1].item() == pytest.approx(-math.log(0.2))


def test_the_monitor_loss_is_the_class_weighted_sum_of_the_episode_means_over_the_number_of_episodes():
    # episode 0: a failure of 4 steps, episode 1: a success of 2 steps padded to 4. MLP cumulative scores below.
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 9.0, 9.0]])
    y = torch.tensor([[1.0, 1, 1, 1], [0.0, 0, 0, 0]])
    valid = torch.tensor([[1.0, 1, 1, 1], [1.0, 1, 0, 0]])
    num, den = monitor_terms("mlp", scores, y, valid, (3.0, 2.0))
    # failure mean(-score) = -2.5, weight 3 -> -7.5; success mean(relu) over the 2 valid steps = 1.0, weight 2 -> 2.0; padding is ignored
    assert den == 2 and float(num) == pytest.approx(-7.5 + 2.0)


def test_the_l2_penalty_counts_every_weight_but_no_bias():
    net = SafeMLP(3, 4)
    expected = sum(float((p ** 2).sum()) for n, p in net.named_parameters() if n.endswith("weight"))
    assert float(l2_penalty(net)) == pytest.approx(expected)
    lstm = SafeLSTM(3, 4)  # an LSTM has bias_ih and bias_hh: both are biases
    assert float(l2_penalty(lstm)) == pytest.approx(sum(float((p ** 2).sum()) for n, p in lstm.named_parameters() if "bias" not in n))


def test_learning_rates_and_penalties_are_written_in_a_short_form():
    assert [tag(v) for v in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2, 1e-1)] == ["1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "1e-2", "1e-1"]


# ---------------------------------------------------------------- the grid and the protocol
def test_each_family_has_safes_twelve_configurations_in_a_fixed_order():
    assert LEARNING_RATES == {"mlp": (1e-5, 1e-4, 3e-4, 1e-3), "lstm": (3e-5, 1e-4, 3e-4, 1e-3)} and LAMBDA_REGS == (1e-3, 1e-2, 1e-1)
    assert len(grid_configs("mlp")) == len(grid_configs("lstm")) == 12
    names = method_names("safe_lstm")
    assert names[:3] == ["safe_lstm__lr3e-5_reg1e-3", "safe_lstm__lr3e-5_reg1e-2", "safe_lstm__lr3e-5_reg1e-1"] and names[-1] == "safe_lstm__lr1e-3_reg1e-1"
    assert len(set(names)) == 12 and sorted(names) != names  # the grid order is not the alphabetical one, so it must be kept explicitly


def test_the_four_families_are_registered_and_all_48_names_get_the_safe_band():
    assert set(SAFE_FAMILIES) == {"safe_mlp", "safe_lstm", "safe_mlp_pre", "safe_lstm_pre"} and set(SAFE_FAMILIES) <= set(FAMILIES)
    for family in SAFE_FAMILIES:
        names = method_names(family)
        assert len(names) == 12 and {calibrator_name(n) for n in names} == {SAFE_BAND} and {family_of(n) for n in names} == {family}
    with pytest.raises(KeyError):
        calibrator_name("safe_lstm__lr9e-9x_reg1e-2")


def test_the_pre_onset_families_are_not_named_like_safe_as_published():
    assert not set(method_names("safe_mlp")) & set(method_names("safe_mlp_pre")) and method_names("safe_lstm_pre")[0].startswith("safe_lstm_pre__")


# ---------------------------------------------------------------- the scorer on a synthetic dataset
@pytest.fixture(scope="module")
def world():
    """100 episodes: failures drift away from the success distribution; every part has both outcomes (the calibration part is large enough
    for the SAFE band)."""
    rng = np.random.default_rng(0)
    episodes, features = {}, {}
    for i in range(100):
        fail = i % 2 == 0
        n = int(rng.integers(12, 20))
        key = f"{'f' if fail else 's'}{i:02d}"
        episodes[key] = Episode(key, not fail, f"t{i % 4}", n, int(rng.integers(6, n - 1)) if fail else None, "a")
        features[key] = (rng.normal(size=(n, D)) + (np.linspace(0, 3, n)[:, None] if fail else 0)).astype(np.float32)
    keys = list(episodes)
    split = Split("task_seed3", keys[:40], keys[40:48], keys[48:78], keys[78:88], keys[88:], seed=3)
    return Inputs("synthetic", episodes, None, {split.name: split}, 3), features, split


def scorer(world, kind="mlp", labels="safe", **kw):
    inputs, features, split = world
    return SafeScorer(kind, inputs.episodes, features, 3, lr=kw.pop("lr", 3e-3), lambda_reg=kw.pop("lambda_reg", 1e-3), labels=labels,
                      max_epochs=kw.pop("max_epochs", 60), patience=kw.pop("patience", 8), seed=kw.pop("seed", 3), **kw)


@pytest.mark.parametrize("kind", ["mlp", "lstm"])
def test_a_trained_scorer_separates_held_out_failures_from_successes(world, kind):
    inputs, _, split = world
    s = scorer(world, kind)
    s.fit(split.train, split.val)
    scores = s.score(split.unseen)
    assert set(scores) == set(split.unseen) and all(v.shape == (inputs.episodes[k].n_steps,) and np.isfinite(v).all() for k, v in scores.items())
    final = [scores[k][-1] for k in split.unseen]
    assert roc_auc_score([not inputs.episodes[k].success for k in split.unseen], final) > 0.9


def test_early_stopping_follows_the_validation_data_loss_and_keeps_the_best_epoch(world):
    _, _, split = world
    s = scorer(world, "lstm", lr=1e-2, lambda_reg=1e-1, max_epochs=200, patience=5)
    s.fit(split.train, split.val)
    r = s.result
    assert r.stopped_early and len(r.val["data"]) < 200 and r.best_epoch == int(np.argmin(r.val["data"]))  # decided by the data loss
    assert set(r.val) == {"data", "l2"} and len(r.val["data"]) == len(r.val["l2"]) == len(r.train["data"])  # the penalty is logged apart


def test_the_recorded_training_details_state_the_deviation_from_safe(world):
    s = scorer(world, "mlp", patience=4)
    assert s.training_details() is None  # not trained
    s.fit(world[2].train, world[2].val)
    d = s.training_details()
    assert d["early_stopping"]["patience"] == 4 and "1000 epochs" in d["early_stopping"]["safe"] and "without the L2 penalty" in d["early_stopping"]["criterion"]
    assert d["label_variant"] == "onset_traj" and d["epochs"] == len(d["val"]["data"]) and d["best_epoch"] < d["epochs"]
    assert s.config["patience"] == 4 and s.config["hidden"] == 256 and s.config["batch"] == 512 and s.config["max_epochs"] == 60


def test_the_defaults_are_safes_batch_and_hidden_size_and_the_user_chosen_patience():
    s = SafeScorer("lstm", {}, {}, 3, lr=3e-4, lambda_reg=1e-2)
    assert (s.batch, s.max_epochs, s.patience, s.config["hidden"]) == (512, 1000, 50, 256) and s.name == "safe_lstm__lr3e-4_reg1e-2"
    assert SafeScorer("mlp", {}, {}, 3, lr=1e-3, lambda_reg=1e-1, labels="pre").name == "safe_mlp_pre__lr1e-3_reg1e-1"


@pytest.mark.parametrize("kind", ["mlp", "lstm"])
def test_running_the_batch_in_chunks_gives_the_same_update_as_the_whole_batch_of_safe(world, kind):
    """SAFE divides its monitor loss by the batch size, so chunks weighted by their share (40 episodes in chunks of 7: five of 7 and one of 5)
    accumulate exactly the gradient of the whole batch; it saves memory and changes nothing else."""
    _, _, split = world
    whole = scorer(world, kind, max_epochs=3, patience=3, batch=40, micro_batch=None)
    chunks = scorer(world, kind, max_epochs=3, patience=3, batch=40, micro_batch=7)
    for s in (whole, chunks):
        s.fit(split.train, split.val)
    for (name, a), (_, b) in zip(whole.model.state_dict().items(), chunks.model.state_dict().items()):
        assert torch.allclose(a, b, atol=1e-5), name
    assert whole.config["micro_batch"] is None and chunks.config["micro_batch"] == 7


def test_the_default_runs_a_batch_of_512_in_chunks_of_128():
    s = SafeScorer("lstm", {}, {}, 3, lr=3e-4, lambda_reg=1e-2)
    assert (s.batch, s.micro_batch, s.config["micro_batch"]) == (512, 128, 128)


def test_the_same_seed_gives_the_same_scores_and_another_seed_does_not(world):
    _, _, split = world
    a, b, c = scorer(world, seed=3), scorer(world, seed=3), scorer(world, seed=4)
    for s in (a, b, c):
        s.fit(split.train, split.val)
    sa, sb, sc = (s.score(split.val) for s in (a, b, c))
    assert all(np.array_equal(sa[k], sb[k]) for k in sa) and not all(np.allclose(sa[k], sc[k]) for k in sa)


def test_saving_and_loading_reproduces_the_scores_and_refuses_another_configuration(world, tmp_path):
    _, _, split = world
    s = scorer(world, "lstm")
    s.fit(split.train, split.val)
    s.save(tmp_path / "w.pt")
    loaded = scorer(world, "lstm")
    loaded.load(tmp_path / "w.pt")
    a, b = s.score(split.val), loaded.score(split.val)
    assert all(np.array_equal(a[k], b[k]) for k in a) and s.state_bytes() == sum(p.numel() * 4 for p in s.model.parameters()) and s.saves_weights
    with pytest.raises(ValueError, match="this scorer is"):
        scorer(world, "lstm", lr=1e-2).load(tmp_path / "w.pt")


def test_scoring_before_fitting_and_unknown_options_are_errors(world):
    with pytest.raises(RuntimeError):
        scorer(world).score(world[2].val)
    with pytest.raises(ValueError):
        SafeScorer("gru", {}, {}, 3, lr=1e-3, lambda_reg=1e-3)
    with pytest.raises(ValueError):
        SafeScorer("mlp", {}, {}, 3, lr=1e-3, lambda_reg=1e-3, labels="onset")


def test_safes_own_labels_keep_failed_episodes_whole_and_the_pre_onset_variant_cuts_them_at_the_onset(world):
    inputs, _, split = world
    failed = [k for k in split.train if not inputs.episodes[k].success][:5]
    whole = {k: it for k, it in zip(failed, scorer(world, labels="safe")._items(failed))}
    cut = {k: it for k, it in zip(failed, scorer(world, labels="pre")._items(failed))}
    for k in failed:
        e = inputs.episodes[k]
        assert len(whole[k].y) == e.n_steps and whole[k].y.min() == 1  # SAFE: every step of a failed rollout is positive
        assert len(cut[k].y) == e.onset + 1 and cut[k].y.min() == 1  # the same loss on the rollout up to the onset
    succ = [k for k in split.train if inputs.episodes[k].success][:3]
    assert all(len(it.y) == inputs.episodes[k].n_steps and it.y.max() == 0 for k, it in zip(succ, scorer(world, labels="pre")._items(succ)))


def test_the_pre_onset_variant_trains_on_less_data_than_safes_labels_but_scores_whole_rollouts(world):
    inputs, _, split = world
    s = scorer(world, labels="pre")
    s.fit(split.train, split.val)
    scores = s.score(split.unseen)
    assert all(len(scores[k]) == inputs.episodes[k].n_steps for k in split.unseen)  # scored over the full rollout, like every method


# ---------------------------------------------------------------- through the baseline driver, evaluation and the report
def test_a_safe_run_has_its_weights_its_training_history_and_is_evaluated_with_the_band(world, tmp_path):
    inputs, features, split = world
    register("safe_test")(lambda ctx, sp: [SafeScorer("mlp", ctx.inputs.episodes, ctx.features, ctx.inputs.window, lr=3e-3, lambda_reg=1e-2,
                                                      max_epochs=30, patience=5, seed=sp.seed, device=ctx.device)])
    try:
        context = Context(inputs, features, "cpu")
        assert run_unit(BaselineUnit("synthetic", "task_seed3", "safe_test"), context, Settings(tmp_path), lambda m: None) == 1
    finally:
        del FAMILIES["safe_test"]
    path = run_dir("synthetic", "task_seed3", "safe_mlp__lr3e-3_reg1e-2", tmp_path)
    assert sorted(p.name for p in path.iterdir()) == ["history.json", "scores.npz", "weights.pt"]
    h = read_history(path)
    assert h["weights_saved"] is True and h["features"] == "raw" and h["seed"] == 3 and h["config"]["lr"] == 3e-3
    assert h["early_stopping"]["patience"] == 5 and h["epochs"] == len(h["val"]["data"]) and h["best_epoch"] < h["epochs"] and h["state_mb"] > 0
    assert evaluate_saved_run("synthetic", "task_seed3", "safe_mlp__lr3e-3_reg1e-2", path, inputs) == "done"


# ---------------------------------------------------------------- against SAFE's own code
def find_safe_root():
    if os.environ.get("SAFE_ROOT"):
        return Path(os.environ["SAFE_ROOT"])
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "baselines/SAFE/failure_prob/model/indep.py").exists():
            return p / "baselines/SAFE"
    return here.parent / "missing"


SAFE_ROOT = find_safe_root()


def load_safe_models(monkeypatch):
    """Execute SAFE's base.py, indep.py and lstm.py (and its real utils.py) with stand-ins for what pulls in hydra, wandb and the data stack."""
    pkg, model_pkg = types.ModuleType("failure_prob"), types.ModuleType("failure_prob.model")
    pkg.__path__, model_pkg.__path__ = [], [str(SAFE_ROOT / "failure_prob/model")]
    stubs = {"failure_prob": pkg, "failure_prob.model": model_pkg, "wandb": types.ModuleType("wandb"),
             "failure_prob.data": types.ModuleType("failure_prob.data"), "failure_prob.data.utils": types.ModuleType("failure_prob.data.utils"),
             "failure_prob.conf": types.ModuleType("failure_prob.conf"), "failure_prob.utils": types.ModuleType("failure_prob.utils"),
             "failure_prob.utils.torch": types.ModuleType("failure_prob.utils.torch")}
    stubs["failure_prob.data"].__path__ = stubs["failure_prob.utils"].__path__ = []
    stubs["failure_prob.data.utils"].Rollout = object
    stubs["failure_prob.conf"].Config = object
    stubs["failure_prob.utils.torch"].move_to_device = lambda batch, device: batch
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)
    loaded = {}
    for name in ("base", "utils", "indep", "lstm"):
        spec = importlib.util.spec_from_file_location(f"failure_prob.model.{name}", SAFE_ROOT / f"failure_prob/model/{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, f"failure_prob.model.{name}", module)
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


def safe_cfg(**overrides):
    model = dict(n_history_steps=1, hidden_dim=256, n_layers=2, final_act_layer="sigmoid", cumsum=True, rmean=False, use_time_weighting=False,
                 use_threshold=False, threshold=50, dropout=0.0, init_weight_scale=1.0, one_loss_per_seq=False, lambda_hard_heg=0.0,
                 hard_neg_margin=0.1, hard_neg_beta=50.0, lambda_reg=0.0)
    model.update(overrides)
    return types.SimpleNamespace(model=types.SimpleNamespace(**model))


def safe_batch(seed=0, B=6, T=11, dim=16):
    """Padded episodes of different lengths; SAFE's labels: 1 = success, 0 = failure."""
    g = torch.Generator().manual_seed(seed)
    lengths = torch.tensor([11, 7, 11, 4, 9, 11])[:B]
    valid = (torch.arange(T)[None, :] < lengths[:, None]).float()
    x = torch.randn(B, T, dim, generator=g) * valid[:, :, None]
    success = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])[:B]
    return {"features": x, "valid_masks": valid, "success_labels": success}


needs_safe = pytest.mark.skipif(not (SAFE_ROOT / "failure_prob/model/indep.py").exists(), reason="SAFE checkout not found (set SAFE_ROOT)")


@needs_safe
@pytest.mark.parametrize("kind", ["mlp", "lstm"])
def test_outputs_losses_and_penalty_equal_those_of_safes_own_models(monkeypatch, kind):
    modules = load_safe_models(monkeypatch)
    batch = safe_batch()
    torch.manual_seed(0)
    if kind == "mlp":
        theirs, mine = modules["indep"].IndepModel(safe_cfg(), 16), SafeMLP(16, 256)
    else:
        theirs, mine = modules["lstm"].LstmModel(safe_cfg(n_history_steps=-1, n_layers=1, cumsum=False), 16), SafeLSTM(16, 256)
    mine.load_state_dict(theirs.state_dict())  # strict: the parameter names are SAFE's
    theirs.eval()
    mine.eval()
    with torch.no_grad():
        their_scores, my_scores = theirs(batch).squeeze(-1), mine(batch["features"])
        assert torch.allclose(their_scores, my_scores, atol=1e-6)
        weights = [2.5, 1.3]  # SAFE's order: (failure, success)
        their_loss, _ = theirs.forward_compute_loss(batch, weights)
        valid, failure = batch["valid_masks"], (1 - batch["success_labels"])
        y = failure[:, None] * valid
        num, den = monitor_terms(kind, my_scores, y, valid, (weights[0], weights[1]))
        assert float(their_loss) == pytest.approx(float(num / den), rel=1e-5)
    their_reg, _ = theirs.compute_regularization_loss(0.37)
    assert float(their_reg) == pytest.approx(0.37 * float(l2_penalty(mine)), rel=1e-5)


@needs_safe
def test_class_weights_equal_those_of_safes_dataset_class(monkeypatch):
    """SAFE: weights = [1 / freq_fail, 1 / freq_success] with freq = (count + 1) / n, read from failure_prob/data/utils.py."""
    source = (SAFE_ROOT / "failure_prob/data/utils.py").read_text()
    assert "freq_0 = (sum([r.episode_success == 0 for r in rollouts]) + 1) / len(rollouts)" in source
    assert "freq_1 = (sum([r.episode_success == 1 for r in rollouts]) + 1) / len(rollouts)" in source
    assert "self.weights = [1./(freq_0), 1./(freq_1)]" in source
    w = class_weights([True] * 13 + [False] * 29)
    assert w == (1.0 / (14 / 42), 1.0 / (30 / 42))
