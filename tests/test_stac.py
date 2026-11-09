"""STAC and STAC-Single on synthetic action chunks, and against the two reference implementations: SAFE's re-implementation
(failure_prob/utils/failure_metrics.py, SAFE_ROOT) and the authors' code (sentinel, STAC_ROOT). The comparisons are skipped when a checkout is absent."""

import dataclasses
import importlib.util
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from fp import config
from fp.baselines import stac
from fp.baselines.registry import FAMILIES, Context, build
from fp.baselines.stac import StacScorer, grid_scorers, method_names, mmd_rbf, overlap, stac_scores, stac_single_scores
from fp.calibration.protocol import MAX_CONFORMAL, calibrator_name
from fp.data.inputs import Inputs
from fp.data.manifest import Episode
from fp.data.splits import Split

T, K, H, A, E = 7, 4, 6, 7, 2  # calls, samples, chunk length, action dimensions, execution horizon


def chunks(seed=0, t=T):
    return np.random.default_rng(seed).normal(size=(t, K, H, A)).astype(np.float32)


def consistent(seed=0):
    """Chunks in which every call repeats, on the overlap, exactly what the previous call had planned for those steps."""
    x = chunks(seed)
    for t in range(1, T):
        x[t, :, : H - E] = x[t - 1, :, E:]
    return x


# ---------------------------------------------------------------- the overlap and the MMD
def test_the_overlap_pairs_the_unexecuted_rest_of_a_call_with_the_start_of_the_next_one():
    x = chunks()
    old, new = overlap(x, E)
    assert old.shape == new.shape == (T - 1, K, H - E, A)
    assert np.array_equal(old[2], x[2, :, E:]) and np.array_equal(new[2], x[3, :, : H - E])


def test_chunks_that_are_not_longer_than_the_execution_horizon_have_no_overlap():
    with pytest.raises(ValueError, match="must be longer"):
        overlap(chunks()[:, :, :E], E)


def test_mmd_is_the_textbook_sum_of_mean_kernels_written_as_loops():
    old, new = overlap(chunks(1), E)
    k = lambda a, b: np.exp(-np.sum((a.astype(np.float64) - b) ** 2))
    for t in range(T - 1):
        x, y = old[t].reshape(K, -1), new[t].reshape(K, -1)
        want = (np.mean([k(a, b) for a, b in itertools.product(x, x)]) + np.mean([k(a, b) for a, b in itertools.product(y, y)])
                - 2 * np.mean([k(a, b) for a, b in itertools.product(x, y)]))
        assert mmd_rbf(old, new)[t] == pytest.approx(want, abs=1e-12)


def test_plans_that_repeat_what_was_planned_give_zero_and_the_order_of_the_samples_does_not_matter():
    x = consistent()
    assert np.allclose(stac_scores(x, E), 0, atol=1e-7)
    shuffled = x.copy()
    for t in range(1, T):
        shuffled[t] = shuffled[t][np.random.default_rng(t).permutation(K)]  # the same set of plans, in another order
    assert np.allclose(stac_scores(shuffled, E), 0, atol=1e-7)


def test_the_first_call_scores_zero_there_is_one_score_per_call_and_the_value_is_not_negative():
    s = stac_scores(chunks(), E)
    assert s.shape == (T,) and s.dtype == np.float32 and s[0] == 0 and (s[1:] > 0).all()
    assert stac_single_scores(chunks()[:, 0], E)[0] == 0


def test_an_episode_with_a_single_policy_call_has_no_pair_of_calls_and_scores_zero():
    one = chunks(t=1)
    assert np.array_equal(stac_scores(one, E), [0.0]) and np.array_equal(stac_single_scores(one[:, 0], E), [0.0])
    assert StacScorer("stac", {"a": one}, E, cumsum=True).score(["a"])["a"].tolist() == [0.0]


def test_the_further_apart_the_plans_the_larger_the_score():
    near, far = consistent(), consistent()
    near[3:] += 0.05
    far[3:] += 0.5
    assert stac_scores(near, E)[3] < stac_scores(far, E)[3]
    assert stac_single_scores(near[:, 0], E)[3] < stac_single_scores(far[:, 0], E)[3]


def test_a_dimension_a_variant_drops_cannot_change_its_score_and_one_it_keeps_does():
    x = 0.1 * chunks()  # close enough together that the kernel does not saturate
    noise = np.random.default_rng(9).normal(size=x.shape[:-1]).astype(np.float32)  # (a constant shift would cancel: the MMD ignores translations)
    gripper, rotation = x.copy(), x.copy()
    gripper[..., 6] += noise
    rotation[..., 3:6] += noise[..., None]
    assert np.array_equal(stac_scores(x, E, 6), stac_scores(gripper, E, 6)) and np.array_equal(stac_scores(x, E, 3), stac_scores(gripper, E, 3))
    assert not np.allclose(stac_scores(x, E, 7), stac_scores(gripper, E, 7))
    assert np.array_equal(stac_scores(x, E, 3), stac_scores(rotation, E, 3)) and not np.allclose(stac_scores(x, E, 6), stac_scores(rotation, E, 6))


def test_stac_single_is_the_mean_over_the_overlap_of_the_l2_norm_over_the_dimensions():
    x = chunks(2)[:, 0]
    got = stac_single_scores(x, E)
    for t in range(1, T):
        diffs = x[t - 1, E:] - x[t, : H - E]  # (H - E, A)
        assert got[t] == pytest.approx(np.linalg.norm(diffs, axis=1).mean(), rel=1e-6)


# ---------------------------------------------------------------- against SAFE's and the authors' code
MISSING = Path(__file__).parent / "missing"
SAFE_FILE = Path(os.environ["SAFE_ROOT"]) / "failure_prob/utils/failure_metrics.py" if os.environ.get("SAFE_ROOT") else MISSING
STAC_SENTINEL = Path(os.environ["STAC_ROOT"]) if os.environ.get("STAC_ROOT") else MISSING

needs_safe = pytest.mark.skipif(not SAFE_FILE.exists(), reason="SAFE checkout not found (set SAFE_ROOT)")
needs_authors = pytest.mark.skipif(not (STAC_SENTINEL / "sentinel").exists(), reason="STAC checkout not found (set STAC_ROOT to its sentinel folder)")


def safe_metrics():
    # loaded by path: SAFE's utils folder holds a torch.py that would shadow PyTorch on sys.path
    spec = importlib.util.spec_from_file_location("safe_failure_metrics", SAFE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_safe
def test_the_all_dimension_stac_and_stac_single_equal_safes_reimplementation():
    import torch

    safe = safe_metrics()
    x = chunks(3)
    mine, theirs = stac_scores(x, E, 7), safe.compute_stac_metrics(torch.from_numpy(x), exec_horizon=E, metric="mmd", rbf_beta=1.0).numpy()
    assert np.allclose(mine, theirs, atol=1e-6) and mine.std() > 0
    single, their_single = stac_single_scores(x[:, 0], E), safe.compute_stac_metrics_single(torch.from_numpy(x[:, 0]), exec_horizon=E).numpy()
    assert np.allclose(single, their_single, atol=1e-6)


@needs_authors
def test_the_mmd_equals_the_authors_function_on_every_pair_of_consecutive_calls():
    sys.path.insert(0, str(STAC_SENTINEL))
    try:
        from sentinel.bc.ood_detection.error_utils import compute_mmd_rbf
    finally:
        sys.path.remove(str(STAC_SENTINEL))
    x = chunks(4)
    old, new = overlap(x, E)
    ours = mmd_rbf(old, new)
    for t in range(T - 1):
        theirs = compute_mmd_rbf(old[t].reshape(K, -1).astype(np.float64), new[t].reshape(K, -1).astype(np.float64), gamma=1.0)
        assert ours[t] == pytest.approx(theirs, abs=1e-9)


@needs_authors
@pytest.mark.parametrize("dims, ignore_rotation", [(6, False), (3, True)])
def test_the_variants_drop_what_the_authors_settings_drop(dims, ignore_rotation):
    """The authors' filters assume their robot's layout, gripper first: [gripper | position | rotation]. Our columns are mapped into it
    and their whole temporal-error function is run with the settings of "mmd_rbf_all_1.0" (rotation kept) and "mmd_rbf_pos" (position only)."""
    sys.path.insert(0, str(STAC_SENTINEL))
    try:
        from sentinel.bc.ood_detection.error_utils import compute_temporal_error
    finally:
        sys.path.remove(str(STAC_SENTINEL))
    x = chunks(5)
    theirs_layout = x[..., [6, 0, 1, 2, 3, 4, 5]]
    ours = stac_scores(x, E, dims)
    for t in range(1, T):
        theirs = compute_temporal_error("mmd_rbf", curr_action=theirs_layout[t].astype(np.float64), prev_action=theirs_layout[t - 1].astype(np.float64),
                                        pred_horizon=H, exec_horizon=E, ignore_gripper=True, ignore_rotation=ignore_rotation, num_robots=1,
                                        action_dim=7, gamma=1.0)
        assert ours[t] == pytest.approx(theirs, abs=1e-6)


# ---------------------------------------------------------------- the scorers
def test_there_are_eight_configurations_each_with_its_own_name_and_the_protocol_gives_them_all_max_conformal():
    names = method_names("stac") + method_names("stac_single")
    assert names == ["stac__mmd7_step", "stac__mmd7_cumsum", "stac__mmd6_step", "stac__mmd6_cumsum", "stac__mmd3_step", "stac__mmd3_cumsum",
                     "stac_single__step", "stac_single__cumsum"]
    assert len(set(names)) == 8 and all(calibrator_name(n) == MAX_CONFORMAL for n in names)
    for wrong in ("stac__mmd7step", "stac__mmd5_step", "stac_single__mmd7_step", "stac__mmd7_sum"):
        with pytest.raises(KeyError):
            calibrator_name(wrong)


def test_cumsum_accumulates_the_step_score_and_each_scorer_scores_the_episodes_it_is_given():
    actions = {"a": chunks(6), "b": chunks(7, t=4)}
    step, cum = StacScorer("stac", actions, E, dims=6), StacScorer("stac", actions, E, dims=6, cumsum=True)
    s, c = step.score(["a", "b"]), cum.score(["a", "b"])
    assert set(s) == {"a", "b"} and s["b"].shape == (4,) and np.allclose(c["a"], np.cumsum(s["a"]), rtol=1e-5)
    assert (np.diff(c["a"]) >= 0).all()  # non-negative steps: the maximum of the accumulated score is its last value
    single = StacScorer("stac_single", actions, E).score(["a"])["a"]
    assert np.array_equal(single, stac_single_scores(actions["a"][:, 0], E))


def test_nothing_is_fitted_nothing_is_kept_and_the_config_says_what_was_computed():
    scorer = StacScorer("stac", {"a": chunks()}, E, dims=3, cumsum=True)
    before = scorer.score(["a"])["a"]
    scorer.fit(["whatever"], ["else"])
    assert np.array_equal(before, scorer.score(["a"])["a"]) and scorer.state_bytes() == 0 and not scorer.saves_weights
    c = scorer.config
    assert (c["family"], c["dimensions"], c["cumsum"], c["exec_horizon"]) == ("stac", 3, True, E) and "mmd_rbf_pos" in c["meaning"]
    assert scorer.input_kind == "sampled action chunks"


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        StacScorer("stac", {}, E, dims=5)
    with pytest.raises(ValueError):
        StacScorer("nope", {}, E)


# ---------------------------------------------------------------- the registry
@pytest.fixture
def world(monkeypatch):
    monkeypatch.setitem(config.REGISTRY, "synthetic", dataclasses.replace(config.REGISTRY["libero10_pi05"], name="synthetic", replan_steps=E))
    episodes = {f"e{i}": Episode(f"e{i}", i % 2 == 0, "t", T, None if i % 2 == 0 else 3, "a") for i in range(4)}
    split = Split("task_seed0", ["e0"], ["e1"], ["e2"], ["e3"], [], seed=0)
    return Inputs("synthetic", episodes, None, {split.name: split}, 3), split


def test_the_families_are_registered_and_built_from_the_action_chunks_with_the_datasets_execution_horizon(world):
    inputs, split = world
    actions = {k: chunks(i) for i, k in enumerate(inputs.episodes)}
    assert {"stac", "stac_single"} <= set(FAMILIES)
    scorers = build("stac", Context(inputs, {}, "cpu", actions), split)
    assert [s.name for s in scorers] == method_names("stac") and all(s.replan_steps == E for s in scorers)
    assert [s.name for s in build("stac_single", Context(inputs, {}, "cpu", actions), split)] == method_names("stac_single")


def test_without_action_chunks_the_family_says_how_to_make_them(world):
    inputs, split = world
    with pytest.raises(ValueError, match="cli.prepare_actions"):
        build("stac", Context(inputs, {}, "cpu"), split)
