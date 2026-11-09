import numpy as np
import pytest
import torch

from fp.baselines import actprobe, failurespot, stac
from fp.baselines.distance import DistanceScorer
from fp.data.manifest import Episode
from fp.data.splits import Split
from fp.eval import latency
from fp.eval.latency import (ActProbeOnline, EuclideanOnline, LatencySpec, LstmOnline, MahalanobisOnline, MlpOnline, PcaKMeansOnline, ProbeOnline, StacOnline, build,
                             measure, spec_for)
from fp.models.ours import Probe, ProbeConfig
from fp.models.safe import SafeLSTM, SafeMLP
from tests.test_distance_baselines import D, make_episodes, split_keys

T = 12
SMALL = LatencySpec(dim=D, n_succ_steps=200, n_fail_steps=200, samples=3, horizon=6, action_dim=7, replan_steps=2)


def stream(scorer, steps):
    scorer.reset()
    return np.array([float(scorer.step(x)) for x in steps])


# ---------------------------------------------------------------- the online step is the batched scorer, step by step
def test_our_probe_online_gives_the_scores_of_the_whole_rollout_and_its_head_outputs():
    torch.manual_seed(0)
    model = Probe(ProbeConfig(modes=True, clip=True, d_modality=4, hidden=16)).eval()
    x = torch.randn(1, T, model.cfg.d_in)
    out = model(x)
    online = ProbeOnline(model)
    got = stream(online, x[0])
    assert np.allclose(got, torch.sigmoid(out["score"])[0].detach().numpy(), atol=1e-5)
    assert np.allclose(online.outputs["mode"].view(-1).detach().numpy(), out["mode"][0, -1].detach().numpy(), atol=1e-5)  # the last step's mode and semantic outputs
    assert online.outputs["sem"].numel() == model.cfg.clip_dim
    assert np.allclose(stream(online, x[0]), got)  # reset starts a new episode


def test_the_online_step_covers_only_the_lstm_trunk_without_attention():
    with pytest.raises(ValueError, match="LSTM trunk"):
        ProbeOnline(Probe(ProbeConfig(trunk="mlp", d_modality=4)))
    with pytest.raises(ValueError, match="LSTM trunk"):
        ProbeOnline(Probe(ProbeConfig(mha=True, d_modality=4)))


def test_safe_and_failurespot_models_online_equal_their_batched_forward():
    torch.manual_seed(1)
    x = torch.randn(1, T, D)
    lstm, mlp = SafeLSTM(D, 8).eval(), SafeMLP(D, 8).eval()
    assert np.allclose(stream(LstmOnline(lstm), x[0]), lstm(x)[0].detach().numpy(), atol=1e-5)
    assert np.allclose(stream(MlpOnline(mlp.projector, cumsum=True), x[0]), mlp(x)[0].detach().numpy(), atol=1e-5)  # SAFE-MLP: the running sum
    spot_lstm, spot_mlp = failurespot.StepLSTM(D, 8).eval(), failurespot.StepMLP(D, 8).eval()
    assert np.allclose(stream(LstmOnline(spot_lstm), x[0]), spot_lstm(x)[0].detach().numpy(), atol=1e-5)
    assert np.allclose(stream(MlpOnline(spot_mlp.net, cumsum=False), x[0]), spot_mlp(x)[0].detach().numpy(), atol=1e-5)  # FailureSpot: no running sum


@pytest.fixture(scope="module")
def fitted():
    episodes, features = make_episodes()
    train, test = split_keys(episodes)
    return episodes, features, train, test


@pytest.mark.parametrize("cumsum", [True, False])
def test_the_distance_baselines_online_equal_their_batched_scores(fitted, cumsum):
    episodes, features, train, test = fitted
    x = torch.as_tensor(features[test[0]])
    for name, make in (
        ("mahala", lambda s: MahalanobisOnline(s.feats_succ_mean, s.inv_cov_succ, s.feats_fail_mean, s.inv_cov_fail, cumsum)),
        ("euclid", lambda s: EuclideanOnline(s.feats_succ, s.feats_fail, s.topk, cumsum)),
        ("pca_kmeans", lambda s: PcaKMeansOnline(torch.as_tensor(s.pca_succ.mean_, dtype=torch.float32), torch.as_tensor(s.pca_succ.components_, dtype=torch.float32),
                                                  s.centroids_succ.float(), s.centroids_fail.float(), cumsum)),
    ):
        scorer = DistanceScorer(name, episodes, features, topk=3, pca_dim=3, n_clusters=3, cumsum=cumsum)
        scorer.fit(train, [])
        batched = scorer.score([test[0]])[test[0]]
        assert np.allclose(stream(make(scorer), x), batched, rtol=1e-3, atol=1e-3), name


def test_stac_online_equals_the_batched_score_of_the_episode_with_the_first_call_at_zero():
    rng = np.random.default_rng(0)
    chunks = rng.standard_normal((T, 3, 6, 7)).astype(np.float32)  # (calls, K, H, A)
    for dims in (7, 6, 3):
        got = stream(StacOnline("stac", dims, 2, cumsum=False), chunks)
        assert got[0] == 0.0 and np.allclose(got, stac.stac_scores(chunks, 2, dims), atol=1e-5)
    single = stream(StacOnline("stac_single", 7, 2, cumsum=True), chunks[:, 0])
    assert np.allclose(single, np.cumsum(stac.stac_single_scores(chunks[:, 0], 2)), atol=1e-5)


def test_actprobe_online_equals_the_network_on_the_whole_episode():
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    net = actprobe.ActProbeNet().eval()
    chunks = rng.standard_normal((T, 6, 7)).astype(np.float32)
    mean, std = np.array([0.3, 0.1], np.float32), np.array([0.9, 1.7], np.float32)
    lang = torch.randn(1, actprobe.LANGUAGE_DIM)
    signals = (actprobe.acm_tce(chunks, 2) - mean) / (std + actprobe.NORM_EPS)
    x = torch.cat([torch.tensor(signals), torch.arange(T).view(-1, 1).float() / actprobe.TIME_SCALE], dim=1).view(1, T, 3)
    expected = net(x, lang, torch.tensor([T])).detach().numpy()[0]
    assert np.allclose(stream(ActProbeOnline(net, mean, std, lang, 2), chunks), expected, atol=1e-5)


# ---------------------------------------------------------------- the methods of the table
@pytest.mark.parametrize("method", ["lstm_modes_clip__onset_pre", "safe_lstm__lr1e-3_reg1e-2", "safe_mlp__lr1e-4_reg1e-2", "failurespot_mlp__eta1_4_eta2_4",
                                    "failurespot_lstm__eta1_1_eta2_1", "mahalanobis__step", "euclidean__k10_cumsum", "pca_kmeans__d32_c64_cumsum", "stac__mmd3_step",
                                    "stac_single__cumsum", "actprobe"])
def test_every_method_of_the_table_has_an_online_step_that_gives_a_finite_score(method):
    spec = LatencySpec(dim=24, n_succ_steps=40, n_fail_steps=40, samples=3, horizon=6, replan_steps=2)
    scorer, make_input = build(method, spec)
    steps = [scorer.step(make_input()) for _ in range(4)]
    assert all(np.isfinite(float(s)) for s in steps)


def test_a_method_without_an_online_step_is_an_error():
    with pytest.raises(KeyError, match="no online step"):
        build("not_a_method", SMALL)


def test_the_benchmark_uses_the_sizes_of_the_spec_and_the_architecture_of_the_name():
    scorer, _ = build("euclidean__k4_step", LatencySpec(dim=8, n_succ_steps=30, n_fail_steps=50))
    assert scorer.feats_succ.shape == (30, 8) and scorer.feats_fail.shape == (50, 8) and scorer.k == 4 and not scorer.cumsum
    scorer, _ = build("pca_kmeans__d5_c7_step", LatencySpec(dim=16))
    assert scorer.components.shape == (5, 16) and scorer.centroids_succ.shape == (7, 5)
    scorer, make_input = build("stac__mmd6_cumsum", SMALL)
    assert make_input().shape == (3, 6, 7) and scorer.cols == stac.DIMENSIONS[6] and scorer.cumsum
    scorer, make_input = build("lstm_plain__onset_pre", LatencySpec(dim=6144))
    assert scorer.model.cfg.modes is False and make_input().shape == (6144,)


def test_the_spec_counts_the_training_steps_of_each_outcome_and_the_replan_steps_of_the_dataset():
    episodes = {"a": Episode("a", True, "t", 10, None), "b": Episode("b", False, "t", 7, 3), "c": Episode("c", True, "t", 5, None), "d": Episode("d", False, "t", 4, 1)}
    inputs = type("I", (), {"episodes": episodes, "dataset": "liberox_groot"})()
    spec = spec_for(inputs, Split("s", ["a", "b", "c"], ["d"], [], [], [], 0))
    assert (spec.n_succ_steps, spec.n_fail_steps, spec.dim, spec.replan_steps, spec.samples) == (15, 7, 6144, 8, 10)


def test_measure_gives_a_median_not_above_the_99th_percentile_and_resets_the_state_first():
    scorer, make_input = build("safe_lstm__lr1e-3_reg1e-2", SMALL)
    scorer.step(make_input())
    assert scorer.state is not None
    result = measure(scorer, make_input, steps=30, warmup=5)
    assert 0 < result["median_ms"] <= result["p99_ms"]
    scorer.reset()
    assert scorer.state is None
