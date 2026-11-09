import functools
import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

from fp import config
from fp.baselines import distance as dist_mod
from fp.baselines.distance import (DistanceScorer, compute_dist_closest_k_by_chunk, compute_euclid_dist, compute_mahala_dist)
from fp.baselines.distance_grid import method_names
from fp.baselines.registry import FAMILIES, Context, build
from fp.data.inputs import Inputs
from fp.data.manifest import Episode
from fp.data.splits import Split

D = 8
CONFORMAL = "baselines/SAFE/failure_prob/utils/conformal"  # only used to locate the SAFE checkout


# ---------------------------------------------------------------- data
def make_episodes(n_succ=24, n_fail=24, seed=0):
    """Success steps ~ N(0, I); a failed episode drifts to a shifted mean over its steps (its early steps look like successes)."""
    rng = np.random.default_rng(seed)
    episodes, features = {}, {}
    for i in range(n_succ + n_fail):
        ok = i < n_succ
        T = int(rng.integers(12, 20))
        x = rng.normal(0, 1, (T, D)).astype(np.float32)
        if not ok:
            x += (np.linspace(0, 1, T)[:, None] * 3.0 * np.ones(D)).astype(np.float32)
        key = f"{'s' if ok else 'f'}{i:02d}"
        episodes[key] = Episode(key, ok, "t", T, None if ok else T - 1)
        features[key] = x
    return episodes, features


def split_keys(episodes, n_test_per_class=6):
    succ = [k for k, e in episodes.items() if e.success]
    fail = [k for k, e in episodes.items() if not e.success]
    return succ[:-n_test_per_class] + fail[:-n_test_per_class], succ[-n_test_per_class:] + fail[-n_test_per_class:]


# ---------------------------------------------------------------- the copied functions
def test_mahalanobis_matches_the_numpy_formula():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 5))
    mean = rng.normal(size=5)
    a = rng.normal(size=(5, 5))
    cov = a @ a.T + 0.1 * np.eye(5)
    expected = np.sqrt(np.einsum("ij,jk,ik->i", x - mean, np.linalg.inv(cov), x - mean))
    got = compute_mahala_dist(torch.tensor(x), torch.tensor(mean), torch.tensor(cov)).numpy()
    assert np.allclose(got, expected)


def test_mahalanobis_with_identity_covariance_is_the_euclidean_distance_to_the_mean():
    rng = np.random.default_rng(1)
    x, mean = rng.normal(size=(30, 4)), rng.normal(size=4)
    got = compute_mahala_dist(torch.tensor(x), torch.tensor(mean), torch.eye(4, dtype=torch.float64)).numpy()
    assert np.allclose(got, np.linalg.norm(x - mean, axis=1))


def test_mahalanobis_gives_the_same_result_whatever_the_chunk_size():
    rng = np.random.default_rng(2)
    x, mean, cov = torch.tensor(rng.normal(size=(50, 3))), torch.zeros(3, dtype=torch.float64), torch.eye(3, dtype=torch.float64) * 2
    assert torch.allclose(compute_mahala_dist(x, mean, cov, chunk_size=7), compute_mahala_dist(x, mean, cov))


def test_pairwise_euclidean_matches_scipy_and_is_never_negative():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(20, 6)), rng.normal(size=(15, 6))
    got = compute_euclid_dist(torch.tensor(x), torch.tensor(y)).numpy()
    assert np.allclose(got, cdist(x, y), atol=1e-6)
    same = compute_euclid_dist(torch.tensor(x), torch.tensor(x)).numpy()
    assert (same >= 0).all() and np.allclose(np.diag(same), 0, atol=1e-5)


@pytest.mark.parametrize("k", [1, 3, 10])
def test_k_closest_distances_are_the_k_smallest_sorted_and_independent_of_the_chunk_size(k):
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=(37, 5)), rng.normal(size=(60, 5))
    expected = np.sort(cdist(x, y), axis=1)[:, :k]
    got = compute_dist_closest_k_by_chunk(torch.tensor(x), torch.tensor(y), k, compute_euclid_dist, chunk_size=8).numpy()
    assert got.shape == (37, k) and np.allclose(got, expected, atol=1e-6)


# ---------------------------------------------------------------- the scorer
@pytest.mark.parametrize("distance", ["mahala", "euclid", "pca_kmeans"])
def test_failed_episodes_get_higher_final_scores_than_successes_on_held_out_episodes(distance):
    episodes, features = make_episodes()
    train, test = split_keys(episodes)
    scorer = DistanceScorer(distance, episodes, features, topk=5, pca_dim=4, n_clusters=3)
    scorer.fit(train, [])
    final = {k: v[-1] for k, v in scorer.score(test).items()}
    labels = [0 if episodes[k].success else 1 for k in test]
    assert roc_auc_score(labels, [final[k] for k in test]) > 0.9


@pytest.mark.parametrize("distance", ["mahala", "euclid", "pca_kmeans"])
def test_cumsum_true_is_the_running_sum_of_the_per_step_scores(distance):
    episodes, features = make_episodes()
    train, test = split_keys(episodes)
    kw = dict(topk=5, pca_dim=4, n_clusters=3)
    step = DistanceScorer(distance, episodes, features, cumsum=False, **kw)
    acc = DistanceScorer(distance, episodes, features, cumsum=True, **kw)
    step.fit(train, [])
    acc.fit(train, [])
    for k in test:
        assert np.allclose(np.cumsum(step.score([k])[k]), acc.score([k])[k], rtol=1e-4, atol=1e-4)


def test_success_only_uses_just_the_distance_to_the_success_set():
    episodes, features = make_episodes()
    train, test = split_keys(episodes)
    scorer = DistanceScorer("euclid", episodes, features, topk=5, use_success_only=True, cumsum=False)
    scorer.fit(train, [])
    k = test[0]
    x = torch.tensor(features[k])
    expected = compute_dist_closest_k_by_chunk(x, scorer.feats_succ, 5, compute_euclid_dist).mean(dim=-1).numpy()
    assert np.allclose(scorer.score([k])[k], expected, atol=1e-5)


def test_every_step_of_a_failed_training_episode_is_a_failure_step_and_validation_keys_are_ignored():
    episodes, features = make_episodes()
    train, test = split_keys(episodes)
    scorer = DistanceScorer("euclid", episodes, features, topk=5)
    scorer.fit(train, test)
    assert len(scorer.feats_fail) == sum(len(features[k]) for k in train if not episodes[k].success)
    assert len(scorer.feats_succ) == sum(len(features[k]) for k in train if episodes[k].success)
    other = DistanceScorer("euclid", episodes, features, topk=5)
    other.fit(train, [])
    k = test[0]
    assert np.array_equal(scorer.score([k])[k], other.score([k])[k])


def test_pca_kmeans_failure_distance_uses_the_success_pca_as_in_safe():
    """SAFE projects x once with the SUCCESS PCA and compares it with both sets of centroids (the failure centroids live in the
    failure PCA's coordinates). Reproduced by hand from the fitted attributes."""
    episodes, features = make_episodes()
    train, test = split_keys(episodes)
    scorer = DistanceScorer("pca_kmeans", episodes, features, pca_dim=4, n_clusters=3, cumsum=False)
    scorer.fit(train, [])
    k = test[0]
    x_pca = scorer.pca_succ.transform(features[k])
    d_succ = cdist(x_pca, scorer.kmeans_succ.cluster_centers_).min(axis=1)
    d_fail_as_in_safe = cdist(x_pca, scorer.kmeans_fail.cluster_centers_).min(axis=1)
    assert np.allclose(scorer.score([k])[k], d_succ - d_fail_as_in_safe, atol=1e-4)
    d_fail_own_pca = cdist(scorer.pca_fail.transform(features[k]), scorer.kmeans_fail.cluster_centers_).min(axis=1)
    assert not np.allclose(d_fail_as_in_safe, d_fail_own_pca, atol=1e-3)  # the two coordinate systems really differ


def test_pca_kmeans_is_reproducible_for_a_fixed_seed():
    episodes, features = make_episodes()
    train, test = split_keys(episodes)
    runs = []
    for _ in range(2):
        s = DistanceScorer("pca_kmeans", episodes, features, pca_dim=4, n_clusters=3, seed=5)
        s.fit(train, [])
        runs.append(s.score(test))
    assert all(np.array_equal(runs[0][k], runs[1][k]) for k in test)


def test_errors():
    episodes, features = make_episodes()
    train, test = split_keys(episodes)
    with pytest.raises(ValueError):
        DistanceScorer("cosine", episodes, features)  # not one of our baselines
    with pytest.raises(RuntimeError):
        DistanceScorer("mahala", episodes, features).score(test)  # not fitted
    with pytest.raises(ValueError):
        DistanceScorer("mahala", episodes, features).fit([k for k in train if episodes[k].success], [])  # no failures
    with pytest.raises(AssertionError):  # SAFE's own check: fewer samples than topk
        DistanceScorer("euclid", episodes, features, topk=10**6).fit(train, [])


def test_the_three_families_build_their_grid_scorers_from_a_preloaded_context():
    import fp.baselines.distance_grid  # noqa: F401  (registers the families)

    assert {"mahalanobis", "euclidean", "pca_kmeans"} <= set(FAMILIES)
    episodes, features = make_episodes()
    split = Split("task_seed1", sorted(episodes)[:30], sorted(episodes)[30:40], [], [], sorted(episodes)[40:], seed=7)
    context = Context(Inputs("synthetic", episodes, None, {split.name: split}, 6), features, device="cpu")
    built = {family: build(family, context, split) for family in ("mahalanobis", "euclidean", "pca_kmeans")}
    assert {f: len(s) for f, s in built.items()} == {"mahalanobis": 2, "euclidean": 6, "pca_kmeans": 18}  # SAFE's 26 configurations
    scorers = [s for ss in built.values() for s in ss]
    assert len({s.name for s in scorers}) == 26 and all(s.seed == 7 and s.device == "cpu" for s in scorers)  # the split's seed
    assert all(s.features is features and s.episodes is episodes for s in scorers)  # shared, never copied or reloaded
    assert [s.name for s in built["euclidean"]] == method_names("euclidean") and method_names("euclidean")[0] == "euclidean__k1_step"  # grid order
    with pytest.raises(KeyError):
        build("no_such_baseline", context, split)


def test_a_scorer_records_every_setting_in_its_config():
    episodes, features = make_episodes()
    scorer = DistanceScorer("euclid", episodes, features, topk=5, cumsum=False, seed=3)
    assert scorer.config == {"distance": "euclid", "topk": 5, "pca_dim": 64, "n_clusters": 16, "use_success_only": False, "cumsum": False, "seed": 3}


def test_the_mahalanobis_pseudo_inverse_is_computed_once_and_gives_the_same_distances():
    rng = torch.Generator().manual_seed(0)
    x, mean = torch.randn(40, D, generator=rng), torch.randn(D, generator=rng)
    a = torch.randn(200, D, generator=rng)
    cov = torch.cov(a.T) + 1e-4 * torch.eye(D)
    assert torch.equal(compute_mahala_dist(x, mean, cov), compute_mahala_dist(x, mean, cov, inv_cov=torch.pinverse(cov)))
    episodes, features = make_episodes()
    scorer = DistanceScorer("mahala", episodes, features)
    train, test = split_keys(episodes)
    scorer.fit(train, [])
    calls = []
    original = torch.pinverse
    torch.pinverse = lambda m: calls.append(1) or original(m)  # count the inversions while scoring
    try:
        scorer.score(test)
    finally:
        torch.pinverse = original
    assert calls == []  # scoring never inverts the covariance again


# ---------------------------------------------------------------- regression against SAFE's own EmbedModel
def find_safe_root():
    if os.environ.get("SAFE_ROOT"):
        return Path(os.environ["SAFE_ROOT"])
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "baselines/SAFE/failure_prob/model/embed.py").exists():
            return p / "baselines/SAFE"
        if p.name == "worktrees" and p.parent.name == ".claude":
            return p.parent.parent / "baselines/SAFE"
    return here.parent / "missing"


SAFE_ROOT = find_safe_root()


def load_safe_embed(monkeypatch):
    """Execute SAFE's embed.py with minimal stand-ins for failure_prob.conf and failure_prob.model.base (their real versions pull in
    hydra, wandb and the whole data stack)."""
    pkg = types.ModuleType("failure_prob")
    pkg.__path__ = []
    model_pkg = types.ModuleType("failure_prob.model")
    model_pkg.__path__ = [str(SAFE_ROOT / "failure_prob/model")]
    base = types.ModuleType("failure_prob.model.base")

    class BaseModel(torch.nn.Module):
        def __init__(self, cfg, input_dim):
            super().__init__()
            self.cfg, self.input_dim = cfg, input_dim

        def get_device(self):
            return torch.device("cpu")

    base.BaseModel = BaseModel
    conf = types.ModuleType("failure_prob.conf")
    conf.Config = object
    for name, mod in (("failure_prob", pkg), ("failure_prob.model", model_pkg), ("failure_prob.model.base", base), ("failure_prob.conf", conf)):
        monkeypatch.setitem(sys.modules, name, mod)
    spec = importlib.util.spec_from_file_location("failure_prob.model.embed", SAFE_ROOT / "failure_prob/model/embed.py")
    embed = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "failure_prob.model.embed", embed)
    spec.loader.exec_module(embed)
    return embed


def padded(episodes, features, keys):
    T = max(len(features[k]) for k in keys)
    x = np.zeros((len(keys), T, D), dtype=np.float32)
    mask = np.zeros((len(keys), T), dtype=np.float32)
    for i, k in enumerate(keys):
        x[i, : len(features[k])] = features[k]
        mask[i, : len(features[k])] = 1
    labels = np.array([1.0 if episodes[k].success else 0.0 for k in keys], dtype=np.float32)  # SAFE: label 1 = success
    return torch.tensor(x), torch.tensor(mask), torch.tensor(labels)


@pytest.mark.skipif(not (SAFE_ROOT / "failure_prob/model/embed.py").exists(), reason="SAFE checkout not found (set SAFE_ROOT)")
@pytest.mark.parametrize("cumsum", [True, False])
@pytest.mark.parametrize("use_success_only", [False, True])
@pytest.mark.parametrize("distance", ["mahala", "euclid", "pca_kmeans"])
def test_scores_equal_those_of_safes_own_embedmodel(monkeypatch, distance, use_success_only, cumsum):
    embed = load_safe_embed(monkeypatch)
    # SAFE leaves PCA and KMeans unseeded; fix the seed there to compare like with like.
    monkeypatch.setattr(embed, "PCA", functools.partial(embed.PCA, random_state=0))
    monkeypatch.setattr(embed, "KMeans", functools.partial(embed.KMeans, random_state=0))
    episodes, features = make_episodes()
    train, test = split_keys(episodes)

    cfg = SimpleNamespace(model=SimpleNamespace(distance=distance, topk=5, use_success_only=use_success_only, cumsum=cumsum,
                                                pca_dim=4, n_clusters=3))
    safe = embed.EmbedModel(cfg, D)
    x, mask, labels = padded(episodes, features, train)
    dataset = SimpleNamespace(get_features=lambda: x, get_valid_masks=lambda: mask, get_labels=lambda: labels)
    safe.train_epoch(None, SimpleNamespace(dataset=dataset))
    xt, _, _ = padded(episodes, features, test)
    reference = safe({"features": xt})[..., 0].detach().numpy()  # (B, T)

    ours = DistanceScorer(distance, episodes, features, topk=5, pca_dim=4, n_clusters=3, use_success_only=use_success_only,
                          cumsum=cumsum, seed=0)
    ours.fit(train, [])
    got = ours.score(test)
    for i, k in enumerate(test):
        T = len(features[k])
        assert np.allclose(got[k], reference[i, :T], rtol=1e-4, atol=1e-4), (distance, k)
