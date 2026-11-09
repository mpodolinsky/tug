"""OursScorer end to end on a small synthetic dataset (tiny widths, CPU)."""

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from fp.data.features import fit_standardiser
from fp.data.manifest import Episode
from fp.data.targets import MODE_NAMES, Targets
from fp.models.ours import ProbeConfig
from fp.train.loop import TrainConfig
from fp.train.ours import OursScorer
from fp.train.standardised import standardise_features

D, WINDOW, N_PHRASES = 8, 3, 20
TRAIN_CFG = TrainConfig(lr=5e-3, batch=8, max_epochs=40, patience=40, seed=0)


def cfg(**kw):
    return ProbeConfig(**{"d_modality": D, "hidden": 16, "proj": 8, "clip_dim": 6, **kw})


@pytest.fixture(scope="module")
def data():
    """40 episodes: failures have a shifted mean in the first columns, a mode, and an onset; every step has a phrase."""
    rng = np.random.default_rng(0)
    episodes, features, ids, modes = {}, {}, {}, {}
    for i in range(40):
        fail = i % 3 == 0
        n = int(rng.integers(14, 22))
        x = rng.normal(size=(n, 3 * D)).astype(np.float32)
        key = f"{'f' if fail else 's'}{i:02d}"
        onset = int(rng.integers(8, n - 2)) if fail else None
        if fail:
            x[:, :4] += 2.0
            modes[key] = int(rng.integers(0, len(MODE_NAMES)))
        episodes[key], features[key] = Episode(key, not fail, f"t{i % 5}", n, onset, "a"), x
        ids[key] = rng.integers(0, N_PHRASES, size=n).astype(np.int32)
    emb = rng.normal(size=(N_PHRASES, 16)).astype(np.float32)
    targets = Targets([f"phrase {j}" for j in range(N_PHRASES)], emb / np.linalg.norm(emb, axis=1, keepdims=True), ids, modes)
    keys = list(episodes)
    standardised = standardise_features({k: v.copy() for k, v in features.items()}, keys[:24])
    return episodes, standardised, targets, keys[:24], keys[24:32], keys[32:], features  # ..., train, val, test, the raw features


def scorer(data, c=None, variant="onset_pre", **kw):
    episodes, features, targets, *_ = data  # the standardised features
    return OursScorer(c or cfg(), variant, episodes, features, targets, WINDOW, TRAIN_CFG, device="cpu", **kw)


@pytest.fixture(scope="module")
def fitted(data):
    s = scorer(data)
    s.fit(data[3], data[4])
    return s


def test_outputs_hold_the_score_and_the_heads_the_architecture_has(data, fitted):
    plain = fitted.outputs(data[5])
    assert set(plain) == {"score"} and fitted.space is None  # the plain probe has no other head and no semantic target space
    s = scorer(data, cfg(modes=True, clip=True))
    s.fit(data[3], data[4])
    out = s.outputs(data[5])
    assert set(out) == {"score", "mode", "sem"} and s.space is not None and s.space.target.shape == (N_PHRASES, 6)
    for key in data[5]:
        n = data[0][key].n_steps
        assert out["mode"][key].shape == (n, len(MODE_NAMES)) and out["sem"][key].shape == (n, 6) and out["score"][key].shape == (n,)
        assert all(out[h][key].dtype == np.float32 and np.isfinite(out[h][key]).all() for h in out)
    assert all(np.array_equal(out["score"][k], s.score(data[5])[k]) for k in data[5])
    assert set(s.outputs(data[5], heads=False)) == {"score"}


def test_the_head_outputs_are_what_the_model_computes_on_the_whole_rollout(data):
    s = scorer(data, cfg(modes=True, clip=True))
    s.fit(data[3], data[4])
    key = data[5][0]
    with torch.no_grad():
        direct = s.model(s.features.arrays[key][None].to(s.device))
    out = s.outputs([key])
    assert np.allclose(out["mode"][key], direct["mode"][0].cpu().numpy(), atol=1e-5)
    assert np.allclose(out["sem"][key], direct["sem"][0].cpu().numpy(), atol=1e-5)
    assert np.allclose(out["score"][key], torch.sigmoid(direct["score"][0]).cpu().numpy(), atol=1e-6)


def test_the_name_is_the_architecture_and_the_label_variant(data):
    assert scorer(data, cfg(mha=True, modes=True, clip=True), "onset").name == "lstm_mha_modes_clip__onset"


def test_scores_are_probabilities_for_every_step_of_every_requested_episode(data, fitted):
    out = fitted.score(data[5])
    assert list(out) == data[5]
    for key, p in out.items():
        assert p.shape == (data[0][key].n_steps,) and p.dtype == np.float32 and ((p > 0) & (p < 1)).all()


@pytest.mark.parametrize("c", [cfg(), cfg(trunk="mlp", clip=True), cfg(mha=True, modes=True, clip=True)], ids=lambda c: c.name)
def test_the_probe_learns_to_separate_failures_from_successes_on_held_out_episodes(data, c):
    s = scorer(data, c)
    s.fit(data[3], data[4])
    episodes = data[0]
    out = s.score(data[5])
    y = [0 if episodes[k].success else 1 for k in data[5]]
    assert roc_auc_score(y, [out[k].mean() for k in data[5]]) > 0.9
    assert s.result.val["score"][s.result.best_epoch] < s.result.val["score"][0]


def test_steps_before_the_variants_cut_get_the_same_score_on_the_cut_rollout(data, fitted):
    """The probe is causal: scoring the rollout cut at step 10 gives the first 10 values of scoring it whole."""
    key = data[5][0]
    full = fitted.score([key])[key]
    arrays = fitted.features.arrays
    original = arrays[key]
    arrays[key] = original[:10]
    try:
        assert np.allclose(fitted.score([key])[key], full[:10], atol=1e-6)
    finally:
        arrays[key] = original


def test_the_same_seed_gives_identical_scores(data, fitted):
    again = scorer(data)
    again.fit(data[3], data[4])
    a, b = fitted.score(data[5]), again.score(data[5])
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_early_stopping_follows_the_validation_score_bce_even_with_the_other_heads(data):
    """The selection value is the weighted BCE of the validation items alone: recomputed here from the fitted model's scores."""
    s = scorer(data, cfg(mha=True, modes=True, clip=True))
    s.fit(data[3], data[4])
    train = [s._item(k) for k in data[3]]
    y = torch.cat([it.y for it in train])
    pos_weight = float((1 - y).sum() / y.sum())
    probs, steps, total = s.score(data[4]), 0, 0.0
    for key in data[4]:
        item = s._item(key)
        p = torch.from_numpy(probs[key][: len(item.y)]).double()
        total += float(-(pos_weight * item.y * torch.log(p) + (1 - item.y) * torch.log(1 - p)).sum())
        steps += len(item.y)
    assert s.result.best_epoch == int(np.argmin(s.result.val["score"]))  # chosen by the score term, whatever the other terms did
    assert s.result.val["score"][s.result.best_epoch] == pytest.approx(total / steps, rel=1e-3)


def test_every_head_term_is_reported_for_every_epoch_and_finite(data):
    s = scorer(data, cfg(modes=True, clip=True))
    s.fit(data[3], data[4])
    for history in (s.result.train, s.result.val):
        assert set(history) == {"score", "mode", "clip"}
        assert all(len(v) == len(s.result.val["score"]) and np.isfinite(v).all() for v in history.values())


def test_the_feature_statistics_come_from_the_training_episodes_only(data, fitted):
    mean, std = fit_standardiser([data[6][k] for k in data[3]])
    assert np.array_equal(fitted.features.mean, mean) and np.array_equal(fitted.features.std, std)


def test_fitting_on_other_training_episodes_than_the_features_were_standardised_with_is_an_error(data):
    with pytest.raises(ValueError, match="other training episodes"):
        scorer(data).fit(data[3][:-1], data[4])


def test_save_and_load_reproduce_the_scores(data, fitted, tmp_path):
    fitted.save(tmp_path / "w.pt")
    loaded = scorer(data)
    loaded.load(tmp_path / "w.pt")
    a, b = fitted.score(data[5]), loaded.score(data[5])
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_loading_weights_into_another_architecture_or_label_variant_is_an_error(data, fitted, tmp_path):
    fitted.save(tmp_path / "w.pt")
    with pytest.raises(ValueError, match="this scorer is"):
        scorer(data, cfg(trunk="mlp")).load(tmp_path / "w.pt")
    with pytest.raises(ValueError, match="this scorer is"):
        scorer(data, variant="onset").load(tmp_path / "w.pt")
    shifted = standardise_features({k: v.copy() for k, v in data[6].items()}, data[3][:-2] + data[4][:2])  # other statistics
    with pytest.raises(ValueError, match="other statistics"):
        OursScorer(cfg(), "onset_pre", data[0], shifted, data[2], WINDOW, TRAIN_CFG, device="cpu").load(tmp_path / "w.pt")


def test_scoring_before_fitting_is_an_error(data):
    with pytest.raises(RuntimeError):
        scorer(data).score(data[5])


def test_a_failed_training_episode_without_a_mode_is_an_error(data):
    episodes, features, targets, train, val, *_ = data
    broken = Targets(targets.phrases, targets.embeddings, targets.phrase_ids, {k: v for k, v in targets.modes.items() if k != train[0]})
    s = OursScorer(cfg(modes=True), "onset", episodes, features, broken, WINDOW, TRAIN_CFG, device="cpu")
    with pytest.raises(KeyError):
        s.fit(train, val)


# ---------------------------------------------------------------- the weights of the mode and CLIP terms
SHORT = TrainConfig(lr=5e-3, batch=8, max_epochs=3, patience=3, seed=0)


def fit_with(data, **kw):
    episodes, features, targets, train, val, *_ = data
    s = OursScorer(cfg(modes=True, clip=True), "onset_pre", episodes, features, targets, WINDOW, SHORT, device="cpu", **kw)  # both heads
    s.fit(train, val)
    return s


def untrained_heads():
    """The heads of a probe as the scorer initialises it (same seed, same construction order)."""
    from fp.models.ours import Probe

    torch.manual_seed(SHORT.seed)
    return Probe(cfg(modes=True, clip=True))


def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_the_weights_of_the_mode_clip_and_score_terms_reach_the_loss(data, monkeypatch):
    from fp.train import ours

    seen = []
    original = ours.make_context
    monkeypatch.setattr(ours, "make_context", lambda *a, **k: seen.append(a[4:]) or original(*a, **k))
    fit_with(data, lam_mode=0.25, lam_clip=0.5, lam_score=2.0)
    fit_with(data)
    assert seen == [(0.25, 0.5, 2.0), (1.0, 1.0, 1.0)]  # the default is the earlier behaviour: all three terms at 1


def test_with_the_score_weight_zero_the_score_head_stays_untrained_and_the_epoch_follows_the_validation_mode_term(data):
    reference = untrained_heads()
    s = fit_with(data, lam_score=0.0)
    assert same(s.model.score, reference.score)  # no gradient reaches the score head
    assert not same(s.model.mode, reference.mode) and not same(s.model.trunk, reference.trunk)  # the mode term trains the head and the trunk
    assert s.result.best_epoch == int(np.argmin(s.result.val["mode"])) and len(s.result.val["score"]) == len(s.result.val["mode"])  # the score is still reported
    default = fit_with(data)
    assert default.result.best_epoch == int(np.argmin(default.result.val["score"]))  # with the score term the selection is unchanged


def test_a_probe_without_a_mode_head_cannot_be_trained_without_the_score_term(data):
    with pytest.raises(ValueError, match="mode head"):
        scorer(data, cfg(), lam_score=0.0)
    with pytest.raises(ValueError, match="mode head"):
        scorer(data, cfg(clip=True), lam_score=0.0)


def test_a_term_with_weight_zero_leaves_its_head_untrained_while_the_other_head_and_the_trunk_learn(data):
    reference = untrained_heads()
    both_off = fit_with(data, lam_mode=0.0, lam_clip=0.0).model
    assert same(both_off.mode, reference.mode) and same(both_off.sem, reference.sem)  # no gradient reaches either head
    assert not same(both_off.trunk, reference.trunk) and not same(both_off.score, reference.score)  # the score term still trains the rest
    mode_only = fit_with(data, lam_mode=1.0, lam_clip=0.0).model
    assert not same(mode_only.mode, reference.mode) and same(mode_only.sem, reference.sem)
    clip_only = fit_with(data, lam_mode=0.0, lam_clip=1.0).model
    assert same(clip_only.mode, reference.mode) and not same(clip_only.sem, reference.sem)
