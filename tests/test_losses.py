import math

import numpy as np
import pytest
import torch

from fp.models.ours import Probe, ProbeConfig
from fp.train.items import Item, collate
from fp.train.losses import TAU, loss_terms, make_context, probe_loss, total_loss
from fp.train.semantic import SemanticSpace

D, N_MODES, DIM = 4, 10, 3
LOG2, LOG10, LOG3 = math.log(2), math.log(10), math.log(3)


def item(y, mode=-1, phrase=None):
    y = torch.tensor(y, dtype=torch.float32)
    return Item(torch.zeros(len(y), D), y, torch.tensor(phrase if phrase is not None else [-1] * len(y)), mode)


def cfg(modes=False, clip=False):
    return ProbeConfig(d_modality=1, n_modalities=D, hidden=8, n_modes=N_MODES, clip_dim=DIM, modes=modes, clip=clip)


def outputs(batch, score=0.0, modes=False, clip=None):
    """Network outputs with a constant score logit, uniform mode logits and a constant semantic vector."""
    out = {"score": torch.full(batch.y.shape, score)}
    if modes:
        out["mode"] = torch.zeros(*batch.y.shape, N_MODES)
    if clip is not None:
        out["sem"] = clip.expand(*batch.y.shape, DIM).clone()
    return out


def space(n_phrases=5, in_bank=(0, 1, 2)):
    target = np.eye(n_phrases, DIM, dtype=np.float32)  # phrase i has the unit vector e_i (i < 3)
    bank_index = np.full(n_phrases, -1, dtype=np.int64)
    bank_index[list(in_bank)] = np.arange(len(in_bank))
    return SemanticSpace(target, target[list(in_bank)], bank_index, 1.0)


# ---------------------------------------------------------------- the weighted BCE
def test_the_score_loss_is_bce_with_the_training_pos_weight():
    train = [item([0, 0, 0, 1])]  # n_neg / n_pos = 3
    ctx = make_context(train, cfg(), None)
    assert float(ctx.pos_weight) == 3.0
    batch = collate(train)
    # logit 0: a positive step costs pos_weight * log 2, a negative one log 2; mean over the 4 real steps
    assert float(probe_loss(outputs(batch), batch, ctx, cfg())) == pytest.approx((3 * LOG2 + 3 * LOG2) / 4)


def test_each_term_is_a_numerator_and_denominator_with_the_mean_by_hand():
    train = [item([0, 1, 1], mode=4, phrase=[0, 1, 2]), item([0, 0, 0, 1])]
    c = cfg(modes=True, clip=True)
    ctx = make_context(train, c, space(), "cpu")
    batch = collate(train)
    terms = loss_terms(outputs(batch, modes=True, clip=torch.ones(DIM)), batch, ctx, c)
    assert set(terms) == {"score", "mode", "clip"}
    num, den = terms["score"]  # 7 real steps: 3 positive (each weighted pos_weight = 4 / 3) and 4 negative, every one costing log 2
    assert float(den) == 7 and float(num) == pytest.approx((3 * float(ctx.pos_weight) + 4) * LOG2)
    num, den = terms["mode"]  # 2 positive steps of the failed episode, mode 4, uniform logits over 10 modes
    assert float(den) == pytest.approx(2 * float(ctx.mode_weight[4])) and float(num / den) == pytest.approx(LOG10)
    num, den = terms["clip"]  # 3 steps with a phrase in the bank, equally close to all 3 targets
    assert float(den) == 3 and float(num / den) == pytest.approx(LOG3)


def test_the_training_loss_is_the_sum_of_the_term_means_weighted_by_lambda():
    train = [item([0, 1, 1], mode=4, phrase=[0, 1, 2]), item([0, 0, 0, 1])]
    c = cfg(modes=True, clip=True)
    ctx = make_context(train, c, space(), "cpu", lam_mode=0.5, lam_clip=2.0)
    batch = collate(train)
    out = outputs(batch, modes=True, clip=torch.ones(DIM))
    t = {name: float(num / den) for name, (num, den) in loss_terms(out, batch, ctx, c).items()}
    assert float(total_loss(loss_terms(out, batch, ctx, c), ctx)) == pytest.approx(t["score"] + 0.5 * t["mode"] + 2.0 * t["clip"])
    assert float(probe_loss(out, batch, ctx, c)) == pytest.approx(float(total_loss(loss_terms(out, batch, ctx, c), ctx)))


def test_the_score_weight_lam_score_scales_the_score_term_and_zero_leaves_only_the_heads():
    train = [item([0, 1, 1], mode=4, phrase=[0, 1, 2]), item([0, 0, 0, 1])]
    c = cfg(modes=True, clip=True)
    batch = collate(train)
    out = outputs(batch, modes=True, clip=torch.ones(DIM))
    for lam in (1.0, 0.5, 0.0):
        ctx = make_context(train, c, space(), "cpu", lam_mode=0.5, lam_clip=2.0, lam_score=lam)
        t = {name: float(num / den) for name, (num, den) in loss_terms(out, batch, ctx, c).items()}
        assert float(total_loss(loss_terms(out, batch, ctx, c), ctx)) == pytest.approx(lam * t["score"] + 0.5 * t["mode"] + 2.0 * t["clip"])
    assert make_context(train, c, space(), "cpu").lam_score == 1.0  # the default is the earlier loss


def test_pooling_the_terms_of_two_batches_gives_the_mean_over_all_their_steps():
    """The point of numerator/denominator pairs: batches with different numbers of eligible steps still pool exactly."""
    c = ProbeConfig(d_modality=2, n_modalities=D // 2, hidden=8, proj=4, n_modes=N_MODES, clip_dim=DIM, modes=True, clip=True)
    g = torch.Generator().manual_seed(0)
    rich = Item(torch.randn(6, D, generator=g), torch.tensor([0., 1, 1, 1, 1, 1]), torch.tensor([0, 1, 2, 0, 1, 2]), 3)  # 5 mode steps, 6 phrases
    poor = Item(torch.randn(6, D, generator=g), torch.tensor([0., 0, 0, 0, 0, 1]), torch.tensor([-1, -1, -1, -1, 0, 1]), 3)  # 1 mode step, 2 phrases
    ctx = make_context([rich, poor], c, space(), "cpu")
    torch.manual_seed(1)
    model = Probe(c)

    def pair(items):
        batch = collate(items)
        return {k: (float(n), float(d)) for k, (n, d) in loss_terms(model(batch.x), batch, ctx, c).items()}

    together, a, b = pair([rich, poor]), pair([rich]), pair([poor])
    assert a["mode"][1] != b["mode"][1] and a["clip"][1] != b["clip"][1]  # the batches really differ in how many steps these terms have
    for name in together:
        assert a[name][0] + b[name][0] == pytest.approx(together[name][0], rel=1e-5)
        assert a[name][1] + b[name][1] == pytest.approx(together[name][1], rel=1e-5)


def test_a_mode_absent_from_the_training_steps_has_weight_zero_and_adds_nothing():
    train = [item([0, 1, 1], mode=4), item([0, 0, 0, 1])]
    c = cfg(modes=True)
    ctx = make_context(train, c, None)
    batch = collate([item([0, 1, 1], mode=7)])  # mode 7 never occurs in the training steps
    terms = loss_terms(outputs(batch, modes=True), batch, ctx, c)
    assert float(terms["mode"][1]) == 0 and float(terms["mode"][0]) == 0
    assert float(total_loss(terms, ctx)) == pytest.approx(float(terms["score"][0] / terms["score"][1]))


def test_pos_weight_counts_the_steps_of_all_training_items():
    ctx = make_context([item([0, 1]), item([0, 0, 0, 0, 0, 1])], cfg(), None)
    assert float(ctx.pos_weight) == pytest.approx(6 / 2)  # 1 + 5 negative steps, 1 + 1 positive


@pytest.mark.parametrize("y", [[0, 0, 0], [1, 1]])
def test_training_steps_without_both_classes_are_an_error(y):
    with pytest.raises(ValueError, match="positive and negative"):
        make_context([item(y)], cfg(), None)


def test_padding_never_counts():
    train = [item([0, 0, 0, 1])]
    ctx = make_context(train, cfg(), None)
    batch = collate(train + [item([0, 1])])  # the second item is padded by 2 steps
    a = outputs(batch)
    b = {"score": a["score"].clone()}
    b["score"][~batch.valid] = 1e3  # whatever the network outputs on padding
    assert float(probe_loss(a, batch, ctx, cfg())) == float(probe_loss(b, batch, ctx, cfg()))


# ---------------------------------------------------------------- the mode loss
def test_mode_weights_balance_the_training_positive_steps_per_mode():
    train = [item([0, 1, 1, 1], mode=0), item([0, 1], mode=2), item([0, 0], mode=-1)]  # mode 0: 3 positive steps, mode 2: 1
    w = make_context(train, cfg(modes=True), None).mode_weight
    assert w[0] == pytest.approx(4 / (2 * 3)) and w[2] == pytest.approx(4 / (2 * 1))
    assert torch.all(w[[1, 3, 4, 5, 6, 7, 8, 9]] == 0)  # modes without training steps get weight 0


def test_the_mode_loss_is_added_on_the_positive_steps_of_failed_episodes():
    train = [item([0, 1, 1], mode=4), item([0, 0, 0, 1], mode=-1)]
    c = cfg(modes=True)
    ctx = make_context(train, c, None)
    batch = collate(train)
    with_mode = float(probe_loss(outputs(batch, modes=True), batch, ctx, c))
    without = float(probe_loss(outputs(batch), batch, ctx, cfg()))
    assert with_mode - without == pytest.approx(LOG10)  # uniform mode logits over 10 classes, however the classes are weighted


def test_no_mode_term_when_no_step_has_a_mode_target():
    train = [item([0, 1, 1], mode=4), item([0, 0, 0, 1], mode=-1)]
    c = cfg(modes=True)
    ctx = make_context(train, c, None)
    batch = collate(train[1:])  # a success only
    assert float(probe_loss(outputs(batch, modes=True), batch, ctx, c)) == float(probe_loss(outputs(batch), batch, ctx, cfg()))


def test_the_mode_weight_lam_mode_scales_the_term():
    train = [item([0, 1, 1], mode=4), item([0, 0, 0, 1])]
    c = cfg(modes=True)
    batch = collate(train)
    base = float(probe_loss(outputs(batch), batch, make_context(train, cfg(), None), cfg()))
    for lam in (0.5, 2.0):
        ctx = make_context(train, c, None, lam_mode=lam)
        assert float(probe_loss(outputs(batch, modes=True), batch, ctx, c)) - base == pytest.approx(lam * LOG10)


# ---------------------------------------------------------------- the CLIP (InfoNCE) loss
def test_infonce_is_log_of_the_bank_size_when_the_output_is_equally_close_to_every_target():
    train = [item([0, 1, 1], phrase=[0, 1, 2]), item([0, 0, 0, 1])]
    c = cfg(clip=True)
    ctx = make_context(train, c, space(), "cpu")
    batch = collate(train)
    equal = torch.ones(DIM)  # the same cosine to e_0, e_1, e_2
    with_clip = float(probe_loss(outputs(batch, clip=equal), batch, ctx, c))
    without = float(probe_loss(outputs(batch), batch, ctx, cfg()))
    assert with_clip - without == pytest.approx(LOG3)


def test_infonce_is_small_when_the_output_points_at_the_steps_phrase_and_large_otherwise():
    train = [item([0, 1, 1, 1], phrase=[0, 0, 0, 0]), item([0, 0, 0, 1])]
    c = cfg(clip=True)
    ctx = make_context(train, c, space(), "cpu")
    batch = collate(train)
    base = float(probe_loss(outputs(batch), batch, ctx, cfg()))
    right = float(probe_loss(outputs(batch, clip=torch.tensor([1.0, 0, 0])), batch, ctx, c)) - base
    wrong = float(probe_loss(outputs(batch, clip=torch.tensor([0, 1.0, 0])), batch, ctx, c)) - base
    assert right < 1e-3 and wrong == pytest.approx(1 / TAU + math.log(1 + 2 * math.exp(-1 / TAU)), rel=1e-3)


def test_steps_with_a_phrase_outside_the_training_bank_are_left_out():
    c = cfg(clip=True)
    train = [item([0, 1, 1], phrase=[0, 1, 2]), item([0, 0, 0, 1])]
    ctx = make_context(train, c, space(), "cpu")
    plain = collate(train)
    foreign = collate([item([0, 1, 1], phrase=[0, 1, 2]), item([0, 0, 0, 1], phrase=[3, 4, -1, -1])])  # phrases 3 and 4 are not in the bank
    sem = torch.ones(DIM)
    assert float(probe_loss(outputs(plain, clip=sem), plain, ctx, c)) == pytest.approx(float(probe_loss(outputs(foreign, clip=sem), foreign, ctx, c)))


def test_gradients_reach_every_head_through_the_loss():
    c = ProbeConfig(d_modality=2, n_modalities=D // 2, hidden=8, proj=4, n_modes=N_MODES, clip_dim=DIM, modes=True, clip=True)
    train = [item([0, 1, 1], mode=3, phrase=[0, 1, 2]), item([0, 0, 0, 1], phrase=[0, 1, 2, 0])]
    ctx = make_context(train, c, space(), "cpu")
    batch = collate(train)
    model = Probe(c)
    probe_loss(model(torch.randn(2, 4, c.d_in)), batch, ctx, c).backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []
