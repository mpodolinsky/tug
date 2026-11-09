import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fp.train.loop import FitResult, Step, TrainConfig, _evaluate, default_lr, fit


class Row:
    """The minimum the toy step function needs from an item."""

    def __init__(self, x, y):
        self.x, self.y = x, y


def dataset(n, seed):
    """Separable toy problem: y = 1 iff the first feature is positive."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randn(1, 2, generator=g)
        out.append(Row(x, (x[:, 0] > 0).float()))
    return out


def noise(n, seed):
    """Labels unrelated to the features: a model can only memorise the training set, so validation gets worse as training goes on."""
    g = torch.Generator().manual_seed(seed)
    return [Row(torch.randn(1, 2, generator=g), (torch.rand(1, generator=g) > 0.5).float()) for _ in range(n)]


def step_fn(model, items):
    """Loss = mean BCE; reported terms: "bce" over all steps and "pos" over the steps with y == 1 only (so its denominators differ per batch)."""
    x, y = torch.cat([it.x for it in items]), torch.cat([it.y for it in items])
    bce = F.binary_cross_entropy_with_logits(model(x).squeeze(-1), y, reduction="none")
    return Step(bce.mean(), {"bce": (float(bce.sum()), float(len(y))), "pos": (float((bce * y).sum()), float(y.sum()))})


def fresh(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))


CFG = TrainConfig(lr=1e-2, batch=8, max_epochs=40, patience=40, seed=0)


def test_the_validation_loss_goes_down_on_a_learnable_problem():
    r = fit(fresh(), dataset(64, 0), dataset(32, 1), step_fn, CFG, select="bce")
    assert isinstance(r, FitResult) and len(r.train["bce"]) == len(r.val["bce"]) == 40
    assert r.val["bce"][r.best_epoch] < 0.5 * r.val["bce"][0] and not r.stopped_early


def test_the_model_ends_with_the_weights_of_its_best_validation_epoch():
    model, val = fresh(), noise(40, 1)
    r = fit(model, noise(12, 0), val, step_fn, TrainConfig(lr=0.05, batch=4, max_epochs=80, patience=80, seed=0), select="bce")
    assert r.val["bce"][-1] > min(r.val["bce"]) + 0.05 and r.best_epoch < len(r.val["bce"]) - 1  # the last epoch was worse: restoring matters
    assert r.best_epoch == int(np.argmin(r.val["bce"]))
    assert _evaluate(model, val, step_fn, 4)["bce"] == pytest.approx(min(r.val["bce"]), rel=1e-5)  # the model in hand is the best one, not the last


def scheduled(a, b):
    """A step function whose validation terms "a" and "b" follow fixed schedules, whatever the model; in training it is the toy loss."""
    a, b = iter(a), iter(b)

    def fn(model, items):
        if model.training:
            return step_fn(model, items)
        return Step(torch.tensor(0.0), {"a": (next(a), 1.0), "b": (next(b), 1.0)})  # one validation batch per epoch

    return fn


def test_the_best_epoch_follows_the_selected_term_and_not_the_training_loss():
    cfg = TrainConfig(lr=1e-2, batch=8, max_epochs=8, patience=8, seed=0)
    a, b = [5.0, 4.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    on_a = fit(fresh(), dataset(32, 0), dataset(4, 1), scheduled(a, b), cfg, select="a")
    on_b = fit(fresh(), dataset(32, 0), dataset(4, 1), scheduled(a, b), cfg, select="b")
    assert on_a.val["a"] == a and on_a.best_epoch == 3 and on_b.best_epoch == 0
    assert on_a.train["bce"][-1] < on_a.train["bce"][3]  # the training loss would have preferred the last epoch


def test_an_unknown_selection_term_is_an_error():
    with pytest.raises(ValueError, match="no term 'nope'"):
        fit(fresh(), dataset(8, 0), dataset(4, 1), step_fn, CFG, select="nope")


def test_a_selection_term_without_any_validation_step_is_an_error():
    only_negatives = [Row(torch.randn(1, 2), torch.zeros(1)) for _ in range(4)]  # "pos" has no eligible step
    with pytest.raises(ValueError, match="no validation step"):
        fit(fresh(), dataset(8, 0), only_negatives, step_fn, CFG, select="pos")


def test_reported_terms_are_pooled_over_batches_so_the_batch_size_does_not_matter():
    """With lr 0 the model never changes: the validation means must be identical for any batching, and equal the mean over all steps."""
    val = dataset(21, 1)
    runs = [fit(fresh(), dataset(8, 0), val, step_fn, TrainConfig(lr=0.0, batch=b, max_epochs=1, patience=1, seed=0), select="bce") for b in (1, 5, 21)]
    for name in ("bce", "pos"):
        assert runs[1].val[name] == pytest.approx(runs[0].val[name])
        assert runs[2].val[name] == pytest.approx(runs[0].val[name])
    model = fresh()
    x, y = torch.cat([it.x for it in val]), torch.cat([it.y for it in val])
    bce = F.binary_cross_entropy_with_logits(model(x).squeeze(-1), y, reduction="none")
    assert runs[0].val["pos"][0] == pytest.approx(float((bce * y).sum() / y.sum()), rel=1e-5)
    old_way = 0.0  # the earlier combination: each batch's mean over its eligible steps, weighted by the batch's number of steps (0 if it has none)
    for i in range(0, len(val), 5):
        chunk = val[i : i + 5]
        num, den = step_fn(model, chunk).parts["pos"]
        old_way += (num / den if den else 0.0) * len(chunk) / len(val)
    assert abs(old_way - runs[0].val["pos"][0]) > 1e-3  # so this test would catch the batch-weighted combination


def test_training_stops_after_patience_epochs_without_a_new_best():
    r = fit(fresh(), dataset(16, 0), dataset(8, 1), step_fn, TrainConfig(lr=0.0, batch=8, max_epochs=50, patience=3, seed=0), select="bce")  # nothing can improve
    assert r.best_epoch == 0 and len(r.val["bce"]) == 4 and r.stopped_early  # epoch 0 is best, then 3 epochs without improvement


def test_the_same_seed_gives_the_same_training():
    a = fit(fresh(), dataset(40, 0), dataset(16, 1), step_fn, CFG, select="bce")
    b = fit(fresh(), dataset(40, 0), dataset(16, 1), step_fn, CFG, select="bce")
    assert a.train == b.train and a.val == b.val


def test_another_seed_changes_the_order_of_the_batches():
    a = fit(fresh(), dataset(40, 0), dataset(16, 1), step_fn, CFG, select="bce")
    b = fit(fresh(), dataset(40, 0), dataset(16, 1), step_fn, TrainConfig(lr=1e-2, batch=8, max_epochs=40, patience=40, seed=1), select="bce")
    assert a.train["bce"] != b.train["bce"]


def test_running_a_batch_in_chunks_gives_the_same_update_for_a_loss_that_is_a_mean_over_the_rollouts():
    """The memory saving (``micro_batch``): chunks of 3 (with a partial last chunk of 2) weighted by their share of the batch of 8."""
    whole = fresh()
    chunked = fresh()
    cfg = TrainConfig(lr=1e-2, batch=8, max_epochs=5, patience=5, seed=0)
    fit(whole, dataset(32, 0), dataset(8, 1), step_fn, cfg, select="bce")
    fit(chunked, dataset(32, 0), dataset(8, 1), step_fn, TrainConfig(lr=1e-2, batch=8, max_epochs=5, patience=5, seed=0, micro_batch=3), select="bce")
    for a, b in zip(whole.parameters(), chunked.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


def test_the_reported_terms_do_not_depend_on_the_chunking():
    a = fit(fresh(), dataset(32, 0), dataset(8, 1), step_fn, TrainConfig(lr=1e-2, batch=8, max_epochs=3, patience=3, seed=0), select="bce")
    b = fit(fresh(), dataset(32, 0), dataset(8, 1), step_fn, TrainConfig(lr=1e-2, batch=8, max_epochs=3, patience=3, seed=0, micro_batch=3), select="bce")
    assert np.allclose(a.train["bce"], b.train["bce"], atol=1e-5) and np.allclose(a.val["pos"], b.val["pos"], atol=1e-5)


def test_empty_sets_are_an_error():
    with pytest.raises(ValueError):
        fit(fresh(), [], dataset(4, 1), step_fn, CFG, select="bce")
    with pytest.raises(ValueError):
        fit(fresh(), dataset(4, 0), [], step_fn, CFG, select="bce")


def test_default_learning_rates_follow_the_trunk():
    assert default_lr("lstm") == 3e-4 and default_lr("mlp") == 1e-3
