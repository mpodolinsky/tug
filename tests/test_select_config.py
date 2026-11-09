import zlib

import numpy as np
import pytest

from fp.data.manifest import Episode
from fp.data.splits import build_split
from fp.eval import select_config
from fp.eval.select_config import choose_best, select_best, select_best_per_group, split_group, validation_area, validation_area_from_scores
from fp.models.base import Scorer

T, ONSET, WINDOW = 20, 10, 3


def make_episodes(seed=0):
    """6 seen tasks and 1 unseen task, 30 episodes each, about 60% successes."""
    rng = np.random.default_rng(seed)
    eps = {}
    for t in range(7):
        for i in range(30):
            ok = rng.random() < 0.6
            key = f"task{t}/ep{i:02d}"
            eps[key] = Episode(key, ok, f"task{t}", T, None if ok else ONSET, "a")
    return eps


def make_splits(eps, seeds=(0, 1)):
    return [build_split(list(eps.values()), f"run{s}", s, lambda e: e.task, {"task6"}) for s in seeds]


class FakeScorer(Scorer):
    """Success: low noise. Failure: rises from 8 steps before the onset with slope ``quality`` (higher is better separated).
    Remembers every key it was asked about."""

    def __init__(self, name, quality, episodes, log):
        self.name, self.quality, self.episodes, self.log = name, quality, episodes, log

    def fit(self, train_keys, val_keys):
        self.log["fit"] += list(train_keys) + list(val_keys)

    def score(self, keys):
        self.log["score"] += list(keys)
        out = {}
        for k in keys:
            rng = np.random.default_rng(zlib.crc32(k.encode()))
            z = -3 + rng.normal(0, 0.3, T)
            if not self.episodes[k].success:
                z = z + self.quality * np.clip(np.arange(T) - (ONSET - 8), 0, None)
            out[k] = 1 / (1 + np.exp(-z))
        return out


QUALITIES = (0.02, 0.06, 0.25, 0.12)  # weak enough that no configuration saturates at an area of 100


def scorers(eps, log, qualities=QUALITIES):
    return lambda: [FakeScorer(f"euclidean__k{i}_cumsum", q, eps, log) for i, q in enumerate(qualities)]


def test_the_configuration_with_the_best_separated_scores_wins():
    eps = make_episodes()
    sel = select_best(scorers(eps, {"fit": [], "score": []}), make_splits(eps), eps, WINDOW)
    assert sel.best == "euclidean__k2_cumsum"  # the highest quality (0.25)
    assert max(sel.mean_val_area.values()) < 100.0
    assert list(sel.mean_val_area) == [f"euclidean__k{i}_cumsum" for i in range(4)]  # grid order kept
    assert all(len(v) == 2 for v in sel.per_run.values())  # one area per run
    assert sel.mean_val_area["euclidean__k2_cumsum"] == max(sel.mean_val_area.values())


def test_selection_only_ever_scores_calibration_and_validation_episodes():
    eps = make_episodes()
    for split in make_splits(eps):  # one split at a time: one split's calibration episodes are another split's held-out ones
        log = {"fit": [], "score": []}
        select_best(scorers(eps, log), [split], eps, WINDOW)
        assert set(log["score"]) <= set(split.calibration + split.val)
        assert not set(log["score"]) & set(split.seen + split.unseen)
        assert set(log["fit"]) <= set(split.train + split.val) and not set(log["fit"]) & set(split.seen + split.unseen)


def test_the_choice_does_not_depend_on_what_the_scorer_does_to_unseen_tasks():
    """A scorer that returns garbage (or would raise) for the unseen and held-out-seen episodes cannot change the selection."""
    eps = make_episodes()
    splits = make_splits(eps)
    reference = select_best(scorers(eps, {"fit": [], "score": []}), splits, eps, WINDOW)

    class Strict(FakeScorer):
        def score(self, keys):
            assert all(k.startswith("task") and not k.startswith("task6") for k in keys), "asked to score an unseen-task episode"
            return super().score(keys)

    again = select_best(lambda: [Strict(f"euclidean__k{i}_cumsum", q, eps, {"fit": [], "score": []}) for i, q in enumerate(QUALITIES)],
                        splits, eps, WINDOW)
    assert again == reference


def test_the_best_is_chosen_on_the_mean_over_runs_not_on_a_single_run(monkeypatch):
    table = {"euclidean__k0_cumsum": [60.0, 10.0], "euclidean__k1_cumsum": [40.0, 40.0]}  # k0 wins run 0, k1 has the better mean
    calls = {}

    def fake_area(scorer, split, episodes, window):
        calls[scorer.name] = calls.get(scorer.name, -1) + 1
        return table[scorer.name][calls[scorer.name]]

    monkeypatch.setattr(select_config, "validation_area", fake_area)
    eps = make_episodes()
    make = lambda: [FakeScorer(n, 0, eps, {"fit": [], "score": []}) for n in table]
    sel = select_best(make, make_splits(eps), eps, WINDOW)
    assert sel.best == "euclidean__k1_cumsum" and sel.mean_val_area == {"euclidean__k0_cumsum": 35.0, "euclidean__k1_cumsum": 40.0}


def test_a_tie_goes_to_the_earlier_configuration(monkeypatch):
    monkeypatch.setattr(select_config, "validation_area", lambda *a, **k: 30.0)
    eps = make_episodes()
    make = lambda: [FakeScorer(f"euclidean__k{i}_cumsum", 0, eps, {"fit": [], "score": []}) for i in (1, 5, 10)]
    assert select_best(make, make_splits(eps), eps, WINDOW).best == "euclidean__k1_cumsum"


def test_validation_without_both_outcomes_is_an_error():
    eps = make_episodes()
    split = make_splits(eps)[0]
    only_success = [k for k in split.val if eps[k].success]
    broken = type(split)(split.name, split.train, only_success, split.calibration, split.seen, split.unseen, split.seed)
    with pytest.raises(ValueError):
        validation_area(FakeScorer("euclidean__k1_cumsum", 1.0, eps, {"fit": [], "score": []}), broken, eps, WINDOW)


def named_splits(eps, names):
    return [build_split(list(eps.values()), n, i, lambda e: e.task, {"task6"}) for i, n in enumerate(names)]


def test_split_names_map_to_their_held_out_set():
    eps = make_episodes()
    names = ["task_seed0", "task_seed2", "signature_seed1", "skill_stack_objects", "skill_toggle_appliance_on", "paper_seed0", "paper_seed2"]
    got = [split_group(s) for s in named_splits(eps, names)]
    assert got == ["task", "task", "signature", "skill", "skill", "all", "all"]


def test_each_held_out_set_gets_its_own_best_configuration(monkeypatch):
    """Task favours k0, signature k1, skill k2; the choice for one set must not leak into another."""
    wins = {"task": "euclidean__k0_cumsum", "signature": "euclidean__k1_cumsum", "skill": "euclidean__k2_cumsum"}
    monkeypatch.setattr(select_config, "validation_area", lambda scorer, split, episodes, window:
                        50.0 if scorer.name == wins[split_group(split)] else 10.0)
    eps = make_episodes()
    names = ["task_seed0", "task_seed1", "task_seed2", "signature_seed0", "signature_seed1", "signature_seed2"] + [f"skill_{i}" for i in range(8)]
    make = lambda: [FakeScorer(f"euclidean__k{i}_cumsum", 0, eps, {"fit": [], "score": []}) for i in range(3)]
    result = select_best_per_group(make, named_splits(eps, names), eps, WINDOW)
    assert {g: sel.best for g, sel in result.items()} == wins
    assert {g: len(next(iter(sel.per_run.values()))) for g, sel in result.items()} == {"task": 3, "signature": 3, "skill": 8}


def test_libero10_runs_are_pooled_into_one_choice(monkeypatch):
    monkeypatch.setattr(select_config, "validation_area", lambda scorer, split, episodes, window: 30.0)
    eps = make_episodes()
    make = lambda: [FakeScorer("euclidean__k1_cumsum", 0, eps, {"fit": [], "score": []})]
    result = select_best_per_group(make, named_splits(eps, ["paper_seed0", "paper_seed1", "paper_seed2"]), eps, WINDOW)
    assert list(result) == ["all"] and len(result["all"].per_run["euclidean__k1_cumsum"]) == 3


def test_choose_best_takes_the_highest_mean_and_gives_ties_to_the_earlier_configuration():
    sel = choose_best({"a": [60.0, 10.0], "b": [40.0, 40.0], "c": [40.0, 40.0]})
    assert sel.best == "b" and sel.mean_val_area == {"a": 35.0, "b": 40.0, "c": 40.0} and sel.per_run["a"] == [60.0, 10.0]


def test_choose_best_refuses_nan_areas_and_unequal_runs():
    with pytest.raises(ValueError, match="NaN"):
        choose_best({"a": [60.0, float("nan")], "b": [40.0, 40.0]})
    with pytest.raises(ValueError, match="same runs"):
        choose_best({"a": [60.0], "b": [40.0, 40.0]})


def test_saved_scores_and_fitted_scorers_are_selected_by_the_same_rule():
    """The scores a scorer produces, saved and then selected from, give the same areas and the same winner as fitting it in the loop."""
    eps = make_episodes()
    splits = make_splits(eps)
    fitted = select_best(scorers(eps, {"fit": [], "score": []}), splits, eps, WINDOW)
    per_run = {}
    for split in splits:
        for scorer in scorers(eps, {"fit": [], "score": []})():
            saved = scorer.score(split.calibration + split.val)  # what a run on disk holds (it holds all episodes; only these are read)
            per_run.setdefault(scorer.name, []).append(validation_area_from_scores(saved, scorer.name, split, eps, WINDOW))
    assert choose_best(per_run) == fitted


def test_the_validation_area_reads_only_calibration_and_validation_scores():
    eps = make_episodes()
    split = make_splits(eps)[0]
    scores = FakeScorer("euclidean__k1_cumsum", 0.25, eps, {"fit": [], "score": []}).score(split.calibration + split.val)
    area = validation_area_from_scores(scores, "euclidean__k1_cumsum", split, eps, WINDOW)  # no seen or unseen key in `scores`: it would KeyError
    assert 0.0 < area <= 100.0 + 1e-9


def test_the_protocol_calibrator_of_the_scorer_is_used():
    """Distance configurations are calibrated with the SAFE band, which needs at least 7 calibration successes."""
    eps = make_episodes()
    split = make_splits(eps)[0]
    area = validation_area(FakeScorer("euclidean__k1_cumsum", 0.25, eps, {"fit": [], "score": []}), split, eps, WINDOW)
    assert 0.0 < area <= 100.0 + 1e-9  # the area is a sum of trapezoids divided by 40: allow rounding error
