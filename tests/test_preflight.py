import numpy as np
import pytest

from fp.data.manifest import Episode
from fp.data.splits import Split
from fp.data.targets import Targets
from fp.train.preflight import check_episodes, check_split

WINDOW, VARIANTS, MIN_PHRASES = 3, ("onset_pre", "onset"), 5


def build(n_fail=6, n_ok=6, n_phrases=12, steps=10):
    episodes, ids, modes = {}, {}, {}
    rng = np.random.default_rng(0)
    for i in range(n_fail + n_ok):
        fail = i < n_fail
        key = f"{'f' if fail else 's'}{i}"
        episodes[key] = Episode(key, not fail, f"t{i}", steps, 6 if fail else None, "a")
        ids[key] = rng.integers(0, n_phrases, steps).astype(np.int32)
        if fail:
            modes[key] = i % 3
    targets = Targets([f"p{j}" for j in range(n_phrases)], np.zeros((n_phrases, 4), np.float32), ids, modes)
    return episodes, targets


def split_of(episodes, **override):
    f, s = [k for k in episodes if k[0] == "f"], [k for k in episodes if k[0] == "s"]
    parts = dict(train=f[:3] + s[:3], val=f[3:4] + s[3:4], calibration=f[4:5] + s[4:5], seen=f[5:] + s[5:], unseen=[])
    return Split("s", seed=0, **{**parts, **override})


def lengths(episodes):
    return {k: e.n_steps for k, e in episodes.items()}


def test_a_consistent_dataset_and_split_have_no_problems():
    episodes, targets = build()
    assert check_episodes(episodes, targets, lengths(episodes)) == []
    assert check_split(split_of(episodes), episodes, targets, VARIANTS, WINDOW, MIN_PHRASES) == []


def test_a_feature_length_that_differs_from_the_manifest_is_reported():
    episodes, targets = build()
    bad = {**lengths(episodes), "s6": 9}
    assert any("s6" in p and "9 feature steps" in p for p in check_episodes(episodes, targets, bad))
    assert any("f0" in p for p in check_episodes(episodes, targets, {k: v for k, v in lengths(episodes).items() if k != "f0"}))


def test_phrase_ids_of_the_wrong_length_or_missing_are_reported():
    episodes, targets = build()
    targets.phrase_ids["s7"] = targets.phrase_ids["s7"][:4]
    del targets.phrase_ids["f1"]
    problems = check_episodes(episodes, targets, lengths(episodes))
    assert any("s7" in p and "4 phrase ids" in p for p in problems) and any("f1" in p and "None phrase ids" in p for p in problems)


def test_a_failure_without_a_mode_or_with_an_invalid_one_is_reported():
    episodes, targets = build()
    del targets.modes["f0"]
    targets.modes["f1"] = 99
    assert sum("without a failure mode" in p for p in check_episodes(episodes, targets, lengths(episodes))) == 2


def test_a_split_that_does_not_partition_the_episodes_is_reported():
    episodes, targets = build()
    s = split_of(episodes)
    dup = split_of(episodes, seen=s.seen + [s.train[0]])
    missing = split_of(episodes, seen=s.seen[:-1])
    for broken in (dup, missing):
        assert any("do not partition" in p for p in check_split(broken, episodes, targets, VARIANTS, WINDOW, MIN_PHRASES))


@pytest.mark.parametrize("part", ["train", "val"])
def test_training_or_validation_with_one_outcome_only_is_reported(part):
    episodes, targets = build()
    s = split_of(episodes)
    only_successes = [k for k in getattr(s, part) if k[0] == "s"]
    broken = split_of(episodes, **{part: only_successes, "seen": s.seen + [k for k in getattr(s, part) if k[0] == "f"]})
    assert any(part in p and "only successes" in p for p in check_split(broken, episodes, targets, VARIANTS, WINDOW, MIN_PHRASES))


def test_too_few_distinct_training_phrases_for_the_semantic_head_are_reported():
    episodes, targets = build(n_phrases=3)
    problems = check_split(split_of(episodes), episodes, targets, VARIANTS, WINDOW, MIN_PHRASES)
    assert any("distinct training phrases" in p for p in problems)


def test_a_variant_whose_training_labels_hold_one_class_is_reported():
    episodes, targets = build()
    no_failure_in_train = split_of(episodes, train=["s6", "s7", "s8"], val=["f0", "s9"], calibration=["f1", "s10"], seen=["f2", "f3", "f4", "f5", "s11"])
    problems = check_split(no_failure_in_train, episodes, targets, VARIANTS, WINDOW, MIN_PHRASES)
    assert any("train holds only successes" in p for p in problems) and any("do not hold both classes" in p for p in problems)
