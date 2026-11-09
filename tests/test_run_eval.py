import numpy as np
import pytest

from fp.data.inputs import Inputs
from fp.data.manifest import Episode
from fp.data.splits import Split
from fp.calibration.max_conformal import MaxConformal
from fp.calibration.safe_band import SafeBand
from fp.eval.run_eval import (EVAL_SETS, calibrator_names, evaluate_ltt, evaluate_run, evaluate_saved_run, evaluate_scores, is_current, rates_at,
                              write_evaluation)
from fp.results import read_json, write_run
from fp.metrics.curves import ALPHAS

T, ONSET, WINDOW = 20, 12, 3
METHOD = "lstm_plain__onset_pre"


def world(n_cal_succ=20):
    """Calibration: n_cal_succ successes and 10 failures. val / seen / unseen: 10 successes and 8 failures each (unseen: 12 and 8)."""
    episodes, keys = {}, {name: [] for name in ("train", "calibration", "val", "seen", "unseen")}
    layout = {"train": (6, 4), "calibration": (n_cal_succ, 10), "val": (10, 8), "seen": (10, 8), "unseen": (12, 8)}
    for part, (n_ok, n_bad) in layout.items():
        for i in range(n_ok + n_bad):
            ok = i < n_ok
            key = f"{part}/{'s' if ok else 'f'}{i:02d}"
            episodes[key] = Episode(key, ok, part, T, None if ok else ONSET, "a")
            keys[part].append(key)
    split = Split("task_seed0", keys["train"], keys["val"], keys["calibration"], keys["seen"], keys["unseen"], seed=3)
    return episodes, split


def separable_scores(episodes, split):
    """Calibration successes peak between 0.10 and 0.40 (distinct, below the 0.5 start point of the LTT grid); every other success stays at
    0.05; every failure is at 0.9 from step 2."""
    out = {}
    for k, e in episodes.items():
        s = np.full(T, 0.05)
        if e.success and k in split.calibration:
            s[3] = 0.10 + 0.005 * int(k.split("s")[-1])
        if not e.success:
            s[2:] = 0.9
        out[k] = s
    return out


def inputs_of(episodes, split):
    return Inputs("synthetic", episodes, None, {split.name: split}, WINDOW)


def test_a_separable_detector_scores_100_on_every_set_with_the_curve_held_flat():
    episodes, split = world()
    results = evaluate_scores(separable_scores(episodes, split), split, episodes, WINDOW, MaxConformal())
    assert set(results) == set(EVAL_SETS)
    for r in results.values():
        assert r.area == pytest.approx(100.0) and r.extended  # TPR early 100 at FPR 0; no alpha reaches FPR 40, so the curve is extended


def test_each_set_holds_the_episodes_of_its_name_and_no_others():
    episodes, split = world()
    results = evaluate_scores(separable_scores(episodes, split), split, episodes, WINDOW, MaxConformal())
    assert (results["val"].n_fail, results["val"].n_succ) == (8, 10) and (results["seen"].n_fail, results["seen"].n_succ) == (8, 10)
    assert (results["unseen"].n_fail, results["unseen"].n_succ) == (8, 12)
    assert all(len(r.rows) == len(ALPHAS) for r in results.values())


def test_a_set_with_one_outcome_has_no_area_not_an_area_of_zero():
    episodes, split = world()
    only_failures = Split(split.name, split.train, split.val, split.calibration, split.seen, [k for k in split.unseen if k.split("/")[1][0] == "f"], 3)
    only_successes = Split(split.name, split.train, [k for k in split.val if k.split("/")[1][0] == "s"], split.calibration, split.seen, split.unseen, 3)
    scores = separable_scores(episodes, split)
    assert np.isnan(evaluate_scores(scores, only_failures, episodes, WINDOW, MaxConformal())["unseen"].area)
    assert np.isnan(evaluate_scores(scores, only_successes, episodes, WINDOW, MaxConformal())["val"].area)
    assert evaluate_scores(scores, only_failures, episodes, WINDOW, MaxConformal())["val"].area == pytest.approx(100.0)  # the other sets are unaffected


def test_the_calibrator_is_fitted_on_the_calibration_successes_only():
    episodes, split = world()
    scores = separable_scores(episodes, split)
    base = evaluate_scores(scores, split, episodes, WINDOW, MaxConformal())
    for kind in ("unseen", "seen", "train"):  # scores of episodes that are not calibration successes cannot move the val rows
        changed = {k: (v * 0 + 0.77 if k.startswith(kind) else v) for k, v in scores.items()}
        after = evaluate_scores(changed, split, episodes, WINDOW, MaxConformal())
        assert after["val"].rows == base["val"].rows
    failures = {k: (v * 0 + 0.77 if k.startswith("calibration/f") else v) for k, v in scores.items()}  # calibration FAILURES are not used either
    assert evaluate_scores(failures, split, episodes, WINDOW, MaxConformal())["val"].rows == base["val"].rows
    moved = {k: (v * 0 + 0.95 if k == split.calibration[0] else v) for k, v in scores.items()}  # a calibration success is: it raises the top threshold above the failures
    assert evaluate_scores(moved, split, episodes, WINDOW, MaxConformal())["val"].rows != base["val"].rows


def test_the_same_calibrator_instance_is_used_for_all_sets():
    """Re-fitting per set would let the set influence its own threshold; the rows of a set must not depend on the order of the sets."""
    episodes, split = world()
    scores = separable_scores(episodes, split)
    scores = {k: (v + np.random.default_rng(len(k)).normal(0, 0.02, T)) for k, v in scores.items()}
    results = evaluate_scores(scores, split, episodes, WINDOW, SafeBand(seed=3))
    alone = evaluate_scores(scores, Split(split.name, split.train, split.val, split.calibration, [], [], 3), episodes, WINDOW, SafeBand(seed=3))
    assert results["val"].rows == alone["val"].rows


def test_calibrator_names_follow_the_protocol_and_only_our_probes_get_a_secondary_one():
    assert calibrator_names(METHOD) == ["max_conformal"]
    assert calibrator_names(METHOD, secondary=True) == ["max_conformal", "safe_band"]
    assert calibrator_names("safe_lstm", secondary=True) == ["safe_band"]
    assert calibrator_names("stac", secondary=True) == ["max_conformal"]
    assert calibrator_names("euclidean__k10_cumsum") == ["safe_band"]


def test_a_calibrator_that_cannot_be_fitted_is_recorded_not_raised(tmp_path):
    episodes, split = world(n_cal_succ=3)  # the SAFE band needs at least two successes in its 30% part, which 3 successes cannot give
    path = tmp_path / "ds" / split.name / METHOD
    write_run(path, separable_scores(episodes, split), {})
    out = evaluate_run(path, METHOD, split, episodes, WINDOW, secondary=True)
    assert "sets" in out["max_conformal"] and "too few" in out["safe_band"]["error"]


def test_evaluation_files_are_written_and_recognised_as_current(tmp_path):
    episodes, split = world()
    path = tmp_path / "ds" / split.name / METHOD
    write_run(path, separable_scores(episodes, split), {})
    inputs = inputs_of(episodes, split)
    assert evaluate_saved_run("ds", split.name, METHOD, path, inputs, secondary=True) == "done"
    summary = read_json(path / "summary.json")
    assert summary["method"] == METHOD and summary["n_calibration_successes"] == 20 and summary["n_calibration_failures"] == 10
    assert set(summary["calibrators"]) == {"max_conformal", "safe_band"} and summary["alphas"] == {"first": 0.005, "last": 0.4, "n": 80}
    unseen = summary["calibrators"]["max_conformal"]["sets"]["unseen"]
    assert unseen["area"] == pytest.approx(100.0) and unseen["n_fail"] == 8 and unseen["n_succ"] == 12
    curve = read_json(path / "curve_max_conformal.json")
    assert curve["calibrator"] == "max_conformal" and set(curve["sets"]) == set(EVAL_SETS) and len(curve["sets"]["unseen"]) == len(ALPHAS) == 80
    assert is_current(path)
    assert evaluate_saved_run("ds", split.name, METHOD, path, inputs) == "skipped"  # current: not recomputed
    assert evaluate_saved_run("ds", split.name, METHOD, path, inputs, force=True) == "done"


def test_an_evaluation_from_other_code_is_not_current_and_is_recomputed(tmp_path):
    episodes, split = world()
    path = tmp_path / "ds" / split.name / METHOD
    write_run(path, separable_scores(episodes, split), {})
    inputs = inputs_of(episodes, split)
    evaluate_saved_run("ds", split.name, METHOD, path, inputs)
    stale = read_json(path / "summary.json")
    stale["code"] = {"git_head": "x", "fp_sha256": "0" * 64}
    (path / "summary.json").write_text(__import__("json").dumps(stale))
    assert not is_current(path) and evaluate_saved_run("ds", split.name, METHOD, path, inputs) == "done" and is_current(path)


# ---------------------------------------------------------------- LTT
def test_ltt_needs_its_targets_to_be_given():
    episodes, split = world()
    with pytest.raises(TypeError):
        evaluate_ltt(separable_scores(episodes, split), split, episodes, WINDOW)


def test_ltt_picks_a_threshold_that_separates_the_classes_and_reports_rates_on_every_set():
    episodes, split = world(n_cal_succ=60)
    scores = separable_scores(episodes, split)
    out = evaluate_ltt(scores, split, episodes, WINDOW, alpha_tpr_early=0.5, alpha_fpr=0.2, delta=0.1)  # 10 calibration failures cannot certify much tighter
    assert out["certified"] and out["tau"] == pytest.approx(0.5) and out["targets"]["delta"] == 0.1  # the lowest accepted tau: the 0.5 start point
    assert set(out["sets"]) == set(EVAL_SETS)
    assert out["sets"]["unseen"]["tpr_early"] == 100.0 and out["sets"]["unseen"]["fpr"] == 0.0
    assert out["sets"]["unseen"]["n_fail"] == 8 and out["sets"]["unseen"]["n_succ"] == 12


def test_ltt_is_not_certified_when_the_targets_cannot_be_met_with_this_many_calibration_episodes():
    episodes, split = world()
    out = evaluate_ltt(separable_scores(episodes, split), split, episodes, WINDOW, alpha_tpr_early=0.001, alpha_fpr=0.001, delta=0.1)
    assert not out["certified"] and out["tau"] is None and "sets" not in out


def test_rates_at_a_constant_threshold_count_an_alarm_where_the_score_reaches_it():
    episodes, split = world()
    scores = separable_scores(episodes, split)
    unseen = [episodes[k] for k in split.unseen]
    r = rates_at(0.9, scores, unseen, WINDOW)  # failures sit at exactly 0.9: ">=" fires; successes stay at 0.05
    assert r["tpr_early"] == 100.0 and r["fpr"] == 0.0
    assert rates_at(0.9000001, scores, unseen, WINDOW)["tpr"] == 0.0
    assert rates_at(0.05, scores, unseen, WINDOW)["fpr"] == 100.0
