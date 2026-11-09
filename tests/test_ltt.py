import numpy as np
import pytest
from scipy.stats import binom

from fp.calibration.ltt import (TAUS, RiskCounts, calibrate_ltt, first_crossing, fixed_sequence_test, joint_p_values, risk_counts,
                                select_threshold)
from fp.metrics.outcomes import alarm_step


def test_taus_are_ascending_probabilities():
    assert len(TAUS) == 181 and np.all(np.diff(TAUS) > 0) and 0 < TAUS[0] < TAUS[-1] < 1


def test_first_crossing_uses_running_max_and_length_when_never():
    s = np.array([0.1, 0.9, 0.2, 0.3])
    taus = np.array([0.05, 0.5, 0.95])
    assert list(first_crossing(s, taus)) == [0, 1, 4]


def test_fixed_sequence_stops_each_walk_at_first_failure():
    p = np.array([0.001] * 4 + [0.9] + [0.001] * 5)  # grid of 10, hypothesis 4 fails
    acc = fixed_sequence_test(p, delta=0.1, n_starts=1)  # one walk from 0
    assert list(np.flatnonzero(acc)) == [0, 1, 2, 3]


def test_multi_start_can_accept_after_a_failure():
    p = np.array([0.001] * 4 + [0.9] + [0.001] * 5)
    acc = fixed_sequence_test(p, delta=0.1, n_starts=2)  # walks from 0 and from 9
    assert acc[9] and acc[:4].all() and not acc[4]


def test_nothing_accepted_when_p_values_are_large():
    assert not fixed_sequence_test(np.full(20, 0.5), 0.1, 5).any()


def make_problem(n_fail=60, n_succ=60, seed=0, fail_peak=0.99, rise_before_onset=2, success_spikes=0, spike_height=0.9):
    """Failures jump to ``fail_peak`` ``rise_before_onset`` steps before the onset; successes stay below 0.2, except
    ``success_spikes`` of them which reach ``spike_height`` once."""
    rng = np.random.default_rng(seed)
    onsets = [int(rng.integers(20, 30)) for _ in range(n_fail)]
    fails = []
    for o in onsets:
        s = np.full(40, 0.05)
        s[o - rise_before_onset :] = fail_peak
        fails.append(s)
    succ = [np.clip(rng.normal(0.05, 0.02, 40), 0, 0.2) for _ in range(n_succ)]
    for s in succ[:success_spikes]:
        s[10] = spike_height
    return succ, fails, onsets


def run(succ, fails, onsets, a_tpr=0.1, a_fpr=0.1, **kw):
    return calibrate_ltt(succ, fails, onsets, alpha_tpr_early=a_tpr, alpha_fpr=a_fpr, **kw)


def test_certifies_a_separable_problem_and_picks_a_threshold_between_the_classes():
    succ, fails, onsets = make_problem()
    res = run(*make_problem())
    assert res.certified and res.n_accepted > 0
    assert 0.2 < res.tau < 0.99  # above every success score, below every failure score


def test_not_certified_when_failures_never_rise():
    succ, _, _ = make_problem()
    fails = [np.full(40, 0.05) for _ in range(60)]
    assert not run(succ, fails, [25] * 60).certified


def test_alarms_after_the_onset_do_not_count_as_early():
    succ, fails, onsets = make_problem(rise_before_onset=-3)  # every failure rises 3 steps AFTER its onset
    res = run(succ, fails, onsets)
    assert not res.certified and res.tau is None and res.n_accepted == 0


def test_alarm_exactly_at_the_onset_counts_as_early():
    succ, fails, onsets = make_problem(rise_before_onset=0)
    assert run(succ, fails, onsets).certified


def test_the_fpr_guarantee_can_block_certification():
    """Success spikes (20% of successes reach 0.9) are above the failures' peak (0.8): no tau has both TPR_early and FPR in range."""
    succ, fails, onsets = make_problem(fail_peak=0.8, success_spikes=12, spike_height=0.9)
    assert not run(succ, fails, onsets, a_fpr=0.1).certified
    loose = run(succ, fails, onsets, a_fpr=0.4)  # allowing 40% false alarms makes a tau below 0.8 admissible
    assert loose.certified and loose.tau <= 0.8


def test_the_tpr_early_guarantee_can_block_certification():
    succ, fails, onsets = make_problem()
    late = [np.concatenate([np.full(o + 1, 0.05), np.full(40 - o - 1, 0.99)]) for o in onsets]  # first high score one step after the onset
    assert not run(succ, late, onsets, a_tpr=0.1).certified
    assert run(succ, late, onsets, a_tpr=0.99).certified is False  # near-1 targets are not a waiver: p stays ~1 when every failure misses


def test_looser_targets_never_hurt_certification():
    succ, fails, onsets = make_problem(fail_peak=0.8, success_spikes=12, spike_height=0.9)
    certified = [run(succ, fails, onsets, a_fpr=a).certified for a in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)]
    assert certified == sorted(certified)  # False... then True...


def test_p_value_is_the_binomial_tail():
    """Hand check of the quantity the test is built on: 2 violations in 30 at alpha 0.1."""
    assert binom.cdf(2, 30, 0.1) == pytest.approx(0.4114, abs=1e-3)


def brute_force(succ, fails, onsets, a_tpr, a_fpr, delta=0.1, n_starts=5):
    """Plain-loop reference: no shared code with fp.calibration.ltt except the grid."""
    p = []
    stats = []
    for tau in TAUS:
        miss = 0
        for s, o in zip(fails, onsets):
            alarm = next((t for t in range(len(s)) if s[t] >= tau), None)
            miss += not (alarm is not None and alarm <= o)
        fa = sum(any(x >= tau for x in s) for s in succ)
        p.append(max(binom.cdf(miss, len(fails), a_tpr), binom.cdf(fa, len(succ), a_fpr)))
        stats.append((1 - miss / len(fails), fa / len(succ)))
    n = len(TAUS)
    starts = sorted({int(round(x)) for x in np.linspace(0, n - 1, n_starts)})
    accepted = set()
    for j in starts:
        while j < n and p[j] <= delta / len(starts):
            accepted.add(j)
            j += 1
    if not accepted:
        return None, 0
    best = min(accepted, key=lambda j: (-stats[j][0], stats[j][1], j))
    return float(TAUS[best]), len(accepted)


def test_matches_a_plain_loop_reference_on_random_problems():
    rng = np.random.default_rng(1)
    n_cert = 0
    for trial in range(40):
        n_f, n_s = int(rng.integers(30, 80)), int(rng.integers(30, 80))
        onsets = [int(rng.integers(10, 25)) for _ in range(n_f)]
        fails, succ = [], []
        for o in onsets:
            t = np.arange(35)
            lead = rng.integers(-2, 12) if trial % 2 else rng.integers(4, 12)  # odd trials: harder, some failures rise late
            z = -4 + 1.5 * np.clip(t - (o - lead), 0, None) * rng.uniform(0.5, 3) + rng.normal(0, 0.4, 35)
            fails.append(1 / (1 + np.exp(-z)))
        for _ in range(n_s):
            succ.append(1 / (1 + np.exp(-(-4 + rng.normal(0, 0.8, 35) + 2 * (rng.random(35) < 0.03)))))
        a_tpr, a_fpr = float(rng.choice([0.1, 0.2, 0.3, 0.4])), float(rng.choice([0.1, 0.2, 0.3, 0.4]))
        tau, n_acc = brute_force(succ, fails, onsets, a_tpr, a_fpr)
        res = run(succ, fails, onsets, a_tpr, a_fpr)
        assert (res.tau, res.n_accepted) == (tau, n_acc), trial
        n_cert += tau is not None
    assert 3 < n_cert < 37  # the comparison covers both certified and uncertified problems


def test_ties_go_to_the_lowest_fpr_then_the_lowest_threshold():
    succ, fails, onsets = make_problem()
    res = run(succ, fails, onsets)
    assert res.tau == brute_force(succ, fails, onsets, 0.1, 0.1)[0]


def test_first_crossing_agrees_with_the_alarm_step_convention_used_by_the_evaluation():
    """LTT and the evaluation must call the same step an alarm: first score >= threshold, no alarm otherwise."""
    rng = np.random.default_rng(5)
    for _ in range(50):
        s = rng.random(30)
        crossings = first_crossing(s)
        for i in (0, 40, 90, 140, 180):
            alarm = alarm_step(s, np.full(30, TAUS[i]))
            assert crossings[i] == (30 if alarm is None else alarm)


def test_risk_counts_on_a_hand_computed_example():
    fails = [np.array([0.2, 0.9, 0.9, 0.9]), np.array([0.2, 0.2, 0.2, 0.9]), np.array([0.1, 0.1, 0.1, 0.1])]
    onsets = [2, 2, 2]  # failure 0 alarms at step 1 (early), failure 1 at step 3 (after the onset), failure 2 never
    succ = [np.array([0.3, 0.3, 0.3]), np.array([0.3, 0.95, 0.3])]
    c = risk_counts(succ, fails, onsets)
    i = int(np.argmin(np.abs(TAUS - 0.5)))  # tau ~ 0.5: only scores of 0.9 and above alarm
    assert (c.n_fail, c.n_succ) == (3, 2)
    assert c.n_miss_early[i] == 2 and c.n_false_alarm[i] == 1
    assert c.tpr_early[i] == pytest.approx(1 / 3) and c.fpr[i] == pytest.approx(0.5)


def test_joint_p_value_is_the_larger_of_the_two_binomial_tails():
    c = RiskCounts(np.array([2, 0]), np.array([1, 5]), n_fail=30, n_succ=20)
    p = joint_p_values(c, alpha_tpr_early=0.1, alpha_fpr=0.2)
    assert p[0] == pytest.approx(max(binom.cdf(2, 30, 0.1), binom.cdf(1, 20, 0.2)))
    assert p[1] == pytest.approx(max(binom.cdf(0, 30, 0.1), binom.cdf(5, 20, 0.2)))


def test_selection_prefers_tpr_early_then_fpr_then_the_lowest_tau():
    n_miss = np.array([5, 2, 2, 2, 0])
    n_fa = np.array([0, 9, 4, 4, 9])
    counts = RiskCounts(n_miss, n_fa, n_fail=10, n_succ=10)
    accepted = np.array([True, True, True, True, False])  # the best-looking tau (index 4) was not accepted
    assert select_threshold(accepted, counts).tau == TAUS[2]  # tpr_early 0.8 beats 0.5; FPR 0.4 beats 0.9; index 2 before 3
    assert select_threshold(np.zeros(5, dtype=bool), counts).tau is None


def test_guarantee_holds_in_simulation_the_family_wise_error_is_below_delta():
    """Known true risks from a large reference sample; over 150 calibration draws the chance that ANY accepted tau truly violates a
    target must be <= delta = 0.1 (the chance of accepting nothing is irrelevant here)."""
    rng = np.random.default_rng(11)
    L = 35

    def fails(n):
        onsets = rng.integers(8, 26, n)
        lead = rng.integers(-3, 12, n)
        slope = rng.uniform(0.4, 2.0, n)
        z = -4 + slope[:, None] * np.clip(np.arange(L)[None] - (onsets - lead)[:, None], 0, None) + rng.normal(0, 0.5, (n, L))
        return list(1 / (1 + np.exp(-z))), list(onsets)

    def succs(n):
        return list(1 / (1 + np.exp(-(-4 + rng.normal(0, 0.9, (n, L)) + 2.5 * (rng.random((n, L)) < 0.03)))))

    f_ref, o_ref = fails(5000)
    ref = risk_counts(succs(5000), f_ref, o_ref)
    a_tpr, a_fpr = 0.6, 0.2
    invalid = ((1 - ref.tpr_early) > a_tpr) | (ref.fpr > a_fpr)
    assert 5 < (~invalid).sum() < len(TAUS) - 5  # the targets sit inside the grid: some taus are valid, some are not
    wrong = certified = 0
    for _ in range(150):
        f, o = fails(60)
        counts = risk_counts(succs(60), f, o)
        accepted = fixed_sequence_test(joint_p_values(counts, a_tpr, a_fpr), 0.1, 5)
        wrong += bool((accepted & invalid).any())
        certified += bool(accepted.any())
    assert wrong / 150 <= 0.10
    assert certified > 10  # and the procedure does certify sometimes, so the check above is not vacuous


def test_requires_failures_and_successes_with_matching_onsets():
    succ, fails, onsets = make_problem(10, 10)
    with pytest.raises(ValueError):
        run(succ, [], [])
    with pytest.raises(ValueError):
        run(succ, fails, onsets[:-1])
