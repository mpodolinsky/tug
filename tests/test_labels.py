import numpy as np
import pytest

from fp.data.labels import VARIANTS, build_labels

N, O, W = 30, 12, 6  # rollout length, onset, window


def positives(variant, n=N, o=O, w=W):
    e = build_labels(variant, n, o, w)
    return e.n, list(np.flatnonzero(e.y))


def test_onset_window_cut_at_onset():
    n, pos = positives("onset")
    assert n == O + 1 and pos == list(range(O - W, O + 1))


def test_onset_pre_is_whole_prefix():
    n, pos = positives("onset_pre")
    assert n == O + 1 and pos == list(range(0, O + 1))


def test_onset_post_is_window_and_everything_after():
    n, pos = positives("onset_post")
    assert n == N and pos == list(range(O - W, N))


def test_onset_traj_is_every_step():
    n, pos = positives("onset_traj")
    assert n == N and pos == list(range(N))


def test_onset_pm_cut_is_window_on_both_sides():
    n, pos = positives("onset_pm_cut")
    assert n == O + W + 1 and pos == list(range(O - W, O + W + 1))


def test_pm_cut_is_clamped_at_rollout_end():
    n, pos = positives("onset_pm_cut", o=N - 2)
    assert n == N and pos[-1] == N - 1


@pytest.mark.parametrize("variant", VARIANTS)
def test_success_is_all_zero_and_uncut(variant):
    e = build_labels(variant, N, None, W)
    assert e.n == N and e.y.sum() == 0


@pytest.mark.parametrize("variant", VARIANTS)
def test_onset_at_step_zero_is_valid(variant):
    e = build_labels(variant, N, 0, W)
    assert e.y[0] == 1 and e.n >= 1


@pytest.mark.parametrize("variant", VARIANTS)
def test_length_matches_label_array(variant):
    e = build_labels(variant, N, O, W)
    assert len(e.y) == e.n <= N


def test_pre_window_and_post_cover_a_failed_rollout():
    """onset_pre (up to the onset) and onset_post (window onwards) overlap exactly on the window."""
    pre = set(positives("onset_pre")[1])
    post = set(positives("onset_post")[1])
    assert pre & post == set(range(O - W, O + 1))
    assert pre | post == set(range(N))


def test_bad_onset_and_variant_raise():
    with pytest.raises(ValueError):
        build_labels("onset", N, N, W)
    with pytest.raises(ValueError):
        build_labels("made_up", N, O, W)
