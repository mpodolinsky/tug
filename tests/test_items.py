import numpy as np
import torch

from fp.data.manifest import Episode
from fp.train.items import collate, make_item

N, ONSET, WINDOW, D = 12, 7, 3, 4


def failed(n=N, onset=ONSET):
    return Episode("f", False, "t", n, onset, "a")


def success(n=9):
    return Episode("s", True, "t", n, None, "a")


def arrays(n):
    return torch.arange(n * D, dtype=torch.float32).reshape(n, D), np.arange(n, dtype=np.int32)


def test_onset_variant_cuts_at_the_onset_and_labels_the_window():
    x, ids = arrays(N)
    it = make_item(failed(), x, ids, mode=3, variant="onset", window=WINDOW)
    assert it.x.shape == (ONSET + 1, D) and it.y.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert it.phrase.tolist() == list(range(ONSET + 1)) and it.phrase.dtype == torch.int64 and it.mode == 3


def test_onset_pre_labels_every_kept_step_and_onset_traj_keeps_the_whole_rollout():
    x, ids = arrays(N)
    assert make_item(failed(), x, ids, 0, "onset_pre", WINDOW).y.tolist() == [1] * (ONSET + 1)
    traj = make_item(failed(), x, ids, 0, "onset_traj", WINDOW)
    assert len(traj.y) == N and traj.y.sum() == N


def test_a_success_is_uncut_all_zero_and_has_no_mode():
    x, ids = arrays(9)
    it = make_item(success(), x, ids, None, "onset_pre", WINDOW)
    assert len(it.y) == 9 and it.y.sum() == 0 and it.mode == -1


def test_the_features_are_the_given_ones_not_a_copy_with_other_values():
    x, ids = arrays(N)
    it = make_item(failed(), x, ids, 0, "onset", WINDOW)
    assert torch.equal(it.x, x[: ONSET + 1]) and it.x.data_ptr() == x.data_ptr()  # a view of the standardised features, not a copy


def test_collate_pads_with_zeros_and_marks_the_real_steps():
    x, ids = arrays(N)
    a = make_item(failed(), x, ids, 2, "onset", WINDOW)  # 8 steps
    b = make_item(success(5), *arrays(5), None, "onset", WINDOW)  # 5 steps
    batch = collate([a, b])
    assert batch.x.shape == (2, 8, D) and batch.y.shape == batch.valid.shape == batch.phrase.shape == batch.mode.shape == (2, 8)
    assert batch.valid.tolist() == [[True] * 8, [True] * 5 + [False] * 3]
    assert torch.all(batch.x[1, 5:] == 0) and torch.all(batch.phrase[1, 5:] == -1)


def test_the_mode_target_sits_on_the_positive_steps_of_failed_episodes_only():
    x, ids = arrays(N)
    batch = collate([make_item(failed(), x, ids, 2, "onset", WINDOW), make_item(success(5), *arrays(5), None, "onset", WINDOW)])
    assert batch.mode[0].tolist() == [-1, -1, -1, -1, 2, 2, 2, 2]  # y == 1 on the window only
    assert torch.all(batch.mode[1] == -1)
