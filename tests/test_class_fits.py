"""The Gaussian fits of the classes of saved points (fp/eval/class_fits.py) and their ellipse figures (fp/eval/tsne_plots.py), on synthetic points."""

import numpy as np
import pytest

from fp.eval import class_fits, tsne_plots


def episodes_points(n_success=6, n_failure=6, steps=20, onset=12, modes=(1, 4), seed=0, centres=((0.0, 0.0), (6.0, 0.0))):
    """Successes around centres[0] and failures around centres[1] (std 1 in the plane), ``steps`` steps each, failures with failure step ``onset`` and a mode from ``modes``."""
    rng = np.random.default_rng(seed)
    n = n_success + n_failure
    success = np.array([True] * n_success + [False] * n_failure)
    episode = np.repeat(np.arange(n), steps)
    centre = np.where(success[episode][:, None], np.array(centres[0]), np.array(centres[1]))
    return {"xy": (centre + rng.normal(size=(n * steps, 2))).astype(np.float32), "episode": episode, "step": np.tile(np.arange(steps), n), "success": success,
            "onset": np.where(success, -1, onset), "n_steps": np.full(n, steps), "mode": np.array([-1] * n_success + [modes[i % len(modes)] for i in range(n_failure)])}


def test_the_late_window_is_the_second_half_of_a_success_and_a_failure_up_to_its_failure_step_and_all_is_every_step():
    p = episodes_points(1, 1, steps=20, onset=12)
    late = class_fits.window_mask(p, "late")
    assert np.flatnonzero(late[:20]).tolist() == list(range(10, 20))  # progress step / 19 >= 0.5 from step 10
    assert np.flatnonzero(late[20:]).tolist() == list(range(6, 13))  # step / 12 >= 0.5 from step 6, nothing after the failure step 12
    assert class_fits.window_mask(p, "all").all()
    with pytest.raises(ValueError):
        class_fits.window_mask(p, "middle")


def test_the_classes_are_success_and_failure_or_success_and_the_failure_mode():
    p = episodes_points(1, 1, steps=3, onset=1, modes=(4,))
    assert class_fits.class_ids(p, "outcome").tolist() == [0, 0, 0, 1, 1, 1]
    assert class_fits.class_ids(p, "mode").tolist() == [0, 0, 0, 5, 5, 5]  # 1 + mode 4
    assert [class_fits.class_name("outcome", i) for i in (0, 1)] == ["success", "failure"] and class_fits.class_name("mode", 5) == "placement_or_insertion_failure"
    assert class_fits.n_classes("outcome") == 2 and class_fits.n_classes("mode") == 11
    with pytest.raises(ValueError):
        class_fits.class_ids(p, "task")


def test_each_class_is_fitted_with_the_mean_and_covariance_of_its_points_in_the_window(monkeypatch):
    monkeypatch.setattr(class_fits, "MIN_POINTS", 10)
    p = episodes_points(40, 40, steps=20, onset=19)
    fits = class_fits.fit_classes(p, "outcome", np.ones(len(p["xy"]), bool), "all")
    assert [f.name for f in fits] == ["success", "failure"] and [f.episodes for f in fits] == [40, 40]
    assert np.allclose(fits[0].mean, (0, 0), atol=0.15) and np.allclose(fits[1].mean, (6, 0), atol=0.15)
    assert np.allclose(fits[0].cov, np.eye(2), atol=0.2) and np.array_equal(fits[1].rows, np.flatnonzero(~p["success"][p["episode"]] & (p["step"] <= 99)))


def test_a_class_with_too_few_episodes_or_points_or_outside_the_subset_is_not_fitted(monkeypatch):
    p = episodes_points(6, 6, steps=20, onset=12)
    everything = np.ones(len(p["xy"]), bool)
    monkeypatch.setattr(class_fits, "MIN_POINTS", 100)
    assert [f.name for f in class_fits.fit_classes(p, "outcome", everything, "all")] == ["success", "failure"]  # 120 points each, 6 episodes each
    monkeypatch.setattr(class_fits, "MIN_POINTS", 200)
    assert class_fits.fit_classes(p, "outcome", everything, "all") == []  # the default minimum of points is 200
    monkeypatch.setattr(class_fits, "MIN_POINTS", 100)
    monkeypatch.setattr(class_fits, "MIN_EPISODES", 7)
    assert class_fits.fit_classes(p, "outcome", everything, "all") == []
    monkeypatch.setattr(class_fits, "MIN_EPISODES", 5)
    assert [f.name for f in class_fits.fit_classes(p, "outcome", everything, "late")] == []  # 60 late points of the successes, 42 of the failures
    only_failures = ~p["success"][p["episode"]]
    assert [f.name for f in class_fits.fit_classes(p, "outcome", only_failures, "all")] == ["failure"]
    by_mode = class_fits.fit_classes(p, "mode", everything, "all")
    assert [f.name for f in by_mode] == ["success"]  # 3 episodes of each mode: below the 5 episodes


def test_the_ellipse_is_the_mahalanobis_circle_of_radius_1_and_holds_39_percent_of_a_gaussian():
    mean, cov = np.array([1.0, -2.0]), np.array([[4.0, 1.5], [1.5, 1.0]])
    outline = class_fits.ellipse(mean, cov)
    diff = outline - mean
    assert np.allclose(np.einsum("ij,jk,ik->i", diff, np.linalg.inv(cov), diff), class_fits.RADIUS**2, atol=1e-6) and class_fits.RADIUS == 1.0
    samples = np.random.default_rng(0).multivariate_normal(mean, cov, size=20000) - mean
    inside = np.einsum("ij,jk,ik->i", samples, np.linalg.inv(cov), samples) <= class_fits.RADIUS**2
    assert abs(inside.mean() - (1 - np.exp(-0.5))) < 0.01  # 39% of the points of a two-dimensional Gaussian lie within one standard deviation


def test_the_distance_between_classes_is_in_standard_deviations_whatever_the_scale_of_a_direction():
    rng = np.random.default_rng(0)
    scale = np.array([1.0, 10.0, 0.1])  # the covariance is the same in both classes and not the identity
    x = np.concatenate([rng.normal(size=(4000, 3)) * scale, rng.normal(size=(4000, 3)) * scale + np.array([3.0, 0.0, 0.0]) * scale])
    counts, means, covs = class_fits.class_stats(x, np.repeat([0, 1], 4000), 3)
    d = class_fits.pairwise_distance(counts, means, covs)
    assert counts.tolist() == [4000, 4000, 0] and np.isnan(means[2]).all()
    assert abs(d[0, 1] - 3.0) < 0.1 and np.allclose(d[0, 1], d[1, 0]) and d[0, 0] == 0 and np.isnan(d[0, 2]).all() and np.isnan(d[2]).all()  # 3 standard deviations; no class 2


def test_the_ellipse_figure_draws_a_fit_per_class_and_returns_them(tmp_path, monkeypatch):
    monkeypatch.setattr(class_fits, "MIN_POINTS", 10)
    p = episodes_points(8, 8, steps=20, onset=15)
    path = tmp_path / "outcome_ellipses_all__all.png"
    fits = tsne_plots.draw_ellipses(p, np.ones(len(p["xy"]), bool), "outcome", "all", "outcome_ellipses_all", "title", path, tsne_plots.limits_of(p["xy"]))
    assert [f.name for f in fits] == ["success", "failure"] and path.stat().st_size > 1000
    assert tsne_plots.class_color("outcome", 0) is tsne_plots.DARK_BLUE and tsne_plots.class_color("outcome", 1) is tsne_plots.RED
    assert np.allclose(tsne_plots.class_color("mode", 3), tsne_plots.matplotlib.colors.to_rgb(tsne_plots.MODE_COLORS[2]))
    assert set(tsne_plots.ELLIPSES) <= set(tsne_plots.FIGURES) and set(tsne_plots.ELLIPSES) <= set(tsne_plots.BOTTOM)


def test_a_gaussian_is_fitted_in_the_space_of_three_axes_and_its_ellipsoid_is_the_mahalanobis_sphere_of_radius_1(monkeypatch):
    monkeypatch.setattr(class_fits, "MIN_POINTS", 10)
    p = episodes_points(40, 40, steps=20, onset=19)
    z = np.random.default_rng(1).normal(size=(len(p["xy"]), 1)).astype(np.float32) * 2.0 + np.where(p["success"][p["episode"]][:, None], 0.0, 5.0)
    p["xyz"] = np.concatenate([p["xy"], z], axis=1)
    fits = class_fits.fit_classes(p, "outcome", np.ones(len(z), bool), "all", "xyz")
    assert [f.name for f in fits] == ["success", "failure"] and fits[0].mean.shape == (3,) and fits[0].cov.shape == (3, 3)
    assert np.allclose(fits[0].mean, (0, 0, 0), atol=0.25) and np.allclose(fits[1].mean, (6, 0, 5), atol=0.25) and abs(fits[0].cov[2, 2] - 4.0) < 0.5
    mean, cov = np.array([1.0, -2.0, 0.5]), np.array([[4.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 1.0]])
    x, y, zz = class_fits.ellipsoid(mean, cov, n=12)
    diff = np.stack([x.ravel(), y.ravel(), zz.ravel()], axis=1) - mean
    assert x.shape == (12, 12) and np.allclose(np.einsum("ij,jk,ik->i", diff, np.linalg.inv(cov), diff), class_fits.RADIUS**2, atol=1e-6)


def test_the_gaussians_only_figure_draws_only_the_centres_and_the_version_with_the_points_all_of_them(tmp_path, monkeypatch):
    monkeypatch.setattr(class_fits, "MIN_POINTS", 10)
    p = episodes_points(8, 8, steps=20, onset=15)
    sizes = []
    monkeypatch.setattr(tsne_plots, "save", lambda fig, path: (sizes.append(sorted(len(c.get_offsets()) for c in fig.axes[0].collections)), tsne_plots.plt.close(fig)))
    args = (p, np.ones(len(p["xy"]), bool), "outcome", "all", "outcome_gaussians_all", "t", tmp_path / "a.png", tsne_plots.limits_of(p["xy"]))
    tsne_plots.draw_ellipses(*args, show_points=False)
    tsne_plots.draw_ellipses(*args)
    assert sizes[0] == [1, 1]  # the two centres and no points
    assert sizes[1] == [1, 1, 160, 160, 320]  # the centres, the points of each class and all the points
    assert {e: v[2] for e, v in tsne_plots.ELLIPSES.items() if e.endswith("_all")} == {"outcome_ellipses_all": True, "mode_ellipses_all": True, "outcome_gaussians_all": False, "mode_gaussians_all": False}


def test_the_typical_successful_episode_is_the_one_closest_to_the_gaussian_of_all_the_successes(monkeypatch):
    monkeypatch.setattr(class_fits, "MIN_EPISODES", 1)
    monkeypatch.setattr(class_fits, "MIN_POINTS", 3)
    rng = np.random.default_rng(0)
    centres = np.array([[3.0, 0.0], [0.0, 0.0], [-3.0, 0.0], [0.0, 3.0], [0.0, -3.0]])  # five successes around the origin; the second sits in the middle
    xy = np.concatenate([c + 0.05 * rng.normal(size=(6, 2)) for c in centres] + [np.array([20.0, 20.0]) + rng.normal(size=(6, 2))])  # and a failure far away
    p = {"xy": xy.astype(np.float32), "episode": np.repeat(np.arange(6), 6), "step": np.tile(np.arange(6), 6), "success": np.array([True] * 5 + [False]), "onset": np.array([-1] * 5 + [3]),
         "n_steps": np.full(6, 6), "mode": np.array([-1] * 5 + [1])}
    assert class_fits.typical_episode(p, np.ones(36, bool), "outcome", 0, "all") == 1  # the failure does not pull the Gaussian of the successes
    assert class_fits.typical_episode(p, ~p["success"][p["episode"]], "outcome", 0, "all") is None  # no successes in the subset
    monkeypatch.setattr(class_fits, "MIN_EPISODES", 6)
    assert class_fits.typical_episode(p, np.ones(36, bool), "outcome", 0, "all") is None  # too few successes to fit


def test_the_smoothing_keeps_the_rows_a_constant_and_a_line_and_averages_a_window(monkeypatch):
    line = np.stack([np.arange(10.0), np.full(10, 2.0)], axis=1)
    out = tsne_plots.smoothed(line, 3)
    assert out.shape == line.shape and np.allclose(out[1:-1], line[1:-1]) and np.allclose(out[:, 1], 2.0)  # a line is its own average away from the ends
    assert np.allclose(out[0], (0.0 + 1.0) / 2, atol=1e-9) or np.allclose(out[0, 0], 0.5)  # fewer rows at the end: the mean of the 2 rows there
    assert np.array_equal(tsne_plots.smoothed(line, 1), line)
    assert tsne_plots.smoothed(line[:3], 9).shape == (3, 2)  # a window larger than the path is cut down to it
    spike = np.zeros((9, 2))
    spike[4] = 9.0
    assert np.allclose(tsne_plots.smoothed(spike, 3)[3:6, 0], 3.0)  # the spike is shared by 3 rows


def test_the_trajectory_figure_draws_one_black_path_with_arrowheads_and_a_ring_and_names_it_in_the_legend(tmp_path, monkeypatch):
    monkeypatch.setattr(class_fits, "MIN_POINTS", 10)
    p = episodes_points(8, 8, steps=20, onset=15)
    p["task"] = np.array([f"task{i}" for i in range(16)])
    seen = []
    monkeypatch.setattr(tsne_plots, "save", lambda fig, path: (seen.append(([(tsne_plots.mcolors.to_hex(line.get_color()), len(line.get_xdata())) for line in fig.axes[0].lines], len(fig.axes[0].texts),
                                                                                [t.get_text() for t in fig.legends[0].get_texts()])), tsne_plots.plt.close(fig)))
    mask = np.ones(len(p["xy"]), bool)
    episode = class_fits.typical_episode(p, mask, "outcome", 0)
    tsne_plots.draw_ellipses(p, mask, "outcome", "late", "outcome_ellipses_path", "t", tmp_path / "a.png", tsne_plots.limits_of(p["xy"]), True, episode)
    tsne_plots.draw_ellipses(p, mask, "outcome", "late", "outcome_ellipses_path", "t", tmp_path / "b.png", tsne_plots.limits_of(p["xy"]))
    failure = int(np.flatnonzero(~p["success"])[0])
    tsne_plots.draw_ellipses(p, mask, "outcome", "late", "outcome_ellipses_paths", "t", tmp_path / "c.png", tsne_plots.limits_of(p["xy"]), True, episode, failure, "grasp_failure")
    with_path, without, both = seen
    black = [n for color, n in with_path[0] if color == "#000000"]
    assert black == [20]  # one black line through the 20 steps of the episode
    assert with_path[1] == 7 and without[1] == 0  # 6 arrowheads (annotations) and the caption with the task with the path, none without
    assert with_path[2][-1] == "one successful episode (ring: first step)" and not any("successful episode" in t for t in without[2])
    assert set(tsne_plots.PATHS) <= set(tsne_plots.FIGURES) and set(tsne_plots.PATHS) <= set(tsne_plots.BOTTOM) and p["success"][episode]


def test_a_one_step_success_is_never_the_typical_one_even_when_it_sits_on_the_centre_of_the_gaussian(monkeypatch):
    monkeypatch.setattr(class_fits, "MIN_EPISODES", 1)
    monkeypatch.setattr(class_fits, "MIN_POINTS", 3)
    rng = np.random.default_rng(0)
    centres = np.array([[3.0, 0.0], [0.0, 0.0], [-3.0, 0.0], [0.0, 3.0], [0.0, -3.0]])
    xy = np.concatenate([c + 0.05 * rng.normal(size=(6, 2)) for c in centres] + [np.array([20.0, 20.0]) + rng.normal(size=(6, 2)), np.zeros((1, 2))])  # the last episode: one point at the centre
    p = {"xy": xy.astype(np.float32), "episode": np.array([0] * 6 + [1] * 6 + [2] * 6 + [3] * 6 + [4] * 6 + [5] * 6 + [6]), "step": np.array([0, 1, 2, 3, 4, 5] * 6 + [0]),
         "success": np.array([True] * 5 + [False, True]), "onset": np.array([-1] * 5 + [3, -1]), "n_steps": np.array([6] * 6 + [1]), "mode": np.array([-1] * 5 + [1, -1])}
    everything = np.ones(len(xy), bool)
    assert class_fits.typical_episode(p, everything, "outcome", 0, "all") == 1  # the 6-step episode in the middle, not the 1-step episode on the centre


def test_the_typical_failure_of_a_mode_is_the_one_whose_path_up_to_its_failure_step_is_closest_to_the_gaussian_of_the_mode(monkeypatch):
    monkeypatch.setattr(class_fits, "MIN_EPISODES", 5)
    monkeypatch.setattr(class_fits, "MIN_POINTS", 10)
    p = episodes_points(10, 12, steps=20, onset=15, modes=(1, 4))  # 6 failures of mode 1 and 6 of mode 4, around (6, 0)
    of_mode_4 = np.flatnonzero((p["mode"] == 4))
    best = int(of_mode_4[2])
    rows = np.flatnonzero(p["episode"] == best)
    p["xy"][rows[p["step"][rows] <= 15]] = np.array([6.0, 0.0]) + 0.01  # before the failure step it is on the centre of the class
    p["xy"][rows[p["step"][rows] > 15]] = 100.0  # after it far away: it does not count
    everything = np.ones(len(p["xy"]), bool)
    chosen = class_fits.typical_episode(p, everything, "mode", class_fits.mode_class_id("stuck_or_no_progress") + 3)  # mode 4 is class 5
    assert class_fits.mode_class_id("grasp_failure") == 6 and class_fits.mode_class_id("wrong_object") == 1
    assert chosen == best and p["mode"][chosen] == 4 and not p["success"][chosen]
    assert class_fits.typical_episode(p, everything, "mode", class_fits.mode_class_id("press_failure")) is None  # no failure of that mode


def test_the_figure_with_both_paths_draws_the_failure_dashed_up_to_its_failure_step_with_a_cross_and_names_it_in_the_legend(tmp_path, monkeypatch):
    monkeypatch.setattr(class_fits, "MIN_POINTS", 10)
    p = episodes_points(8, 8, steps=20, onset=15)
    p["task"] = np.array([f"task{i}" for i in range(16)])
    seen = []
    monkeypatch.setattr(tsne_plots, "save", lambda fig, path: (seen.append(([(tsne_plots.mcolors.to_hex(line.get_color()), line.get_linestyle(), len(line.get_xdata())) for line in fig.axes[0].lines],
                                                                             len(fig.axes[0].texts), [t.get_text() for t in fig.legends[0].get_texts()],
                                                                             [len(c.get_offsets()) for c in fig.axes[0].collections])), tsne_plots.plt.close(fig)))
    mask = np.ones(len(p["xy"]), bool)
    success, failure = class_fits.typical_episode(p, mask, "outcome", 0), class_fits.typical_episode(p, mask, "outcome", 1)
    tsne_plots.draw_ellipses(p, mask, "outcome", "late", "outcome_ellipses_paths", "t", tmp_path / "a.png", tsne_plots.limits_of(p["xy"]), True, success, failure, "grasp_failure")
    lines, texts, legend, offsets = seen[0]
    black = sorted((style, n) for color, style, n in lines if color == "#000000")
    assert black == [("-", 20), ("--", 16)]  # the success has its 20 steps, the dashed failure the 16 steps up to its failure step 15
    assert texts == 14 and legend[-2:] == ["one successful episode (ring: first step)", "one grasp_failure episode (X: failure step)"]  # 2 x 6 arrowheads and 2 captions
    assert offsets.count(1) >= 5  # the centres, the two rings and the cross
    assert not p["success"][failure] and set(tsne_plots.PATHS) <= set(tsne_plots.BOTTOM) and tsne_plots.PATHS["mode_ellipses_paths"] == ("mode", "late", True)


def test_the_caption_of_a_path_names_the_task_and_is_cut_when_it_is_too_long():
    fig, ax = tsne_plots.plt.subplots()
    p = {"xy": np.zeros((3, 2), np.float32), "episode": np.zeros(3, int), "step": np.arange(3), "onset": np.array([-1]), "task": np.array(["short_task"])}
    tsne_plots.add_path(ax, p, 0)
    p["task"] = np.array(["x" * 200])
    tsne_plots.add_path(ax, p, 0, False, 1)
    captions = [t.get_text() for t in ax.texts if t.get_text()]
    assert captions[0] == "successful episode: short_task" and len(captions[1]) == tsne_plots.CAPTION_CHARS == 80 and captions[1].endswith("…")
    tsne_plots.plt.close(fig)


def pair_points():
    """12 successes of 20 steps and 12 failures of 20 steps (failure step 15) around two centres, plus a 1-step success and a 2-step one."""
    p = episodes_points(12, 12, steps=20, onset=15)
    extra_xy = np.array([[0.0, 0.0], [0.0, 0.0], [0.1, 0.1]], np.float32)
    return {"xy": np.concatenate([p["xy"], extra_xy]), "episode": np.concatenate([p["episode"], [24, 25, 25]]), "step": np.concatenate([p["step"], [0, 0, 1]]),
            "success": np.concatenate([p["success"], [True, True]]), "onset": np.concatenate([p["onset"], [-1, -1]]), "n_steps": np.concatenate([p["n_steps"], [1, 2]]),
            "mode": np.concatenate([p["mode"], [-1, -1]])}


def test_random_pairs_are_distinct_random_successes_and_failures_never_the_very_short_ones_and_the_same_for_the_same_seed():
    p = pair_points()
    everything = np.ones(len(p["xy"]), bool)
    a = class_fits.random_pairs(p, everything, 5, np.random.default_rng(0))
    b = class_fits.random_pairs(p, everything, 5, np.random.default_rng(0))
    c = class_fits.random_pairs(p, everything, 5, np.random.default_rng(1))
    assert a == b and a != c and len(a) == 5
    for pairs in (a, c):
        assert all(p["success"][s] and not p["success"][f] for s, f in pairs)
        assert len({s for s, _ in pairs}) == 5 and len({f for _, f in pairs}) == 5  # distinct
        assert not {24, 25} & {s for s, _ in pairs}  # the 1- and 2-step successes are shorter than the median
    for seed in range(5):  # asking for all of them: every successful episode of the median length or more, and still not the 1-step one
        every = class_fits.random_pairs(p, everything, 12, np.random.default_rng(seed))
        assert len(every) == 12 and 24 not in [s for s, _ in every]


def test_random_pairs_are_fewer_when_a_class_has_fewer_episodes_and_none_without_one_of_the_classes():
    p = pair_points()
    only_two_failures = np.isin(p["episode"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])  # 12 successes and 2 failures
    assert len(class_fits.random_pairs(p, only_two_failures, 5, np.random.default_rng(0))) == 2
    assert class_fits.random_pairs(p, p["success"][p["episode"]], 5, np.random.default_rng(0)) == []  # no failure in the subset
    assert class_fits.random_pairs(p, ~p["success"][p["episode"]], 5, np.random.default_rng(0)) == []  # no success


def test_render_draws_a_figure_per_pair_only_when_asked_for(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from fp.eval import tsne

    p = pair_points()
    keys = [f"e{i}" for i in range(26)]
    table = {"task": np.array([f"t{i % 4}" for i in range(26)]), "skill": np.array([""] * 26), "success": p["success"], "onset": p["onset"].astype(np.int32), "mode": p["mode"].astype(np.int16),
             "n_steps": p["n_steps"].astype(np.int32)}
    path = tmp_path / "vla" / "points.npz"
    tsne.save_points(path, p["xy"], p["episode"].astype(np.int32), p["step"].astype(np.int32), keys, table, {"dataset": "x", "embedding": "vla", "split": None})
    split = SimpleNamespace(train=[], val=[], calibration=keys[:20], seen=keys[20:], unseen=[])
    monkeypatch.setattr(tsne_plots, "load_inputs", lambda dataset: SimpleNamespace(splits={"task_seed0": split}))
    written = tsne_plots.render(path, "task_seed0", colorings=["outcome_ellipses_pairs"], subsets=["all"], n_pairs=3)
    assert [w.name for w in written] == [f"outcome_ellipses_pair{k}__all.png" for k in (1, 2, 3)] and all(w.stat().st_size > 1000 for w in written)
    assert "outcome_ellipses_pairs" not in tsne_plots.FIGURES and tsne_plots.PAIRS == {"outcome_ellipses_pairs": ("outcome", "late")} and tsne_plots.N_PAIRS == 5


def test_the_post_window_is_the_steps_after_the_failure_step_for_a_failure_and_the_late_half_for_a_success():
    p = episodes_points(1, 1, steps=20, onset=12)
    post = class_fits.window_mask(p, "post")
    assert np.flatnonzero(post[20:]).tolist() == list(range(13, 20))  # the failure: only the steps after its failure step 12
    assert np.array_equal(post[:20], class_fits.window_mask(p, "late")[:20]) and np.flatnonzero(post[:20]).tolist() == list(range(10, 20))  # the success: as the late window
    assert not (post & class_fits.window_mask(p, "late"))[20:].any()  # no failure step is in both windows
    assert "post" in class_fits.WINDOWS and {"outcome_ellipses_post", "mode_ellipses_post"} <= set(tsne_plots.ELLIPSES)
    assert all(v[1] in class_fits.WINDOWS and v[1] in tsne_plots.WINDOW_TITLES for v in tsne_plots.ELLIPSES.values())


def test_the_gaussian_of_the_failures_after_the_failure_step_is_fitted_on_those_steps_only(monkeypatch):
    monkeypatch.setattr(class_fits, "MIN_POINTS", 10)
    p = episodes_points(8, 8, steps=20, onset=10)
    after = (~p["success"][p["episode"]]) & (p["step"] > 10)
    p["xy"][after] += np.array([30.0, 0.0], np.float32)  # the steps after the failure step are far from the earlier ones
    everything = np.ones(len(p["xy"]), bool)
    post = {f.name: f for f in class_fits.fit_classes(p, "outcome", everything, "post")}
    late = {f.name: f for f in class_fits.fit_classes(p, "outcome", everything, "late")}
    assert abs(post["failure"].mean[0] - 36.0) < 0.5 and abs(late["failure"].mean[0] - 6.0) < 0.5  # centres (6, 0) before, (36, 0) after
    assert np.allclose(post["success"].mean, late["success"].mean) and np.array_equal(post["failure"].rows, np.flatnonzero(after))


def test_the_pre_window_is_every_step_up_to_the_failure_step_for_a_failure_and_every_step_for_a_success():
    p = episodes_points(1, 1, steps=20, onset=12)
    pre, late, post, everything = (class_fits.window_mask(p, w) for w in ("pre", "late", "post", "all"))
    assert pre[:20].all() and np.flatnonzero(pre[20:]).tolist() == list(range(13))  # the success: all 20 steps; the failure: steps 0 to its failure step 12
    assert (late[20:] <= pre[20:]).all() and np.flatnonzero(late[20:]).tolist() == list(range(6, 13))  # late is the second half of it
    assert np.array_equal(pre[20:] | post[20:], everything[20:]) and not (pre[20:] & post[20:]).any()  # for a failure pre and post make all of it
    assert {"outcome_ellipses_pre", "mode_ellipses_pre"} <= set(tsne_plots.ELLIPSES) and class_fits.WINDOWS == ("late", "all", "post", "pre")


def seen_unseen_masks(p, unseen_failures=True):
    """Seen = the even episodes, unseen = the odd ones (of the 8 successes and 8 failures of ``episodes_points(8, 8)``); without ``unseen_failures`` the unseen set has no failure."""
    even = (np.arange(16) % 2 == 0)[p["episode"]]
    odd = ~even if unseen_failures else (~even) & p["success"][p["episode"]]
    return even, odd


def test_the_train_unseen_figure_has_two_panels_on_the_same_axes_one_legend_with_both_counts_and_no_title(monkeypatch, tmp_path):
    monkeypatch.setattr(class_fits, "MIN_EPISODES", 1)
    monkeypatch.setattr(class_fits, "MIN_POINTS", 10)
    p = episodes_points(8, 8, steps=20, onset=15)
    seen, unseen = seen_unseen_masks(p)
    captured = []
    monkeypatch.setattr(tsne_plots, "save", lambda fig, path: (captured.append((fig, path)), None)[1])
    fits_seen, fits_unseen = tsne_plots.draw_train_unseen(p, seen, unseen, "outcome", "pre", tmp_path / "a.png", tsne_plots.limits_of(p["xy"]))
    fig, path = captured[0]
    ax_seen, ax_unseen = fig.axes
    assert ax_seen.get_xlim() == ax_unseen.get_xlim() and ax_seen.get_ylim() == ax_unseen.get_ylim()  # the same axes
    assert [t.get_text() for t in ax_seen.texts if t.get_text()] == ["Train"] and [t.get_text() for t in ax_unseen.texts if t.get_text()] == ["Unseen"]  # a label per panel
    assert not any(ax.get_title() for ax in fig.axes) and fig._suptitle is None  # no title
    assert [f.name for f in fits_seen] == ["success", "failure"] and [f.name for f in fits_unseen] == ["success", "failure"]
    assert len(ax_seen.lines) == 2 and len(ax_unseen.lines) == 2  # an ellipse outline per class and panel
    legend = fig.legends[0]
    assert legend.get_title().get_text() == "episodes (train / unseen)" and [t.get_text() for t in legend.get_texts()] == ["success (4 / 4)", "failure (4 / 4)"]
    assert fig.get_size_inches()[0] == 2 * tsne_plots.SIZE
    tsne_plots.plt.close(fig)


def test_a_class_that_is_fitted_in_only_one_panel_has_a_dash_for_the_other_in_the_legend(monkeypatch, tmp_path):
    monkeypatch.setattr(class_fits, "MIN_EPISODES", 1)
    monkeypatch.setattr(class_fits, "MIN_POINTS", 10)
    p = episodes_points(8, 8, steps=20, onset=15)
    seen, unseen = seen_unseen_masks(p, unseen_failures=False)
    captured = []
    monkeypatch.setattr(tsne_plots, "save", lambda fig, path: captured.append(fig))
    tsne_plots.draw_train_unseen(p, seen, unseen, "outcome", "pre", tmp_path / "a.png", tsne_plots.limits_of(p["xy"]))
    assert [t.get_text() for t in captured[0].legends[0].get_texts()] == ["success (4 / 4)", "failure (4 / –)"]
    tsne_plots.plt.close(captured[0])


def test_the_train_unseen_panels_are_as_tall_as_the_data_within_limits(monkeypatch, tmp_path):
    monkeypatch.setattr(class_fits, "MIN_EPISODES", 1)
    monkeypatch.setattr(class_fits, "MIN_POINTS", 10)
    p = episodes_points(8, 8, steps=20, onset=15)
    seen, unseen = seen_unseen_masks(p)
    captured = []
    monkeypatch.setattr(tsne_plots, "save", lambda fig, path: captured.append(fig))
    sizes = []
    for limits in (((0.0, 10.0), (0.0, 10.0)), ((0.0, 10.0), (0.0, 6.0)), ((0.0, 10.0), (0.0, 1.0))):
        tsne_plots.draw_train_unseen(p, seen, unseen, "outcome", "pre", tmp_path / "a.png", limits)
        sizes.append(captured[-1].get_size_inches()[1])
    # the height of the panels (the figure without the legend margin that make_room adds): square, 60% of the width, and the lower limit of 45% for very flat data
    assert sizes[0] > sizes[1] > sizes[2]
    assert sizes[1] - sizes[2] == pytest.approx((0.6 - tsne_plots.PANEL_ASPECT[0]) * tsne_plots.SIZE, abs=0.35)
    for fig in captured:
        tsne_plots.plt.close(fig)


def test_the_trajectory_candidates_are_random_successes_and_failures_of_the_most_common_modes_never_the_very_short_ones_and_the_same_for_the_same_seed(monkeypatch):
    monkeypatch.setattr(class_fits, "MIN_EPISODES", 1)
    p = pair_points()  # 12 successes + 12 failures (modes 1 and 4 alternate) of 20 steps, and a 1-step and a 2-step success
    p["mode"] = np.concatenate([p["mode"][:12], np.array([1, 1, 1, 4, 4, 6, 1, 4, 6, 1, 4, 1]), p["mode"][24:]])  # failures: mode 1 x 7, mode 4 x 4, mode 6 x 2... (12 entries)
    everything = np.ones(len(p["xy"]), bool)
    a = class_fits.trajectory_candidates(p, everything, 5, 3, np.random.default_rng(0))
    b = class_fits.trajectory_candidates(p, everything, 5, 3, np.random.default_rng(0))
    c = class_fits.trajectory_candidates(p, everything, 5, 3, np.random.default_rng(1))
    assert a == b and a != c
    successes, failures = a
    assert len(successes) == 5 and len(set(successes)) == 5 and all(p["success"][e] for e in successes) and not {24, 25} & set(successes)  # none of the 1- and 2-step successes
    assert len(failures) == 3 and not any(p["success"][e] for e in failures)
    assert sorted(int(p["mode"][e]) for e in failures) == [1, 4, 6]  # one failure of each of the three most common modes
    only_two_modes = class_fits.trajectory_candidates(p, everything, 1, 5, np.random.default_rng(0))[1]
    assert len(only_two_modes) == 5 and len(set(only_two_modes)) == 5  # four modes exist: one each, then a random one
    fewer = class_fits.trajectory_candidates(p, np.isin(p["episode"], [0, 1, 2, 12, 13]), 5, 5, np.random.default_rng(0))
    assert len(fewer[0]) == 3 and len(fewer[1]) == 2  # the subset has 3 successes and 2 failures


def test_a_second_draw_of_trajectory_candidates_skips_the_earlier_episodes_and_takes_the_modes_the_first_one_did_not_show_first(monkeypatch):
    monkeypatch.setattr(class_fits, "MIN_EPISODES", 1)
    p = pair_points()
    p["mode"] = np.concatenate([p["mode"][:12], np.array([1, 1, 1, 4, 4, 6, 1, 4, 6, 1, 4, 1]), p["mode"][24:]])  # failures: mode 1 x 6, mode 4 x 4, mode 6 x 2
    everything = np.ones(len(p["xy"]), bool)
    first = class_fits.trajectory_candidates(p, everything, 3, 2, np.random.default_rng(0))
    assert sorted(int(p["mode"][e]) for e in first[1]) == [1, 4]  # the two most common modes
    second = class_fits.trajectory_candidates(p, everything, 3, 2, np.random.default_rng(1), set(first[0]) | set(first[1]))
    assert not set(second[0]) & set(first[0]) and not set(second[1]) & set(first[1])  # new episodes
    assert sorted(int(p["mode"][e]) for e in second[1]) == [1, 6]  # the mode that was not shown (6) first, then the most common of the others (1)
    assert class_fits.trajectory_candidates(p, everything, 3, 2, np.random.default_rng(0), frozenset()) == first  # no exclusion: as before


def test_the_failures_of_the_trajectory_candidates_are_only_of_modes_that_have_enough_episodes_for_a_gaussian():
    p = pair_points()
    p["mode"] = np.concatenate([p["mode"][:12], np.array([1, 1, 1, 4, 4, 6, 1, 4, 6, 1, 4, 1]), p["mode"][24:]])  # failures: mode 1 x 6, mode 4 x 4, mode 6 x 2
    everything = np.ones(len(p["xy"]), bool)
    assert class_fits.MIN_EPISODES == 5
    _, failures = class_fits.trajectory_candidates(p, everything, 1, 3, np.random.default_rng(0))
    assert len(failures) == 3 and {int(p["mode"][e]) for e in failures} == {1}  # modes 4 (4 episodes) and 6 (2) have no Gaussian: only mode 1 is left
