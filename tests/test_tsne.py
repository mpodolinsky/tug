"""The t-SNE of the embeddings (fp/eval/tsne.py) and its figures (fp/eval/tsne_plots.py), on synthetic episodes."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fp.data.manifest import Episode
from fp.data.splits import Split
from fp.eval import tsne, tsne_plots
from fp.models.ours import Probe, ProbeConfig

D = 4  # columns per modality of the synthetic probes


def episode(key, success, task="t", n_steps=10, onset=None):
    return Episode(key=key, success=success, task=task, n_steps=n_steps, onset=onset)


def split(**kw):
    return Split(**{"name": "task_seed0", "train": [], "val": [], "calibration": [], "seen": [], "unseen": [], "seed": 0} | kw)


def test_the_subsets_are_all_the_seen_tasks_the_seen_tasks_never_trained_on_and_the_unseen_tasks():
    s = split(train=["a", "b"], val=["c"], calibration=["d"], seen=["e"], unseen=["f", "g"])
    masks = tsne.subset_masks(list("abcdefg"), s)
    assert list(masks) == list(tsne.SUBSETS)
    assert masks["all"].all()
    assert np.flatnonzero(masks["seen"]).tolist() == [0, 1, 2, 3, 4]  # everything but the held-out tasks
    assert np.flatnonzero(masks["seen_held_out"]).tolist() == [3, 4]  # not in the training or the validation episodes
    assert np.flatnonzero(masks["unseen"]).tolist() == [5, 6]


def test_with_nothing_held_out_the_unseen_subset_is_empty_and_the_seen_one_is_everything():
    masks = tsne.subset_masks(["a", "b"], split(train=["a"], calibration=["b"]))
    assert masks["seen"].all() and not masks["unseen"].any()


def test_the_skill_of_an_episode_is_the_skill_fold_that_holds_its_task_out():
    inputs = SimpleNamespace(episodes={k: None for k in "abcd"}, splits={"skill_pick": split(unseen=["a", "b"]), "skill_place": split(unseen=["c"]), "task_seed0": split(unseen=["d"])})
    assert tsne.episode_skills(inputs) == {"a": "pick", "b": "pick", "c": "place", "d": ""}  # the task splits do not name a skill


def test_the_standardisation_gives_every_column_zero_mean_and_unit_spread_whatever_the_chunk_size(monkeypatch):
    x = np.random.default_rng(0).normal(5.0, 3.0, size=(53, 6)).astype(np.float32)
    whole = tsne.standardise_all(x.copy())
    monkeypatch.setattr(tsne, "CHUNK", 7)
    chunked = tsne.standardise_all(x.copy())
    assert np.allclose(whole.mean(axis=0), 0, atol=1e-5) and np.allclose(whole.std(axis=0), 1, atol=1e-4)
    assert np.allclose(whole, chunked, atol=1e-5) and chunked.dtype == np.float32


def blobs(n=150, d=20, gap=12.0):
    rng = np.random.default_rng(0)
    return np.concatenate([rng.normal(0, 1, (n, d)), rng.normal(0, 1, (n, d)) + gap]).astype(np.float32), np.repeat([0, 1], n)


def test_two_separated_clusters_stay_separated_in_two_dimensions():
    x, label = blobs()
    xy = tsne.embed_2d(x, perplexity=20, pca_dim=10, n_jobs=1)
    assert xy.shape == (300, 2) and xy.dtype == np.float32
    centres = [xy[label == k].mean(axis=0) for k in (0, 1)]
    spread = max(np.linalg.norm(xy[label == k] - centres[k], axis=1).mean() for k in (0, 1))
    assert np.linalg.norm(centres[0] - centres[1]) > 3 * spread


def test_the_embedding_is_the_same_for_the_same_seed_and_differs_for_another():
    x, _ = blobs(60)
    a, b, c = (tsne.embed_2d(x.copy(), seed=s, perplexity=15, pca_dim=8, n_jobs=1) for s in (0, 0, 1))
    assert np.array_equal(a, b) and not np.allclose(a, c)


def test_the_points_are_saved_and_loaded_with_their_metadata_and_settings(tmp_path):
    keys = ["e0", "e1"]
    table = tsne.episode_table(SimpleNamespace(episodes={"e0": episode("e0", True, "pick", 3), "e1": episode("e1", False, "place", 4, onset=2)},
                                               splits={"skill_pick": split(unseen=["e0"])}, targets=SimpleNamespace(modes={"e1": 4})), keys)
    xy = np.arange(14, dtype=np.float32).reshape(7, 2)
    path = tmp_path / "a" / "points.npz"
    tsne.save_points(path, xy, np.array([0, 0, 0, 1, 1, 1, 1], np.int32), np.array([0, 1, 2, 0, 1, 2, 3], np.int32), keys, table, {"embedding": "vla", "seed": 3})
    got = tsne.load_points(path)
    assert np.array_equal(got["xy"], xy) and got["config"] == {"embedding": "vla", "seed": 3} and got["keys"].tolist() == keys
    assert got["task"].tolist() == ["pick", "place"] and got["skill"].tolist() == ["pick", ""] and got["success"].tolist() == [True, False]
    assert got["onset"].tolist() == [-1, 2] and got["mode"].tolist() == [-1, 4] and got["n_steps"].tolist() == [3, 4]  # a success has neither an onset nor a mode


def test_the_vla_points_are_every_step_of_every_episode_in_the_order_of_the_keys(monkeypatch):
    raw = {"a": np.full((2, 3), 1.0, np.float32), "b": np.full((3, 3), 2.0, np.float32)}
    monkeypatch.setattr(tsne, "load_features", lambda dataset, keys: {k: raw[k].copy() for k in keys})
    x, episode_index, step = tsne.vla_points("any", ["b", "a"])
    assert x[:, 0].tolist() == [2, 2, 2, 1, 1] and episode_index.tolist() == [0, 0, 0, 1, 1] and step.tolist() == [0, 1, 2, 0, 1]


@pytest.mark.parametrize("mha", [False, True])
def test_the_probe_states_are_the_hidden_states_of_the_saved_probe_on_the_standardised_features(tmp_path, mha):
    torch.manual_seed(0)
    cfg = ProbeConfig(trunk="lstm", mha=mha, modes=True, clip=True, hidden=8, d_modality=D, proj=8, n_attention_heads=2)
    model = Probe(cfg).eval()
    rng = np.random.default_rng(0)
    mean, std = rng.normal(size=3 * D), rng.uniform(1, 2, size=3 * D)
    path = tmp_path / "weights.pt"
    torch.save({"cfg": asdict(cfg), "variant": "onset_pre", "state": model.state_dict(), "mean": mean, "std": std}, path)
    raw = {"a": rng.normal(size=(5, 3 * D)).astype(np.float32), "b": rng.normal(size=(3, 3 * D)).astype(np.float32)}
    got = tsne.probe_states(path, raw, ["a", "b"])
    assert got.shape == (8, 8) and got.dtype == np.float32
    for key, rows in (("a", slice(0, 5)), ("b", slice(5, 8))):  # each episode alone, with the training statistics
        with torch.no_grad():
            expected = model.hidden(torch.from_numpy(((raw[key] - mean) / std).astype(np.float32))[None])[0].numpy()
        assert np.allclose(got[rows], expected, atol=1e-6)


def points(onset_failure=4):
    """Two episodes: a success of 5 steps and a failure of 6 steps (failure step ``onset_failure``) of mode 2."""
    return {"xy": np.random.default_rng(0).normal(size=(11, 2)).astype(np.float32), "episode": np.array([0] * 5 + [1] * 6), "step": np.array(list(range(5)) + list(range(6))),
            "success": np.array([True, False]), "onset": np.array([-1, onset_failure]), "n_steps": np.array([5, 6]), "mode": np.array([-1, 2]), "task": np.array(["x", "y"]),
            "skill": np.array(["pick", "place"]), "keys": np.array(["e0", "e1"]), "config": {"dataset": "any", "embedding": "vla", "split": None}}


def test_a_success_goes_from_light_to_dark_blue_and_a_failure_from_light_blue_to_red_at_its_failure_step_and_stays_red():
    p = points(onset_failure=4)
    colors = tsne_plots.by_outcome(p)
    assert np.allclose(colors[0], tsne_plots.LIGHT_BLUE) and np.allclose(colors[4], tsne_plots.DARK_BLUE)  # first and last step of the success
    assert np.allclose(colors[5], tsne_plots.LIGHT_BLUE)  # first step of the failure
    assert np.allclose(colors[5 + 2], 0.5 * (tsne_plots.LIGHT_BLUE + tsne_plots.RED))  # halfway to the failure step
    assert np.allclose(colors[5 + 4], tsne_plots.RED) and np.allclose(colors[5 + 5], tsne_plots.RED)  # the failure step and after


def test_a_failure_at_the_first_step_is_red_throughout():
    assert np.allclose(tsne_plots.by_outcome(points(onset_failure=0))[5:], tsne_plots.RED)


def test_a_failure_ends_in_the_colour_of_its_mode_and_the_successes_are_coloured_as_by_outcome():
    p = points(onset_failure=4)
    colors = tsne_plots.by_mode(p)
    assert np.allclose(colors[5 + 4], matplotlib_rgb(tsne_plots.MODE_COLORS[2])) and np.allclose(colors[5], tsne_plots.LIGHT_BLUE)
    assert np.allclose(colors[:5], tsne_plots.by_outcome(p)[:5])


def matplotlib_rgb(hex_color):
    import matplotlib.colors as mcolors

    return np.array(mcolors.to_rgb(hex_color))


def test_every_point_of_a_task_has_the_same_colour_and_different_tasks_differ():
    p = points()
    colors = tsne_plots.by_task(p)
    assert (colors[:5] == colors[0]).all() and (colors[5:] == colors[5]).all() and not np.allclose(colors[0], colors[5])


def test_the_skills_are_coloured_by_their_alphabetical_order_and_an_episode_without_a_skill_is_grey():
    p = points()
    p["skill"] = np.array(["place", ""])
    colors = tsne_plots.by_skill(p)
    assert np.allclose(colors[0], matplotlib_rgb(tsne_plots.SKILL_COLORS[0])) and np.allclose(colors[5], (0.7, 0.7, 0.7))


def test_there_is_a_mode_colour_for_every_mode_and_a_skill_colour_for_every_skill_of_libero_x():
    from fp.data.targets import MODE_NAMES

    assert len(tsne_plots.MODE_COLORS) == len(MODE_NAMES) and len(set(tsne_plots.MODE_COLORS)) == len(MODE_NAMES) and len(set(tsne_plots.SKILL_COLORS)) == 8


def test_render_writes_a_png_per_colouring_and_subset_on_the_same_axes_and_skips_empty_subsets(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n_eps, steps = 6, 8
    keys = [f"e{i}" for i in range(n_eps)]
    path = tmp_path / "liberox" / "vla" / "points.npz"
    table = {"task": np.array([f"t{i % 3}" for i in range(n_eps)]), "skill": np.array(["pick", "pick", "place", "place", "pick", "place"]),
             "success": np.array([True, False] * 3), "onset": np.array([-1, 3] * 3, np.int32), "mode": np.array([-1, 1, -1, 4, -1, 7], np.int16), "n_steps": np.full(n_eps, steps, np.int32)}
    tsne.save_points(path, rng.normal(size=(n_eps * steps, 2)).astype(np.float32), np.repeat(np.arange(n_eps, dtype=np.int32), steps), np.tile(np.arange(steps, dtype=np.int32), n_eps),
                     keys, table, {"dataset": "liberox", "embedding": "vla", "split": None})
    s = split(train=keys[:2], val=keys[2:3], calibration=keys[3:4], seen=keys[4:5], unseen=keys[5:])
    monkeypatch.setattr(tsne_plots, "load_inputs", lambda dataset: SimpleNamespace(splits={"task_seed0": s}))
    written = tsne_plots.render(path, "task_seed0")
    assert sorted(p.name for p in written) == sorted(f"{c}__{u}.png" for c in tsne_plots.FIGURES if c not in tsne_plots.NEEDS_XYZ for u in tsne.SUBSETS)  # no 3D plots without a third axis
    assert all(p.parent == path.parent / "figures" / "task_seed0" and p.stat().st_size > 1000 for p in written)
    only = tsne_plots.render(path, "task_seed0", tmp_path / "out", colorings=["outcome"], subsets=["unseen", "all"])
    assert [p.name for p in only] == ["outcome__unseen.png", "outcome__all.png"] and all(p.parent == tmp_path / "out" for p in only)
    monkeypatch.setattr(tsne_plots, "load_inputs", lambda dataset: SimpleNamespace(splits={"task_seed0": split(train=keys)}))
    assert not any(p.name.endswith("__unseen.png") for p in tsne_plots.render(path, "task_seed0", tmp_path / "none"))  # nothing held out: no unseen figure


def test_the_axis_limits_cover_nearly_all_points_whatever_the_outliers():
    xy = np.random.default_rng(0).normal(size=(5000, 2)).astype(np.float32)
    xy[0] = (1e6, -1e6)
    (x0, x1), (y0, y1) = tsne_plots.limits_of(xy)
    assert x1 < 10 and y0 > -10 and x0 < -3 and y1 > 3


def test_keeping_every_kth_step_keeps_the_steps_0_k_2k_of_every_episode():
    step = np.array([0, 1, 2, 3, 4, 0, 1, 2])
    assert np.flatnonzero(tsne.every_kth_step(step, 2)).tolist() == [0, 2, 4, 5, 7]
    assert tsne.every_kth_step(step, 1).all() and np.flatnonzero(tsne.every_kth_step(step, 4)).tolist() == [0, 4, 5]
    with pytest.raises(ValueError):
        tsne.every_kth_step(step, 0)


def test_an_episode_is_the_mean_of_its_rows_up_to_the_failure_step_and_a_success_the_mean_of_all_its_rows():
    x = np.array([[0.0], [2.0], [4.0], [100.0], [10.0], [20.0], [30.0]])  # a failure of 4 steps (failure step 2) and a success of 3 steps
    table = {"onset": np.array([2, -1], np.int32), "n_steps": np.array([4, 3], np.int32)}
    means, episode, last = tsne.episode_means(x, np.array([0, 0, 0, 0, 1, 1, 1]), table)
    assert means[:, 0].tolist() == [2.0, 20.0] and means.dtype == np.float32  # the step after the failure step (100) is not in the mean
    assert episode.tolist() == [0, 1] and last.tolist() == [2, 2]  # the failure step, and the last step of the success


def test_the_ramps_of_an_episode_point_end_in_red_for_a_failure_and_dark_blue_for_a_success():
    p = {"episode": np.array([0, 1]), "step": np.array([2, 2]), "success": np.array([False, True]), "onset": np.array([2, -1]), "n_steps": np.array([4, 3]), "mode": np.array([2, -1])}
    colors = tsne_plots.by_outcome(p)
    assert np.allclose(colors[0], tsne_plots.RED) and np.allclose(colors[1], tsne_plots.DARK_BLUE)
    assert np.allclose(tsne_plots.by_mode(p)[0], matplotlib_rgb(tsne_plots.MODE_COLORS[2]))


def test_the_line_plots_sample_up_to_15_successes_and_25_failures_of_the_subset_the_same_ones_for_the_same_seed():
    success = np.array([True] * 40 + [False] * 60)
    subset = np.arange(5, 100)  # 35 successes and 60 failures
    a = tsne_plots.sample_episodes(subset, success, np.random.default_rng(0))
    b = tsne_plots.sample_episodes(subset, success, np.random.default_rng(0))
    c = tsne_plots.sample_episodes(subset, success, np.random.default_rng(1))
    assert np.array_equal(a, b) and not np.array_equal(a, c) and np.array_equal(a, np.sort(a))
    assert success[a].sum() == tsne_plots.LINE_SUCCESSES == 15 and (~success[a]).sum() == tsne_plots.LINE_FAILURES == 25 and set(a) <= set(subset)
    few = tsne_plots.sample_episodes(np.array([1, 2, 3]), np.array([True, True, False, False]), np.random.default_rng(0))
    assert few.tolist() == [1, 2, 3]  # fewer episodes than the limits: all of them


def test_the_lines_join_consecutive_points_of_a_chosen_episode_and_never_two_episodes():
    p = {"xy": np.arange(14, dtype=np.float32).reshape(7, 2), "episode": np.array([0, 0, 0, 1, 1, 2, 2])}  # 3, 2 and 2 points
    segments, ends, first = tsne_plots.trajectory_segments(p, np.array([0, 1]))
    assert segments.shape == (3, 2, 2) and ends.tolist() == [1, 2, 4] and first.tolist() == [0, 3]  # 2 + 1 segments; no segment from point 2 to 3
    assert segments[0].tolist() == [[0, 1], [2, 3]] and segments[2].tolist() == [[6, 7], [8, 9]]
    assert tsne_plots.trajectory_segments(p, np.array([2]))[0].shape == (1, 2, 2) and tsne_plots.trajectory_segments(p, np.array([], int))[0].shape == (0, 2, 2)


def test_one_point_per_episode_has_no_line_plots(tmp_path, monkeypatch):
    keys = ["e0", "e1"]
    path = tmp_path / "vla" / "points.npz"
    table = {"task": np.array(["a", "b"]), "skill": np.array(["", ""]), "success": np.array([True, False]), "onset": np.array([-1, 3], np.int32), "mode": np.array([-1, 1], np.int16),
             "n_steps": np.array([5, 5], np.int32)}
    tsne.save_points(path, np.random.default_rng(0).normal(size=(2, 2)).astype(np.float32), np.arange(2, dtype=np.int32), np.array([4, 3], np.int32), keys, table,
                     {"dataset": "x", "embedding": "vla", "split": None, "unit": "episode"})
    monkeypatch.setattr(tsne_plots, "load_inputs", lambda dataset: SimpleNamespace(splits={"task_seed0": split(train=keys)}))
    names = sorted(p.name for p in tsne_plots.render(path, "task_seed0", subsets=["all"]))
    assert names == sorted(["mode__all.png", "outcome__all.png", "task__all.png"] + [f"{e}__all.png" for e in tsne_plots.ELLIPSES])  # no skill (none), no lines, no 3D


def test_the_two_principal_axes_follow_the_directions_of_largest_variance_and_report_their_share():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 6)) * np.array([5.0, 3.0, 1.0, 1.0, 1.0, 1.0])
    xy, share = tsne.pca_axes(x.astype(np.float32), 2, fit_points=400)
    assert xy.shape == (400, 2) and xy.dtype == np.float32 and share.shape == (2,)
    assert share[0] >= share[1] > 0 and share.sum() < 1  # the columns are standardised, so the first two axes hold a part of the variance
    assert abs(np.corrcoef(xy[:, 0], xy[:, 1])[0, 1]) < 1e-3  # the projections on two principal axes are uncorrelated


def test_the_projection_is_linear_so_that_a_point_halfway_between_two_lies_halfway_between_their_projections():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(100, 5)).astype(np.float32)
    x[2] = (x[0] + x[1]) / 2
    xy, _ = tsne.pca_axes(x, 2, fit_points=100)
    assert np.allclose(xy[2], (xy[0] + xy[1]) / 2, atol=1e-4)  # the standardisation is also linear, so the midpoint stays the midpoint


def test_the_mode_legend_has_a_ramp_for_the_successes_and_one_for_each_mode_that_occurs_and_no_red_ramp():
    p = points()
    entries = tsne_plots.mode_entries(p)
    assert [label for label, _ in entries] == ["success", "wrong_target"]  # the only failure mode of the points is 2
    assert np.allclose(entries[0][1], tsne_plots.DARK_BLUE) and np.allclose(entries[1][1], matplotlib_rgb(tsne_plots.MODE_COLORS[2]))
    assert not any(np.allclose(end, tsne_plots.RED) for _, end in entries)


def three_d_file(tmp_path, n_eps=4, steps=10):
    """Points with a third axis, as the PCA saves them: ``xy`` is the first two columns of ``xyz``."""
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(n_eps * steps, 3)).astype(np.float32)
    table = {"task": np.array(["a", "b", "a", "b"]), "skill": np.array(["", "", "", ""]), "success": np.array([True, False, True, False]), "onset": np.array([-1, 6, -1, 6], np.int32),
             "mode": np.array([-1, 1, -1, 4], np.int16), "n_steps": np.full(n_eps, steps, np.int32)}
    path = tmp_path / "pca" / "points.npz"
    tsne.save_points(path, xyz[:, :2], np.repeat(np.arange(n_eps, dtype=np.int32), steps), np.tile(np.arange(steps, dtype=np.int32), n_eps), [f"e{i}" for i in range(n_eps)], table,
                     {"dataset": "x", "embedding": "probe", "split": "task_seed0", "reducer": "pca", "explained_variance": [0.04, 0.033, 0.0251]}, {"xyz": xyz})
    return path, xyz


def test_the_pca_keeps_its_third_axis_next_to_the_two_of_the_plane(tmp_path):
    path, xyz = three_d_file(tmp_path)
    got = tsne.load_points(path)
    assert np.array_equal(got["xyz"], xyz) and np.array_equal(got["xy"], xyz[:, :2])


def test_the_3d_plots_are_drawn_only_for_points_with_a_third_axis_one_png_each_with_the_colours_of_the_outcome_or_the_mode(tmp_path, monkeypatch):
    path, _ = three_d_file(tmp_path)
    keys = [f"e{i}" for i in range(4)]
    monkeypatch.setattr(tsne_plots, "load_inputs", lambda dataset: SimpleNamespace(splits={"task_seed0": split(train=keys[:2], calibration=keys[2:3], seen=keys[3:])}))
    written = tsne_plots.render(path, colorings=["outcome_3d", "mode_3d"], subsets=["all", "seen"])
    assert [p.name for p in written] == ["outcome_3d__all.png", "outcome_3d__seen.png", "mode_3d__all.png", "mode_3d__seen.png"] and all(p.stat().st_size > 1000 for p in written)
    assert set(tsne_plots.THREE_D) <= set(tsne_plots.FIGURES) and tsne_plots.BASE["outcome_3d"] == "outcome" and tsne_plots.BASE["mode_lines"] == "mode"


def test_the_limits_have_one_range_per_column_and_a_3d_plot_samples_at_most_the_limit_of_points(tmp_path, monkeypatch):
    assert len(tsne_plots.limits_of(np.arange(30, dtype=np.float32).reshape(10, 3))) == 3 and len(tsne_plots.limits_of(np.arange(20, dtype=np.float32).reshape(10, 2))) == 2
    path, xyz = three_d_file(tmp_path)
    monkeypatch.setattr(tsne_plots, "THREE_D_POINTS", 7)
    drawn = []
    names = []
    monkeypatch.setattr(tsne_plots, "finish", lambda fig, ax, coloring, points, path: (drawn.append([len(a.collections[0].get_offsets()) for a in fig.axes]),
                                                                                       names.append([(a.get_xlabel(), a.get_ylabel(), a.get_zlabel()) for a in fig.axes])))
    points = tsne.load_points(path)
    tsne_plots.draw_3d(points, tsne_plots.by_outcome(points), np.ones(len(xyz), bool), "outcome", "t", path.parent / "a.png", tsne_plots.limits_of(xyz), np.random.default_rng(0))
    assert drawn == [[7, 7, 7]]  # three views of the same 7 points
    assert names == [[("PC1 (4.0%)", "PC2 (3.3%)", "PC3 (2.5%)")] * 3]  # every view has the coordinate system


def test_the_axes_of_pca_points_are_labelled_with_the_principal_axis_and_its_share_of_the_variance_and_those_of_other_points_are_not():
    pca = {"config": {"reducer": "pca", "explained_variance": [0.0404, 0.0331, 0.0249]}}
    assert tsne_plots.axis_labels(pca) == ("PC1 (4.0%)", "PC2 (3.3%)", "PC3 (2.5%)")
    assert tsne_plots.axis_labels({"config": {"reducer": "pca"}}) == ("PC1", "PC2", "PC3")  # points saved without the shares
    assert tsne_plots.axis_labels({"config": {"reducer": "tsne"}}) is None and tsne_plots.axis_labels({"config": {}}) is None


def test_a_2d_figure_of_pca_points_has_arrows_for_its_two_axes_and_one_of_t_sne_points_has_none():
    limits = ((-1.0, 1.0), (-1.0, 1.0))
    fig, ax = tsne_plots.canvas("outcome", "t", limits, ("PC1 (4.0%)", "PC2 (3.3%)"))
    assert [t.get_text() for t in ax.texts if t.get_text()] == ["PC1 (4.0%)", "PC2 (3.3%)"]  # the arrows are annotations without text
    plain_fig, plain = tsne_plots.canvas("outcome", "t", limits)
    assert len(plain.texts) == 0  # no arrows, no labels
    assert len(ax.texts) == 4  # two arrows (annotations) and two labels
    tsne_plots.plt.close("all")


def test_the_3d_gaussian_figures_draw_an_ellipsoid_and_a_centre_per_class_with_or_without_the_points(tmp_path, monkeypatch):
    from fp.eval import class_fits

    monkeypatch.setattr(class_fits, "MIN_EPISODES", 1)
    monkeypatch.setattr(class_fits, "MIN_POINTS", 3)
    path, xyz = three_d_file(tmp_path)
    points = tsne.load_points(path)
    mask = np.ones(len(xyz), bool)
    fits = class_fits.fit_classes(points, "outcome", mask, "late", "xyz")
    assert [f.name for f in fits] == ["success", "failure"] and fits[0].mean.shape == (3,) and fits[0].cov.shape == (3, 3)
    seen = []

    def kinds(fig):
        return {name: sum(type(c).__name__ == name for c in fig.axes[0].collections) for name in ("Poly3DCollection", "Path3DCollection")}

    monkeypatch.setattr(tsne_plots, "finish", lambda fig, ax, coloring, points, path: (seen.append(("with", kinds(fig))), tsne_plots.plt.close(fig)))
    monkeypatch.setattr(tsne_plots, "save", lambda fig, path: (seen.append(("only", kinds(fig))), tsne_plots.plt.close(fig)))
    for show_points in (True, False):
        tsne_plots.draw_3d(points, tsne_plots.by_outcome(points), mask, "outcome", "t", tmp_path / "a.png", tsne_plots.limits_of(xyz), np.random.default_rng(0), fits, "outcome", show_points)
    assert seen == [("with", {"Poly3DCollection": 2, "Path3DCollection": 3}), ("only", {"Poly3DCollection": 2, "Path3DCollection": 2})]  # 2 ellipsoids; the points and 2 centres, or 2 centres


def test_render_draws_the_3d_gaussian_figures_only_for_points_with_a_third_axis(tmp_path, monkeypatch):
    from fp.eval import class_fits

    monkeypatch.setattr(class_fits, "MIN_EPISODES", 1)
    monkeypatch.setattr(class_fits, "MIN_POINTS", 3)
    path, _ = three_d_file(tmp_path)
    keys = [f"e{i}" for i in range(4)]
    monkeypatch.setattr(tsne_plots, "load_inputs", lambda dataset: SimpleNamespace(splits={"task_seed0": split(train=keys[:2], calibration=keys[2:3], seen=keys[3:])}))
    written = tsne_plots.render(path, colorings=list(tsne_plots.THREE_D_FITS), subsets=["all"])
    assert sorted(p.name for p in written) == sorted(f"{c}__all.png" for c in tsne_plots.THREE_D_FITS) and all(p.stat().st_size > 1000 for p in written)
    assert set(tsne_plots.THREE_D_FITS) <= set(tsne_plots.FIGURES) and set(tsne_plots.THREE_D_FITS) <= set(tsne_plots.NEEDS_XYZ) and tsne_plots.BASE["mode_3d_gaussians"] == "mode"


def saved_probe(tmp_path, **cfg_kwargs):
    torch.manual_seed(0)
    cfg = ProbeConfig(trunk="lstm", hidden=8, d_modality=D, **cfg_kwargs)
    model = Probe(cfg).eval()
    mean, std = np.zeros(3 * D), np.ones(3 * D)
    path = tmp_path / "weights.pt"
    torch.save({"cfg": asdict(cfg), "variant": "onset_pre", "state": model.state_dict(), "mean": mean, "std": std}, path)
    return model, path, {"a": np.random.default_rng(0).normal(size=(5, 3 * D)).astype(np.float32)}


def test_the_probe_layers_are_the_trunk_state_the_64_units_of_the_mode_head_and_the_64_d_output_of_the_clip_head(tmp_path):
    model, path, raw = saved_probe(tmp_path, modes=True, clip=True)
    x = torch.from_numpy(raw["a"])[None]
    with torch.no_grad():
        out, h = model(x), model.hidden(x)
        expected = {"hidden": h[0].numpy(), "mode_hidden": torch.relu(model.mode[0](h))[0].numpy(), "score_hidden": torch.relu(model.score[0](h))[0].numpy(), "clip": out["sem"][0].numpy()}
        assert np.allclose(model.mode[2](torch.from_numpy(expected["mode_hidden"])).numpy(), out["mode"][0].numpy(), atol=1e-6)  # the logits are the next layer of the mode head
        assert np.allclose(model.score[2](torch.from_numpy(expected["score_hidden"])).squeeze(-1).numpy(), out["score"][0].numpy(), atol=1e-6)  # and the failure logit that of the score head
    assert tsne.LAYERS == ("hidden", "mode_hidden", "score_hidden", "clip")
    for layer, want in expected.items():
        got = tsne.probe_states(path, raw, ["a"], layer=layer)
        assert got.shape == (5, {"hidden": 8, "mode_hidden": 64, "score_hidden": 64, "clip": 64}[layer]) and np.allclose(got, want, atol=1e-6)
    assert np.array_equal(tsne.probe_states(path, raw, ["a"]), tsne.probe_states(path, raw, ["a"], layer="hidden"))  # the trunk state is the default


def test_a_layer_the_probe_does_not_have_is_an_error_and_the_title_names_the_layer(tmp_path):
    _, path, raw = saved_probe(tmp_path)  # no mode head, no CLIP head
    for layer in ("mode_hidden", "clip"):
        with pytest.raises(ValueError, match="no .* head"):
            tsne.probe_states(path, raw, ["a"], layer=layer)
    with pytest.raises(ValueError, match="unknown layer"):
        tsne.probe_states(path, raw, ["a"], layer="logits")
    assert tsne.probe_states(path, raw, ["a"], layer="score_hidden").shape == (5, 64)  # the score head is always there
    config = {"embedding": "probe", "split": "task_seed0"}
    assert [tsne_plots.embedding_title(config | {"layer": layer}) for layer in (None, "hidden", "mode_hidden", "score_hidden", "clip")] == [
        "probe state (trained on task_seed0)", "probe state (trained on task_seed0)", "probe mode-head layer (64-d) (trained on task_seed0)",
        "probe score-head layer (64-d) (trained on task_seed0)", "probe CLIP head (64-d) (trained on task_seed0)"]


def test_the_3d_gaussian_figures_fit_the_window_of_their_name_late_without_a_suffix_and_pre_with_one(tmp_path, monkeypatch):
    from fp.eval import class_fits

    path, _ = three_d_file(tmp_path)
    keys = [f"e{i}" for i in range(4)]
    monkeypatch.setattr(tsne_plots, "load_inputs", lambda dataset: SimpleNamespace(splits={"task_seed0": split(train=keys[:2], calibration=keys[2:3], seen=keys[3:])}))
    windows = []
    original = class_fits.fit_classes
    monkeypatch.setattr(class_fits, "fit_classes", lambda points, kind, subset, window, columns="xy": (windows.append((window, columns)), original(points, kind, subset, window, columns))[1])
    for figure in ("mode_3d_gaussians", "mode_3d_gaussians_pre", "outcome_3d_ellipsoids_pre", "outcome_3d_gaussians"):
        windows.clear()
        tsne_plots.render(path, colorings=[figure], subsets=["all"])
        assert windows == [({"mode_3d_gaussians": "late", "mode_3d_gaussians_pre": "pre", "outcome_3d_ellipsoids_pre": "pre", "outcome_3d_gaussians": "late"}[figure], "xyz")]
    assert all(window in class_fits.WINDOWS and window in tsne_plots.WINDOW_TITLES for _, _, window in tsne_plots.THREE_D_FITS.values())
    assert tsne_plots.WINDOW_TITLES["late"] == "second half of the steps before the failure"


def test_a_figure_is_saved_in_every_format_without_its_titles_when_the_export_has_none_and_the_setting_is_restored(tmp_path):
    fig, ax = tsne_plots.canvas("outcome", "the title", ((-1.0, 1.0), (-1.0, 1.0)))
    fig.suptitle("the suptitle")
    calls = []
    fig.savefig = lambda file, **kw: calls.append((file.name, kw, ax.get_title(), fig.get_suptitle()))
    with tsne_plots.exporting(tsne_plots.Export(("png", "pdf"), title=False, tight=True)):
        assert tsne_plots.EXPORT.formats == ("png", "pdf")
        tsne_plots.save(fig, tmp_path / "a.png")
    assert [(name, title, suptitle) for name, _, title, suptitle in calls] == [("a.png", "", ""), ("a.pdf", "", "")]  # a file per format, no title
    assert all(kw == {"dpi": tsne_plots.DPI, "bbox_inches": "tight", "pad_inches": 0.05} for _, kw, _, _ in calls)
    assert tsne_plots.EXPORT == tsne_plots.Export() == tsne_plots.Export(("png",), True, False)  # back to the default: png, with the title, not cropped
    with pytest.raises(RuntimeError):
        with tsne_plots.exporting(tsne_plots.Export(("pdf",))):
            raise RuntimeError
    assert tsne_plots.EXPORT == tsne_plots.Export()  # also after an error
    fig2, _ = tsne_plots.canvas("outcome", "kept", ((-1.0, 1.0), (-1.0, 1.0)))
    seen = []
    fig2.savefig = lambda file, **kw: seen.append((file.name, kw, fig2.axes[0].get_title()))
    tsne_plots.save(fig2, tmp_path / "b.png")
    assert seen == [("b.png", {"dpi": tsne_plots.DPI}, "kept")]  # by default one png, with its title, uncropped


def test_render_writes_the_figures_with_and_without_titles_in_two_folders_in_every_format(tmp_path, monkeypatch):
    path, _ = three_d_file(tmp_path)
    keys = [f"e{i}" for i in range(4)]
    monkeypatch.setattr(tsne_plots, "load_inputs", lambda dataset: SimpleNamespace(splits={"task_seed0": split(train=keys[:2], calibration=keys[2:3], seen=keys[3:])}))
    out = tmp_path / "paper"
    written = tsne_plots.render(path, colorings=["outcome"], subsets=["all"], out_dir=out, titles=(True, False), formats=("png", "pdf"), tight=True)
    assert sorted(w.relative_to(out).as_posix() for w in written) == ["with_title/outcome__all.pdf", "with_title/outcome__all.png", "without_title/outcome__all.pdf", "without_title/outcome__all.png"]
    assert all(w.exists() and w.stat().st_size > 1000 for w in written)
    assert tsne_plots.EXPORT == tsne_plots.Export()  # the export does not leak out of render
    one = tsne_plots.render(path, colorings=["outcome"], subsets=["all"], out_dir=tmp_path / "one", titles=(False,), formats=("pdf",))
    assert [w.relative_to(tmp_path / "one").as_posix() for w in one] == ["outcome__all.pdf"] and one[0].exists()  # one setting: no sub-folder, and only the pdf
    default = tsne_plots.render(path, colorings=["outcome"], subsets=["all"], out_dir=tmp_path / "default")
    assert [w.name for w in default] == ["outcome__all.png"]  # as before: one png with its title


def test_the_standardisation_and_the_pca_fitted_on_some_rows_do_not_depend_on_the_others_which_are_projected_onto_the_axes():
    rng = np.random.default_rng(0)
    first = rng.normal(0, 3.0, 300)
    fit_part = np.column_stack([first, first + rng.normal(0, 0.5, 300), rng.normal(0, 1.0, 300)])
    other = np.column_stack([rng.normal(0, 1.0, 100), rng.normal(0, 30.0, 100), rng.normal(0, 1.0, 100)])
    x = np.concatenate([fit_part, other]).astype(np.float32)
    fit = np.array([True] * 300 + [False] * 100)
    standardised = tsne.standardise_all(x.copy(), fit)
    assert np.allclose(standardised[fit].mean(axis=0), 0, atol=1e-4) and np.allclose(standardised[fit].std(axis=0), 1, atol=1e-3)
    assert standardised[~fit][:, 1].std() > 10  # the other rows are scaled with the statistics of the fit rows: the column that varies only in them stays large
    with pytest.raises(ValueError, match="no row"):
        tsne.standardise_all(x.copy(), np.zeros(400, bool))
    fitted, share = tsne.pca_axes(x.copy(), 2, fit_points=1000, fit=fit)
    alone, share_alone = tsne.pca_axes(x[fit].copy(), 2, fit_points=1000)
    everything, _ = tsne.pca_axes(x.copy(), 2, fit_points=1000)
    assert np.allclose(np.abs(fitted[fit]), np.abs(alone), atol=1e-3) and np.allclose(share, share_alone, atol=1e-5)  # the same axes as with the fit rows alone
    assert fitted.shape == (400, 2) and not np.allclose(np.abs(everything[fit]), np.abs(alone), atol=1e-2)  # the other rows would change the axes if they were used


def test_the_fit_rows_are_the_steps_up_to_the_failure_step_of_the_training_episodes_only():
    keys = ["a", "b", "c"]  # a: a training failure (failure step 2), b: a training success, c: not a training episode
    episode = np.array([0] * 5 + [1] * 3 + [2] * 4)
    step = np.array([0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 2, 3])
    onset = np.array([2, -1, 1])
    sp = split(train=["a", "b"], val=["c"])
    fit = tsne.fit_rows(keys, episode, step, onset, sp, "train_pre")
    assert np.flatnonzero(fit).tolist() == [0, 1, 2, 5, 6, 7]  # the failure up to its failure step 2, the success all, nothing of c
    assert tsne.fit_rows(keys, episode, step, onset, sp, "all") is None and tsne.FIT_SETS == ("all", "train_pre")
    with pytest.raises(ValueError, match="unknown fit set"):
        tsne.fit_rows(keys, episode, step, onset, sp, "val")


def test_the_title_says_when_the_axes_are_fitted_on_the_training_episodes_and_the_2d_pre_window_gaussians_exist():
    config = {"embedding": "vla", "split": None, "reducer": "pca"}
    assert tsne_plots.embedding_title(config) == "VLA embedding, PCA"
    assert tsne_plots.embedding_title(config | {"fit_set": "train_pre"}) == "VLA embedding, PCA (axes from the training episodes)"
    assert tsne_plots.embedding_title(config | {"fit_set": "all"}) == "VLA embedding, PCA" and tsne_plots.embedding_title({**config, "reducer": "tsne", "fit_set": "train_pre"}) == "VLA embedding"
    assert tsne_plots.ELLIPSES["mode_gaussians_pre"] == ("mode", "pre", False) and tsne_plots.ELLIPSES["outcome_gaussians_pre"] == ("outcome", "pre", False)
    assert {"mode_gaussians_pre", "outcome_gaussians_pre"} <= set(tsne_plots.BOTTOM) and {"mode_gaussians_pre", "outcome_gaussians_pre"} <= set(tsne_plots.FIGURES)


def test_failures_only_draws_the_failure_mode_gaussians_without_the_successes_and_skips_every_other_figure(tmp_path, monkeypatch):
    from fp.eval import class_fits

    monkeypatch.setattr(class_fits, "MIN_EPISODES", 1)
    monkeypatch.setattr(class_fits, "MIN_POINTS", 3)
    path, xyz = three_d_file(tmp_path)  # episodes 0 and 2 are successes, 1 and 3 failures of the modes 1 and 4
    keys = [f"e{i}" for i in range(4)]
    monkeypatch.setattr(tsne_plots, "load_inputs", lambda dataset: SimpleNamespace(splits={"task_seed0": split(train=keys[:2], calibration=keys[2:3], seen=keys[3:])}))
    drawn, fitted, limits = [], [], []
    monkeypatch.setattr(tsne_plots, "save", lambda fig, file: (drawn.append((file.name, [t.get_text() for t in fig.legends[0].get_texts()], fig.get_suptitle() or fig.axes[0].get_title())), tsne_plots.plt.close(fig)))
    original = class_fits.fit_classes
    monkeypatch.setattr(class_fits, "fit_classes", lambda points, kind, subset, window, columns="xy": (fitted.append((kind, subset.copy(), points["success"][points["episode"]])), original(points, kind, subset, window, columns))[1])
    monkeypatch.setattr(tsne_plots, "limits_of", lambda x: limits.append(len(x)) or ((0.0, 1.0),) * x.shape[1])
    figures = ["mode_gaussians_pre", "outcome_gaussians_pre", "mode_3d_gaussians_pre", "mode_ellipses_pre", "mode_ellipses_paths", "mode", "mode_3d"]
    written = tsne_plots.render(path, colorings=figures, subsets=["all"], failures_only=True)
    assert [w.name for w in written] == ["mode_gaussians_pre_failures__all.png", "mode_3d_gaussians_pre_failures__all.png", "mode_ellipses_pre_failures__all.png"]  # the others are skipped
    assert [name for name, _, _ in drawn] == [w.name for w in written]
    assert all(not (subset & success).any() for kind, subset, success in fitted) and all(kind == "mode" for kind, _, _ in fitted)  # nothing of a success is fitted
    legends = [legend for _, legend, _ in drawn]
    assert all(len(legend) == 2 and not any("success" in entry for entry in legend) for legend in legends)  # the two failure modes only
    assert all(title.endswith(", failures only") for _, _, title in drawn)  # the 2D titles are on the axes, the 3D one is the title of the figure
    assert limits == [20, 20]  # the axis limits are those of the 20 points of the two failures (of 40), for the plane and for the three axes
    assert tsne_plots.MODE_FIT_FIGURES and all(f.startswith("mode") for f in tsne_plots.MODE_FIT_FIGURES) and "mode_ellipses_paths" not in tsne_plots.MODE_FIT_FIGURES
    drawn.clear()
    tsne_plots.render(path, colorings=["mode_gaussians_pre"], subsets=["all"])
    assert drawn[0][0] == "mode_gaussians_pre__all.png" and any("success" in entry for entry in drawn[0][1])  # by default: the successes are in, no suffix


def test_the_font_scale_multiplies_every_font_size_and_the_class_legend_takes_fewer_columns_and_more_room():
    from fp.data.targets import MODE_NAMES

    fits = [SimpleNamespace(id=i, name=name, episodes=100) for i, name in enumerate(["success", "wrong_object", "stuck_or_no_progress", "wrong_target", "placement_or_insertion_failure"])]
    limits = ((0.0, 1.0), (0.0, 1.0))
    fig, ax = tsne_plots.canvas("mode_gaussians_pre", "t", limits, ("PC1", "PC2"))
    columns = tsne_plots.legend_layout(["placement_or_insertion_failure (100 episodes)"] * 5, 2, fig)[0]
    needed = tsne_plots.add_class_legend(fig, "mode", fits, 2)
    base = (ax.title.get_fontsize(), [t.get_fontsize() for t in ax.texts if t.get_text()], fig.legends[0].get_texts()[0].get_fontsize(), fig.legends[0]._ncols)
    assert base == (11, [8, 8], 7.5, 2) and columns == 2 and needed < 0.2 and tsne_plots.EXPORT.font_scale == 1.0  # the defaults: as before, and BOTTOM has the room
    with tsne_plots.exporting(tsne_plots.Export(font_scale=2.0)):
        big, big_ax = tsne_plots.canvas("mode_gaussians_pre", "t", limits, ("PC1", "PC2"))
        big_needed = tsne_plots.add_class_legend(big, "mode", fits, 2)
        scaled = (big_ax.title.get_fontsize(), [t.get_fontsize() for t in big_ax.texts if t.get_text()], big.legends[0].get_texts()[0].get_fontsize(), big.legends[0]._ncols)
        assert tsne_plots.fs(7.5) == 15.0
    assert scaled == (22, [16, 16], 15.0, 1)  # every size doubled; the long entries no longer fit two to a row
    assert big_needed > 2 * needed and tsne_plots.EXPORT.font_scale == 1.0  # more rows of larger text need more room, and the setting is restored
    short = tsne_plots.legend_layout(["success (200 episodes)", "failure (230 episodes)"], 3, fig)
    assert short[0] == 3 and short[1] < 0.06  # two short entries stay in one row
    assert len(MODE_NAMES) == 10


def test_making_room_grows_the_figure_so_that_the_plot_keeps_its_size_and_does_nothing_when_there_is_room():
    fig, _ = tsne_plots.plt.subplots(figsize=(6.0, 6.0))
    fig.subplots_adjust(bottom=0.2, top=0.91)
    before = (fig.subplotpars.top - fig.subplotpars.bottom) * 6.0  # inches of plot
    tsne_plots.make_room(fig, 0.1)
    assert tuple(fig.get_size_inches()) == (6.0, 6.0) and fig.subplotpars.bottom == 0.2  # there is room
    tsne_plots.make_room(fig, 0.4)
    width, height = fig.get_size_inches()
    assert (width, height) == (6.0, pytest.approx(7.2))  # 2.4 inches of margin instead of 1.2
    assert fig.subplotpars.bottom * height == pytest.approx(2.4) and (fig.subplotpars.top - fig.subplotpars.bottom) * height == pytest.approx(before)
    assert (1 - fig.subplotpars.top) * height == pytest.approx(0.09 * 6.0)  # the margin above is as many inches as before
    tsne_plots.plt.close(fig)


def test_the_axis_arrows_grow_with_the_font_scale():
    limits = ((0.0, 1.0), (0.0, 1.0))
    _, ax = tsne_plots.canvas("outcome", "t", limits, ("PC1", "PC2"))
    small = [t.get_position() for t in ax.texts if t.get_text()]
    with tsne_plots.exporting(tsne_plots.Export(font_scale=2.0)):
        _, big_ax = tsne_plots.canvas("outcome", "t", limits, ("PC1", "PC2"))
        big = [t.get_position() for t in big_ax.texts if t.get_text()]
    assert small[0][0] == pytest.approx(0.16) and small[1][1] == pytest.approx(0.16)  # as before at the default scale
    assert big[0][0] == pytest.approx(0.29) and big[1][1] == pytest.approx(0.29)  # 0.03 + 0.13 * 2
    tsne_plots.plt.close("all")


def test_render_draws_the_train_unseen_figure_once_without_a_subset_in_its_name_and_the_mode_one_only_without_successes_when_asked_for_failures(tmp_path, monkeypatch):
    from fp.eval import class_fits

    monkeypatch.setattr(class_fits, "MIN_EPISODES", 1)
    monkeypatch.setattr(class_fits, "MIN_POINTS", 3)
    path, _ = three_d_file(tmp_path)  # e0 and e2 are successes, e1 and e3 failures (modes 1 and 4)
    keys = [f"e{i}" for i in range(4)]
    monkeypatch.setattr(tsne_plots, "load_inputs", lambda dataset: SimpleNamespace(splits={"task_seed0": split(train=keys[:2], calibration=keys[2:3], unseen=keys[3:])}))
    out = tmp_path / "pairs"
    written = tsne_plots.render(path, colorings=["outcome_ellipses_pre_train_unseen", "mode_ellipses_pre_train_unseen"], out_dir=out)
    assert sorted(w.name for w in written) == ["mode_ellipses_pre_train_unseen.png", "outcome_ellipses_pre_train_unseen.png"] and all(w.stat().st_size > 1000 for w in written)
    only = tsne_plots.render(path, colorings=["outcome_ellipses_pre_train_unseen", "mode_ellipses_pre_train_unseen"], out_dir=tmp_path / "failures", failures_only=True)
    assert [w.name for w in only] == ["mode_ellipses_pre_train_unseen_failures.png"]  # the outcome figure needs the successes
    assert set(tsne_plots.TRAIN_UNSEEN) <= set(tsne_plots.MODE_FIT_FIGURES + tuple(tsne_plots.TRAIN_UNSEEN)) and not set(tsne_plots.TRAIN_UNSEEN) & set(tsne_plots.FIGURES)  # not drawn by default
    assert "mode_ellipses_pre_train_unseen" in tsne_plots.MODE_FIT_FIGURES and "outcome_ellipses_pre_train_unseen" not in tsne_plots.MODE_FIT_FIGURES


def test_the_train_mask_marks_the_training_episodes_of_the_split_and_render_uses_it_for_the_left_panel():
    keys = ["a", "b", "c", "d"]
    sp = split(train=["a", "c"], val=["b"], unseen=["d"])
    assert tsne.train_mask(keys, sp).tolist() == [True, False, True, False]
    assert not set(tsne.train_mask(keys, sp).nonzero()[0]) & set(tsne.subset_masks(keys, sp)["unseen"].nonzero()[0])  # none of them is unseen
