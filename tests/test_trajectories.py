"""The contact sheet of candidate trajectories (fp/eval/trajectories.py), on synthetic points."""

import numpy as np
import pytest

from fp.eval import class_fits, trajectories, tsne_plots
from tests.test_class_fits import episodes_points


def points_with_tasks():
    p = episodes_points(8, 8, steps=20, onset=15, modes=(1, 4))
    p["task"] = np.array(["EXTENSION_KITCHEN_SCENE30_LEVEL3__T172_place_the_ramekin_on_the_bottom_layer"] * 8 + ["EXTENSION_STUDY_SCENE2_pick_up_the_mug"] * 8)
    p["keys"] = np.array([f"LEVEL1/t/ep{i:03d}" for i in range(16)])
    p["n_steps"] = np.full(16, 20)
    return p


def test_a_task_name_is_written_as_words_without_the_scene_and_level_prefixes():
    assert trajectories.readable_task("EXTENSION_KITCHEN_SCENE30_LEVEL3__T172_place_the_ramekin_on_the_bottom_layer") == "place the ramekin on the bottom layer"
    assert trajectories.readable_task("EXTENSION_STUDY_SCENE2_pick_up_the_mug") == "pick up the mug"
    assert trajectories.readable_task("something_else") == "something else"


def test_a_title_has_the_label_the_mode_and_failure_time_of_a_failure_and_the_wrapped_task():
    p = points_with_tasks()
    assert trajectories.title_of(p, 0, "S1", 0.4) == "S1\nplace the ramekin on the\nbottom layer"
    failure = trajectories.title_of(p, 9, "F2", 0.4)  # episode 9 is a failure of mode 4 (placement_or_insertion_failure) with failure step 15 = 6.0 s
    assert failure.split("\n")[0] == "F2: placement or insertion failure, 6.0 s" and "pick up the mug" in failure


def test_the_table_of_the_candidates_has_a_row_per_trajectory_with_the_failure_labels_only_for_failures():
    p = points_with_tasks()
    rows = trajectories.candidate_rows(p, [0, 1], [8, 9], 0.4)
    assert [r["label"] for r in rows] == ["S1", "S2", "F1", "F2"] and [r["outcome"] for r in rows] == ["success", "success", "failure", "failure"]
    assert rows[0]["failure_mode"] == "" and rows[0]["failure_time_s"] == "" and rows[2]["failure_mode"] == "stuck_or_no_progress" and rows[2]["failure_time_s"] == 6.0
    assert rows[0]["key"] == "LEVEL1/t/ep000" and rows[0]["steps"] == 20


def test_the_contact_sheet_has_a_panel_per_candidate_a_path_in_each_and_the_failures_marked(monkeypatch, tmp_path):
    monkeypatch.setattr(class_fits, "MIN_EPISODES", 1)
    monkeypatch.setattr(class_fits, "MIN_POINTS", 10)
    p = points_with_tasks()
    captured = []
    monkeypatch.setattr(trajectories, "save", lambda fig, path: captured.append(fig))
    fits = trajectories.draw_candidates(p, np.ones(len(p["xy"]), bool), [0, 1, 2], [8, 9], 0.4, tmp_path / "a.png", tsne_plots.limits_of(p["xy"]))
    fig = captured[0]
    shown = [ax for ax in fig.axes if ax.get_title("left")]
    assert len(fig.axes) == 6 and [ax.get_title("left").split("\n")[0] for ax in shown][:3] == ["S1", "S2", "S3"] and [ax.get_title("left").split("\n")[0][:2] for ax in shown][3:] == ["F1", "F2"]
    black = lambda ax: [line for line in ax.lines if tsne_plots.mcolors.to_hex(line.get_color()) == "#000000"]  # noqa: E731
    assert all(len(black(ax)) == 1 for ax in shown)  # one path per panel, and no caption text of the path
    assert all(not [t for t in ax.texts if t.get_text()] for ax in shown[1:])
    assert [line.get_linestyle() for ax in shown for line in black(ax)] == ["-", "-", "-", "--", "--"]  # the failures are dashed
    assert all(len(black(ax)[0].get_xdata()) == (20 if i < 3 else 16) for i, ax in enumerate(shown))  # a success over its 20 steps, a failure up to its failure step 15
    assert len(fig.legends) == 1 and [f.name for f in fits][0] == "success"
    tsne_plots.plt.close(fig)


def test_the_candidates_are_written_as_an_image_and_a_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(class_fits, "MIN_EPISODES", 1)
    monkeypatch.setattr(class_fits, "MIN_POINTS", 10)
    p = points_with_tasks()
    files = trajectories.write_candidates(p, np.ones(len(p["xy"]), bool), [0, 1], [8, 9], 0.4, tmp_path / "out" / "sheet", ("png",))
    assert [f.name for f in files] == ["sheet.png", "sheet.csv"] and files[0].stat().st_size > 1000
    lines = files[1].read_text().splitlines()
    assert lines[0] == "label,key,task,outcome,failure_mode,failure_time_s,steps" and len(lines) == 5


def test_a_task_name_that_is_too_long_for_its_lines_ends_in_an_ellipsis():
    p = points_with_tasks()
    p["task"] = p["task"].astype(object)  # a fixed-width string array would cut the long name
    p["task"][0] = "EXTENSION_STUDY_SCENE45_" + "_".join(["place the orange book".replace(" ", "_")] * 4)
    lines = trajectories.title_of(p, 0, "S1", 0.4).split("\n")
    assert len(lines) == 1 + trajectories.TASK_LINES and lines[-1].endswith("…")  # the label and three lines of the task, the last one cut
    assert all(len(line) <= trajectories.TITLE_WIDTH + 2 for line in lines[1:])


def test_the_numbering_of_a_second_sheet_continues_after_the_first():
    p = points_with_tasks()
    rows = trajectories.candidate_rows(p, [0, 1], [8, 9], 0.4, start=(6, 6))
    assert [r["label"] for r in rows] == ["S6", "S7", "F6", "F7"]
    assert trajectories.title_of(p, 0, "S6", 0.4).startswith("S6\n")


def embedding_panels():
    """Two synthetic spaces over the same episodes (the second one shifted), the second without the successes."""
    a, b = points_with_tasks(), points_with_tasks()
    b["xy"] = b["xy"] + 5.0
    everything = np.ones(len(a["xy"]), bool)
    return [("space A", a, everything, False), ("space B (failures only)", b, everything, True)], a


def test_the_comparison_draws_the_chosen_episodes_in_every_space_with_the_same_dash_and_the_background_of_each_space(monkeypatch, tmp_path):
    monkeypatch.setattr(class_fits, "MIN_EPISODES", 1)
    monkeypatch.setattr(class_fits, "MIN_POINTS", 10)
    panels, a = embedding_panels()
    captured = []
    monkeypatch.setattr(trajectories, "save", lambda fig, path: captured.append(fig))
    fits = trajectories.draw_embeddings(panels, [("S1", 0), ("F1", 9)], 0.4, tmp_path / "a.png")
    fig = captured[0]
    left, right = fig.axes[:2]
    assert [[t.get_text() for t in ax.texts if t.get_text().startswith("(")] for ax in (left, right)] == [["(a) space A"], ["(b) space B (failures only)"]]  # labeled (a), (b), ...
    assert [ax.get_position().bounds[2:] for ax in (left, right)][0] == pytest.approx([ax.get_position().bounds[2:] for ax in (left, right)][1])  # the same size
    assert left.texts[0].get_position() == right.texts[0].get_position() or [t.get_position() for t in left.texts if t.get_text().startswith("(")] == [t.get_position() for t in right.texts if t.get_text().startswith("(")]  # the label is at the same place in both
    paths = lambda ax: [line for line in ax.lines if tsne_plots.mcolors.to_hex(line.get_color()) in (trajectories.SUCCESS_PATH, trajectories.FAILURE_PATH)]  # noqa: E731
    assert [line.get_linestyle() for line in paths(left)] == ["-", "--"] and [line.get_linestyle() for line in paths(right)] == ["-", "--"]  # a success solid, a failure dashed, in both
    assert [tsne_plots.mcolors.to_hex(line.get_color()) for line in paths(left)] == [trajectories.SUCCESS_PATH, trajectories.FAILURE_PATH]  # green for the success, red for the failure
    assert all(line.get_path_effects() for line in paths(left) + paths(right))  # with an outline
    assert len(left.lines) - 2 == 3 and len(right.lines) - 2 == 2  # the ellipses: success and two modes, and the two modes only without successes
    assert [f.name for f in fits][0] == "success" and len(fits) == 3  # the legend has the classes of the panel with successes
    legend = [t.get_text() for t in fig.legends[0].get_texts()]
    assert legend[-2:] == ["S1: success", "F1: placement or insertion failure, 6.0 s"] and all(f"({n} episodes)" in l for l, n in zip(legend[:1], [8]))
    assert left.get_xlim() != right.get_xlim()  # each space has its own axes
    tsne_plots.plt.close(fig)


def test_a_path_that_lies_outside_the_background_of_a_space_is_still_in_view(monkeypatch, tmp_path):
    monkeypatch.setattr(class_fits, "MIN_EPISODES", 1)
    monkeypatch.setattr(class_fits, "MIN_POINTS", 10)
    panels, a = embedding_panels()
    name, b, mask, failures_only = panels[1]
    b["xy"][b["episode"] == 0] += 40.0  # the success is far from the failures that form the background of the space without successes
    captured = []
    monkeypatch.setattr(trajectories, "save", lambda fig, path: captured.append(fig))
    trajectories.draw_embeddings([panels[0], (name, b, mask, failures_only)], [("S1", 0)], 0.4, tmp_path / "a.png")
    right = captured[0].axes[1]
    x_path = np.array([b["xy"][i, 0] for i in np.flatnonzero(b["episode"] == 0)])
    assert right.get_xlim()[0] <= np.percentile(x_path, 0.1) and right.get_xlim()[1] >= np.percentile(x_path, 99.9) - 1e-6
    tsne_plots.plt.close(captured[0])


def test_the_pastel_colours_of_the_paths_are_light_green_and_light_red():
    import matplotlib.colors as mcolors

    green, red = (mcolors.to_rgb(c) for c in (trajectories.SUCCESS_PATH, trajectories.FAILURE_PATH))
    assert green[1] > green[0] and green[1] > green[2] and red[0] > red[1] and red[0] > red[2]  # the dominant channel
    assert min(green) > 0.4 and min(red) > 0.25  # soft: no channel is dark
    assert red[0] - max(red[1], red[2]) > 0.4  # the red is clearly red (stronger than a pastel pink)
