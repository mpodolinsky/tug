"""The labeling example figure (fp/eval/label_example.py), on synthetic labels, frames and a synthetic video."""

import json

import numpy as np
import pytest

from fp.eval import label_example as le


def write_labels(folder, success=False, onset=7.5, mode="wrong_target", windows=4, last_end=None):
    folder.mkdir(parents=True, exist_ok=True)
    timeline = [{"t_start_sec": 3.0 * i, "t_end_sec": 3.0 * (i + 1), "phrase": f"phrase {i}"} for i in range(windows)]
    if last_end is not None:
        timeline[-1]["t_end_sec"] = last_end
    labels = {"instruction": "place the pan on the shelf", "success": success, "semantic_timeline": timeline}
    if not success:
        labels["vlm_failure"] = {"vlm_failure_mode": mode, "vlm_failure_onset_seconds": onset}
    (folder / "labels.json").write_text(json.dumps(labels))
    return folder


def frames(n):
    return [np.full((8, 8, 3), 40 * i, np.uint8) for i in range(n)]


def test_the_labels_of_a_failed_episode_are_its_windows_failure_mode_and_failure_time_and_those_of_a_success_have_no_failure(tmp_path):
    failed = le.read_labels(write_labels(tmp_path / "a"))
    assert [(w.start, w.end, w.phrase) for w in failed.windows] == [(0.0, 3.0, "phrase 0"), (3.0, 6.0, "phrase 1"), (6.0, 9.0, "phrase 2"), (9.0, 12.0, "phrase 3")]
    assert (failed.mode, failed.onset, failed.instruction) == ("wrong_target", 7.5, "place the pan on the shelf")
    won = le.read_labels(write_labels(tmp_path / "b", success=True))
    assert (won.mode, won.onset) == (None, None) and len(won.windows) == 4


def test_the_frame_of_a_window_is_the_one_at_its_centre():
    windows = [le.Window(0.0, 3.0, "a"), le.Window(3.0, 6.0, "b"), le.Window(6.0, 9.0, "c")]
    assert le.frame_indices(windows, 20.0) == [30, 90, 150] and le.frame_indices(windows[:1], 10.0) == [15]


def test_a_failed_row_has_a_frame_and_a_description_per_window_and_marks_the_failure_time_and_mode(tmp_path):
    rollout = le.read_labels(write_labels(tmp_path))
    fig = le.build([(rollout, frames(4))], 4)
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert len(ax.images) == 4 and all(f"phrase {i}" in texts for i in range(4)) and le.failure_text(7.5, "wrong_target") in texts
    tag = [t for t in ax.texts if t.get_text() == "Timeline"]
    assert len(tag) == 1 and tag[0].get_position()[0] == pytest.approx(6.0) and tag[0].get_position()[1] < le.FRAME_BOTTOM  # centred under the descriptions (0.1 to 11.9 s), below the frames
    bracket = [line for line in ax.lines if line.get_xdata()[0] == pytest.approx(0.1)]
    assert len(bracket) == 1 and list(bracket[0].get_xdata()) == pytest.approx([0.1, 0.1, 11.9, 11.9])  # the bracket spans the descriptions
    assert [t for t in texts if t.endswith(" s") and "failure" not in t] == ["12 s"]  # the time axis: 0, 3, 6, 9 and 12 s
    assert len([line for line in ax.lines if line.get_xdata()[0] == 7.5]) == 1  # one mark at the failure time
    outlined = [p for p in ax.patches if p.get_fill() is False and p.get_edgecolor()[:3] == le.matplotlib.colors.to_rgb(le.FAIL_COLOR)]
    assert len(outlined) == 1 and outlined[0].get_x() == pytest.approx(7.5 - le.FRAME_SIZE / 2, abs=1.5)  # only the frame of the window that holds the failure time
    tinted = [p for p in ax.patches if p.get_fill() and p.get_facecolor()[:3] == le.FAIL_FILL]
    assert len(tinted) == 1 and tinted[0].get_x() == pytest.approx(6.05)  # the window 6 to 9 s holds 7.5 s
    le.plt.close(fig)


def test_two_rows_share_the_time_axis_the_success_on_top_is_green_without_a_failure_mark_and_the_failure_below_it(tmp_path):
    won = le.read_labels(write_labels(tmp_path / "w", success=True, windows=3, last_end=8.0))
    failed = le.read_labels(write_labels(tmp_path / "f", windows=5, onset=10.3))
    fig = le.build([(won, frames(3)), (failed, frames(5))], 6)
    ax = fig.axes[0]
    assert len(ax.images) == 8 and fig.get_size_inches()[1] > 2 * 1.5  # a taller figure for two rows
    texts = {t.get_text(): t for t in ax.texts}
    failure = le.failure_text(10.3, "wrong_target")
    assert "success" in texts and failure in texts and len([t for t in ax.texts if t.get_text() == "Timeline"]) == 2  # a Timeline tag with a bracket per row
    assert texts["success"].get_position()[1] > texts[failure].get_position()[1]  # the first row is on top
    assert texts["success"].get_position()[0] == pytest.approx(7.5)  # "success" is centred on the time axis (0 to 15 s), not on the shorter bar of the success (0 to 8 s)
    assert all(t.get_fontsize() >= 8 for t in ax.texts)  # large text: descriptions, annotations and tick labels
    assert [t for t in texts if t.endswith(" s") and "failure" not in t] == ["15 s"]  # the axis ends at the whole window after the longest row (5 windows: 15 s)
    greens = [p for p in ax.patches if p.get_fill() and p.get_facecolor()[:3] == le.SUCCESS_FILL]
    assert len(greens) == 3 and greens[-1].get_width() == pytest.approx(8.0 - 6.0 - 0.1)  # the bar of the last window of the success has its true length
    assert not any(line.get_xdata()[0] == 8.0 for line in ax.lines)  # no failure mark for a success
    le.plt.close(fig)


def test_the_bracket_of_the_timeline_spans_the_slots_of_the_first_to_the_last_window():
    windows = [le.Window(0.0, 3.0, "a"), le.Window(3.0, 6.0, "b"), le.Window(6.0, 7.0, "c")]  # the last window is shorter
    assert le.timeline_span(windows) == pytest.approx((0.1, 8.9))
    assert le.timeline_span(windows[:1]) == pytest.approx((0.1, 2.9))


def test_the_failure_line_names_both_labels_in_bold_and_writes_the_mode_in_words():
    text = le.failure_text(10.34, "grasp_failure")
    assert text == r"$\mathbf{Failure\ time}$ 10.3 s, $\mathbf{Failure\ mode}$: grasp failure"


def test_a_frame_sits_in_the_3_second_slot_of_its_window_so_that_a_short_last_window_does_not_overlap_the_frame_before_it(tmp_path):
    rollout = le.read_labels(write_labels(tmp_path, success=True, windows=3, last_end=7.0))  # the last window is 1 s long
    fig = le.build([(rollout, frames(3))], 3)
    extents = [image.get_extent() for image in fig.axes[0].images]
    assert [round((e[0] + e[1]) / 2, 6) for e in extents] == [1.5, 4.5, 7.5]  # centred in the slots 0-3, 3-6 and 6-9
    assert extents[1][1] < extents[2][0]  # no overlap
    le.plt.close(fig)


def test_wrong_numbers_of_frames_are_an_error_and_the_figure_is_written_in_every_format_and_closed(tmp_path):
    rollout = le.read_labels(write_labels(tmp_path / "labels", success=True, windows=3))
    with pytest.raises(ValueError, match="3 windows but 2 frames"):
        le.build([(rollout, frames(2))], 3)
    fig = le.build([(rollout, frames(3))], 3)
    files = le.write(fig, tmp_path / "out" / "example", ("png", "pdf"), dpi=50)
    assert [f.name for f in files] == ["example.png", "example.pdf"] and all(f.stat().st_size > 500 for f in files)
    assert fig.number not in le.plt.get_fignums()


def test_the_frames_are_read_in_the_order_asked_for_and_a_video_that_is_too_short_is_an_error(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (32, 32))
    if not writer.isOpened():
        pytest.skip("this OpenCV cannot write mp4")
    for i in range(10):
        writer.write(np.full((32, 32, 3), 25 * i, np.uint8))
    writer.release()
    got = le.read_frames(path, [7, 2])  # asked for out of order
    assert [round(float(f.mean()), -1) for f in got] == [170.0, 50.0] and got[0].shape == (32, 32, 3)  # frame 7 is 175 and frame 2 is 50 (video compression changes the values slightly)
    with pytest.raises(ValueError, match="needs frame 20"):
        le.read_frames(path, [20])
