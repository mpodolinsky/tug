"""Example trajectories on the latent space of the probe, to choose one success and one failure from: a contact sheet of candidate episodes, each drawn as an arrow on the plane of the
first two principal axes of saved PCA points (``fp.eval.tsne``), over the points and the Gaussians of the pre window of a subset.

The candidates are chosen by ``fp.eval.class_fits.trajectory_candidates`` (random successes; failures of the most common modes). A success is drawn over all its steps, a failure up to its
failure step, ending in a cross (``fp.eval.tsne_plots.add_path``: smoothed over a few steps, a ring at the first step). Each panel is titled with a number (S1 to S5, F1 to F5) and the task,
and for a failure its mode and failure time.
"""

import csv
import re
import textwrap
from pathlib import Path

import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D

from . import class_fits
from .tsne_plots import (SIZE, Export, add_axes_arrows, add_class_legend, add_path, axis_labels, draw_fits, exporting, fs, limits_of, make_room, plt, save)
from ..data.targets import MODE_NAMES

PANEL = 3.0  # inches: the width of a panel
TITLE_WIDTH = 30  # characters before the task in a title wraps
TASK_LINES = 3  # lines of a title that the task may take


def readable_task(name: str) -> str:
    """The task name of a LIBERO-X scene as words: ``EXTENSION_KITCHEN_SCENE30_LEVEL3__T172_place_the_ramekin_on_the_bottom_layer`` -> ``place the ramekin on the bottom layer``."""
    name = re.sub(r"^EXTENSION_[A-Z]+_SCENE\d+_", "", name)
    name = re.sub(r"^LEVEL\d__T\d+_", "", name)
    return name.replace("_", " ")


def title_of(points: dict, episode: int, label: str, seconds_per_step: float) -> str:
    """The title of the panel of ``episode`` (``label`` is S1, F2, ...): the label, for a failure its mode and failure time, then the task, wrapped."""
    task = textwrap.wrap(readable_task(str(points["task"][episode])), TITLE_WIDTH, max_lines=TASK_LINES, placeholder=" …")  # a task that is too long ends in an ellipsis
    head = label if points["success"][episode] else f"{label}: {MODE_NAMES[points['mode'][episode]].replace('_', ' ')}, {points['onset'][episode] * seconds_per_step:.1f} s"
    return "\n".join([head] + task)


def candidate_rows(points: dict, successes: list[int], failures: list[int], seconds_per_step: float, start: tuple[int, int] = (1, 1)) -> list[dict]:
    """The table of the candidates: label (numbered from ``start``: of the successes, of the failures), episode key, task, outcome, failure mode and failure time (s) of a failure,
    number of steps."""
    rows = []
    for prefix, episodes, first in (("S", successes, start[0]), ("F", failures, start[1])):
        for i, e in enumerate(episodes, first):
            failed = not points["success"][e]
            rows.append({"label": f"{prefix}{i}", "key": str(points["keys"][e]), "task": readable_task(str(points["task"][e])), "outcome": "failure" if failed else "success",
                         "failure_mode": MODE_NAMES[points["mode"][e]] if failed else "", "failure_time_s": round(float(points["onset"][e]) * seconds_per_step, 1) if failed else "",
                         "steps": int(points["n_steps"][e])})
    return rows


def draw_candidates(points: dict, mask: np.ndarray, successes: list[int], failures: list[int], seconds_per_step: float, path: Path, limits, start: tuple[int, int] = (1, 1)) -> list:
    """The contact sheet: a row of the ``successes`` over a row of the ``failures``, one panel each (the same axes, the PC arrows in the first panel), over the points of ``mask`` and the
    Gaussians of the classes (success and failure modes) of the pre window fitted on them, with one legend; the panels are numbered from ``start`` (of the successes, of the failures);
    returns the fits."""
    n = max(len(successes), len(failures), 1)
    (x0, x1), (y0, y1) = limits[0], limits[1]
    height = PANEL * min(max((y1 - y0) / (x1 - x0), 0.6), 1.0)
    fig, axes = plt.subplots(2, n, figsize=(PANEL * n, 2 * (height + 0.75)), squeeze=False)
    fig.subplots_adjust(left=0.03, right=0.995, top=0.93, bottom=0.04, wspace=0.03, hspace=0.32)
    fits = []
    for row, (prefix, episodes) in enumerate((("S", successes), ("F", failures))):
        for col in range(n):
            ax = axes[row, col]
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
            ax.set_aspect("equal")
            ax.axis("off")
            if col >= len(episodes):
                continue
            fits = draw_fits(ax, points, mask, "mode", "pre")
            add_path(ax, points, episodes[col], failure=prefix == "F", caption=False)
            ax.set_title(title_of(points, episodes[col], f"{prefix}{(start[0] if prefix == 'S' else start[1]) + col}", seconds_per_step), fontsize=fs(8), loc="left", color="0.15")
    labels = axis_labels(points)
    if labels:
        add_axes_arrows(axes[0, 0], labels)
    make_room(fig, add_class_legend(fig, "mode", fits, 5))
    save(fig, path)
    return fits


def write_candidates(points: dict, mask: np.ndarray, successes: list[int], failures: list[int], seconds_per_step: float, stem: Path, formats=("png",), tight: bool = True,
                     start: tuple[int, int] = (1, 1)) -> list[Path]:
    """Draw the contact sheet of the candidates to ``<stem>.<format>`` and write their table to ``<stem>.csv``; returns the files."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    with exporting(Export(tuple(formats), True, tight)):
        draw_candidates(points, mask, successes, failures, seconds_per_step, stem.with_suffix(".png"), limits_of(points["xy"][mask]), start)
    rows = candidate_rows(points, successes, failures, seconds_per_step, start)
    with open(stem.with_suffix(".csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return [stem.with_suffix(f".{fmt}") for fmt in formats] + [stem.with_suffix(".csv")]


EMBEDDING_PANEL = 6.6  # inches: the width of a panel of the comparison of the embeddings
EMBEDDING_ASPECT = 0.6  # the height of a panel relative to its width: the same for every space, so that the panels line up
LEGEND_COLUMNS = 6  # columns of the legend of the comparison (it takes fewer if the labels do not fit in the width of the figure)
PANEL_LETTERS = "abcdefgh"
SUCCESS_PATH, FAILURE_PATH, PATH_OUTLINE = "#81c995", "#f0524d", "0.4"  # pastel green for a success, a soft but clearly red tone for a failure, with an outline so that they show over the Gaussians


def label_of(points: dict, episode: int, label: str, seconds_per_step: float) -> str:
    """The first line of the title of ``episode`` (``label`` is S10, F8, ...): ``F8: press failure, 56.0 s`` for a failure, ``S10: success`` for a success."""
    return f"{label}: success" if points["success"][episode] else title_of(points, episode, label, seconds_per_step).split("\n")[0]


def draw_embeddings(panels: list[tuple[str, dict, np.ndarray, bool]], chosen: list[tuple[str, int]], seconds_per_step: float, path: Path, note: str = "") -> list:
    """The same episodes ``chosen`` (label, episode index) drawn in each space of ``panels`` (title, saved PCA points, mask over the points, whether to leave out the successes), two panels
    per row: each panel has the points and the Gaussians of the classes (success and failure modes, pre window) of its mask and the path of every chosen episode (a success solid, a failure
    dashed up to its failure step and ending in a cross; pastel green and pastel red with a dark outline), on the axes of its own space (PC arrows with its own variance shares). The panels
    all have the same size (``EMBEDDING_ASPECT``) and are labeled (a), (b), ... with the title at the same place above each; ``note`` is written in small type under the legend (e.g. which
    episodes the points are). One legend: the classes of the first panel that has a success class, and the chosen episodes. Returns the fits of the panel used for the legend."""
    rows = -(-len(panels) // 2)
    height = EMBEDDING_PANEL * EMBEDDING_ASPECT
    fig, axes = plt.subplots(rows, 2, figsize=(2 * EMBEDDING_PANEL, rows * (height + 0.45)), squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.95, bottom=0.03, wspace=0.03, hspace=0.2)
    legend_fits = []
    for k, (ax, (title, points, mask, failures_only)) in enumerate(zip(axes.ravel(), panels)):
        use = mask & (~points["success"][points["episode"]] if failures_only else np.ones(len(mask), bool))
        ends = [points["xy"][np.flatnonzero(points["episode"] == e)] for _, e in chosen]
        ax.set_xlim(*limits_of(np.concatenate([points["xy"][use]] + ends))[0])
        ax.set_ylim(*limits_of(np.concatenate([points["xy"][use]] + ends))[1])
        ax.set_aspect("equal", adjustable="datalim")  # the same box for every space: the limits grow to fill it, never shrink
        ax.axis("off")
        fits = draw_fits(ax, points, use, "mode", "pre")
        if not legend_fits and not failures_only:
            legend_fits = fits
        for _, e in chosen:
            add_path(ax, points, e, failure=not points["success"][e], caption=False, color=SUCCESS_PATH if points["success"][e] else FAILURE_PATH, outline=PATH_OUTLINE)
        ax.text(0.0, 1.03, f"({PANEL_LETTERS[k]}) {title}", transform=ax.transAxes, ha="left", va="bottom", fontsize=fs(11), color="0.15")  # the same place in every panel
        labels = axis_labels(points)
        if labels:
            add_axes_arrows(ax, labels)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    first = panels[0][1]
    extra = [Line2D([0], [0], color=SUCCESS_PATH if first["success"][e] else FAILURE_PATH, linewidth=2.0, linestyle="-" if first["success"][e] else "--", label=label_of(first, e, label, seconds_per_step),
                    path_effects=[patheffects.withStroke(linewidth=3.0, foreground=PATH_OUTLINE)]) for label, e in chosen]
    room = add_class_legend(fig, "mode", legend_fits, LEGEND_COLUMNS, extra)
    if note:
        fig.text(0.5, 0.004, note, ha="center", va="bottom", fontsize=fs(8), color="0.35")
        room += 0.025
    make_room(fig, room)
    save(fig, path)
    return legend_fits


def write_embeddings(panels: list[tuple[str, dict, np.ndarray, bool]], chosen: list[tuple[str, int]], seconds_per_step: float, stem: Path, formats=("png",), tight: bool = True,
                     note: str = "") -> list[Path]:
    """Draw ``draw_embeddings`` to ``<stem>.<format>`` for each format; returns the files."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    with exporting(Export(tuple(formats), True, tight)):
        draw_embeddings(panels, chosen, seconds_per_step, stem.with_suffix(".png"), note)
    return [stem.with_suffix(f".{fmt}") for fmt in formats]
