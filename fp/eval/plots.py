"""The curve figures: TPR early (or on-time TPR) against FPR, one panel per setting in a single row, one line per method, the legend below.

Our probes (one line per label variant) are drawn thick in four colours sampled from viridis; every baseline is thin and neutral: SAFE black,
the action-signal baselines dark gray, the embedding-distance baselines light gray. Methods that share a colour are told apart by line style, and
ours stand out by width as well as by hue, because a mid gray can look like a viridis blue or teal to a colour-blind reader (the viridis
samples are separated by a colour-blind distance of 14.7 and a normal-vision one of 16.4, checked with a port of the dataviz skill's
palette validator). Axis labels, ticks and spines are black, only horizontal grid lines are drawn, and the legend has no category headings.
The text uses matplotlib's default font (DejaVu Sans); the PDF embeds it as TrueType.
"""

import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from ..metrics.curves import FPR_MAX  # noqa: E402
from ..metrics.ppv import ppv_bound  # noqa: E402
from .figures import GRID, Entry, Setting  # noqa: E402
from .tables import LttRow  # noqa: E402

VIRIDIS_AT = {"onset_pre": 0.55, "onset_traj": 0.10, "onset": 0.33, "onset_post": 0.75}  # positions on the viridis map, one per label variant
NEUTRAL = {"safe": "#000000", "action": "#595959", "distance": "#a6a6a6"}
STYLES = {"solid": "-", "dashed": (0, (5, 2)), "dotted": (0, (1, 1.5)), "dashdot": (0, (5, 2, 1, 2))}
INK, GRID_COLOR = "#000000", "#e6e5e1"
WIDTH = {"ours": 2.4}
DEFAULT_WIDTH = 1.1
PANEL_WIDTH, PANEL_HEIGHT = 2.5, 2.9  # inches per panel
PANEL_TITLES = {  # the title of the panel of each setting: the benchmark and the policy, then the held-out split, with the names used in the text
    "LIBERO-10 π0.5": "LIBERO-10 π0.5\nleave-three-out", "LIBERO-10 GR00T": "LIBERO-10 GR00T\nleave-three-out", "LIBERO-X task": "LIBERO-X GR00T\nnew-task",
    "LIBERO-X signature": "LIBERO-X GR00T\nnew-signature", "LIBERO-X skill": "LIBERO-X GR00T\nnew-skill", "Real robot": "Real robot GR00T\nleave-one-out"}  # any other setting keeps its label
SPLIT_COLORS = {"LIBERO-X task": "#DE8E46", "LIBERO-X signature": "#4A8C64", "LIBERO-X skill": "#3E5496"}  # the colour of the split name in a title: taskOrange, sigGreen and skillNavy
# of the paper (RGB 222,142,70, 74,140,100 and 62,84,150); the other settings' split names stay black
TITLE_LINE, SPLIT_OFFSET = 1.3, 4  # the split name is drawn 4 points over the panel and the first title line 1.3 font heights above that, so the gap grows with the font
LEGEND_INCHES = 0.5  # height reserved below the panels for the legend
NARROW_FIGURE = 9.0  # inches: below this width a single row of seven legend entries does not fit at the usual spacing
PPV_PANEL = (3.0, 2.8)  # inches per panel of the PPV figure
PPV_FPR_TARGETS = (0.2, 0.3, 0.4)  # the FPR targets of the LTT rows it shows


def color_of(entry: Entry):
    return matplotlib.colormaps["viridis"](VIRIDIS_AT[entry.variant]) if entry.kind == "ours" else NEUTRAL[entry.kind]


def style_axes(scale: float = 1.0) -> None:
    """The style of the figures; ``scale`` multiplies the font sizes (9 pt for the text, 10 pt for the titles)."""
    plt.rcParams.update({"font.size": 9 * scale, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
                         "grid.color": GRID_COLOR, "grid.linewidth": 0.6, "axes.titlesize": 10 * scale, "axes.titlecolor": INK, "legend.frameon": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})  # 42: the font is embedded as TrueType, not as outlines of Type 3 glyphs


def set_panel_title(ax, label: str) -> None:
    """The title of the panel of the setting ``label``: its first line (``PANEL_TITLES``) as the title and the second, the held-out split, as a text of its own under it
    in the colour of the split (``SPLIT_COLORS``; a title cannot be coloured in parts). A setting without a table entry gets its label as a one-line title."""
    first, _, split = PANEL_TITLES.get(label, label).partition("\n")
    ax.set_title(first, pad=SPLIT_OFFSET + TITLE_LINE * ax.title.get_fontsize() if split else matplotlib.rcParams["axes.titlepad"])
    if split:
        ax.annotate(split, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, SPLIT_OFFSET), textcoords="offset points", ha="center", va="bottom",
                    color=SPLIT_COLORS.get(label, INK), fontsize=ax.title.get_fontsize())


def line_kwargs(entry: Entry) -> dict:
    return dict(color=color_of(entry), linestyle=STYLES[entry.style], linewidth=WIDTH.get(entry.kind, DEFAULT_WIDTH), solid_capstyle="round")


def curves_figure(rows: list[tuple[str, dict[tuple[str, str], list[float]]]], settings: list[Setting], entries: list[Entry], ylabel: str,
                  panel_width: float = PANEL_WIDTH, legend_rows: int = 2, height_scale: float = 1.0, font_scale: float = 1.0) -> Figure:
    """``rows`` is a list of (row label, curves), top to bottom; ``curves[(setting label, entry label)]`` is the mean curve on ``GRID`` (percent)
    and entries or settings without one are left out. One panel per setting in each row, the setting names above the top row, the FPR label
    below the bottom one, the row label with ``ylabel`` on the left; the legend below, one entry per method drawn (in the order of ``entries``).
    Each panel is ``panel_width`` inches wide (a figure with fewer settings can keep the width of a full one by making its panels wider); the legend
    has ``legend_rows`` rows. ``height_scale`` shrinks the height of the panels and the room for the legend, not the width and not the text, which is then
    larger relative to the figure. ``font_scale`` multiplies all font sizes (``style_axes``) and the legend's. A figure narrower than ``NARROW_FIGURE`` sets the legend
    entries closer together so that a single row still fits."""
    style_axes(font_scale)
    n_rows = len(rows)
    height = PANEL_HEIGHT * n_rows * height_scale
    width = panel_width * len(settings)
    fig, axes = plt.subplots(n_rows, len(settings), figsize=(width, height), squeeze=False, sharex="col")
    drawn = []
    for r, ((row_label, curves), row_axes) in enumerate(zip(rows, axes)):
        for ax, setting in zip(row_axes, settings):
            for entry in sorted(entries, key=lambda e: e.kind == "ours"):  # ours on top
                y = curves.get((setting.label, entry.label))
                if y is None:
                    continue
                ax.plot(GRID, y, **line_kwargs(entry), zorder=3 if entry.kind == "ours" else 2)
                if entry not in drawn:
                    drawn.append(entry)
            ax.set_xlim(0, FPR_MAX)
            ax.set_ylim(bottom=0)
            if r == 0:
                set_panel_title(ax, setting.label)
            if r == n_rows - 1:
                ax.set_xlabel("FPR (%)")
        row_axes[0].set_ylabel(f"{row_label}\n{ylabel}")
    shown = [e for e in entries if e in drawn]
    fig.tight_layout(rect=(0, LEGEND_INCHES * height_scale / height, 1, 1))
    spacing = (dict(fontsize=9 * font_scale, handlelength=2.8, columnspacing=1.6) if width >= NARROW_FIGURE
               else dict(fontsize=8.5 * font_scale, handlelength=1.8, columnspacing=0.9, handletextpad=0.5))
    fig.legend([Line2D([], [], **line_kwargs(e)) for e in shown], [e.label for e in shown], loc="lower center", ncol=max(1, math.ceil(len(shown) / legend_rows)),
               labelspacing=0.5, bbox_to_anchor=(0.5, 0.0), **spacing)
    return fig


def save(fig: Figure, stem: str, out_dir: Path) -> list[Path]:
    """Write ``stem``.png (200 dpi) and ``stem``.pdf into ``out_dir`` and close the figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def ppv_figure(rows: list[LttRow], fpr_targets: tuple[float, ...] = PPV_FPR_TARGETS) -> Figure:
    """The PPV bound (``metrics.ppv``) against the failure prevalence for the LTT rows of ``fpr_targets``, one panel each: the bound is a dashed line, the realised
    PPV of the seen and of the unseen episodes (the mean over the certified runs) a filled and a hollow marker at the mean prevalence of those sets. Needs at
    least one LTT row with a TPR-early target and cells."""
    shown = [r for f in fpr_targets for r in rows if r.highlight and abs(r.fpr_target - f) < 1e-9 and r.tpr_target is not None and r.seen is not None]
    if not shown:
        raise ValueError("no LTT row with a TPR-early target for the PPV figure")
    style_axes()
    fig, axes = plt.subplots(1, len(shown), figsize=(PPV_PANEL[0] * len(shown), PPV_PANEL[1]), sharey=True, squeeze=False)
    colour = matplotlib.colormaps["viridis"](VIRIDIS_AT["onset_pre"])
    prevalence = np.linspace(0, 1, 201)
    for ax, row in zip(axes[0], shown):
        ax.plot(prevalence, [100 * ppv_bound(row.tpr_target, row.fpr_target, p) for p in prevalence], color=INK, linestyle=STYLES["dashed"], linewidth=1.4, zorder=2)
        for cells, filled in ((row.seen, True), (row.unseen, False)):
            if cells is not None and cells.ppv is not None:
                ax.plot([cells.prevalence], [cells.ppv], marker="o", markersize=7, linestyle="none", color=colour, markerfacecolor=colour if filled else "white",
                        markeredgewidth=1.6, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        ax.set_title(rf"FPR $\leq$ {100 * row.fpr_target:.0f}%, TPR early $\geq$ {100 * row.tpr_target:.0f}%")
        ax.set_xlabel(r"failure prevalence $\pi$")
    axes[0][0].set_ylabel("PPV (%)")
    keys = [Line2D([], [], color=INK, linestyle=STYLES["dashed"], linewidth=1.4), Line2D([], [], marker="o", markersize=7, linestyle="none", color=colour),
            Line2D([], [], marker="o", markersize=7, linestyle="none", color=colour, markerfacecolor="white", markeredgewidth=1.6)]
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.legend(keys, ["guaranteed bound", "realised, seen tasks", "realised, unseen tasks"], loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    return fig
