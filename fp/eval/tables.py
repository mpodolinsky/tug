"""The main comparison table as LaTeX and CSV (pure formatting; ``fp.eval.figures`` supplies the numbers).

One row per method, a seen and an unseen column for each setting (the area under TPR early vs FPR up to 40%, in %), a column saying which
calibrator the protocol gives the method (MC max-conformal, FC the functional conformal band of SAFE), and the training cost on LIBERO-X (fit time, state kept to score
a new episode, scoring time per policy step). In every numeric column the best value is bold and the second best italic, judged on the
displayed numbers (so equal displayed values are marked alike); the areas are better when higher, the costs when lower. Our rows are shaded.
The LaTeX needs booktabs and xcolor with the table option (\\rowcolor).
"""

from dataclasses import dataclass

CALIBRATOR_CODES = {"max_conformal": "MC", "safe_band": "FC"}  # MC: max-conformal; FC: functional conformal (the SAFE band)
COST_COLUMNS = (("fit_seconds", "Fit (s)", 1), ("state_mb", "State (MB)", 1), ("latency_gpu_ms", "GPU (ms)", 2), ("latency_cpu_ms", "CPU (ms)", 2))
MISSING = "--"


@dataclass(frozen=True)
class Row:
    label: str
    ours: bool
    calibrator: str  # 'max_conformal' or 'safe_band'
    areas: list[tuple[float, float] | None]  # (mean, std) per setting and set, in the order of ``columns``; None where there is no run
    costs: list[float | None]  # in the order of COST_COLUMNS


def columns(settings: list[str]) -> list[tuple[str, str]]:
    """(setting, set) for every area column: seen then unseen for each setting."""
    return [(s, e) for s in settings for e in ("seen", "unseen")]


def number(value: float, digits: int) -> str:
    """Fixed decimals; two significant digits below 0.1 (a scoring time of 0.0008 ms must not read 0.00); no decimals from 1000 on."""
    if value == 0:
        return f"{0:.{digits}f}"
    if abs(value) < 0.1:
        return f"{value:.2g}"
    return f"{value:.0f}" if abs(value) >= 1000 else f"{value:.{digits}f}"


def marks(values: list[float | None], digits: int, higher_is_better: bool) -> list[str | None]:
    """'best' / 'second' / None for each value, judged on the number as displayed (``number``): every value equal to the best is 'best', every
    one equal to the next distinct value is 'second'."""
    shown = [None if v is None else float(number(v, digits)) for v in values]
    distinct = sorted({s for s in shown if s is not None}, reverse=higher_is_better)
    best, second = (distinct + [None, None])[:2]
    return ["best" if s is not None and s == best else "second" if s is not None and s == second else None for s in shown]


def escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace("π0.5", r"$\pi_{0.5}$")


def style(text: str, mark: str | None) -> str:
    return {"best": rf"\textbf{{{text}}}", "second": rf"\textit{{{text}}}"}.get(mark, text)


def tabular(rows: list[Row], settings: list[str], with_std: bool = False) -> str:
    """The bare ``tabular``: header, rows (ours shaded, best and second best marked) and rules."""
    n_area = len(columns(settings))
    area_marks = [marks([r.areas[i][0] if r.areas[i] else None for r in rows], 1, True) for i in range(n_area)]
    cost_marks = [marks([r.costs[i] for r in rows], d, False) for i, (_, _, d) in enumerate(COST_COLUMNS)]
    lines = [r"\begin{tabular}{ll" + "r" * n_area + "r" * len(COST_COLUMNS) + "}", r"\toprule",
             " & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{escape(s)}}}" for s in settings) + rf" & \multicolumn{{{len(COST_COLUMNS)}}}{{c}}{{Cost (LIBERO-X)}} \\"]
    lines += [r"".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(settings))) + rf"\cmidrule(lr){{{3 + n_area}-{2 + n_area + len(COST_COLUMNS)}}}"]
    lines += ["Method & Cal. & " + " & ".join(e.capitalize() for _ in settings for e in ("seen", "unseen")) + " & " + " & ".join(name for _, name, _ in COST_COLUMNS) + r" \\",
              r"\midrule"]
    previous_ours = rows[0].ours if rows else False
    for k, r in enumerate(rows):
        if k and r.ours != previous_ours:
            lines.append(r"\midrule")
        previous_ours = r.ours
        cells = []
        for i, area in enumerate(r.areas):
            if area is None:
                cells.append(MISSING)
                continue
            text = number(area[0], 1) + (rf"{{\scriptsize$\pm${area[1]:.1f}}}" if with_std else "")
            cells.append(style(text, area_marks[i][k]))
        cells += [MISSING if c is None else style(number(c, d), cost_marks[i][k]) for i, ((_, _, d), c) in enumerate(zip(COST_COLUMNS, r.costs))]
        prefix = r"\rowcolor{gray!15} " if r.ours else ""
        lines.append(prefix + f"{escape(r.label)} & {CALIBRATOR_CODES[r.calibrator]} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


CAPTION = (r"Early failure detection on seen and unseen tasks. Each cell is the area under the curve of TPR early against FPR up to 40\% "
           r"(in \%, higher is better), the mean over the splits of the setting; seen: held-out episodes of tasks the method trained on, "
           r"unseen: held-out tasks. Cal.: calibrator (MC max-conformal, FC functional conformal). The shaded row is our probe, trained with the "
           r"pre-onset trajectory (\texttt{onset\_pre}) as the positive label (the other label variants are in the ablation table); it uses one "
           r"architecture in every setting. SAFE-LSTM and SAFE-MLP use one configuration on the three LIBERO-X splits and their best on validation "
           r"elsewhere; every other baseline uses the configuration with the best validation area in each setting. Cost, measured on LIBERO-X (lower is better): seconds to fit, megabytes kept to score "
           r"a new episode, and the online latency in milliseconds to score one policy step on a GPU and on a CPU (four threads); STAC's GPU latency includes "
           r"sampling its ten plans from the policy, which it needs and no other method does, and it has no CPU latency. Real robot: GR00T on eight tasks, one task held out at a time, the mean over five seeds "
           r"(40 splits; the configurations are those chosen on seed 0); STAC, ActProbe and FailureSpot need the planned action chunks, which "
           r"were not saved there. Best in bold, second best in italics.")
STD_CAPTION = r" Values after $\pm$ are the standard deviation over the splits."


def latex_table(rows: list[Row], settings: list[str], with_std: bool = False, caption: str | None = None, label: str | None = None,
                source: str = "python -m cli.figures") -> str:
    """The table as a full-width ``table*`` float: the caption above, a label, the tabular scaled to the text width. Needs booktabs, graphicx
    and xcolor with the table option in the preamble."""
    caption = caption or CAPTION + (STD_CAPTION if with_std else "")
    label = label or ("tab:main_std" if with_std else "tab:main")
    return "\n".join([f"% {source}: areas and costs read from results/, see docs/FIGURES.md; needs booktabs, graphicx and xcolor with the table option",
                      r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\setlength{\tabcolsep}{4pt}",
                      r"\resizebox{\textwidth}{!}{%", tabular(rows, settings, with_std).rstrip("\n") + "%", "}", r"\end{table*}", ""])


def csv_rows(rows: list[Row], settings: list[str]) -> list[dict]:
    """One dict per method: the area mean and std of every setting and set, and the costs."""
    out = []
    for r in rows:
        row = {"method": r.label, "calibrator": r.calibrator}
        for (setting, eval_set), area in zip(columns(settings), r.areas):
            row[f"{setting} {eval_set} mean"], row[f"{setting} {eval_set} std"] = (None, None) if area is None else (round(area[0], 3), round(area[1], 3))
        row.update({name: None if c is None else round(c, 5) for (_, name, _), c in zip(COST_COLUMNS, r.costs)})
        out.append(row)
    return out


# ---------------------------------------------------------------- the failure-mode table
# The mode table has fewer numeric columns than the main table's 15, so scaling it to the full text width would make its text larger than the main
# table's. It is set at a fraction of the text width instead, the main table's scale (it fills the text width) applied to the mode table's own
# natural width, so that both have the same text size. Measured with pdflatex at the normal font size in a two-column article: the natural
# width of the main table is 785.7 pt (10.87 in), that of the mode table 480.6 pt for three settings and 616.4 pt for four, i.e. 72.8 pt for the
# label column plus 135.9 pt per setting: a fraction of 0.093 + 0.173 per setting of the main table's width, 0.61 and 0.78 for three and four
# settings; see docs/FIGURES.md (the sizes need not be identical).
MODE_TABLE_WIDTH_BASE, MODE_TABLE_WIDTH_PER_SETTING = 0.093, 0.173


def mode_table_width(n_settings: int) -> float:
    """The width of the mode table as a fraction of the text width (see above)."""
    return round(MODE_TABLE_WIDTH_BASE + MODE_TABLE_WIDTH_PER_SETTING * n_settings, 2)
MODE_CAPTION = (r"Failure-mode prediction on LIBERO-X (10 modes)REAL: top-1 and top-3 accuracy in \%, the mean over the splits of each kind. "
                r"Most common: always the most common training mode (top-3: the three most common); random: a uniform guess among the training "
                r"modes. Shaded rows are our probe (one architecture), by label variant. The predicted mode is the one with the highest mean "
                r"probability over the WINDOW before an anchor step and the anchor itself: the probe's first alarm (own detection, "
                r"max-conformal at $\alpha = 0.2$, scored on the failures that have an alarm, COVERAGE of them) or the true failure onset (oracle "
                r"onset, an upper bound a deployed system cannot reach, not marked). Best in bold, second best in italics, among the rows other "
                r"than the oracle.")


@dataclass(frozen=True)
class ModeRow:
    label: str
    group: str  # 'baseline', 'detected' (our probe at its own alarm), 'oracle' (at the true onset) or 'oracle_on_detected' (CSV only)
    top1: list[float | None]  # percent, one per setting and set in the order of ``columns``
    top3: list[float | None]
    share: list[float | None] | None = None  # 'detected' rows: percent of the failed episodes that get an alarm, same order
    in_latex: bool = True


REAL_ROBOT = "Real robot"  # the label of the real-robot setting (``fp.eval.figures.SETTINGS``)
REAL_CAPTION = (r" and on the real robot (GR00T on eight tasks, one held out at a time, the mean over five seeds, 40 splits; seven of the ten modes "
                r"occur there, so the random baseline guesses among at most those)")


MODE_BALANCED_CAPTION = (r"Failure-mode prediction on LIBERO-XREAL: balanced top-1 and top-3 accuracy in \%, the mean over the failure classes of the share of their "
                         r"failures predicted correctly, pooled over the splits of each kind. Rare modes are merged until every class holds at least 5\% of the "
                         r"failures of its dataset, and the predictions of the mode head (trained on the 10 modes) are merged by summing their probabilities. Most "
                         r"common: always the most common training class (top-3: the three most common); random: a uniform guess among the training classes. "
                         r"Shaded rows are our probe (one architecture), by label variant. The predicted class is the one with the highest mean probability over the "
                         r"WINDOW before an anchor step and the anchor itself: the probe's first alarm (own detection, max-conformal at $\alpha = 0.2$, scored on the "
                         r"failures that have an alarm, COVERAGE of them) or the true failure onset (oracle onset, an upper bound a deployed system cannot reach, not "
                         r"marked). Best in bold, second best in italics, among the rows other than the oracle.")
REAL_BALANCED_CAPTION = r" and on the real robot (GR00T on eight tasks, one held out at a time, the mean over five seeds, 40 splits)"


def mode_caption(rows: list[ModeRow], settings: list[str] | None = None, balanced: bool = False) -> str:
    """``MODE_CAPTION`` (``MODE_BALANCED_CAPTION`` when ``balanced``) with the range of the share of failures that get an alarm read from the rows, and the real
    robot described when it is one of the ``settings`` (the window is 2.4 s on both datasets: six policy steps on LIBERO-X, 20 on the real robot)."""
    shares = [v for r in rows if r.share for v in r.share if v is not None]
    coverage = rf"{min(shares):.0f} to {max(shares):.0f}\%" if shares else "see the CSV"
    real = REAL_ROBOT in (settings or [])
    window = "2.4 s (six steps on LIBERO-X, 20 on the real robot)" if real else "six steps"
    template, real_text = (MODE_BALANCED_CAPTION, REAL_BALANCED_CAPTION) if balanced else (MODE_CAPTION, REAL_CAPTION)
    return template.replace("COVERAGE", coverage).replace("REAL", real_text if real else "").replace("WINDOW", window)


def mode_header(settings: list[str]) -> list[str]:
    """The opening and the header lines of a failure-mode tabular: top-1 and top-3 side by side, a column per setting and set."""
    m, n = len(settings), 2 * len(settings)
    short = [escape(s.removeprefix("LIBERO-X ")) for s in settings]
    return [r"\begin{tabular}{l" + "r" * (2 * n) + "}", r"\toprule",
            rf" & \multicolumn{{{n}}}{{c}}{{Top-1 accuracy (\%)}} & \multicolumn{{{n}}}{{c}}{{Top-3 accuracy (\%)}} \\",
            rf"\cmidrule(lr){{2-{1 + n}}}\cmidrule(lr){{{2 + n}-{1 + 2 * n}}}",
            " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{s}}}" for _ in range(2) for s in short) + r" \\",
            "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(2 * m)),
            "Method & " + " & ".join(e.capitalize() for _ in range(2 * m) for e in ("seen", "unseen")) + r" \\"]


def mode_float(lines: list[str], n_settings: int, caption: str, label: str, source: str, width: float | None = None) -> str:
    """The ``table*`` float around the tabular ``lines`` of a failure-mode table, scaled to ``width`` (a fraction of the text width; by default the one
    ``mode_table_width`` gives)."""
    return "\n".join([f"% {source}: read from results/, see docs/FIGURES.md; needs booktabs, graphicx and xcolor with the table option",
                      r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\setlength{\tabcolsep}{4pt}",
                      rf"\resizebox{{{width or mode_table_width(n_settings)}\textwidth}}{{!}}{{%", "\n".join(lines) + "%", "}", r"\end{table*}", ""])


def mode_latex_table(rows: list[ModeRow], settings: list[str], caption: str | None = None, label: str = "tab:modes",
                     source: str = "python -m cli.figures") -> str:
    """The failure-mode table as a ``table*`` float (see ``latex_table``): top-1 and top-3 side by side, one row per method, the baselines first,
    then our probe at its own alarm and at the oracle onset under their own headings. Only the rows with ``in_latex``."""
    n = 2 * len(settings)  # numeric columns per metric
    shown = [r for r in rows if r.in_latex]
    compared = [r for r in shown if r.group != "oracle"]  # best and second best are judged among these
    marks_of = {metric: [marks([getattr(r, metric)[i] for r in compared], 1, True) for i in range(n)] for metric in ("top1", "top3")}
    lines = mode_header(settings)
    headings = {"detected": "Ours, own detection (first alarm)", "oracle": "Ours, oracle onset (upper bound)"}
    previous = None
    for r in shown:
        if r.group != previous:
            lines.append(r"\midrule")
            if r.group in headings:
                lines.append(rf"\multicolumn{{{1 + 2 * n}}}{{l}}{{{headings[r.group]}}} \\")
        previous = r.group
        index = compared.index(r) if r in compared else None

        def cell(value, metric, i):
            return MISSING if value is None else style(number(value, 1), marks_of[metric][i][index] if index is not None else None)

        cells = [cell(v, "top1", i) for i, v in enumerate(r.top1)] + [cell(v, "top3", i) for i, v in enumerate(r.top3)]
        lines.append(("" if r.group == "baseline" else r"\rowcolor{gray!15} ") + f"{escape(r.label)} & " + " & ".join(cells) + r" \\")
    return mode_float(lines + [r"\bottomrule", r"\end{tabular}"], len(settings), caption or mode_caption(rows, settings), label, source)


def mode_csv_rows(rows: list[ModeRow], settings: list[str]) -> list[dict]:
    """Every row, also those left out of the LaTeX: a column per metric, setting and set."""
    out = []
    for r in rows:
        record = {"method": r.label, "group": r.group}
        for metric, name in (("top1", "top-1"), ("top3", "top-3"), ("share", "alarm share")):
            values = getattr(r, metric)
            if values is not None:
                record.update({f"{name} {s} {e}": None if v is None else round(v, 3) for (s, e), v in zip(columns(settings), values)})
        out.append(record)
    return out


# ---------------------------------------------------------------- what the CLIP head and the score term add to the mode prediction
MODE_ABLATION_CAPTION = (r"What the CLIP head and the score term add to failure-mode prediction: top-1 and top-3 accuracy in \%, the mean over the splits of each kind "
                         r"(three, three and eight on LIBERO-X, the eight leave-one-task-out splits of one seed on the real robot). All probes have the LSTM trunk "
                         r"and the pre-onset label; ``Score'' marks training with the failure-score loss, the other probes are trained on the mode (and CLIP) "
                         r"terms alone and have no alarm of their own. The mode is the one with the highest mean probability over the 2.4 s before an anchor "
                         r"step and the anchor itself; the anchor is the same for all four probes: the first alarm of our probe (max-conformal at $\alpha = 0.2$, "
                         r"the failures that get one) or the true failure onset (an upper bound a deployed system cannot reach). The shaded row is our probe. "
                         r"Best in bold, second best in italics, per column and anchor.")


@dataclass(frozen=True)
class ModeAblationRow:
    label: str
    ours: bool
    top1: list[float | None]  # mean over the splits, percent, one per setting and set in the order of ``columns``
    top3: list[float | None]
    top1_std: list[float | None]  # population standard deviation over the splits, same order
    top3_std: list[float | None]


@dataclass(frozen=True)
class ModeAblationBlock:
    title: str  # the anchor the mode is read at
    rows: list[ModeAblationRow]


def mode_ablation_latex_table(blocks: list[ModeAblationBlock], settings: list[str], with_std: bool = False, caption: str | None = None,
                              label: str = "tab:mode_ablation", source: str = "python -m cli.figures") -> str:
    """The mode-ablation table as a ``table*`` float (see ``mode_latex_table``): a block per anchor, a row per probe, the best and the second best of a
    column marked within its block. With ``with_std`` each cell has the standard deviation over the splits after $\\pm$ and the table is set at the full text
    width (the cells are wider, so the fraction of the plain table would make the text smaller)."""
    n = 2 * len(settings)
    lines = mode_header(settings)
    for block in blocks:
        marks_of = {metric: [marks([getattr(r, metric)[i] for r in block.rows], 1, True) for i in range(n)] for metric in ("top1", "top3")}
        lines += [r"\midrule", rf"\multicolumn{{{1 + 2 * n}}}{{l}}{{{block.title}}} \\"]
        for j, r in enumerate(block.rows):
            cells = [MISSING if v is None else style(number(v, 1) + (rf"{{\scriptsize$\pm${sd:.1f}}}" if with_std else ""), marks_of[metric][i][j])
                     for metric in ("top1", "top3") for i, (v, sd) in enumerate(zip(getattr(r, metric), getattr(r, f"{metric}_std")))]
            lines.append((r"\rowcolor{gray!15} " if r.ours else "") + f"{escape(r.label)} & " + " & ".join(cells) + r" \\")
    caption = (caption or MODE_ABLATION_CAPTION) + (STD_CAPTION if with_std else "")
    return mode_float(lines + [r"\bottomrule", r"\end{tabular}"], len(settings), caption, label, source, 1.0 if with_std else None)


def mode_ablation_csv_rows(blocks: list[ModeAblationBlock], settings: list[str]) -> list[dict]:
    """Every cell with its standard deviation: a column per metric, setting and set, and a row per anchor and probe."""
    out = []
    for block in blocks:
        for r in block.rows:
            record = {"anchor": block.title, "method": r.label}
            for metric, name in (("top1", "top-1"), ("top3", "top-3")):
                for (s, e), v, sd in zip(columns(settings), getattr(r, metric), getattr(r, f"{metric}_std")):
                    record[f"{name} {s} {e}"] = None if v is None else round(v, 3)
                    record[f"{name} {s} {e} std"] = None if sd is None else round(sd, 3)
            out.append(record)
    return out


# ---------------------------------------------------------------- the LTT against max conformal and functional conformal table
LTT_CAPTION = (r"Learn-then-Test (LTT) against max conformal (MC) and functional conformal (FC) calibration of the same probe at the same "
               r"false-positive-rate (FPR) target, over RUNS "
               r"runs (one per splitKINDS). LTT (STARTS fixed-sequence walks, $\delta = DELTA$) certifies the FPR target together with the TPR-early target "
               r"of the second column, the highest one that at least half of the runs certify; it can abstain, so its percentages are over the runs it "
               r"certifies (column Runs). MC and FC always return a threshold (the FC band varies with the step) and have no TPR-early target. Holds: the share of runs in which the "
               r"target is met on the held-out episodes of the tasks the calibration used (seen) or on held-out tasks (unseen); FPR and TPR early "
               r"(in \%): the mean over the runs and, after $\pm$, the standard deviation. PPV bound: the lower bound "
               r"$\beta\pi/(\beta\pi+\alpha(1-\pi))$ on the probability of failure given an alarm, from the two targets ($\beta$: TPR early, $\alpha$: FPR) and the "
               r"failure prevalence $\pi$ of the calibration set (mean over the certified runs); MC and FC have no TPR-early target, so no bound. PPV: the "
               r"realised share of failed rollouts among those with an alarm (mean over the runs that have one); PPV holds: the share of those runs in which it is at "
               r"least the bound, computed with the calibration set's prevalence or, in the next column, with the prevalence of the evaluated set. The failure "
               r"prevalence is SEEN_PI\% on the seen and UNSEEN_PI\% on the unseen episodes (mean over the runs). Shaded rows are "
               r"LTT. The guarantee covers the seen episodes only, and the TPR-early targets were chosen from these runs.")


@dataclass(frozen=True)
class LttCells:
    """One evaluation set of one row of the LTT table: the share of runs (%) in which each target holds, the mean and standard deviation over
    the runs of the FPR and of the TPR early (%), and the PPV: its mean over the runs that have an alarm (%) and the share of those runs (%) in which
    it is at least the bound computed with the calibration set's prevalence and with the set's own."""
    fpr_holds: float
    tpr_holds: float | None  # None for a method without a TPR-early target
    fpr: tuple[float, float]
    tpr_early: tuple[float, float]
    ppv: float | None = None  # None when no run has an alarm in the set
    ppv_holds_calibration: float | None = None  # None without a bound (no TPR-early target)
    ppv_holds_own: float | None = None
    prevalence: float | None = None  # the mean failure prevalence of the set over the runs (a fraction)


@dataclass(frozen=True)
class LttRow:
    fpr_target: float  # a fraction
    tpr_target: float | None  # a fraction; None for max-conformal, and for LTT when no TPR-early target is certified by enough runs
    method: str
    highlight: bool  # shaded: the method the table argues for
    n_certified: int  # runs with a threshold
    n_runs: int
    seen: LttCells | None  # None when no run has a threshold
    unseen: LttCells | None
    ppv_bound: float | None = None  # % : the mean over the certified runs of the PPV bound at the calibration prevalence; None without a TPR-early target


def percent(value: float) -> str:
    return rf"{value:.0f}\%"


def optional(value: float | None, show) -> str:
    return MISSING if value is None else show(value)


def ltt_cells(cells: LttCells | None) -> list[str]:
    """FPR holds, TPR holds, FPR, TPR early, PPV, PPV holds and PPV holds with the set's own prevalence, of one set."""
    if cells is None:
        return [MISSING] * 7
    spread = lambda mean_std: number(mean_std[0], 1) + rf"{{\scriptsize$\pm${mean_std[1]:.1f}}}"
    return [percent(cells.fpr_holds), optional(cells.tpr_holds, percent), spread(cells.fpr), spread(cells.tpr_early), optional(cells.ppv, lambda v: number(v, 1)),
            optional(cells.ppv_holds_calibration, percent), optional(cells.ppv_holds_own, percent)]


def ltt_tabular(rows: list[LttRow]) -> str:
    """The bare ``tabular``: the two targets and the PPV bound they give first, then the method and its runs, then the seen and the unseen columns; one block per
    FPR target."""
    per_set = ["FPR holds", "TPR holds", "FPR", "TPR early", "PPV", r"\shortstack{PPV\\holds}", r"\shortstack{PPV holds\\(own $\pi$)}"]  # two-line headers keep the columns narrow
    lines = [r"\begin{tabular}{llrlr" + "r" * 2 * len(per_set) + "}", r"\toprule",
             r" & & & & & \multicolumn{7}{c}{Seen tasks} & \multicolumn{7}{c}{Unseen tasks} \\",
             r"\cmidrule(lr){6-12}\cmidrule(lr){13-19}",
             r"FPR target & TPR-early target & PPV bound & Method & Runs & " + " & ".join(per_set * 2) + r" \\", r"\midrule"]
    previous = None
    for r in rows:
        if r.fpr_target != previous and previous is not None:
            lines.append(r"\midrule")
        first = percent(100 * r.fpr_target) if r.fpr_target != previous else ""
        previous = r.fpr_target
        target = MISSING if r.tpr_target is None else percent(100 * r.tpr_target)
        bound = optional(r.ppv_bound, lambda v: number(v, 1))
        runs = percent(100 * r.n_certified / r.n_runs) + f" ({r.n_certified}/{r.n_runs})"
        cells = ltt_cells(r.seen) + ltt_cells(r.unseen)
        lines.append((r"\rowcolor{gray!15} " if r.highlight else "") + " & ".join([first, target, bound, escape(r.method), runs] + cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def ltt_caption(rows: list[LttRow], delta: float, n_starts: int, kinds: str = "") -> str:
    """``LTT_CAPTION`` with the number of runs, what the runs are (``kinds``, e.g. "3 task seeds, 8 skill folds"; left out when empty), the walks, delta and the
    failure prevalence of the seen and the unseen set (a property of the data, the same for every row: read from a max-conformal row, which has every run)."""
    every_run = next((r for r in rows if not r.highlight and r.seen is not None), None)
    prevalence = lambda cells: "--" if cells is None or cells.prevalence is None else f"{100 * cells.prevalence:.1f}"
    return (LTT_CAPTION.replace("RUNS", str(max(r.n_runs for r in rows))).replace("KINDS", f": {kinds}" if kinds else "").replace("STARTS", str(n_starts))
            .replace("DELTA", f"{delta:g}")
            .replace("UNSEEN_PI", prevalence(every_run and every_run.unseen)).replace("SEEN_PI", prevalence(every_run and every_run.seen)))  # (UNSEEN_PI contains SEEN_PI)


def ltt_latex_table(rows: list[LttRow], delta: float, n_starts: int, caption: str | None = None, label: str = "tab:ltt",
                    source: str = "python -m cli.ltt_table", kinds: str = "") -> str:
    """The LTT table as a ``table*`` float (see ``latex_table``); ``kinds`` says in the caption what the runs are (``ltt_caption``). Needs booktabs, graphicx and
    xcolor with the table option."""
    return ltt_float(ltt_tabular(rows), caption or ltt_caption(rows, delta, n_starts, kinds), label, source)


def ltt_float(tabular: str, caption: str, label: str, source: str, width: float = 1.0, needs: str = "", before: tuple[str, ...] = (), placement: str = "t") -> str:
    """The ``table*`` float around an LTT ``tabular``, scaled to ``width`` times the text width. ``needs``: further packages the table uses, named in the header
    comment; ``before``: lines placed before the float (command definitions)."""
    return "\n".join([f"% {source}: read from results/, see docs/FIGURES.md; needs booktabs, graphicx, {needs}xcolor with the table option", *before,
                      rf"\begin{{table*}}[{placement}]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                      r"\setlength{\tabcolsep}{4pt}", rf"\resizebox{{{width}\textwidth}}{{!}}{{%", tabular.rstrip("\n") + "%", "}", r"\end{table*}", ""])


# The compact version for the paper's text: three FPR targets, no PPV columns (the PPV bound is in the text), no spread, and in each cell the mean rate with the
# share of runs in which the target holds in parentheses. The full table above goes to the appendix.
COMPACT_FPR_TARGETS = (0.2, 0.3, 0.4)
LTT_COMPACT_CAPTION = (r"Learn-then-Test (LTT) against max conformal (MC) and functional conformal (FC) calibration of our probe at three false-positive-rate (FPR) targets, over RUNS runs (one per split). "
                       r"Mean FPR and TPR early (in \%) over the runs, with the share of runs in which the target holds in parentheses. LTT ($\delta = DELTA$) "
                       r"certifies the FPR target together with the TPR-early target shown and may abstain, so its numbers are over the runs it certifies (Runs). "
                       r"MC and FC have no TPR-early target. Seen: held-out episodes of the calibration tasks; unseen: held-out tasks. Shaded: LTT.")
# The marks of the share of runs in which a target holds: a tick when it holds in every run, a wave on the unseen tasks (where no guarantee applies) when it holds
# in at least ``WAVE_FROM`` percent of the runs, a cross otherwise. A rate without a target has no mark. The commands are defined with \providecommand, so the
# file can be input beside a paper that defines them.
WAVE_FROM = 50.0
MARK_COMMANDS = (r"\providecommand{\gtick}{\textcolor{green!60!black}{\ding{51}}}", r"\providecommand{\owave}{\textcolor{orange}{$\boldsymbol{\sim}$}}",
                 r"\providecommand{\rcross}{\textcolor{red}{\ding{55}}}", "")
LTT_COMPACT_LEGEND = (r" \gtick: target holds in all runs; \owave: empirically approaches the guarantee (on unseen tasks, holds in at least half of the runs); "
                      r"\rcross: does not hold.")
LTT_COMPACT_TABLE_WIDTH = 0.74  # of the text width: the table's natural width (579.6 pt with the marks, measured with pdflatex as for the mode table) over the main
# table's (785.7 pt), so that its text is as large as the main table's


def ltt_compact_rows(rows: list[LttRow], targets=COMPACT_FPR_TARGETS) -> list[LttRow]:
    """The rows of the FPR ``targets`` only (a target the rows do not have is left out)."""
    return [r for r in rows if any(abs(r.fpr_target - t) < 1e-9 for t in targets)]


def holds_mark(holds: float, unseen: bool) -> str:
    """The mark of a target that holds in ``holds`` percent of the runs (see ``WAVE_FROM``)."""
    if holds >= 100:
        return r"\gtick"
    return r"\owave" if unseen and holds >= WAVE_FROM else r"\rcross"


def rate_with_holds(rate: float, holds: float | None, unseen: bool = False) -> str:
    """The mean ``rate`` (%), in parentheses the share of runs (%) in which its target holds and the mark of that share; without a target only the rate."""
    return number(rate, 1) + ("" if holds is None else rf" ({holds:.0f}\%) {holds_mark(holds, unseen)}")


def ltt_compact_cells(cells: LttCells | None, unseen: bool = False) -> list[str]:
    """FPR and TPR early of one set (``unseen``: the unseen tasks), each with the share of runs in which the target holds."""
    if cells is None:
        return [MISSING] * 2
    return [rate_with_holds(cells.fpr[0], cells.fpr_holds, unseen), rate_with_holds(cells.tpr_early[0], cells.tpr_holds, unseen)]


def ltt_compact_tabular(rows: list[LttRow]) -> str:
    """The bare ``tabular``: the two targets, the method, the runs it certifies, then FPR and TPR early on the seen and on the unseen episodes; a block per FPR target."""
    lines = [r"\begin{tabular}{lllrrrrr}", r"\toprule", r" & & & & \multicolumn{2}{c}{Seen tasks} & \multicolumn{2}{c}{Unseen tasks} \\",
             r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}", r"FPR target & TPR-early target & Method & Runs & FPR & TPR early & FPR & TPR early \\", r"\midrule"]
    previous = None
    for r in rows:
        if r.fpr_target != previous and previous is not None:
            lines.append(r"\midrule")
        first = percent(100 * r.fpr_target) if r.fpr_target != previous else ""
        previous = r.fpr_target
        target = MISSING if r.tpr_target is None else percent(100 * r.tpr_target)
        cells = ltt_compact_cells(r.seen) + ltt_compact_cells(r.unseen, unseen=True)
        lines.append((r"\rowcolor{gray!15} " if r.highlight else "") + " & ".join([first, target, escape(r.method), f"{r.n_certified}/{r.n_runs}"] + cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def ltt_compact_latex_table(rows: list[LttRow], delta: float, caption: str | None = None, label: str = "tab:ltt_compact",
                            source: str = "python -m cli.ltt_table") -> str:
    """The compact LTT table as a ``table*`` float: only the rows of ``COMPACT_FPR_TARGETS`` (``ltt_compact_rows``), see above."""
    rows = ltt_compact_rows(rows)
    caption = caption or LTT_COMPACT_CAPTION.replace("RUNS", str(max(r.n_runs for r in rows))).replace("DELTA", f"{delta:g}") + LTT_COMPACT_LEGEND
    return ltt_float(ltt_compact_tabular(rows), caption, label, source, LTT_COMPACT_TABLE_WIDTH, "pifont, amsmath, ", MARK_COMMANDS, "htb")


def ltt_csv_rows(rows: list[LttRow]) -> list[dict]:
    """One dict per row: the targets (%), the method, the runs, and for the seen and the unseen set every column of the table."""
    out = []
    for r in rows:
        rounded = lambda v: None if v is None else round(v, 3)
        record = {"fpr target": round(100 * r.fpr_target, 3), "tpr-early target": None if r.tpr_target is None else round(100 * r.tpr_target, 3),
                  "ppv bound": rounded(r.ppv_bound), "method": r.method, "runs certified": r.n_certified, "runs": r.n_runs}
        for name, cells in (("seen", r.seen), ("unseen", r.unseen)):
            if cells is not None:
                record.update({f"{name} fpr holds": round(cells.fpr_holds, 3), f"{name} tpr holds": rounded(cells.tpr_holds),
                               f"{name} fpr mean": round(cells.fpr[0], 3), f"{name} fpr std": round(cells.fpr[1], 3),
                               f"{name} tpr early mean": round(cells.tpr_early[0], 3), f"{name} tpr early std": round(cells.tpr_early[1], 3),
                               f"{name} ppv": rounded(cells.ppv), f"{name} ppv holds": rounded(cells.ppv_holds_calibration),
                               f"{name} ppv holds own prevalence": rounded(cells.ppv_holds_own), f"{name} prevalence": rounded(cells.prevalence)})
        out.append(record)
    return out


# ---------------------------------------------------------------- the failure time moved later
ONSET_SHIFT_CAPTION = (r"Early-detection area (TPR early against FPR up to 40\%, in \%, higher is better) on the SET episodes when the failure time of every failed episode "
                       r"is moved later by 0, 1 and 2\,s, the mean over the splits of each setting. The saved scores are scored again against the shifted failure times, "
                       r"with the same configurations and thresholds as the main table; nothing is retrained. Ours is shaded. "
                       r"Best in bold, second best in italics, in each column.")


@dataclass(frozen=True)
class ShiftRow:
    label: str
    ours: bool
    values: list[float | None]  # for each setting, the area at each shift


def onset_shift_latex_table(rows: list[ShiftRow], settings: list[str], shifts: list[float], eval_set: str, caption: str | None = None, label: str = "tab:onset_shift",
                            source: str = "python -m cli.onset_shift") -> str:
    """The table of ``cli.onset_shift`` as a full-width ``table*`` float (see ``latex_table``): a group of columns per setting with a column per shift, best and second
    best marked in each column, ours shaded."""
    n, k = len(settings) * len(shifts), len(shifts)
    column_marks = [marks([r.values[i] for r in rows], 1, True) for i in range(n)]
    header = [r"\begin{tabular}{l" + "r" * n + "}", r"\toprule", " & " + " & ".join(rf"\multicolumn{{{k}}}{{c}}{{{escape(s)}}}" for s in settings) + r" \\",
              "".join(rf"\cmidrule(lr){{{2 + k * i}-{1 + k * (i + 1)}}}" for i in range(len(settings))),
              "Method & " + " & ".join((f"+{s:g}\\,s" if s else "0\\,s") for _ in settings for s in shifts) + r" \\", r"\midrule"]
    lines, previous_ours = list(header), rows[0].ours if rows else False
    for j, r in enumerate(rows):
        if j and r.ours != previous_ours:
            lines.append(r"\midrule")
        previous_ours = r.ours
        cells = [MISSING if v is None else style(number(v, 1), column_marks[i][j]) for i, v in enumerate(r.values)]
        lines.append((r"\rowcolor{gray!15} " if r.ours else "") + f"{escape(r.label)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = caption or ONSET_SHIFT_CAPTION.replace("SET", eval_set)
    return "\n".join([f"% {source}: read from results/, see docs/FIGURES.md; needs booktabs, graphicx and xcolor with the table option", r"\begin{table*}[t]", r"\centering",
                      rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\setlength{\tabcolsep}{4pt}", r"\resizebox{\textwidth}{!}{%", "\n".join(lines) + "%", "}",
                      r"\end{table*}", ""])


# ---------------------------------------------------------------- the ablation table
ABLATION_CAPTION = (r"Ablations of our probe in the simulated settings (LIBERO-10 with $\pi_{0.5}$ and with GR00T, and the three kinds of LIBERO-X held-out set): "
                    r"the area under the curve of TPR early against FPR up to 40\% (in \%, higher is better), the mean "
                    r"over the splits of each setting and, after $\pm$, the standard deviation; in brackets the change relative to our probe "
                    r"(shaded, the same model in both blocks). SPLITS Seen: held-out episodes of tasks the probe trained on; unseen: held-out tasks, "
                    r"signatures or skills. Heads: the architecture (LSTM trunk, no attention) with the pre-onset label, so the rows differ only in the "
                    r"mode head and the CLIP head. Training strategy: the positive label of our architecture: the whole pre-onset trajectory "
                    r"(\texttt{onset\_pre}), every step of the failed rollout (\texttt{onset\_traj}), the window before the onset only (\texttt{onset}), "
                    r"or the window and everything after it (\texttt{onset\_post}). Best in bold, second best in italics, within each block.")
# The table is set at the full text width. Its natural width (863.5 pt with the five settings, measured with pdflatex in a two-column article as for the mode
# table, see above) is larger than the main table's (785.7 pt), so its text is about 9% smaller than the main table's (it would be 559.9 pt, and 0.71 of the
# text width to match, with the three LIBERO-X settings alone).
ABLATION_TABLE_WIDTH = 1.0


@dataclass(frozen=True)
class AblationCell:
    mean: float  # the area, in %
    std: float  # population std over the splits of the setting
    n: int  # number of splits
    change: float | None  # relative change of the mean against the baseline row of the block, in %; None on the baseline row


@dataclass(frozen=True)
class AblationRow:
    label: str
    ours: bool
    cells: list[AblationCell | None]  # one per setting and set in the order of ``columns``


@dataclass(frozen=True)
class AblationBlock:
    title: str
    rows: list[AblationRow]  # the baseline (our probe) first


def signed_percent(value: float) -> str:
    """'+6\\%' or '$-$27\\%', rounded to whole percent; the sign follows the rounded value."""
    shown = round(value)
    return ("+" if shown >= 0 else "$-$") + percent(abs(shown))


def ablation_caption(blocks: list[AblationBlock], settings: list[str]) -> str:
    """``ABLATION_CAPTION`` with the number of splits of each setting read from the first row."""
    counts = [f"{escape(s)} {c.n}" for s, c in zip(settings, blocks[0].rows[0].cells[::2]) if c is not None]
    return ABLATION_CAPTION.replace("SPLITS", f"Number of splits: {', '.join(counts)}.")


def ablation_tabular(blocks: list[AblationBlock], settings: list[str]) -> str:
    """The bare ``tabular``: one block of rows per ablation under its heading, the baseline (shaded) first, best and second best marked per block."""
    m, n = len(settings), 2 * len(settings)
    names = [escape(s) for s in settings]
    lines = [r"\begin{tabular}{l" + "r" * n + "}", r"\toprule", " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{s}}}" for s in names) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(m)), " & " + " & ".join(e.capitalize() for _ in range(m) for e in ("seen", "unseen")) + r" \\"]
    for block in blocks:
        lines += [r"\midrule", rf"\multicolumn{{{1 + n}}}{{l}}{{{escape(block.title)}}} \\"]
        column_marks = [marks([r.cells[i].mean if r.cells[i] else None for r in block.rows], 1, True) for i in range(n)]
        for k, r in enumerate(block.rows):
            cells = []
            for i, cell in enumerate(r.cells):
                if cell is None:
                    cells.append(MISSING)
                    continue
                text = style(number(cell.mean, 1), column_marks[i][k]) + rf"{{\scriptsize$\pm${cell.std:.1f}}}"
                cells.append(text + ("" if cell.change is None else rf" {{\scriptsize({signed_percent(cell.change)})}}"))
            lines.append((r"\rowcolor{gray!15} " if r.ours else "") + f"{escape(r.label)} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def ablation_latex_table(blocks: list[AblationBlock], settings: list[str], caption: str | None = None, label: str = "tab:ablations",
                         source: str = "python -m cli.figures") -> str:
    """The ablation table as a ``table*`` float (see ``latex_table``). Needs booktabs, graphicx and xcolor with the table option."""
    return "\n".join([f"% {source}: read from results/, see docs/FIGURES.md; needs booktabs, graphicx and xcolor with the table option",
                      r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption or ablation_caption(blocks, settings)}}}", rf"\label{{{label}}}", r"\setlength{\tabcolsep}{4pt}",
                      rf"\resizebox{{{ABLATION_TABLE_WIDTH}\textwidth}}{{!}}{{%", ablation_tabular(blocks, settings).rstrip("\n") + "%", "}", r"\end{table*}", ""])


def ablation_csv_rows(blocks: list[AblationBlock], settings: list[str]) -> list[dict]:
    """One dict per row: its block and label, and the mean, std and change of every setting and set."""
    out = []
    for block in blocks:
        for r in block.rows:
            record = {"block": block.title, "variant": r.label, "ours": r.ours}
            for (setting, eval_set), cell in zip(columns(settings), r.cells):
                if cell is not None:
                    record.update({f"{setting} {eval_set} mean": round(cell.mean, 3), f"{setting} {eval_set} std": round(cell.std, 3),
                                   f"{setting} {eval_set} change %": None if cell.change is None else round(cell.change, 3)})
            out.append(record)
    return out
