"""What the CLIP head and the score term add to the failure-mode prediction of our probe, from the saved runs (no model is run).

Four probes with the same LSTM trunk and the pre-onset label, each trained on every split of LIBERO-X and on the leave-one-task-out splits of the real robot (seed 0):

    score + modes + CLIP   ours (``figures.OURS_ARCHITECTURE``), trained on all three terms of the loss: results/
    score + modes          the same without the CLIP head: results/
    modes + CLIP           the same two heads but trained without the score term (``--lam-score 0``): results/mode_only/
    modes                  the mode head alone, without score term and CLIP head: results/mode_only/

The mode head is scored as in ``fp.eval.modes`` (mean softmax over the on-time window ending at an anchor step, top-1 and top-3 accuracy on the failed
episodes), at two anchors. A probe trained without the score term has no alarm of its own, so all four are read at the anchors of the same episodes:

    first alarm   the first alarm of our probe (score + modes + CLIP, max-conformal at ``modes.ALPHA``), on the failed episodes that get one
    true onset    the true failure onset, on all failed episodes (an upper bound: a deployed system does not know it)

The numbers are per split, then the mean over the splits of a setting and, for the spread, their population standard deviation, in percent, for the seen and
the unseen episodes. The settings are the three kinds of LIBERO-X split and the real robot's seed-0 splits (the other seeds were run for our probe only).
"""

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..calibration.protocol import make_calibrator
from ..results import read_scores, run_dir
from . import figures, modes
from .select_config import group_of
from .tables import ModeAblationBlock, ModeAblationRow

REAL_SEED0 = figures.Setting("Real robot", "real8_groot", "all")  # the pooled leave-one-task-out splits of seed 0, without the extra seeds (``figures.SETTINGS``)
SETTINGS = (*modes.LIBEROX, REAL_SEED0)
VARIANT = "onset_pre"  # the label of our probe
FULL_DIR, MODE_ONLY_DIR = "results", "mode_only"  # the runs with all three terms (the root of the sweep) and without the score term (a folder in it)
# label, whether the probe is trained with the score term, architecture
MODELS = (("Score + modes + CLIP (ours)", True, figures.OURS_ARCHITECTURE), ("Score + modes", True, "lstm_modes"),
          ("Modes + CLIP", False, figures.OURS_ARCHITECTURE), ("Modes", False, "lstm_modes"))
MODEL_NAMES = tuple(label for label, _, _ in MODELS)
ANCHORS = (("alarm", "At the first alarm of our probe"), ("oracle", "At the true failure onset (upper bound)"))


def mode_ablation_results(inputs, root: Path, settings=SETTINGS, alpha: float = modes.ALPHA) -> dict[tuple, list[float]]:
    """{(model label, anchor, metric, setting label, set): [value per split, in percent]}: ``anchor`` is 'alarm' or 'oracle', ``metric`` 'top1' or 'top3'.
    ``root`` holds the runs with all terms; the runs without the score term are in ``root / MODE_ONLY_DIR``. Sets without a failed episode are left out of a split;
    for the alarm anchor so are the splits in which no failed episode of the set gets an alarm."""
    per_split = defaultdict(list)
    for setting in settings:
        for name in (n for n in inputs.splits if group_of(n) == setting.group):
            split = inputs.splits[name]
            _, sets, true_modes = modes.split_modes(inputs, split)
            reference = run_dir(inputs.dataset, name, f"{figures.OURS_ARCHITECTURE}__{VARIANT}", root)
            scores = read_scores(reference)
            calibrator = make_calibrator(reference.name, seed=split.seed).fit([scores[k] for k in split.calibration if inputs.episodes[k].success])
            for label, with_score, architecture in MODELS:
                folder = run_dir(inputs.dataset, name, f"{architecture}__{VARIANT}", root if with_score else root / MODE_ONLY_DIR)
                with np.load(folder / "mode_logits.npz") as z:
                    logits = {k: z[k] for k in true_modes}
                for eval_set, keys in sets.items():
                    result = modes.evaluate_split(logits, scores, inputs.episodes, true_modes, keys, calibrator, alpha, inputs.window)
                    for anchor, _ in ANCHORS:
                        for k in modes.TOP_K:
                            value = result[f"{'detected' if anchor == 'alarm' else 'oracle'}{k}"]
                            if not np.isnan(value):
                                per_split[(label, anchor, f"top{k}", setting.label, eval_set)].append(100 * value)
    return dict(per_split)


def mode_ablation_blocks(results: dict[tuple, list[float]], settings=SETTINGS) -> list[ModeAblationBlock]:
    """One block per anchor, a row per model with the mean and the standard deviation over the splits for each metric, setting and set
    (None for a cell without a split)."""
    columns = [(s.label, e) for s in settings for e in modes.SETS]

    def cells(label: str, anchor: str, metric: str) -> tuple[list[float | None], list[float | None]]:
        values = [results.get((label, anchor, metric, s, e)) for s, e in columns]
        return [None if v is None else float(np.mean(v)) for v in values], [None if v is None else float(np.std(v)) for v in values]

    blocks = []
    for anchor, title in ANCHORS:
        rows = []
        for k, (label, _, _) in enumerate(MODELS):
            (top1, top1_std), (top3, top3_std) = cells(label, anchor, "top1"), cells(label, anchor, "top3")
            rows.append(ModeAblationRow(label, k == 0, top1, top3, top1_std, top3_std))
        blocks.append(ModeAblationBlock(title, rows))
    return blocks
