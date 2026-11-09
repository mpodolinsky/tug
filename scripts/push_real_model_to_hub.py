"""Push the real-robot probe (trained on all 8 tasks) to the Hub: safetensors weights, a config.json with the alarm thresholds, and a model card.

    python scripts/push_real_model_to_hub.py            # dry run: the choice, the thresholds, the staged files and the card
    python scripts/push_real_model_to_hub.py --push     # create the repo (private; --public for a public one) and upload

The model is our main-table probe (``ARCHITECTURE``) with the pre-onset labels, trained on every task with nothing held out, on the best of the
three ``full_seed*`` splits by validation area (the protocol's rule: never the held-out "seen" results). The card gives the thresholds for a set of
levels alpha (max-conformal, ``fp.calibration.max_conformal``, from the maxima of the calibration successes stored in the file) next to what each
threshold did on the split's held-out episodes, the data, the other seeds, and the same probe's result on tasks it had not seen. The staged
tensors are read back before anything is uploaded. Run it with an interpreter that has torch, safetensors and ``huggingface_hub`` (the openpi one).
"""

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fp.calibration.max_conformal import MaxConformal  # noqa: E402
from fp.config import RESULTS, get  # noqa: E402
from fp.data.targets import MODE_NAMES  # noqa: E402
from fp.export import calibration_maxima, hub_api, load_checkpoint, push_folder, verify_tensors, write_tensors  # noqa: E402
from fp.models.ours import ProbeConfig  # noqa: E402

OWNER = "podolinsky"
DATASET = "real8_groot"
ARCHITECTURE = "lstm_modes_clip"
VARIANT = "onset_pre"
SPLITS = ("full_seed0", "full_seed1", "full_seed2")
ALPHAS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)  # all on the evaluation grid (fp.metrics.curves.ALPHAS)
LABEL_TEXT = "positive on every step from the start of the episode to the failure onset, the rollout cut at the onset"


def repo_name(split: str) -> str:
    return f"groot-real8-probe-lstm-{VARIANT.replace('_', '-')}-mode-clip-{split.replace('_', '-')}"


def run_path(root: Path, split: str, dataset: str = DATASET) -> Path:
    return Path(root) / dataset / split / f"{ARCHITECTURE}__{VARIANT}"


def read_run(root: Path, split: str) -> dict:
    """The evaluation of a run under max-conformal: the areas and counts of each set (summary.json) and the curve rows of the seen set."""
    run = run_path(root, split)
    sets = json.loads((run / "summary.json").read_text())["calibrators"]["max_conformal"]["sets"]
    rows = json.loads((run / "curve_max_conformal.json").read_text())["sets"]["seen"]
    return {"sets": sets, "seen_rows": rows}


def select_split(root: Path) -> str:
    """The split with the best validation area among those with weights and an evaluation (ties: the earlier name)."""
    ready = [s for s in SPLITS if (run_path(root, s) / "weights.pt").exists() and (run_path(root, s) / "summary.json").exists()]
    if not ready:
        raise ValueError(f"no {ARCHITECTURE}__{VARIANT} run on {SPLITS} has weights and an evaluation (python -m cli.train, cli.evaluate)")
    return max(ready, key=lambda s: read_run(root, s)["sets"]["val"]["area"])


def thresholds(maxima: np.ndarray, alphas=ALPHAS) -> dict[float, float | None]:
    """{alpha: alarm threshold} of the max-conformal calibrator fitted on the calibration maxima; None where no finite threshold exists."""
    calibrator = MaxConformal().fit([np.array([m]) for m in maxima])
    return {a: (None if np.isinf(v := float(calibrator.threshold(a, 1)[0])) else v) for a in alphas}


def operating_points(rows: list[dict], alphas=ALPHAS) -> dict[float, dict]:
    """The seen-set rates the evaluation recorded at each alpha: {alpha: {fpr, tpr_early, tpr_on_time}} (percentages)."""
    by_alpha = {round(r["alpha"], 3): r for r in rows}
    missing = [a for a in alphas if a not in by_alpha]
    if missing:
        raise ValueError(f"the evaluation has no row for alpha {missing}")
    return {a: {k: by_alpha[a][k] for k in ("fpr", "tpr_early", "tpr_on_time")} for a in alphas}


def seed_summary(root: Path) -> dict:
    """Mean and population std over the seeds of the validation and seen areas."""
    areas = {k: [read_run(root, s)["sets"][k]["area"] for s in SPLITS if (run_path(root, s) / "summary.json").exists()] for k in ("val", "seen")}
    return {"n_seeds": len(areas["val"]), **{k: (float(np.mean(v)), float(np.std(v))) for k, v in areas.items()}}


def unseen_task_summary(root: Path) -> dict | None:
    """The same probe in the leave-one-task-out runs (one task unseen each): mean seen and unseen area and the number of runs; None if absent."""
    runs = sorted((Path(root) / DATASET).glob(f"loto_*/{ARCHITECTURE}__{VARIANT}/summary.json"))
    if not runs:
        return None
    sets = [json.loads(p.read_text())["calibrators"]["max_conformal"]["sets"] for p in runs]
    return {"n_runs": len(runs), "seen": float(np.mean([s["seen"]["area"] for s in sets])), "unseen": float(np.mean([s["unseen"]["area"] for s in sets]))}


def read_prompts(manifest: Path) -> dict[str, str]:
    """{task: instruction} from the dataset's manifest.csv."""
    with open(manifest, newline="") as f:
        return {r["task"]: r["prompt"] for r in csv.DictReader(f)}


def data_summary(inputs, prompts: dict[str, str], split: str) -> dict:
    """What the model was trained on: the tasks (instruction, episodes, successes), the failure modes and the sizes of the split's parts."""
    episodes, s = inputs.episodes, inputs.splits[split]
    tasks = {t: {"instruction": prompts[t], "episodes": sum(e.task == t for e in episodes.values()), "successes": sum(e.success for e in episodes.values() if e.task == t)}
             for t in sorted({e.task for e in episodes.values()})}
    modes = {}
    for index in inputs.targets.modes.values():
        modes[MODE_NAMES[index]] = modes.get(MODE_NAMES[index], 0) + 1
    count = lambda keys: {"episodes": len(keys), "successes": sum(episodes[k].success for k in keys)}
    return {"tasks": tasks, "failure_modes": dict(sorted(modes.items(), key=lambda kv: -kv[1])),
            "split": {part: count(getattr(s, part)) for part in ("train", "val", "calibration", "seen")}}


def build_config(root: Path, split: str, inputs, prompts: dict[str, str], maxima: np.ndarray, saved: dict, tensors: dict) -> dict:
    history = json.loads((run_path(root, split) / "history.json").read_text())
    run = read_run(root, split)
    cut, points = thresholds(maxima), operating_points(run["seen_rows"])
    return {
        "name": f"vla-probe {ARCHITECTURE} ({VARIANT}), real robot, all 8 tasks, split {split}", "architecture": ARCHITECTURE, "architecture_config": saved["cfg"],
        "label_variant": VARIANT, "label_description": LABEL_TEXT, "dataset": DATASET, "split": split,
        "selection": f"best validation area of the {len(SPLITS)} seeds",
        "input": {"features": "pooled per-modality token means of the policy's backbone, one vector per policy step: [base image | wrist image | language], 2048 columns each (6144)",
                  "standardisation": "(x - feature_mean) / feature_std, with the two tensors of model.safetensors (statistics of the split's training episodes)",
                  "on_time_window_steps": history["window"]},
        "outputs": {"score": "failure probability per policy step: the sigmoid of the score head's logit", "mode": "failure-mode logits per step over mode_names",
                    "sem": f"semantic embedding per step ({ProbeConfig(**saved['cfg']).clip_dim}-d, trained toward the CLIP text feature of the step's phrase)"},
        "mode_names": list(MODE_NAMES),
        "alarm": "alarm at the first step whose score >= threshold; thresholds[alpha] is the k-th smallest of calibration_maxima, k = ceil((n + 1)(1 - alpha)), n = len(calibration_maxima); "
                 "null (no finite threshold, the detector never alarms) if k > n",
        "thresholds": {str(a): v for a, v in cut.items()},
        "seen_set_at_threshold": {str(a): v for a, v in points.items()},
        "seen_set": {"episodes": run["sets"]["seen"]["n_fail"] + run["sets"]["seen"]["n_succ"], "failures": run["sets"]["seen"]["n_fail"], "successes": run["sets"]["seen"]["n_succ"]},
        "areas": {"metric": "area under TPR early against FPR up to 40%, in %, max-conformal alarms", "validation": run["sets"]["val"]["area"], "seen": run["sets"]["seen"]["area"]},
        "seeds": seed_summary(root), "unseen_tasks": unseen_task_summary(root), "data": data_summary(inputs, prompts, split),
        "training": {"seed": history["seed"], "best_epoch": history["best_epoch"], "epochs": history["epochs"], "stopped_early": history["stopped_early"], "split_sha256": history["split_sha256"]},
        "code": history.get("code"), "tensors": {name: list(t.shape) for name, t in tensors.items()},
    }


def model_card(config: dict) -> str:
    """The card, as the Hub shows it: what the model is, the thresholds with what they did on held-out episodes, how to use it, the statistics and the caveats."""
    a, seeds, unseen, data, seen = config["areas"], config["seeds"], config["unseen_tasks"], config["data"], config["seen_set"]
    rows = "\n".join(
        f"| {alpha} | " + (f"{config['thresholds'][alpha]:.4f}" if config["thresholds"][alpha] is not None else "none (never alarms)")
        + (f" | {p['fpr']:.1f} | {p['tpr_early']:.1f} | {p['tpr_on_time']:.1f} |" if config["thresholds"][alpha] is not None else " | - | - | - |")
        for alpha, p in config["seen_set_at_threshold"].items())
    n_cal = config["tensors"]["calibration_maxima"][0]
    example_alpha, example = next((a, f"{v:.4f}") for a, v in config["thresholds"].items() if v is not None)
    tasks = "\n".join(f"| {t} | {v['instruction']} | {v['episodes']} | {v['successes']} |" for t, v in data["tasks"].items())
    modes = ", ".join(f"{m} {n}" for m, n in data["failure_modes"].items())
    s = data["split"]
    unseen_line = (f"- **On a task it has not seen the same probe is much weaker:** in {unseen['n_runs']} leave-one-task-out runs (one task unseen each, trained on the other seven) "
                   f"the mean area is {unseen['seen']:.1f} on seen tasks and {unseen['unseen']:.1f} on the unseen one. Do not expect these thresholds to transfer to a new task.\n") if unseen else ""
    return f"""---
library_name: pytorch
tags:
  - robotics
  - failure-detection
  - vision-language-action
---

# GR00T real-robot failure probe (pre-onset, all 8 tasks)

An early-failure detector for a fine-tuned GR00T-N1.7 policy on a real robot: a small LSTM over the policy's pooled backbone features that gives, at every policy step, the probability that the episode is heading for a failure. It is trained on the pre-onset labels (positive from the start of the episode to the failure onset), with a failure-mode head and a semantic head. Trained on all 8 tasks with nothing held out; split `{config['split']}`, the best of {seeds['n_seeds']} seeds on validation. Internal use: no licence is granted yet.

## Thresholds

Alarm at the first policy step whose score is **at or above** the threshold. One constant threshold per level alpha (max-conformal: alpha is the target false-alarm rate on successful episodes, from the maxima of the {n_cal} calibration successes stored in `model.safetensors`; with so few, a level below {1 / (n_cal + 1):.3f} has no finite threshold). The rates are measured on {seen['episodes']} held-out episodes of the same 8 tasks ({seen['successes']} successes, {seen['failures']} failures) that were never used for training or calibration. TPR early: an alarm at or before the failure onset; on-time: within the 2.4 s (20 policy steps) before it.

| alpha | threshold | FPR % | TPR early % | TPR on-time % |
|---|---|---|---|---|
{rows}

## Use

```python
import json, torch
from safetensors.torch import load_file
from fp.models.ours import Probe, ProbeConfig  # the vla-probe code (github.com/mpodolinsky/vla-probe)

t = load_file("model.safetensors")
model = Probe(ProbeConfig(**json.load(open("config.json"))["architecture_config"])).eval()
model.load_state_dict({{k: v for k, v in t.items() if k not in ("feature_mean", "feature_std", "calibration_maxima")}})
x = ((feats - t["feature_mean"]) / t["feature_std"]).float()  # feats: (1, T, 6144) pooled [base | wrist | language], float64
score = torch.sigmoid(model(x)["score"])                    # (1, T) failure probability, causal in time
alarm = bool((score >= {example})[0].any())                  # {example} is the threshold for alpha = {example_alpha}; `calibration_maxima` gives any other alpha
```

`config.json` has the architecture, the thresholds, the mode names and the exact alarm rule.

## Statistics

- **Area under TPR early against FPR up to 40%** (the paper's metric, %): validation {a['validation']:.1f}, held-out seen episodes {a['seen']:.1f}. Over the {seeds['n_seeds']} seeds: validation {seeds['val'][0]:.1f} ± {seeds['val'][1]:.1f}, seen {seeds['seen'][0]:.1f} ± {seeds['seen'][1]:.1f}.
- **Data:** 399 rollouts (145 successes, 254 failures) of 8 tasks; this split trains on {s['train']['episodes']} episodes, validates on {s['val']['episodes']}, calibrates on {s['calibration']['episodes']} and tests on {s['seen']['episodes']}. Failure labels come from Gemini and are not verified by a person (the success flag is the operator's). Failure modes: {modes}.

| task | instruction | episodes | successes |
|---|---|---|---|
{tasks}

## Caveats

{unseen_line}- **The false-alarm rate is only roughly alpha**: with {n_cal} calibration successes the thresholds are coarse, the FPR is measured on {seen['successes']} successes (steps of {100 / seen['successes']:.1f} points), and the guarantee holds for episodes like the calibration ones, not for a new setting.
- **Few alarms fall in the last 2.4 s before the onset** (on-time TPR in the table): most alarms that count as early come well before that window.
- {seen['episodes']} test episodes are few: differences of a few points between seeds or alphas are within noise.
- It reads the features of this fine-tuned GR00T policy, pooled as described in `config.json`; it will not work on another policy or pooling.
"""


def stage(root: Path, inputs, prompts: dict[str, str], folder: Path) -> tuple[str, dict]:
    """Stage the chosen split's model in ``folder`` and verify it; returns the split and the config."""
    split = select_split(root)
    run = run_path(root, split)
    saved = load_checkpoint(run)
    cfg = ProbeConfig(**saved["cfg"])
    if cfg.name != ARCHITECTURE or saved["variant"] != VARIANT:
        raise ValueError(f"{run} holds {cfg.name} / {saved['variant']}, expected {ARCHITECTURE} / {VARIANT}")
    maxima = calibration_maxima(run, inputs.splits[split], inputs.episodes)
    tensors = write_tensors(saved, maxima, folder)
    verify_tensors(saved, folder, split)
    config = build_config(root, split, inputs, prompts, maxima, saved, tensors)
    (folder / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    (folder / "README.md").write_text(model_card(config))
    return split, config


def describe(folder: Path) -> list[str]:
    return [f"  {p.name}  {p.stat().st_size / 1e6:8.2f} MB" for p in sorted(folder.iterdir())]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=RESULTS)
    p.add_argument("--owner", default=OWNER)
    p.add_argument("--stage", type=Path, help="stage the files in this (empty) folder instead of a temporary one, to inspect them")
    p.add_argument("--public", action="store_true", help="create the repo public (default: private; an existing repo keeps its visibility)")
    p.add_argument("--push", action="store_true")
    args = p.parse_args()

    from fp.data.inputs import load_inputs

    inputs = load_inputs(DATASET)
    prompts = read_prompts(get(DATASET).data_dir / "manifest.csv")
    with tempfile.TemporaryDirectory() as tmp:
        folder = args.stage if args.stage else Path(tmp)
        if args.stage and any(folder.glob("*")):
            sys.exit(f"{folder} is not empty")
        folder.mkdir(parents=True, exist_ok=True)
        split, config = stage(args.root, inputs, prompts, folder)
        repo = f"{args.owner}/{repo_name(split)}"
        print(f"{ARCHITECTURE} / {VARIANT} on {split} ({config['selection']}) -> {repo}, {'public' if args.public else 'private'}")
        print("thresholds:", {a: (None if v is None else round(v, 4)) for a, v in config["thresholds"].items()})
        print("staged and verified (tensors read back, probe outputs identical):")
        print("\n".join(describe(folder)))
        if not args.push:
            print("\n--- README.md ---\n" + (folder / "README.md").read_text())
            print("dry run: nothing was uploaded (--push creates the repo above and uploads the staged files)")
            return
        listed = push_folder(folder, repo, "safetensors weights, config with the alarm thresholds, and card of the real-robot probe", private=not args.public, api=hub_api())
        absent = [f.name for f in folder.iterdir() if f.name not in listed]
        print(f"pushed https://huggingface.co/{repo}: {len(listed)} files listed; missing: {absent or 'none'}")
        if absent:
            sys.exit(f"some staged files are not on the Hub: {absent}")


if __name__ == "__main__":
    main()
