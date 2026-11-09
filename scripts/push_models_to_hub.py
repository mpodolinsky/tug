"""Push the LIBERO-X probes of the main table to the Hub, one public model repo each: safetensors weights, a config.json and a short card.

    python scripts/push_models_to_hub.py            # dry run: the picks, the repos, the staged files with their sizes, and the text of one card
    python scripts/push_models_to_hub.py --push     # create the repos (public) and upload; safe to repeat

Which models: our probe (``ARCHITECTURE``) with the two labels of the table's rows, ``onset_pre`` and ``onset_traj``, on five LIBERO-X splits:
the best split of the task kind and the best of the signature kind (by the validation area of ``onset_pre``: the protocol's rule, never the unseen
results), and ``N_RANDOM_SKILL`` splits of the skill kind drawn at random with a fixed seed. Both labels of a split are pushed, so they can be
compared on the same split: 10 models, 10 repos, named like the earlier probe repos (``groot-libero-x-probe-lstm-onset-mode-clip``) with the label
and the split added.

Per repo: ``model.safetensors`` (the weights, the feature mean and std the probe was trained with, and the sorted maxima of the calibration
successes' scores, from which the alarm threshold of any level is read), ``config.json`` and ``README.md`` (the card). No pickle is uploaded. The
staged files are checked before anything leaves the machine: the tensors are read back and a probe built from them gives the original outputs.
Run it with an interpreter that has torch, safetensors and ``huggingface_hub`` (the openpi one).
"""

import argparse
import json
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fp.config import RESULTS  # noqa: E402
from fp.data.targets import MODE_NAMES  # noqa: E402
from fp.export import calibration_maxima, hub_api, load_checkpoint, push_folder, verify_tensors, write_tensors  # noqa: E402
from fp.models.ours import ProbeConfig  # noqa: E402

OWNER = "podolinsky"
DATASET = "liberox_groot"
DATASET_REPO = "podolinsky/gr00t-libero-x-features-v3"
ARCHITECTURE = "lstm_modes_clip"  # fp.eval.figures.OURS_ARCHITECTURE (a test checks that they agree)
VARIANTS = ("onset_pre", "onset_traj")
KINDS = ("task", "signature", "skill")
N_RANDOM_SKILL = 3
SEED = 0
LICENSE = "cc-by-4.0"
LABEL_TEXT = {"onset_pre": "positive on every step from the start of the episode to the failure onset, the rollout cut at the onset (our `onset_pre`)",
              "onset_traj": "positive on every step of a failed episode, the label of SAFE (our `onset_traj`)"}


@dataclass(frozen=True)
class Pick:
    split: str
    kind: str
    how: str  # why this split
    val_area: float  # validation area of onset_pre on this split


def repo_name(split: str, variant: str) -> str:
    """e.g. groot-libero-x-probe-lstm-onset-pre-mode-clip-task-seed1: the pattern of the earlier probe repos (groot-libero-x-probe-lstm-onset-mode-clip)
    with the label variant and the split."""
    return f"groot-libero-x-probe-lstm-{variant.replace('_', '-')}-mode-clip-{split.replace('_', '-')}"


def run_path(root: Path, split: str, variant: str) -> Path:
    return Path(root) / DATASET / split / f"{ARCHITECTURE}__{variant}"


def read_areas(root: Path, split: str, variant: str) -> dict[str, float]:
    """{val, seen, unseen: area} of a run under max-conformal, from its summary.json."""
    sets = json.loads((run_path(root, split, variant) / "summary.json").read_text())["calibrators"]["max_conformal"]["sets"]
    return {name: s["area"] for name, s in sets.items()}


def available_splits(root: Path) -> dict[str, list[str]]:
    """The splits of each kind whose runs have weights for every label of the table, sorted."""
    found = {kind: [] for kind in KINDS}
    for folder in sorted((Path(root) / DATASET).glob("*")):
        kind = folder.name.split("_")[0]
        if kind in found and all((run_path(root, folder.name, v) / "weights.pt").exists() for v in VARIANTS):
            found[kind].append(folder.name)
    return found


def select_models(root: Path, seed: int = SEED, n_random: int = N_RANDOM_SKILL) -> list[Pick]:
    """The best split of the task and of the signature kind on the validation area of onset_pre (ties: the earlier name), and ``n_random`` skill
    splits drawn with ``random.Random(seed)``."""
    splits = available_splits(root)
    picks = []
    for kind in ("task", "signature"):
        if not splits[kind]:
            raise ValueError(f"no {kind} split has the weights of {ARCHITECTURE} for {VARIANTS}")
        best = max(splits[kind], key=lambda s: read_areas(root, s, "onset_pre")["val"])
        picks.append(Pick(best, kind, f"best of the {len(splits[kind])} {kind} splits on the validation area of onset_pre", read_areas(root, best, "onset_pre")["val"]))
    if len(splits["skill"]) < n_random:
        raise ValueError(f"{len(splits['skill'])} skill splits have weights, {n_random} are asked for")
    for name in sorted(random.Random(seed).sample(splits["skill"], n_random)):
        picks.append(Pick(name, "skill", f"drawn at random (seed {seed}) from the {len(splits['skill'])} skill splits", read_areas(root, name, "onset_pre")["val"]))
    return picks


def export_model(root: Path, pick: Pick, variant: str, out_dir: Path, inputs, owner: str = OWNER) -> dict:
    """Write model.safetensors, config.json and README.md of one model into ``out_dir``; returns the config."""
    run = run_path(root, pick.split, variant)
    saved = load_checkpoint(run)  # our own file; the pickle is why the Hub gets safetensors
    cfg = ProbeConfig(**saved["cfg"])
    if cfg.name != ARCHITECTURE or saved["variant"] != variant:
        raise ValueError(f"{run} holds {cfg.name} / {saved['variant']}, expected {ARCHITECTURE} / {variant}")
    maxima = calibration_maxima(run, inputs.splits[pick.split], inputs.episodes)
    tensors = write_tensors(saved, maxima, out_dir)
    history = json.loads((run / "history.json").read_text())
    areas = read_areas(root, pick.split, variant)
    config = {
        "name": f"vla-probe {ARCHITECTURE} ({variant}), LIBERO-X {pick.kind} split {pick.split}", "architecture": ARCHITECTURE, "architecture_config": saved["cfg"],
        "label_variant": variant, "label_description": LABEL_TEXT[variant], "dataset": DATASET, "split": pick.split, "split_kind": pick.kind, "selection": pick.how,
        "input": {"features": "pooled per-modality token means of the policy's backbone, one vector per policy step: [base image | wrist image | language], 2048 columns each (6144)",
                  "standardisation": "(x - feature_mean) / feature_std, with the two tensors of model.safetensors (statistics of the split's training episodes)",
                  "on_time_window_steps": history["window"]},
        "outputs": {"score": "failure probability per policy step: the sigmoid of the score head's logit", "mode": "failure-mode logits per step over mode_names",
                    "sem": f"semantic embedding per step ({cfg.clip_dim}-d, trained toward the CLIP text feature of the step's phrase)"},
        "mode_names": list(MODE_NAMES),
        "alarm": "alarm at the first step whose score >= threshold; for level alpha the threshold is the k-th smallest of calibration_maxima with k = ceil((n + 1)(1 - alpha)), n = len(calibration_maxima); +inf (never an alarm) if k > n",
        "training": {"seed": history["seed"], "best_epoch": history["best_epoch"], "epochs": history["epochs"], "stopped_early": history["stopped_early"], "split_sha256": history["split_sha256"]},
        "results_on_this_split": {"metric": "area under TPR early against FPR up to 40%, in %, alarms by max-conformal calibration on the calibration successes",
                                  "validation": round(areas["val"], 2), "seen": round(areas["seen"], 2), "unseen": round(areas["unseen"], 2)},
        "license": LICENSE, "code": history.get("code"),
        "tensors": {name: list(t.shape) for name, t in tensors.items()},
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "README.md").write_text(model_card(config, len(maxima), owner))
    return config


def verify_export(root: Path, pick: Pick, variant: str, out_dir: Path) -> None:
    """Read the staged tensors back: they equal the saved ones, and a probe built from them gives the original outputs. Raises on any difference."""
    verify_tensors(load_checkpoint(run_path(root, pick.split, variant)), out_dir, f"{pick.split}/{variant}")


def model_card(config: dict, n_calibration: int, owner: str = OWNER) -> str:
    """The card of one model, as the Hub shows it: front matter, then a short description."""
    r = config["results_on_this_split"]
    other = next(v for v in VARIANTS if v != config["label_variant"])
    sibling = f"https://huggingface.co/{owner}/{repo_name(config['split'], other)}"
    return f"""---
license: {LICENSE}
library_name: pytorch
tags:
  - robotics
  - failure-detection
  - vision-language-action
  - libero
datasets:
  - {DATASET_REPO}
---

# vla-probe {config["architecture"]} - {config["label_variant"]} - LIBERO-X {config["split_kind"]} split `{config["split"]}`

A small failure detector for a VLA policy (GR00T on LIBERO-X): an LSTM probe on the policy's pooled backbone features, with a per-step failure
score, a failure-mode head (10 modes) and a semantic head. Trained on the training episodes of one split of the
[LIBERO-X features](https://huggingface.co/datasets/{DATASET_REPO}); the episodes of the held-out {config["split_kind"]} of that split are the "unseen" set.

- **Labels:** {config["label_description"]}.
- **Split:** `{config["split"]}` ({config["selection"]}; a split is chosen by its validation results, never by the unseen ones). The numbers in the paper are means over all splits of the kind.
- **Result on this split** (area under TPR early against FPR up to 40%, in %, max-conformal alarms): validation {r["validation"]}, seen {r["seen"]}, unseen {r["unseen"]}.
- **The other label on the same split:** [{repo_name(config["split"], other)}]({sibling}).
- **Files:** `model.safetensors` (weights, `feature_mean`, `feature_std`, and `calibration_maxima`: the {n_calibration} calibration successes' maximum scores, sorted) and `config.json`
  (architecture, input, outputs, how an alarm is read, training).

```python
from safetensors.torch import load_file
t = load_file("model.safetensors")
mean, std, maxima = t["feature_mean"], t["feature_std"], t["calibration_maxima"]  # standardise the 6144-d step features with mean and std
weights = {{k: v for k, v in t.items() if k not in ("feature_mean", "feature_std", "calibration_maxima")}}
# alarm at level alpha: threshold = sorted maxima[k - 1], k = ceil((n + 1) * (1 - alpha)); alarm at the first step with score >= threshold
```

The architecture is the `{config["architecture"]}` probe of the vla-probe code (https://github.com/mpodolinsky/vla-probe); `config.json` has its exact configuration.
Licence: CC BY 4.0.
"""


def stage(root: Path, picks: list[Pick], inputs, folder: Path, owner: str = OWNER) -> dict[str, dict]:
    """Stage every model in ``folder/<repo name>/`` and verify it; returns the config of each repo."""
    configs = {}
    for pick in picks:
        for variant in VARIANTS:
            out = folder / repo_name(pick.split, variant)
            configs[out.name] = export_model(root, pick, variant, out, inputs, owner)
            verify_export(root, pick, variant, out)
    return configs


def describe(folder: Path) -> list[str]:
    return [f"  {p.relative_to(folder)}  {p.stat().st_size / 1e6:8.2f} MB" for p in sorted(folder.rglob("*")) if p.is_file()]


def push(folder: Path, repo: str, api=None) -> list[str]:
    """Create the public model repo if needed, upload ``folder`` to it in one commit, and return the files the Hub lists afterwards."""
    return push_folder(folder, repo, "safetensors weights, config and card of a LIBERO-X probe of the main table", private=False, api=api or hub_api())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=RESULTS)
    p.add_argument("--owner", default=OWNER)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--stage", type=Path, help="stage the files in this (empty) folder instead of a temporary one, to inspect them")
    p.add_argument("--push", action="store_true")
    args = p.parse_args()

    from fp.data.inputs import load_inputs

    inputs = load_inputs(DATASET)
    picks = select_models(args.root, args.seed)
    print(f"{len(picks)} splits x {len(VARIANTS)} labels = {len(picks) * len(VARIANTS)} models of {ARCHITECTURE}, one public repo each under {args.owner}/")
    for pick in picks:
        print(f"  {pick.split:26s} {pick.kind:10s} validation {pick.val_area:5.1f}  ({pick.how})")
    with tempfile.TemporaryDirectory() as tmp:
        folder = args.stage if args.stage else Path(tmp)
        if args.stage and any(folder.glob("*")):
            sys.exit(f"{folder} is not empty")
        folder.mkdir(parents=True, exist_ok=True)
        configs = stage(args.root, picks, inputs, folder, args.owner)
        print("\nstaged and verified (tensors read back, probe outputs identical):")
        print("\n".join(describe(folder)))
        total = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
        print(f"total {total / 1e6:.0f} MB in {len(configs)} repos")
        if not args.push:
            first = sorted(configs)[0]
            print(f"\n--- {first}/README.md ---\n" + (folder / first / "README.md").read_text())
            print("dry run: nothing was uploaded (--push creates the public repos above and uploads the staged files)")
            return
        missing = {}
        for name in sorted(configs):
            listed = push(folder / name, f"{args.owner}/{name}")
            absent = [f for f in sorted(q.name for q in (folder / name).iterdir()) if f not in listed]
            print(f"pushed https://huggingface.co/{args.owner}/{name}: {len(listed)} files listed; missing: {absent or 'none'}")
            if absent:
                missing[name] = absent
        if missing:
            sys.exit(f"some staged files are not on the Hub: {missing}")


if __name__ == "__main__":
    main()
