import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fp.data.manifest import Episode
from fp.data.splits import Split
from fp.models.ours import Probe, ProbeConfig

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "push_models_to_hub.py"
spec = importlib.util.spec_from_file_location("push_models_to_hub", SCRIPT)
hub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hub)

TINY = dict(trunk="lstm", mha=False, modes=True, clip=True, hidden=8, d_modality=4, proj=8, n_attention_heads=2, n_modes=10, clip_dim=3)
SKILLS = [f"skill_{c}" for c in "abcdefgh"]


def episodes():
    return {f"S{i}": Episode(f"S{i}", True, "t", 6, None, "a") for i in range(6)} | {f"F{i}": Episode(f"F{i}", False, "t", 6, 3, "a") for i in range(2)}


def inputs_for(names):
    split = Split("any", [], [], ["S0", "S1", "S2", "S3"], [], [], seed=0)  # the calibration successes are S0..S3
    return SimpleNamespace(episodes=episodes(), splits={n: split for n in names})


def write_run(root, split, variant, val, seen=60.0, unseen=40.0, cfg=TINY, seed=0, weights=True):
    run = Path(root) / "liberox_groot" / split / f"{hub.ARCHITECTURE}__{variant}"
    run.mkdir(parents=True)
    if weights:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        torch.save({"cfg": cfg, "variant": variant, "state": Probe(ProbeConfig(**cfg)).state_dict(), "mean": rng.normal(size=12), "std": np.abs(rng.normal(size=12)) + 0.5,
                    "best_epoch": 3}, run / "weights.pt")
    np.savez(run / "scores.npz", **{k: np.random.default_rng(i + seed).random(6) for i, k in enumerate(episodes())})
    (run / "history.json").write_text(json.dumps({"seed": 1, "best_epoch": 3, "epochs": 12, "stopped_early": True, "window": 6, "split_sha256": "abc", "code": {"git_head": "x"}}))
    (run / "summary.json").write_text(json.dumps({"calibrators": {"max_conformal": {"sets": {"val": {"area": float(val)}, "seen": {"area": seen}, "unseen": {"area": unseen}}}}}))


@pytest.fixture
def results(tmp_path):
    root = tmp_path / "results"
    # onset_pre validation areas decide; the onset_traj ones are deliberately different (and ignored)
    for split, val_pre, val_traj in [("task_seed0", 50, 90), ("task_seed1", 70, 10), ("task_seed2", 60, 95),
                                     ("signature_seed0", 80, 10), ("signature_seed1", 80, 99), ("signature_seed2", 70, 99)] + [(s, 55 + i, 50) for i, s in enumerate(SKILLS)]:
        for variant, val in (("onset_pre", val_pre), ("onset_traj", val_traj)):
            write_run(root, split, variant, val, seed=hash(split) % 7)
    return root


# ---------------------------------------------------------------- the choice of models
def test_the_architecture_is_the_one_of_the_main_table():
    figures = pytest.importorskip("fp.eval.figures")  # needs scikit-learn, which the Hub interpreter lacks
    assert hub.ARCHITECTURE == figures.OURS_ARCHITECTURE and hub.VARIANTS == ("onset_pre", "onset_traj")


def test_the_best_task_and_signature_split_are_chosen_on_the_validation_area_of_onset_pre_with_ties_going_to_the_earlier_split(results):
    picks = hub.select_models(results)
    by_kind = {p.kind: p for p in picks}
    assert by_kind["task"].split == "task_seed1" and by_kind["task"].val_area == 70  # 70 against 50 and 60, although onset_traj says seed 2
    assert by_kind["signature"].split == "signature_seed0"  # a tie of 80 with seed 1: the earlier; onset_traj's 99 for seed 1 is ignored
    assert "best of the 3 task splits" in by_kind["task"].how


def test_two_skill_splits_are_drawn_at_random_with_a_fixed_seed_and_are_not_chosen_by_their_results(results):
    first = [p.split for p in hub.select_models(results) if p.kind == "skill"]
    assert first == sorted(random.Random(0).sample(SKILLS, 3)) and len(set(first)) == 3 and first == [p.split for p in hub.select_models(results) if p.kind == "skill"]
    others = {tuple(p.split for p in hub.select_models(results, seed=s) if p.kind == "skill") for s in range(1, 8)}
    assert tuple(first) not in others or len(others) > 1  # another seed gives another draw
    assert [p.kind for p in hub.select_models(results)] == ["task", "signature", "skill", "skill", "skill"]


def test_a_split_without_the_weights_of_both_labels_is_not_available(results):
    (hub.run_path(results, "task_seed1", "onset_traj") / "weights.pt").unlink()
    assert hub.available_splits(results)["task"] == ["task_seed0", "task_seed2"]
    assert {p.kind: p.split for p in hub.select_models(results)}["task"] == "task_seed2"  # the best split that has both


def test_missing_splits_are_an_error_not_a_smaller_upload(results, tmp_path):
    with pytest.raises(ValueError, match="8 skill splits have weights, 9 are asked for"):
        hub.select_models(results, n_random=9)
    empty = tmp_path / "empty"
    write_run(empty, "task_seed0", "onset_pre", 1.0)  # no onset_traj weights, no signature or skill splits
    with pytest.raises(ValueError, match="no task split has the weights"):
        hub.select_models(empty)


# ---------------------------------------------------------------- the export
@pytest.fixture
def exported(results, tmp_path):
    pytest.importorskip("safetensors")
    names = ["task_seed1"]
    pick = hub.Pick("task_seed1", "task", "best of the 3 task splits on the validation area of onset_pre", 70.0)
    out = tmp_path / "out"
    config = hub.export_model(results, pick, "onset_pre", out, inputs_for(names))
    return results, pick, out, config


def test_the_weights_are_written_as_safetensors_with_the_statistics_and_the_calibration_maxima_and_read_back_equal(exported):
    from safetensors.torch import load_file

    results, pick, out, config = exported
    t = load_file(str(out / "model.safetensors"))
    saved = torch.load(hub.run_path(results, "task_seed1", "onset_pre") / "weights.pt", weights_only=False)
    assert all(torch.equal(t[k], v) for k, v in saved["state"].items())
    assert t["feature_mean"].dtype == torch.float64 and np.array_equal(t["feature_mean"].numpy(), saved["mean"]) and np.array_equal(t["feature_std"].numpy(), saved["std"])
    scores = np.load(hub.run_path(results, "task_seed1", "onset_pre") / "scores.npz")
    assert np.array_equal(t["calibration_maxima"].numpy(), np.sort([scores[k].max() for k in ("S0", "S1", "S2", "S3")]))  # the calibration successes only
    raw = (out / "model.safetensors").read_bytes()
    header = json.loads(raw[8 : 8 + int.from_bytes(raw[:8], "little")])  # a safetensors file is a JSON header and raw tensors: no pickle
    assert header["feature_mean"]["dtype"] == "F64" and header["__metadata__"] == {"format": "pt"}
    hub.verify_export(results, pick, "onset_pre", out)  # does not raise


def test_the_config_says_what_the_model_takes_gives_and_how_an_alarm_is_read(exported):
    _, _, out, config = exported
    assert json.loads((out / "config.json").read_text()) == config
    assert config["architecture"] == "lstm_modes_clip" and config["label_variant"] == "onset_pre" and config["split_kind"] == "task" and config["license"] == "cc-by-4.0"
    assert config["architecture_config"]["hidden"] == 8 and len(config["mode_names"]) == 10 and config["input"]["on_time_window_steps"] == 6
    assert config["results_on_this_split"] == {"metric": config["results_on_this_split"]["metric"], "validation": 70.0, "seen": 60.0, "unseen": 40.0}
    assert "ceil((n + 1)(1 - alpha))" in config["alarm"] and "score >= threshold" in config["alarm"] and config["training"]["best_epoch"] == 3
    assert config["tensors"]["calibration_maxima"] == [4] and config["tensors"]["feature_mean"] == [12] and "trunk.weight_ih_l0" in config["tensors"]


def test_a_staged_file_that_differs_from_the_saved_weights_is_caught_before_upload(exported):
    from safetensors.torch import load_file, save_file

    results, pick, out, _ = exported
    t = load_file(str(out / "model.safetensors"))
    t["score.2.bias"] = t["score.2.bias"] + 1.0  # a corrupted tensor
    save_file(t, str(out / "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="differ from the saved ones"):
        hub.verify_export(results, pick, "onset_pre", out)


def test_weights_of_another_architecture_or_label_are_refused(results, tmp_path):
    pytest.importorskip("safetensors")
    other = tmp_path / "other"
    write_run(other, "task_seed1", "onset_pre", 1.0, cfg={**TINY, "trunk": "mlp"})
    pick = hub.Pick("task_seed1", "task", "x", 1.0)
    with pytest.raises(ValueError, match="expected lstm_modes_clip"):
        hub.export_model(other, pick, "onset_pre", tmp_path / "o1", inputs_for(["task_seed1"]))
    wrong_label = tmp_path / "wrong_label"
    write_run(wrong_label, "task_seed1", "onset_pre", 1.0)
    saved = torch.load(hub.run_path(wrong_label, "task_seed1", "onset_pre") / "weights.pt", weights_only=False)
    saved["variant"] = "onset"
    torch.save(saved, hub.run_path(wrong_label, "task_seed1", "onset_pre") / "weights.pt")
    with pytest.raises(ValueError, match="expected lstm_modes_clip / onset_pre"):
        hub.export_model(wrong_label, pick, "onset_pre", tmp_path / "o2", inputs_for(["task_seed1"]))


# ---------------------------------------------------------------- the cards
def test_the_model_card_is_short_names_the_split_and_the_labels_and_explains_the_alarm(exported):
    _, _, out, config = exported
    card = (out / "README.md").read_text()
    assert card.startswith("---\nlicense: cc-by-4.0\nlibrary_name: pytorch\n") and "  - podolinsky/gr00t-libero-x-features-v3" in card  # the Hub's front matter
    assert "\n# vla-probe lstm_modes_clip - onset_pre - LIBERO-X task split `task_seed1`\n" in card and len(card.splitlines()) < 45
    assert "huggingface.co/podolinsky/groot-libero-x-probe-lstm-onset-traj-mode-clip-task-seed1" in card  # points to the other label on the same split
    assert "validation 70.0, seen 60.0, unseen 40.0" in card and "the episodes of the held-out task of that split" in card and "calibration successes' maximum scores" in card
    assert "CC BY 4.0" in card and "weights = {k: v for k, v in t.items()" in card  # the snippet's own braces are the only ones
    assert "TODO" not in card and "{config" not in card and "{'" not in card and "{r[" not in card  # nothing left unformatted


def test_repo_names_follow_the_earlier_probe_repos_with_the_label_and_the_split():
    assert hub.repo_name("task_seed1", "onset_pre") == "groot-libero-x-probe-lstm-onset-pre-mode-clip-task-seed1"
    assert hub.repo_name("skill_toggle_appliance_on", "onset_traj") == "groot-libero-x-probe-lstm-onset-traj-mode-clip-skill-toggle-appliance-on"


def test_staging_makes_one_folder_per_model_with_tensors_json_and_a_card_only(results, tmp_path):
    pytest.importorskip("safetensors")
    picks = hub.select_models(results)
    folder = tmp_path / "stage"
    configs = hub.stage(results, picks, inputs_for([p.split for p in picks]), folder)
    assert sorted(configs) == sorted(hub.repo_name(p.split, v) for p in picks for v in hub.VARIANTS) and len(configs) == 10
    for name in configs:
        assert sorted(f.name for f in (folder / name).iterdir()) == ["README.md", "config.json", "model.safetensors"]  # tensors, JSON and markdown: no pickle
        assert (folder / name / "README.md").read_text().startswith("---\nlicense: cc-by-4.0\n")
    assert not (folder / "README.md").exists()  # nothing at the staging root: every repo is its own folder


# ---------------------------------------------------------------- the upload
class FakeApi:
    def __init__(self):
        self.calls = []
        self.files = []

    def create_repo(self, repo, **kwargs):
        self.calls.append(("create_repo", repo, kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))
        self.files = sorted(str(p.relative_to(kwargs["folder_path"])) for p in Path(kwargs["folder_path"]).rglob("*") if p.is_file()) + [".gitattributes"]

    def list_repo_files(self, repo, repo_type):
        self.calls.append(("list_repo_files", repo, repo_type))
        return self.files


def test_push_creates_a_public_model_repo_uploads_the_folder_once_and_lists_the_files_back(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.txt").write_text("x")
    api = FakeApi()
    listed = hub.push(tmp_path, "podolinsky/vla-probe", api=api)
    assert api.calls[0] == ("create_repo", "podolinsky/vla-probe", {"repo_type": "model", "private": False, "exist_ok": True})  # public, a model repo
    uploads = [c for c in api.calls if c[0] == "upload_folder"]
    assert len(uploads) == 1 and uploads[0][1]["folder_path"] == str(tmp_path) and uploads[0][1]["repo_type"] == "model" and uploads[0][1]["repo_id"] == "podolinsky/vla-probe"
    assert listed == sorted(["a/x.txt", ".gitattributes"])


def run_main(monkeypatch, results, tmp_path, *extra):
    pytest.importorskip("safetensors")
    names = hub.available_splits(results)
    inputs = inputs_for([n for kind in names.values() for n in kind])
    monkeypatch.setattr("fp.data.inputs.load_inputs", lambda dataset: inputs)
    monkeypatch.setattr(sys, "argv", ["push_models_to_hub", "--root", str(results), "--stage", str(tmp_path / "stage"), *extra])
    hub.main()


def test_a_dry_run_stages_and_verifies_but_never_touches_the_hub(results, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hub, "hub_api", lambda: pytest.fail("the dry run must not create a Hub client"))
    monkeypatch.setattr(hub, "push", lambda *a, **k: pytest.fail("the dry run must not upload"))
    run_main(monkeypatch, results, tmp_path)
    out = capsys.readouterr().out
    assert "10 models of lstm_modes_clip, one public repo each under podolinsky/" in out and "staged and verified" in out and "dry run: nothing was uploaded" in out
    assert len(list((tmp_path / "stage").rglob("model.safetensors"))) == 10 and "total" in out


def test_push_uploads_every_staged_folder_to_its_own_repo_and_fails_when_a_file_is_missing_on_the_hub(results, tmp_path, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(hub, "push", lambda folder, repo, api=None: seen.append((Path(folder), repo)) or sorted(f.name for f in Path(folder).iterdir()))
    run_main(monkeypatch, results, tmp_path, "--push")
    names = sorted(f.name for f in (tmp_path / "stage").iterdir())
    assert len(names) == 10 and seen == [(tmp_path / "stage" / n, f"podolinsky/{n}") for n in names] and capsys.readouterr().out.count("missing: none") == 10
    monkeypatch.setattr(hub, "push", lambda folder, repo, api=None: ["README.md"])  # the Hub lists almost nothing back
    (tmp_path / "stage2").mkdir()
    monkeypatch.setattr(sys, "argv", ["push_models_to_hub", "--root", str(results), "--stage", str(tmp_path / "stage2"), "--push"])
    with pytest.raises(SystemExit, match="not on the Hub"):
        hub.main()


def test_staging_into_a_folder_that_is_not_empty_is_refused(results, tmp_path, monkeypatch):
    (tmp_path / "stage").mkdir()
    (tmp_path / "stage" / "old.txt").write_text("old")
    with pytest.raises(SystemExit, match="is not empty"):
        run_main(monkeypatch, results, tmp_path)
