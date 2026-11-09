import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fp.calibration.max_conformal import MaxConformal
from fp.data.manifest import Episode
from fp.data.splits import Split
from fp.data.targets import MODE_NAMES
from fp.models.ours import Probe, ProbeConfig

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "push_real_model_to_hub.py"
spec = importlib.util.spec_from_file_location("push_real_model_to_hub", SCRIPT)
real = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real)

TINY = dict(trunk="lstm", mha=False, modes=True, clip=True, hidden=8, d_modality=4, proj=8, n_attention_heads=2, n_modes=10, clip_dim=3)
CALIBRATION = [f"S{i}" for i in range(10)]  # ten calibration successes: alpha below 1/11 has no finite threshold
ROWS = [{"alpha": round(a, 3), "fpr": 100 * a, "tpr_early": 50 + 100 * a, "tpr_on_time": 5.0, "tpr": 90.0, "ldr": 0.0} for a in np.arange(0.005, 0.4001, 0.005)]


def episodes():
    eps = {k: Episode(k, True, "basket", 6, None, "basket") for k in CALIBRATION}
    eps.update({f"F{i}": Episode(f"F{i}", False, "tomato", 6, 3, "tomato") for i in range(4)})
    return eps


def inputs_for():
    split = Split("any", ["F0"], ["F1"], CALIBRATION, ["F2", "F3"], [], seed=0)
    return SimpleNamespace(episodes=episodes(), splits={n: split for n in real.SPLITS}, targets=SimpleNamespace(modes={"F0": 5, "F1": 5, "F2": 1, "F3": 0}))


PROMPTS = {"basket": "Flip the basket upright", "tomato": "Put the tomato in the bin"}


def write_run(root, split, val, seen=60.0, weights=True, evaluated=True, seed=0):
    run = real.run_path(root, split)
    run.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    if weights:
        torch.manual_seed(seed)
        torch.save({"cfg": TINY, "variant": real.VARIANT, "state": Probe(ProbeConfig(**TINY)).state_dict(), "mean": rng.normal(size=12), "std": np.abs(rng.normal(size=12)) + 0.5,
                    "best_epoch": 3}, run / "weights.pt")
    # the maxima of the ten calibration successes are 0.10, 0.20, ..., 1.00; the failures' scores do not matter here
    np.savez(run / "scores.npz", **{**{k: np.array([0.0, 0.1 * (i + 1)]) for i, k in enumerate(CALIBRATION)}, **{f"F{i}": np.array([0.5, 0.6]) for i in range(4)}})
    (run / "history.json").write_text(json.dumps({"seed": 1, "best_epoch": 3, "epochs": 12, "stopped_early": True, "window": 20, "split_sha256": "abc", "code": {"git_head": "x"}}))
    if evaluated:
        sets = {"val": {"area": val, "n_fail": 4, "n_succ": 2}, "seen": {"area": seen, "n_fail": 2, "n_succ": 2}}
        (run / "summary.json").write_text(json.dumps({"calibrators": {"max_conformal": {"sets": sets}}}))
        (run / "curve_max_conformal.json").write_text(json.dumps({"calibrator": "max_conformal", "sets": {"val": ROWS, "seen": ROWS}}))


@pytest.fixture
def results(tmp_path):
    root = tmp_path / "results"
    for i, (split, val) in enumerate((("full_seed0", 43.0), ("full_seed1", 49.0), ("full_seed2", 48.0))):
        write_run(root, split, val, seen=50.0 + i, seed=i)
    return root


# ---------------------------------------------------------------- the choice and the numbers
def test_the_split_is_chosen_on_the_validation_area_never_on_the_seen_one(results):
    assert real.select_split(results) == "full_seed1"  # val 49 beats 43 and 48, although seed 2 has the best seen area
    assert real.repo_name("full_seed1") == "groot-real8-probe-lstm-onset-pre-mode-clip-full-seed1"


def test_a_split_without_weights_or_without_an_evaluation_is_not_a_candidate(tmp_path):
    root = tmp_path / "results"
    write_run(root, "full_seed0", 90.0, weights=False)
    write_run(root, "full_seed1", 80.0, evaluated=False)
    write_run(root, "full_seed2", 10.0)
    assert real.select_split(root) == "full_seed2"
    with pytest.raises(ValueError, match="has weights and an evaluation"):
        real.select_split(tmp_path / "nothing")


def test_the_thresholds_are_the_max_conformal_ones_and_none_where_no_finite_threshold_exists():
    maxima = np.array([0.1 * (i + 1) for i in range(10)])
    cut = real.thresholds(maxima, (0.05, 0.1, 0.2, 0.4))
    calibrator = MaxConformal().fit([np.array([m]) for m in maxima])
    assert cut[0.05] is None  # k = ceil(11 * 0.95) = 11 > 10
    assert cut[0.1] == pytest.approx(1.0) == calibrator.threshold(0.1, 1)[0]  # k = 10, the largest maximum
    assert cut[0.2] == pytest.approx(0.9) and cut[0.4] == pytest.approx(0.7)  # k = 9 and 7
    assert all(v is None or v in maxima for v in cut.values())


def test_the_operating_points_are_read_at_the_requested_alphas_and_a_missing_alpha_is_an_error():
    points = real.operating_points(ROWS, (0.1, 0.2))
    assert points[0.1] == {"fpr": pytest.approx(10.0), "tpr_early": pytest.approx(60.0), "tpr_on_time": 5.0}
    with pytest.raises(ValueError, match="no row for alpha"):
        real.operating_points(ROWS, (0.1, 0.123))


def test_the_seed_and_unseen_task_summaries(results, tmp_path):
    summary = real.seed_summary(results)
    assert summary["n_seeds"] == 3 and summary["val"][0] == pytest.approx(np.mean([43.0, 49.0, 48.0])) and summary["seen"][0] == pytest.approx(51.0)
    assert real.unseen_task_summary(results) is None  # no leave-one-task-out runs in this folder
    loto = results / real.DATASET / "loto_basket" / f"{real.ARCHITECTURE}__{real.VARIANT}"
    loto.mkdir(parents=True)
    (loto / "summary.json").write_text(json.dumps({"calibrators": {"max_conformal": {"sets": {"seen": {"area": 46.0}, "unseen": {"area": 20.0}}}}}))
    assert real.unseen_task_summary(results) == {"n_runs": 1, "seen": 46.0, "unseen": 20.0}


def test_the_data_summary_counts_tasks_modes_and_the_parts_of_the_split():
    data = real.data_summary(inputs_for(), PROMPTS, "full_seed1")
    assert data["tasks"]["basket"] == {"instruction": "Flip the basket upright", "episodes": 10, "successes": 10}
    assert data["tasks"]["tomato"]["successes"] == 0 and data["failure_modes"] == {MODE_NAMES[5]: 2, MODE_NAMES[1]: 1, MODE_NAMES[0]: 1}
    assert data["split"]["calibration"] == {"episodes": 10, "successes": 10} and data["split"]["seen"] == {"episodes": 2, "successes": 0}


# ---------------------------------------------------------------- the staged model
@pytest.fixture
def staged(results, tmp_path):
    pytest.importorskip("safetensors")
    folder = tmp_path / "stage"
    folder.mkdir()
    split, config = real.stage(results, inputs_for(), PROMPTS, folder)
    return folder, split, config


def test_the_folder_holds_tensors_config_and_card_and_the_tensors_include_the_calibration_maxima(staged):
    from safetensors.torch import load_file

    folder, split, config = staged
    assert split == "full_seed1" and sorted(p.name for p in folder.iterdir()) == ["README.md", "config.json", "model.safetensors"]
    t = load_file(str(folder / "model.safetensors"))
    assert np.allclose(t["calibration_maxima"].numpy(), [0.1 * (i + 1) for i in range(10)]) and t["feature_mean"].dtype == torch.float64
    assert json.loads((folder / "config.json").read_text()) == json.loads(json.dumps(config))  # (JSON turns the tuples into lists)


def test_the_config_has_the_thresholds_the_rates_at_them_and_the_alarm_rule(staged):
    _, _, config = staged
    assert config["thresholds"]["0.05"] is None and config["thresholds"]["0.1"] == pytest.approx(1.0) and config["thresholds"]["0.4"] == pytest.approx(0.7)
    assert config["seen_set_at_threshold"]["0.1"]["fpr"] == pytest.approx(10.0)
    assert "score >= threshold" in config["alarm"] and "ceil((n + 1)(1 - alpha))" in config["alarm"] and config["input"]["on_time_window_steps"] == 20
    assert config["areas"]["validation"] == 49.0 and config["seeds"]["n_seeds"] == 3 and config["seen_set"] == {"episodes": 4, "failures": 2, "successes": 2}


def test_the_card_is_concise_gives_the_threshold_table_and_the_statistics_and_leaves_nothing_unformatted(staged):
    folder, _, config = staged
    card = (folder / "README.md").read_text()
    assert card.startswith("---\nlibrary_name: pytorch\n") and "\n# GR00T real-robot failure probe (pre-onset, all 8 tasks)\n" in card and len(card.splitlines()) < 80
    assert "| 0.05 | none (never alarms) | - | - | - |" in card and "| 0.1 | 1.0000 | 10.0 | 60.0 | 5.0 |" in card and "| 0.4 | 0.7000 |" in card
    assert "alpha = 0.1" in card and "score >= 1.0000" in card  # the example uses a threshold of the table
    assert "validation 49.0, held-out seen episodes 51.0" in card and "Over the 3 seeds" in card and "Flip the basket upright" in card and "no licence is granted yet" in card
    assert card.count("{") == card.count("}") == 1  # the only braces are the dict comprehension of the usage snippet
    assert "{config" not in card and "{seeds" not in card and "{n_cal" not in card and "{'" not in card


def test_without_leave_one_task_out_runs_the_card_leaves_that_caveat_out(staged):
    assert "much weaker" not in (staged[0] / "README.md").read_text()


# ---------------------------------------------------------------- the upload
class FakeApi:
    def __init__(self):
        self.calls, self.files = [], []

    def create_repo(self, repo, **kwargs):
        self.calls.append(("create_repo", repo, kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))
        self.files = sorted(p.name for p in Path(kwargs["folder_path"]).iterdir()) + [".gitattributes"]

    def list_repo_files(self, repo, repo_type):
        return self.files


def run_main(monkeypatch, results, tmp_path, *extra):
    pytest.importorskip("safetensors")
    monkeypatch.setattr("fp.data.inputs.load_inputs", lambda dataset: inputs_for())
    monkeypatch.setattr(real, "read_prompts", lambda path: PROMPTS)
    monkeypatch.setattr(sys, "argv", ["push_real_model_to_hub", "--root", str(results), "--stage", str(tmp_path / "stage"), *extra])
    real.main()


def test_a_dry_run_stages_and_verifies_but_never_creates_a_hub_client(results, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(real, "hub_api", lambda: pytest.fail("the dry run must not create a Hub client"))
    monkeypatch.setattr(real, "push_folder", lambda *a, **k: pytest.fail("the dry run must not upload"))
    run_main(monkeypatch, results, tmp_path)
    out = capsys.readouterr().out
    assert "full_seed1" in out and "private" in out and "staged and verified" in out and "dry run: nothing was uploaded" in out and (tmp_path / "stage" / "model.safetensors").exists()


def test_push_creates_a_private_repo_by_default_and_a_public_one_on_request(results, tmp_path, monkeypatch, capsys):
    api = FakeApi()
    monkeypatch.setattr(real, "hub_api", lambda: api)
    run_main(monkeypatch, results, tmp_path, "--push")
    assert api.calls[0] == ("create_repo", "podolinsky/groot-real8-probe-lstm-onset-pre-mode-clip-full-seed1", {"repo_type": "model", "private": True, "exist_ok": True})
    assert len([c for c in api.calls if c[0] == "upload_folder"]) == 1 and "missing: none" in capsys.readouterr().out
    api2 = FakeApi()
    monkeypatch.setattr(real, "hub_api", lambda: api2)
    (tmp_path / "stage2").mkdir()
    monkeypatch.setattr(sys, "argv", ["push_real_model_to_hub", "--root", str(results), "--stage", str(tmp_path / "stage2"), "--push", "--public"])
    real.main()
    assert api2.calls[0][2]["private"] is False


def test_a_file_missing_on_the_hub_after_the_upload_is_an_error(results, tmp_path, monkeypatch):
    api = FakeApi()
    api.list_repo_files = lambda repo, repo_type: ["README.md"]
    monkeypatch.setattr(real, "hub_api", lambda: api)
    with pytest.raises(SystemExit, match="not on the Hub"):
        run_main(monkeypatch, results, tmp_path, "--push")


def test_weights_of_another_architecture_or_label_are_refused(results, tmp_path):
    pytest.importorskip("safetensors")
    run = real.run_path(results, "full_seed1")
    saved = torch.load(run / "weights.pt", weights_only=False)
    saved["variant"] = "onset"
    torch.save(saved, run / "weights.pt")
    folder = tmp_path / "stage"
    folder.mkdir()
    with pytest.raises(ValueError, match="expected lstm_modes_clip / onset_pre"):
        real.stage(results, inputs_for(), PROMPTS, folder)
