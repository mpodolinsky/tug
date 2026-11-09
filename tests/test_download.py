from fnmatch import fnmatch
from pathlib import Path

import pytest

from fp import config
from fp.data.download import download_dataset, hub_patterns


class FakeHub:
    """Stands in for huggingface_hub.snapshot_download: writes the files of a fake repo that match allow_patterns."""

    def __init__(self, files):
        self.files, self.calls = files, []

    def __call__(self, repo_id, repo_type, allow_patterns, local_dir):
        self.calls.append((repo_id, repo_type, tuple(allow_patterns)))
        for name, content in self.files.items():
            if any(fnmatch(name, pattern) for pattern in allow_patterns):
                path = Path(local_dir) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        (Path(local_dir) / ".cache").mkdir(exist_ok=True)  # huggingface_hub leaves a bookkeeping folder behind


REPO_FILES = {
    "pooled/features.npz": b"features", "pooled/targets.npz": b"targets", "pooled/meta.json": b"{}", "pooled/README.md": b"card",
    "manifest.csv": b"manifest",
    "rollout.json": b"raw metadata", "normal/t0/ep000/rollout.npz": b"raw rollout",
    "splits/leave_task_out/test.txt": b"a.bddl\n", "splits/leave_skill_out/pick/test.txt": b"b.bddl\n",
}


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA", tmp_path)
    return tmp_path


def test_libero10_gets_features_targets_and_manifest_only(data):
    hub = FakeHub(REPO_FILES)
    folder = download_dataset("libero10_groot", fetch=hub)
    assert hub.calls == [("podolinsky/groot-libero-10-features-v3", "dataset", ("pooled/features.npz", "pooled/targets.npz", "manifest.csv"))]
    assert sorted(p.name for p in folder.iterdir()) == [".cache", "features.npz", "manifest.csv", "targets.npz"]
    assert (folder / "features.npz").read_bytes() == b"features" and (folder / "targets.npz").read_bytes() == b"targets"


def test_liberox_also_gets_the_split_files(data):
    folder = download_dataset("liberox_groot", fetch=FakeHub(REPO_FILES))
    assert (folder / "splits/leave_task_out/test.txt").read_bytes() == b"a.bddl\n"
    assert (folder / "splits/leave_skill_out/pick/test.txt").exists()


def test_raw_rollouts_and_extra_pooled_files_are_not_downloaded(data):
    folder = download_dataset("libero10_pi05", fetch=FakeHub(REPO_FILES))
    names = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()}
    assert names == {"features.npz", "targets.npz", "manifest.csv"}


def test_the_pooled_folder_is_gone_after_the_move(data):
    folder = download_dataset("libero10_groot", fetch=FakeHub(REPO_FILES))
    assert not (folder / "pooled").exists()


@pytest.mark.parametrize("existing", ["features.npz", "targets.npz"])
def test_an_existing_pooled_file_is_never_overwritten(data, existing):
    folder = data / "libero10_groot"
    folder.mkdir()
    (folder / existing).write_bytes(b"mine")
    hub = FakeHub(REPO_FILES)
    with pytest.raises(FileExistsError):
        download_dataset("libero10_groot", fetch=hub)
    assert (folder / existing).read_bytes() == b"mine" and hub.calls == []  # nothing was fetched either


@pytest.mark.parametrize("missing", ["features.npz", "targets.npz"])
def test_a_repo_without_a_pooled_file_is_an_error(data, missing):
    without = {k: v for k, v in REPO_FILES.items() if k != f"pooled/{missing}"}
    with pytest.raises(FileNotFoundError, match=f"has no pooled/{missing}"):
        download_dataset("libero10_groot", fetch=FakeHub(without))


def test_patterns_ask_for_splits_only_where_they_exist():
    assert hub_patterns(config.get("liberox_groot")) == ["pooled/features.npz", "pooled/targets.npz", "manifest.csv", "splits/*"]
    assert hub_patterns(config.get("libero10_groot")) == ["pooled/features.npz", "pooled/targets.npz", "manifest.csv"]
