import csv
import json

import numpy as np
import pytest

from fp import config
from fp.data import task_embeddings as te
from fp.data.task_embeddings import last_token_pool, load_task_embeddings, normalise, prepare_task_embeddings, shipped_embeddings, task_prompts

PROMPTS = {"t0": "open the drawer", "t1": "close the drawer", "t2": "open the drawer"}  # t0 and t2 share an instruction (as in LIBERO-X)


def unit(seed, dim=te.DIM):
    v = np.random.default_rng(seed).normal(size=dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA", tmp_path / "data")
    folder = tmp_path / "data" / "libero10_groot"
    folder.mkdir(parents=True)
    with open(folder / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dir", "task", "prompt"])
        w.writeheader()
        for i, (task, prompt) in enumerate(PROMPTS.items()):
            for episode in range(2):
                w.writerow(dict(dir=f"{task}/ep{episode}", task=task, prompt=prompt))
    return folder


@pytest.fixture
def shipped(tmp_path):
    """ActProbe's layout: task_instructions.json {"task_<id>": text} and task_embeddings.npy, a pickled {id: vector}."""
    folder = tmp_path / "shipped"
    folder.mkdir()
    json.dump({"task_3": "open the drawer", "task_7": "close the drawer"}, open(folder / "task_instructions.json", "w"))
    np.save(folder / "task_embeddings.npy", {3: unit(3), 7: unit(7)}, allow_pickle=True)
    return folder


# ---------------------------------------------------------------- pure functions
@pytest.mark.parametrize("side", ["left", "right"])
def test_the_last_token_is_the_last_real_one_whatever_the_side_of_the_padding(side):
    hidden = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2)
    mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]]) if side == "right" else np.array([[1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]])
    want_index = [3, 1, 0] if side == "right" else [3, 3, 3]
    assert np.array_equal(last_token_pool(hidden, mask), hidden[np.arange(3), want_index])


def test_normalise_gives_unit_float32_rows():
    v = normalise(np.random.default_rng(0).normal(size=(5, 8)) * 7)
    assert v.dtype == np.float32 and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-6)


# ---------------------------------------------------------------- prompts and the shipped source
def test_every_task_maps_to_its_instruction_and_two_instructions_for_one_task_are_an_error(dataset):
    assert task_prompts("libero10_groot") == PROMPTS
    with open(dataset / "manifest.csv", "a", newline="") as f:
        csv.writer(f).writerow(["t1/ep9", "t1", "something else"])
    with pytest.raises(ValueError, match="more than one instruction"):
        task_prompts("libero10_groot")


def test_the_shipped_vectors_are_read_by_instruction_text(shipped):
    got = shipped_embeddings(shipped)
    assert set(got) == {"open the drawer", "close the drawer"} and np.array_equal(got["close the drawer"], unit(7))


def test_the_shipped_source_gives_each_task_the_vector_of_its_instruction_and_records_where_it_came_from(dataset, shipped):
    prepare_task_embeddings("libero10_groot", "shipped", shipped)
    vectors, source = load_task_embeddings("libero10_groot")
    assert source == "shipped" and set(vectors) == set(PROMPTS)
    assert np.array_equal(vectors["t0"], unit(3)) and np.array_equal(vectors["t2"], unit(3)) and np.array_equal(vectors["t1"], unit(7))
    assert all(v.dtype == np.float32 and v.shape == (te.DIM,) for v in vectors.values())


def test_an_instruction_without_a_shipped_vector_is_an_error_and_nothing_is_written(dataset, shipped):
    json.dump({"task_3": "open the drawer"}, open(shipped / "task_instructions.json", "w"))
    with pytest.raises(KeyError, match="1 instructions have no shipped vector"):
        prepare_task_embeddings("libero10_groot", "shipped", shipped)
    assert not (dataset / "task_embeddings.npz").exists() and not list(dataset.glob("*.tmp*"))


def test_the_shipped_source_needs_its_folder(dataset):
    with pytest.raises(ValueError, match="needs the folder"):
        prepare_task_embeddings("libero10_groot", "shipped")


# ---------------------------------------------------------------- the qwen3 source
def test_the_qwen3_source_embeds_each_distinct_instruction_once(dataset, monkeypatch):
    seen = []

    def fake(texts, batch=16, device="cpu"):
        seen.append(list(texts))
        return np.stack([unit(len(t)) for t in texts])

    monkeypatch.setattr(te, "qwen3_embeddings", fake)
    prepare_task_embeddings("libero10_groot", "qwen3")
    vectors, source = load_task_embeddings("libero10_groot")
    assert source == "qwen3" and seen == [["close the drawer", "open the drawer"]]  # two distinct instructions, sorted
    assert np.array_equal(vectors["t0"], vectors["t2"]) and not np.array_equal(vectors["t0"], vectors["t1"])


def test_vectors_that_are_not_unit_vectors_of_the_right_width_are_refused(dataset, monkeypatch):
    monkeypatch.setattr(te, "qwen3_embeddings", lambda texts, batch=16, device="cpu": np.stack([2 * unit(i) for i, _ in enumerate(texts)]))
    with pytest.raises(ValueError, match="not finite unit vectors"):
        prepare_task_embeddings("libero10_groot", "qwen3")
    monkeypatch.setattr(te, "qwen3_embeddings", lambda texts, batch=16, device="cpu": np.stack([unit(i, 8) for i, _ in enumerate(texts)]))
    with pytest.raises(ValueError, match="width 1024"):
        prepare_task_embeddings("libero10_groot", "qwen3")
    assert not (dataset / "task_embeddings.npz").exists() and not list(dataset.glob("*.tmp*"))


# ---------------------------------------------------------------- files
def test_it_never_overwrites_and_load_says_how_to_make_a_missing_file(dataset, shipped):
    with pytest.raises(FileNotFoundError, match="cli.embed_tasks"):
        load_task_embeddings("libero10_groot")
    prepare_task_embeddings("libero10_groot", "shipped", shipped)
    with pytest.raises(FileExistsError):
        prepare_task_embeddings("libero10_groot", "shipped", shipped)
    with pytest.raises(ValueError, match="source must be"):
        prepare_task_embeddings("libero10_pi05", "bert")


def test_a_file_with_an_unknown_source_is_refused(dataset):
    np.savez(dataset / "task_embeddings.npz", t0=unit(0), _source=np.array("bert"))
    with pytest.raises(ValueError, match="expected one of"):
        load_task_embeddings("libero10_groot")
