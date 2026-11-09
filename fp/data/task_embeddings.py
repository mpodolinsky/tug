"""Text embeddings of the task instructions (the language conditioning of ActProbe).

Output is ``data/<dataset>/task_embeddings.npz``: one float32 vector (1024,) per task, keyed by the manifest's task name, plus the entry
``_source`` saying where the vectors come from. Every task has exactly one instruction (the manifest's ``prompt``), so one vector per task is one
vector per instruction.

Two sources, because ActProbe ships vectors for the ten LIBERO-10 instructions that the recipe below does not reproduce (cosine 0.6 to 0.75;
the pairwise similarities between the tasks do agree, r = 0.92):

    ``shipped``  ActProbe's own vectors (benchmarks/pi0_libero/data of its repository), matched to the tasks by instruction text
    ``qwen3``    Qwen3-Embedding-0.6B, the model ActProbe names: the hidden state of the last token (left padding), L2-normalised, no
                 instruction prefix. Used where ActProbe has no vectors (LIBERO-X: 218 tasks, 152 distinct instructions).

Each dataset is trained on separately, so the two sources never meet in one model; the source is stored with the vectors and recorded in the
history of every ActProbe run.
"""

import csv
import json
from pathlib import Path

import numpy as np

from ..config import get

DIM = 1024
QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
SOURCES = ("shipped", "qwen3")
SOURCE_KEY = "_source"


def task_prompts(dataset: str) -> dict[str, str]:
    """{task name: instruction} from the manifest; a task with two different instructions is an error."""
    prompts: dict[str, str] = {}
    with open(get(dataset).data_dir / "manifest.csv", newline="") as f:
        for row in csv.DictReader(f):
            if prompts.setdefault(row["task"], row["prompt"]) != row["prompt"]:
                raise ValueError(f"task {row['task']} has more than one instruction: {prompts[row['task']]!r} and {row['prompt']!r}")
    return prompts


def last_token_pool(hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """(n, d) hidden state of each text's last real token; ``hidden`` is (n, tokens, d), padded on the left or on the right."""
    last = attention_mask.shape[1] - 1 - np.argmax(attention_mask[:, ::-1] > 0, axis=1)  # index of the last 1 in each row
    return hidden[np.arange(len(hidden)), last]


def normalise(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def qwen3_embeddings(texts: list[str], batch: int = 16, device: str = "cpu") -> np.ndarray:
    """(n, 1024) unit vectors from Qwen3-Embedding-0.6B (needs `transformers` and the model weights)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, padding_side="left")
    model = AutoModel.from_pretrained(QWEN_MODEL, torch_dtype=torch.float32).to(device).eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tokenizer(texts[i : i + batch], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state.float().cpu().numpy()
        out.append(last_token_pool(hidden, enc["attention_mask"].cpu().numpy()))
    return normalise(np.concatenate(out))


def shipped_embeddings(folder: str | Path) -> dict[str, np.ndarray]:
    """{instruction: unit vector} from ActProbe's ``task_instructions.json`` and ``task_embeddings.npy`` in ``folder``."""
    folder = Path(folder)
    instructions = json.load(open(folder / "task_instructions.json"))  # {"task_<id>": text}
    vectors = np.load(folder / "task_embeddings.npy", allow_pickle=True).item()  # {id: vector}; a pickled dict of arrays, from ActProbe itself
    return {text: np.asarray(vectors[int(name.split("_")[1])], dtype=np.float32) for name, text in instructions.items()}


def prepare_task_embeddings(dataset: str, source: str, shipped_dir: str | Path | None = None, device: str = "cpu") -> Path:
    """Write data/<dataset>/task_embeddings.npz from the given source. Never overwrites."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    out = get(dataset).data_dir / "task_embeddings.npz"
    if out.exists():
        raise FileExistsError(f"{out} already exists; remove it to prepare again")
    prompts = task_prompts(dataset)
    if source == "shipped":
        if shipped_dir is None:
            raise ValueError("the shipped source needs the folder with task_instructions.json and task_embeddings.npy")
        known = shipped_embeddings(shipped_dir)
        missing = sorted({p for p in prompts.values() if p not in known})
        if missing:
            raise KeyError(f"{len(missing)} instructions have no shipped vector, e.g. {missing[:2]}")
        vectors = {task: known[prompt] for task, prompt in prompts.items()}
    else:
        distinct = sorted(set(prompts.values()))
        by_text = dict(zip(distinct, qwen3_embeddings(distinct, device=device)))
        vectors = {task: by_text[prompt] for task, prompt in prompts.items()}
    bad = [t for t, v in vectors.items() if v.shape != (DIM,) or not np.isfinite(v).all() or abs(np.linalg.norm(v) - 1) > 1e-3]
    if bad:
        raise ValueError(f"{len(bad)} vectors are not finite unit vectors of width {DIM}, e.g. {bad[:2]}")
    tmp = out.with_name("task_embeddings.tmp.npz")  # (np.savez appends .npz to a name that lacks it)
    try:
        np.savez(tmp, **vectors, **{SOURCE_KEY: np.array(source)})
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def load_task_embeddings(dataset: str) -> tuple[dict[str, np.ndarray], str]:
    """({task: (1024,) float32}, source) from data/<dataset>/task_embeddings.npz."""
    path = get(dataset).data_dir / "task_embeddings.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; python -m cli.embed_tasks {dataset} --source shipped|qwen3")
    with np.load(path) as z:
        source = str(z[SOURCE_KEY])
        vectors = {k: z[k] for k in z.files if k != SOURCE_KEY}
    if source not in SOURCES:
        raise ValueError(f"{path} has source {source!r}, expected one of {SOURCES}")
    return vectors, source
