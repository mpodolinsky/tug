"""python -m cli.embed_tasks <dataset> --source shipped|qwen3 [--shipped-dir DIR] [--device cuda]

Write data/<dataset>/task_embeddings.npz: one 1024-d vector per task, the language conditioning of ActProbe.
    --source shipped   ActProbe's own vectors for the LIBERO-10 instructions; --shipped-dir is the folder holding its task_instructions.json and
                       task_embeddings.npy (benchmarks/pi0_libero/data in its repository)
    --source qwen3     computed with Qwen3-Embedding-0.6B (needs `transformers` and the model weights); for LIBERO-X
"""

import argparse

from fp.config import REGISTRY
from fp.data.task_embeddings import SOURCES, prepare_task_embeddings


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=sorted(REGISTRY))
    p.add_argument("--source", choices=SOURCES, required=True)
    p.add_argument("--shipped-dir")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    print(prepare_task_embeddings(a.dataset, a.source, a.shipped_dir, a.device))


if __name__ == "__main__":
    main()
