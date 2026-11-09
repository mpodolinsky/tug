"""python -m cli.targets <dataset> <labels_root>

Build data/<dataset>/targets.npz (failure modes, timeline phrases and their CLIP text embeddings) from the labels.json / labels.npz of each
episode under <labels_root>/<manifest dir>/. Needs `transformers` and the CLIP weights to embed the phrases.
"""

import argparse

from fp.config import REGISTRY
from fp.data.targets import build_targets


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=sorted(REGISTRY))
    p.add_argument("labels_root")
    a = p.parse_args()
    print(build_targets(a.dataset, a.labels_root))


if __name__ == "__main__":
    main()
