"""python -m cli.import_real <dataset> <source folder>

Build data/<dataset>/ (manifest, pooled features, targets) for the real-robot dataset from its collection folder, e.g.
`python -m cli.import_real real8_groot outputs/gr00t-real8`. Needs `transformers` and the CLIP weights to embed the timeline phrases.
"""

import argparse

from fp.config import REGISTRY
from fp.data.real_import import import_real


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=sorted(d for d, spec in REGISTRY.items() if spec.suite == "real8"))
    p.add_argument("source")
    a = p.parse_args()
    print(import_real(a.dataset, a.source))


if __name__ == "__main__":
    main()
