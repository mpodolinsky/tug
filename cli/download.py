"""python -m cli.download <dataset>

Fetch the pooled features, the manifest and (LIBERO-X) the split files of a dataset from the Hugging Face Hub into data/<dataset>/.
The raw rollouts are not downloaded.
"""

import argparse

from fp.config import REGISTRY
from fp.data.download import download_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=sorted(REGISTRY))
    print(download_dataset(p.parse_args().dataset))


if __name__ == "__main__":
    main()
