"""python -m cli.prepare <dataset> <rollout_root> [--workers N]

Pool the raw rollouts under <rollout_root> (one <manifest dir>/rollout.npz per episode) into data/<dataset>/features.npz.
The only command that reads raw rollouts.
"""

import argparse

from fp.data.pooling import prepare_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("rollout_root")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    print(prepare_dataset(a.dataset, a.rollout_root, a.workers))


if __name__ == "__main__":
    main()
