"""python -m cli.prepare_actions <dataset> <rollout_root> [--workers N]

Extract the sampled action chunks of every episode from the raw rollouts under <rollout_root> (one <manifest dir>/rollout.npz per episode)
into data/<dataset>/actions.npz. Needed by STAC, ActProbe and FailureSpot; with cli.prepare, the only command that reads raw rollouts.
"""

import argparse

from fp.data.actions import prepare_actions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("rollout_root")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    print(prepare_actions(a.dataset, a.rollout_root, a.workers))


if __name__ == "__main__":
    main()
