"""python -m cli.latency [--dataset liberox_groot] [--split task_seed0] [--chosen results/figures/chosen.csv] [--methods ...] [--steps 2000]
                       [--vla-sampling results/latency/vla_sampling.json] [--out results/latency/latency.json]

The online latency of the methods of the main table (fp/eval/latency.py): the time to turn one new policy step into a score, on the GPU and on the CPU with 4 and
with 1 thread, median and 99th percentile over --steps steps. The methods are those the main table chose on LIBERO-X (--chosen, written by python -m cli.figures;
or --methods), at the sizes of the dataset's --split. The weights are random: the time of a layer does not depend on its values.

--vla-sampling is the file of the extra time STAC needs to sample its K plans from the policy at every call (measured with the policy itself, see
docs/COMPUTE.md); it is stored with the results and added to STAC's GPU latency by the main table. Writes --out (JSON, with the hardware).
"""

import argparse
import csv
import json
import os
import platform
from pathlib import Path

import torch

from fp.config import RESULTS
from fp.data.inputs import load_inputs
from fp.eval import latency

NUMPY_ONLY = ("stac", "stac_single")  # scored with numpy on the CPU: no GPU time


def chosen_methods(path: Path) -> list[str]:
    """The distinct methods the main table chose on LIBERO-X, in the order of the file."""
    with open(path) as f:
        return list(dict.fromkeys(r["method"] for r in csv.DictReader(f) if r["setting"].startswith("LIBERO-X")))


def hardware() -> dict:
    return {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
            "torch": torch.__version__, "python": platform.python_version()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="liberox_groot")
    p.add_argument("--split", default="task_seed0", help="the split whose training steps give the sizes a k-NN stores")
    p.add_argument("--chosen", type=Path, default=RESULTS / "figures" / "chosen.csv")
    p.add_argument("--methods", nargs="+", help="method names; default: those the main table chose on LIBERO-X")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--vla-sampling", type=Path, help="JSON with the extra time STAC needs to sample its K plans (key extra_ms)")
    p.add_argument("--out", type=Path, default=RESULTS / "latency" / "latency.json")
    args = p.parse_args()

    inputs = load_inputs(args.dataset)
    spec = latency.spec_for(inputs, inputs.splits[args.split])
    methods = args.methods or chosen_methods(args.chosen)
    results = {}
    for method in methods:
        family = method.split("__")[0]
        record = {}
        for label, device, threads in (("gpu", "cuda", None), ("cpu4", "cpu", 4), ("cpu1", "cpu", 1)):
            if device == "cuda" and (not torch.cuda.is_available() or family in NUMPY_ONLY):
                record[label] = None
                continue
            torch.set_num_threads(threads or torch.get_num_threads())
            scorer, make_input = latency.build(method, spec, device)
            record[label] = latency.measure(scorer, make_input, device, args.steps)
            print(f"{method:42s} {label:5s} median {record[label]['median_ms']:8.4f} ms   p99 {record[label]['p99_ms']:8.4f} ms", flush=True)
        results[method] = record
    out = {"hardware": hardware(), "spec": spec.__dict__, "dataset": args.dataset, "split": args.split, "steps": args.steps, "methods": results}
    if args.vla_sampling:
        out["vla_sampling"] = json.loads(args.vla_sampling.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
