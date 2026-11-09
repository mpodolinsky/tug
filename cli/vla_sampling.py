"""python cli/vla_sampling.py --gr00t-scripts <repo>/scripts/groot-libero-10 --model-path <GR00T checkpoint> [--samples 10] [--calls 50] [--out results/latency/vla_sampling.json]

How much longer a GR00T policy call takes when it samples K plans instead of one: the extra time STAC needs at every call and no other method does (every other method
reads what a single call already produces). Run it with the Python of the GR00T environment (``submodules/Isaac-GR00T/.venv`` of the data-collection repository; it
needs ``gr00t`` and CUDA, and is the only part of this repository that does), because it loads the policy; it imports nothing else from this repository.

A call is timed end to end (``Gr00tSimPolicyWrapper.get_action``: preprocessing, the backbone, the action head, decoding) on a synthetic LIBERO observation, with the
GPU synchronised, in three ways:

    one plan                      the ordinary call
    K plans, batched head         the backbone once, the action head once on K copies of its inputs (K independent noise draws in one batch)
    K plans, sequential head      the backbone once, the action head K times
    K plans, K calls              K ordinary calls: an upper bound

``extra_ms`` is the time of the faster of the two shared-backbone ways to get K plans, minus the time for one plan (the cheapest STAC could be, which is favourable to
it), so that STAC's latency is its scoring time plus ``extra_ms``. The times do not depend
on the weights, only on the architecture and the observation size. Writes --out (JSON, with the checkpoint and the hardware).
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


def fake_observation(groot_obs) -> dict:
    """A LIBERO observation of the right shapes and a prompt, in the flat format of the policy wrapper."""
    rng = np.random.default_rng(0)
    obs = {"agentview_image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8), "robot0_eye_in_hand_image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
           "robot0_eef_pos": rng.standard_normal(3), "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]), "robot0_gripper_qpos": rng.standard_normal(2)}
    return groot_obs.build_flat_obs(obs, "put the black bowl in the bottom drawer of the cabinet and close it")


def tile(x, k: int):
    """``x`` (a tensor, or a dict of them, possibly nested) with every batch-1 tensor repeated ``k`` times along the batch dimension."""
    if isinstance(x, torch.Tensor):
        return x.repeat_interleave(k, dim=0) if x.ndim and x.shape[0] == 1 else x
    if hasattr(x, "items"):
        return type(x)({name: tile(value, k) for name, value in x.items()})
    return x


def synchronised(fn, calls: int, warmup: int = 10) -> float:
    """The median time of ``fn()`` in milliseconds over ``calls`` calls after ``warmup`` untimed ones."""
    times = []
    for i in range(calls + warmup):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(1000 * (time.perf_counter() - start))
    return statistics.median(times)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gr00t-scripts", type=Path, required=True, help="the scripts/groot-libero-10 folder of the data-collection repository (groot_obs.py)")
    p.add_argument("--model-path", required=True)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--calls", type=int, default=50)
    p.add_argument("--out", type=Path, default=Path("results/latency/vla_sampling.json"))
    args = p.parse_args()
    sys.path.insert(0, str(args.gr00t_scripts))

    import groot_obs
    from gr00t.data.types import MessageType
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper, _rec_to_dtype

    class SamplingPolicy(Gr00tPolicy):
        """The ordinary policy that runs the action head ``n_samples`` times on one backbone pass."""

        n_samples = 1
        batched = False

        def _get_action(self, observation, options=None):
            unbatched = self._unbatch_observation(observation)
            states, processed = [], []
            for obs in unbatched:
                vla = self._to_vla_step_data(obs)
                states.append(vla.states)
                processed.append(self.processor([{"type": MessageType.EPISODE_STEP.value, "content": vla}]))
            collated = _rec_to_dtype(self.collate_fn(processed), dtype=torch.bfloat16)
            inputs = collated["inputs"] if "inputs" in collated else collated
            with torch.inference_mode():
                backbone_inputs, action_inputs = self.model.prepare_input(inputs)
                backbone_outputs = self.model.backbone(backbone_inputs)
                if self.batched and self.n_samples > 1:
                    plan = self.model.action_head.get_action(tile(backbone_outputs, self.n_samples), tile(action_inputs, self.n_samples), options)["action_pred"]
                    plans = [plan[:1]]
                else:
                    plans = [self.model.action_head.get_action(backbone_outputs, action_inputs, options)["action_pred"] for _ in range(self.n_samples)]
            batched_states = {k: np.stack([s[k] for s in states], axis=0) for k in self.modality_configs["state"].modality_keys}
            decoded = self.processor.decode_action(plans[0].float().cpu().numpy(), self.embodiment_tag, batched_states)
            return {k: v.astype(np.float32) for k, v in decoded.items()}, {}

    policy = SamplingPolicy(embodiment_tag="LIBERO_PANDA", model_path=args.model_path, device="cuda:0", strict=False)
    wrapper = Gr00tSimPolicyWrapper(policy, strict=False)
    obs = fake_observation(groot_obs)

    def call(n: int, batched: bool = False) -> float:
        policy.n_samples, policy.batched = n, batched
        return synchronised(lambda: wrapper.get_action(obs), args.calls)

    one, sequential, batched = call(1), call(args.samples), call(args.samples, batched=True)
    result = {"model_path": args.model_path, "samples": args.samples, "calls": args.calls, "one_plan_ms": one, "sequential_head_k_plans_ms": sequential,
              "batched_head_k_plans_ms": batched, "extra_ms": min(sequential, batched) - one, "k_calls_extra_ms": (args.samples - 1) * one,
              "hardware": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "python": platform.python_version()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
