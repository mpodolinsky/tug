"""Online latency of the methods of the main table: the time to turn one new policy step into a score.

The batched scorers of the sweeps (``score_ms_per_step``) run a whole dataset through a model at once, which says how fast a recorded dataset is scored and not
how long a robot waits for the score of the step it has just taken. Online, one embedding (or one action chunk) arrives per policy step, nothing can be batched
over time, and a recurrent model carries its state from the previous step. Each method here has an online step that does exactly that, from the same modules and
helper functions as its batched scorer (``tests/test_latency.py`` checks that the online scores equal the batched ones):

    ours, SAFE-LSTM, FailureSpot-LSTM   one LSTM step with the carried state (ours: and the score, mode and CLIP heads)
    SAFE-MLP, FailureSpot-MLP           one MLP pass, with SAFE's running sum for SAFE-MLP
    Mahalanobis, Euclidean, PCA-KMeans  the distances of one embedding to the stored statistics, centroids or training steps, and the running sum
    STAC, STAC-Single, ActProbe         the score of one call from its action chunk(s) (numpy; the signals ActProbe reads are computed here too)

``measure`` times one step after a warm-up, with the input prepared beforehand (an embedding already on the device it is scored on) and the device synchronised:
the median and the 99th percentile over many steps. The sizes of a method's stored data come from the dataset (``LatencySpec``); the weights are random, because the
time of a dense layer or an LSTM does not depend on its values (the equality with the batched scorers is tested with the same modules at a small size).
STAC samples K plans from the policy at every call, which no other method needs; the time to sample them is not part of its scoring step and is recorded separately
(``cli.latency --vla-sampling``) and added by the main table.
"""

import re
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..baselines import actprobe, distance, failurespot, stac
from ..calibration.protocol import parse_probe_name
from ..models.ours import Probe, ProbeConfig
from ..models.safe import SafeLSTM, SafeMLP


@dataclass(frozen=True)
class LatencySpec:
    """The sizes the timed methods depend on: the embedding width, the training steps a k-NN stores (successes, failures), and the action chunks STAC and ActProbe
    read (K sampled plans of H steps with A values, of which ``replan_steps`` are executed per call)."""
    dim: int = 6144
    n_succ_steps: int = 60_000
    n_fail_steps: int = 64_000
    samples: int = 10
    horizon: int = 16
    action_dim: int = 7
    replan_steps: int = 8


def spec_for(inputs, split) -> LatencySpec:
    """The ``LatencySpec`` of a split: its training steps by outcome, and the action chunk format of the dataset (GR00T: K = 10, H = 16, A = 7, 8 executed)."""
    from ..config import FEATURE_DIM, get

    train = [inputs.episodes[k] for k in split.train]
    return LatencySpec(FEATURE_DIM, sum(e.n_steps for e in train if e.success), sum(e.n_steps for e in train if not e.success), replan_steps=get(inputs.dataset).replan_steps)


# ---------------------------------------------------------------- the online steps
class OnlineScorer(ABC):
    """One policy step in, one score out. ``reset`` starts a new episode (the state is cleared)."""

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def step(self, x): ...


class ProbeOnline(OnlineScorer):
    """Our probe with the LSTM trunk: one LSTM step with the carried state, then the score head and the other heads the architecture has (their outputs are in
    ``outputs``)."""

    def __init__(self, model: Probe):
        if model.cfg.trunk != "lstm" or model.cfg.mha:
            raise ValueError("the online step covers the LSTM trunk without attention")
        self.model, self.state, self.outputs = model.eval(), None, {}

    def reset(self) -> None:
        self.state = None

    @torch.no_grad()
    def step(self, x: torch.Tensor) -> torch.Tensor:
        u, self.state = self.model.trunk(x.view(1, 1, -1), self.state)
        self.outputs = {name: head(u) for name, head in (("mode", self.model.mode), ("sem", self.model.sem)) if head is not None}
        return torch.sigmoid(self.model.score(u)).reshape(())


class LstmOnline(OnlineScorer):
    """SAFE-LSTM and FailureSpot's LSTM detector (both: an ``lstm`` and an ``fc`` layer, a sigmoid on the state): one step with the carried state."""

    def __init__(self, model: nn.Module):
        self.model, self.state = model.eval(), None

    def reset(self) -> None:
        self.state = None

    @torch.no_grad()
    def step(self, x: torch.Tensor) -> torch.Tensor:
        u, self.state = self.model.lstm(x.view(1, 1, -1), self.state)
        return torch.sigmoid(self.model.fc(u)).reshape(())


class MlpOnline(OnlineScorer):
    """SAFE-MLP (``net`` = its projector, ``cumsum``: the failure score is the running sum of the step probabilities) and FailureSpot's MLP detector (no sum)."""

    def __init__(self, net: nn.Module, cumsum: bool):
        self.net, self.cumsum, self.total = net.eval(), cumsum, 0.0

    def reset(self) -> None:
        self.total = 0.0

    @torch.no_grad()
    def step(self, x: torch.Tensor) -> torch.Tensor:
        p = self.net(x).reshape(())
        self.total = self.total + p
        return self.total if self.cumsum else p


class DistanceOnline(OnlineScorer, ABC):
    """The embedding-distance baselines: the step score is the distance to the success set minus the distance to the failure set, accumulated when ``cumsum``."""

    def __init__(self, cumsum: bool):
        self.cumsum, self.total = cumsum, 0.0

    def reset(self) -> None:
        self.total = 0.0

    @abstractmethod
    def distances(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

    @torch.no_grad()
    def step(self, x: torch.Tensor) -> torch.Tensor:
        d_succ, d_fail = self.distances(x.view(1, -1))
        score = (d_succ - d_fail).reshape(())
        self.total = self.total + score
        return self.total if self.cumsum else score


class MahalanobisOnline(DistanceOnline):
    def __init__(self, mean_succ, inv_succ, mean_fail, inv_fail, cumsum: bool):
        super().__init__(cumsum)
        self.mean_succ, self.inv_succ, self.mean_fail, self.inv_fail = mean_succ, inv_succ, mean_fail, inv_fail

    def distances(self, x):
        return (distance.compute_mahala_dist(x, self.mean_succ, None, inv_cov=self.inv_succ), distance.compute_mahala_dist(x, self.mean_fail, None, inv_cov=self.inv_fail))


class EuclideanOnline(DistanceOnline):
    """The mean distance to the ``k`` nearest stored training steps of each outcome."""

    def __init__(self, feats_succ, feats_fail, k: int, cumsum: bool):
        super().__init__(cumsum)
        self.feats_succ, self.feats_fail, self.k = feats_succ, feats_fail, k

    def distances(self, x):
        nearest = lambda y: distance.compute_dist_closest_k_by_chunk(x, y, self.k, distance.compute_euclid_dist).mean(dim=-1)
        return nearest(self.feats_succ), nearest(self.feats_fail)


class PcaKMeansOnline(DistanceOnline):
    """The distance to the nearest centroid of each outcome, after SAFE's projection of the embedding with the success PCA (``mean``, ``components``)."""

    def __init__(self, mean, components, centroids_succ, centroids_fail, cumsum: bool):
        super().__init__(cumsum)
        self.mean, self.components, self.centroids_succ, self.centroids_fail = mean, components, centroids_succ, centroids_fail

    def distances(self, x):
        z = (x - self.mean) @ self.components.T
        nearest = lambda c: distance.compute_dist_closest_k_by_chunk(z, c, 1, distance.compute_euclid_dist).mean(dim=-1)
        return nearest(self.centroids_succ), nearest(self.centroids_fail)


class StacOnline(OnlineScorer):
    """STAC from the plans of consecutive calls: ``step`` takes the K sampled plans of this call, (K, H, A); STAC-Single (``family`` 'stac_single') the executed plan
    only, (H, A). The score of a call compares the unexecuted rest of the previous plan with the first actions of the new one; the first call scores 0."""

    def __init__(self, family: str, dims: int, replan_steps: int, cumsum: bool):
        self.family, self.cols, self.e, self.cumsum = family, stac.DIMENSIONS[dims], replan_steps, cumsum
        self.reset()

    def reset(self) -> None:
        self.prev, self.total = None, 0.0

    def step(self, chunks: np.ndarray) -> float:
        score = 0.0
        if self.prev is not None:
            if self.family == "stac":
                old, new = self.prev[..., self.e:, self.cols], chunks[..., :-self.e, self.cols]
                score = float(stac.mmd_rbf(old[None], new[None])[0])
            else:
                score = float(np.linalg.norm(self.prev[self.e:].astype(np.float64) - chunks[:-self.e], axis=-1).mean())
        self.prev = chunks
        self.total += score
        return self.total if self.cumsum else score


class ActProbeOnline(OnlineScorer):
    """ActProbe from the executed plan of each call, (H, A): ACM and TCE (``actprobe.acm_tce``), normalised with the training mean and standard deviation, and the
    time feature; one LSTM step starting from the state the task embedding sets, then the MLP with the skip connection."""

    def __init__(self, net: actprobe.ActProbeNet, mean: np.ndarray, std: np.ndarray, lang: torch.Tensor, replan_steps: int):
        self.net, self.mean, self.std, self.lang, self.e = net.eval(), mean, std, lang, replan_steps
        self.reset()

    def reset(self) -> None:
        self.prev, self.t, self.state = None, 0, None

    @torch.no_grad()
    def step(self, chunk: np.ndarray) -> torch.Tensor:
        if self.state is None:
            z = self.net.lang_proj(self.lang)
            self.state = (self.net.h0_proj(z).view(1, 1, -1).contiguous(), self.net.c0_proj(z).view(1, 1, -1).contiguous())
        pair = chunk[None] if self.prev is None else np.stack([self.prev, chunk])
        signals = actprobe.acm_tce(pair, self.e)[-1]
        x = torch.tensor([*((signals - self.mean) / (self.std + actprobe.NORM_EPS)), self.t / actprobe.TIME_SCALE], dtype=torch.float32, device=self.lang.device).view(1, 1, -1)
        out, self.state = self.net.lstm(x, self.state)
        self.prev, self.t = chunk, self.t + 1
        return torch.sigmoid(self.net.mlp(torch.cat([out, x], dim=-1))).reshape(())


# ---------------------------------------------------------------- a method name -> an online step with random weights of the right size
def build(method: str, spec: LatencySpec, device: str = "cpu", seed: int = 0) -> tuple[OnlineScorer, callable]:
    """The online step of the main-table method ``method`` (its name in ``chosen.csv``) with random weights of the sizes ``spec`` gives, and a function that makes
    the input of one step (an embedding on ``device``, or action chunks on the CPU). Raises ``KeyError`` for a method without an online step."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    embedding = lambda: torch.randn(spec.dim, device=device)
    family = method.split("__")[0]
    cumsum = method.endswith("cumsum")
    if re.match(r"(lstm|mlp)_", family):
        cfg = parse_probe_name(method)
        config = ProbeConfig(trunk=cfg["trunk"], mha=cfg["mha"], modes="modes" in cfg["heads"], clip="clip" in cfg["heads"], d_modality=spec.dim // 3)  # three modalities
        return ProbeOnline(Probe(config).to(device)), embedding
    if family == "safe_lstm":
        return LstmOnline(SafeLSTM(spec.dim).to(device)), embedding
    if family == "safe_mlp":
        return MlpOnline(SafeMLP(spec.dim).projector.to(device), True), embedding
    if family in ("failurespot_lstm", "failurespot_mlp"):
        model = (failurespot.StepLSTM if family.endswith("lstm") else failurespot.StepMLP)(spec.dim).to(device)
        return (LstmOnline(model) if family.endswith("lstm") else MlpOnline(model.net, False)), embedding
    if family == "mahalanobis":
        def inverse_covariance():  # positive semi-definite, as a covariance's pseudo-inverse is (the distance is a square root)
            a = torch.randn(spec.dim, spec.dim, device=device) / spec.dim ** 0.5
            return a @ a.T

        return MahalanobisOnline(torch.randn(spec.dim, device=device), inverse_covariance(), torch.randn(spec.dim, device=device), inverse_covariance(), cumsum), embedding
    if family == "euclidean":
        k = int(re.search(r"k(\d+)", method).group(1))
        return EuclideanOnline(torch.randn(spec.n_succ_steps, spec.dim, device=device), torch.randn(spec.n_fail_steps, spec.dim, device=device), k, cumsum), embedding
    if family == "pca_kmeans":
        d, c = (int(v) for v in re.search(r"d(\d+)_c(\d+)", method).groups())
        r = lambda *shape: torch.randn(*shape, device=device)
        return PcaKMeansOnline(r(spec.dim), r(d, spec.dim), r(c, d), r(c, d), cumsum), embedding
    if family == "stac":
        dims = int(re.search(r"mmd(\d)", method).group(1))
        return StacOnline("stac", dims, spec.replan_steps, cumsum), lambda: rng.standard_normal((spec.samples, spec.horizon, spec.action_dim)).astype(np.float32)
    if family == "stac_single":
        return StacOnline("stac_single", 7, spec.replan_steps, cumsum), lambda: rng.standard_normal((spec.horizon, spec.action_dim)).astype(np.float32)
    if family == "actprobe":
        lang = torch.randn(1, actprobe.LANGUAGE_DIM, device=device)
        net = actprobe.ActProbeNet().to(device)
        return (ActProbeOnline(net, np.zeros(actprobe.N_FEATURES, np.float32), np.ones(actprobe.N_FEATURES, np.float32), lang, spec.replan_steps),
                lambda: rng.standard_normal((spec.horizon, spec.action_dim)).astype(np.float32))
    raise KeyError(f"no online step for {method!r}")


def measure(scorer: OnlineScorer, make_input, device: str = "cpu", steps: int = 2000, warmup: int = 200, max_seconds: float = 20.0) -> dict[str, float]:
    """The time of one online step in milliseconds: the median and the 99th percentile over ``steps`` steps after ``warmup`` untimed ones. The inputs are made
    beforehand; a GPU is synchronised before and after each step, a CPU result is read out. A slow step (a k-NN over 124,000 stored steps takes seconds on one CPU
    thread) would take hours to repeat 2,000 times, so after three trial steps the counts are cut to what takes about ``max_seconds`` (at least 20 timed steps)."""

    def once(x) -> float:
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = scorer.step(x)
        if device == "cuda":
            torch.cuda.synchronize()
        elif isinstance(out, torch.Tensor):
            out.item()
        return 1000 * (time.perf_counter() - start)

    scorer.reset()
    trial = statistics.median(once(make_input()) for _ in range(3)) / 1000  # seconds per step
    steps = max(20, min(steps, int(max_seconds / max(trial, 1e-9))))
    warmup = min(warmup, max(3, int(max_seconds / 4 / max(trial, 1e-9))))
    for _ in range(warmup):
        once(make_input())
    times = sorted(once(x) for x in [make_input() for _ in range(steps)])
    return {"median_ms": statistics.median(times), "p99_ms": times[int(0.99 * len(times))], "steps": steps}
