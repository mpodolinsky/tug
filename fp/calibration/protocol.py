"""Which calibrator each method gets in the main results. The single place that says so.

    SAFE-MLP, SAFE-LSTM        SAFE functional band (their own). All 12 configurations of each, named
                               <family>__lr<lr>_reg<lambda_reg> (see baselines/safe_grid.py), get it; so do the same models trained on
                               rollouts cut at the onset (families safe_mlp_pre, safe_lstm_pre), which are still SAFE models.
    FailureSpot                SAFE functional band (the paper's own conformal band). The nine (eta1, eta2) configurations of
                               each detector, named failurespot_<mlp|lstm>__eta1_<a>_eta2_<b> (see baselines/failurespot.py).
    our probes                 max-conformal
    STAC (original, single)    max-conformal. STAC accumulates non-negative per-step scores, so the maximum over steps is
                               the final value of the accumulated score.
    ActProbe                   max-conformal (as in the ActProbe paper)
    Mahalanobis, Euclidean,
    PCA-KMeans                 SAFE band, as SAFE itself evaluates them (their score can be negative, so the maximum over
                               steps is not the final value, unlike STAC). All configurations of a baseline, named
                               <baseline>__<tag> (see baselines/distance.py), get the same calibrator.

Our probes may additionally be evaluated with the SAFE band as a secondary table, and LTT is a separate evaluation; neither
changes this table.
"""

import re

from ..data.labels import VARIANTS
from .base import Calibrator
from .max_conformal import MaxConformal
from .safe_band import SafeBand

SAFE_BAND = "safe_band"
MAX_CONFORMAL = "max_conformal"


PROTOCOL: dict[str, str] = {
    "safe_mlp": SAFE_BAND,
    "safe_lstm": SAFE_BAND,
    "safe_mlp_pre": SAFE_BAND,
    "safe_lstm_pre": SAFE_BAND,
    "failurespot_mlp": SAFE_BAND,
    "failurespot_lstm": SAFE_BAND,
    "mahalanobis": SAFE_BAND,
    "euclidean": SAFE_BAND,
    "pca_kmeans": SAFE_BAND,
    "stac": MAX_CONFORMAL,
    "stac_single": MAX_CONFORMAL,
    "actprobe": MAX_CONFORMAL,
}

# Our probes are named <ProbeConfig.name>__<label variant>, e.g. lstm_mha_modes_clip__onset_pre (see models/ours.py, train/ours.py).
_OURS = re.compile(rf"^(lstm|mlp)(_mha)?_(plain|modes|clip|modes_clip)__({'|'.join(VARIANTS)})$")
# One configuration of an embedding-distance baseline: <baseline>__<tag>, e.g. euclidean__k10_cumsum (see baselines/distance.py).
_DISTANCE_CONFIG = re.compile(r"^(mahalanobis|euclidean|pca_kmeans)__[a-z0-9_]+$")
# One configuration of SAFE-MLP / SAFE-LSTM: <family>__lr<learning rate>_reg<lambda_reg>, e.g. safe_lstm__lr3e-4_reg1e-2 (see baselines/safe_grid.py).
_SAFE_CONFIG = re.compile(r"^(safe_mlp|safe_lstm)(_pre)?__lr[0-9]e-[0-9]_reg[0-9]e-[0-9]$")


# One configuration of STAC: stac__mmd<dimensions>_<step|cumsum> or stac_single__<step|cumsum> (see baselines/stac.py).
_STAC_CONFIG = re.compile(r"^(stac__mmd[763]_|stac_single__)(step|cumsum)$")
# One configuration of FailureSpot: failurespot_<mlp|lstm>__eta1_<a>_eta2_<b> with a, b in {0p25, 1, 4} (see baselines/failurespot.py).
_FAILURESPOT_CONFIG = re.compile(r"^failurespot_(mlp|lstm)__eta1_(0p25|1|4)_eta2_(0p25|1|4)$")


def calibrator_name(method: str) -> str:
    """'safe_band' or 'max_conformal' for the method; KeyError for an unknown method."""
    if method in PROTOCOL:
        return PROTOCOL[method]
    if _OURS.match(method):
        return MAX_CONFORMAL
    if any(pattern.match(method) for pattern in (_DISTANCE_CONFIG, _SAFE_CONFIG, _STAC_CONFIG, _FAILURESPOT_CONFIG)):
        return PROTOCOL[method.split("__")[0]]
    raise KeyError(f"unknown method {method!r}; known: {sorted(PROTOCOL)}, our probes <lstm|mlp>_<heads>__<variant>, "
                   f"distance configurations <baseline>__<tag>, SAFE configurations <family>__lr<lr>_reg<lambda_reg>, STAC and "
                   f"FailureSpot configurations (see baselines/stac.py, baselines/failurespot.py)")


def parse_probe_name(method: str) -> dict | None:
    """The factors of one of our probes, e.g. lstm_mha_modes_clip__onset_pre -> trunk lstm, mha True, heads modes_clip, variant onset_pre;
    None if ``method`` is not one of our probes."""
    match = _OURS.match(method)
    if match is None:
        return None
    trunk, mha, heads, variant = match.groups()
    return {"trunk": trunk, "mha": mha is not None, "heads": heads, "variant": variant}


def calibrator_by_name(name: str, seed: int = 0) -> Calibrator:
    """An unfitted calibrator: 'safe_band' (``seed`` is its shuffle seed, the split seed) or 'max_conformal'."""
    if name not in (SAFE_BAND, MAX_CONFORMAL):
        raise KeyError(f"unknown calibrator {name!r}; choose from {SAFE_BAND!r}, {MAX_CONFORMAL!r}")
    return SafeBand(seed=seed) if name == SAFE_BAND else MaxConformal()


def make_calibrator(method: str, seed: int = 0) -> Calibrator:
    """An unfitted calibrator for the method, the one the protocol says it gets."""
    return calibrator_by_name(calibrator_name(method), seed)
