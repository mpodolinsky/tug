import itertools

import numpy as np
import pytest

from fp.calibration.max_conformal import MaxConformal
from fp.baselines.distance_grid import all_grid_scorers
from fp.calibration.protocol import MAX_CONFORMAL, SAFE_BAND, calibrator_by_name, calibrator_name, make_calibrator, parse_probe_name
from fp.calibration.safe_band import SafeBand
from fp.data.labels import VARIANTS
from fp.models.ours import all_configs


@pytest.mark.parametrize("method", ["safe_mlp", "safe_lstm", "failurespot_mlp", "failurespot_lstm"])
def test_safe_models_and_failurespot_use_the_safe_band(method):
    assert calibrator_name(method) == SAFE_BAND


@pytest.mark.parametrize("method", ["stac", "stac_single", "actprobe"])
def test_stac_and_actprobe_use_max_conformal(method):
    assert calibrator_name(method) == MAX_CONFORMAL


def test_every_probe_name_gets_max_conformal():
    """The names come from the model code itself, so the protocol cannot drift from the naming of the probes."""
    for cfg, variant in itertools.product(all_configs(), VARIANTS):
        assert calibrator_name(f"{cfg.name}__{variant}") == MAX_CONFORMAL


@pytest.mark.parametrize("method", ["mahalanobis", "euclidean", "pca_kmeans"])
def test_the_embedding_distance_baselines_use_the_safe_band(method):
    assert calibrator_name(method) == SAFE_BAND


def test_every_configuration_of_the_distance_baselines_gets_the_safe_band():
    episodes = {}  # configurations are named without needing any data
    for scorer in all_grid_scorers(episodes, {}):
        assert calibrator_name(scorer.name) == SAFE_BAND


def test_a_distance_configuration_of_an_unknown_baseline_is_an_error():
    with pytest.raises(KeyError):
        calibrator_name("cosine__k10_cumsum")


@pytest.mark.parametrize("name", ["safe_gru", "lstm_modes__not_a_variant", "gru_plain__onset", ""])
def test_unknown_names_are_an_error_not_a_default(name):
    with pytest.raises(KeyError):
        calibrator_name(name)


def test_make_calibrator_returns_the_right_class_and_passes_the_seed():
    band = make_calibrator("safe_lstm", seed=7)
    assert isinstance(band, SafeBand) and band.seed == 7 and not band.finite_sample  # SAFE-faithful quantile
    assert isinstance(make_calibrator("lstm_modes_clip__onset_pre"), MaxConformal)


def test_probe_names_are_parsed_into_their_factors_and_everything_else_is_not_a_probe():
    assert parse_probe_name("lstm_mha_modes_clip__onset_pre") == {"trunk": "lstm", "mha": True, "heads": "modes_clip", "variant": "onset_pre"}
    assert parse_probe_name("mlp_plain__onset_traj") == {"trunk": "mlp", "mha": False, "heads": "plain", "variant": "onset_traj"}
    for name in ("safe_lstm", "euclidean__k10_cumsum", "lstm_plain", "lstm_plain__nope", "gru_plain__onset"):
        assert parse_probe_name(name) is None
    for cfg, variant in itertools.product(all_configs(), VARIANTS):  # every probe name the package builds parses back to its parts
        parsed = parse_probe_name(f"{cfg.name}__{variant}")
        assert (parsed["trunk"], parsed["mha"], parsed["variant"]) == (cfg.trunk, cfg.mha, variant)


def test_calibrators_can_be_built_by_name_and_an_unknown_name_is_an_error():
    assert isinstance(calibrator_by_name(MAX_CONFORMAL), MaxConformal)
    assert isinstance(calibrator_by_name(SAFE_BAND, seed=4), SafeBand) and calibrator_by_name(SAFE_BAND, seed=4).seed == 4
    with pytest.raises(KeyError):
        calibrator_by_name("ltt")


def test_stac_max_over_steps_is_the_final_value_because_its_scores_are_non_negative():
    """The reason max-conformal on STAC equals thresholding the final accumulated score."""
    rng = np.random.default_rng(0)
    cumulated = [np.cumsum(rng.random(rng.integers(20, 60))) for _ in range(30)]  # non-negative per-step scores
    m = MaxConformal().fit(cumulated)
    assert np.array_equal(m.maxima, np.sort([c[-1] for c in cumulated]))
