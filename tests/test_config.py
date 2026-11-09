import pytest

from fp import config


def test_window_seconds_follow_replan_and_hz():
    assert config.get("libero10_groot").window_seconds == pytest.approx(2.4)
    assert config.get("liberox_groot").window_seconds == pytest.approx(2.4)
    assert config.get("libero10_pi05").window_seconds == pytest.approx(2.5)


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        config.get("nope")


def test_each_dataset_points_to_its_hub_repo():
    assert config.get("liberox_groot").hf_repo == "podolinsky/gr00t-libero-x-features-v3"
    assert config.get("libero10_groot").hf_repo == "podolinsky/groot-libero-10-features-v3"
    assert config.get("libero10_pi05").hf_repo == "podolinsky/pi0.5-libero-10-features-v3"


def test_paths_are_inside_the_package_folder():
    spec = config.get("libero10_groot")
    assert spec.data_dir.parent == config.DATA
    assert spec.results_dir.parent == config.RESULTS
