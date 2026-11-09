from fp.baselines.distance_grid import GRID, all_grid_scorers, grid_configs, grid_scorers


def test_the_grid_has_the_configurations_safe_sweeps_26_in_all():
    assert len(grid_configs("mahala")) == 2  # cumsum
    assert len(grid_configs("euclid")) == 6  # topk x cumsum
    assert len(grid_configs("pca_kmeans")) == 18  # pca_dim x n_clusters x cumsum
    assert len(all_grid_scorers({}, {})) == 26


def test_the_values_are_those_of_safes_sweep_script():
    assert GRID["euclid"]["topk"] == (1, 5, 10)
    assert GRID["pca_kmeans"]["pca_dim"] == (32, 64, 128)
    assert GRID["pca_kmeans"]["n_clusters"] == (16, 32, 64)
    assert all(cfg["cumsum"] in (False, True) for d in GRID for cfg in grid_configs(d))


def test_names_are_unique_and_say_what_they_are():
    names = [s.name for s in all_grid_scorers({}, {})]
    assert len(set(names)) == 26
    assert "mahalanobis__cumsum" in names and "mahalanobis__step" in names
    assert "euclidean__k5_cumsum" in names and "euclidean__k1_step" in names
    assert "pca_kmeans__d128_c64_step" in names


def test_safes_default_configuration_is_in_the_grid():
    names = {s.name for s in all_grid_scorers({}, {})}
    assert {"euclidean__k10_cumsum", "mahalanobis__cumsum", "pca_kmeans__d64_c16_cumsum"} <= names


def test_a_scorer_is_built_for_every_configuration_with_its_parameters():
    for cfg, scorer in zip(grid_configs("pca_kmeans"), grid_scorers("pca_kmeans", {}, {})):
        assert (scorer.pca_dim, scorer.n_clusters, scorer.cumsum) == (cfg["pca_dim"], cfg["n_clusters"], cfg["cumsum"])
        assert not scorer.use_success_only  # SAFE's sweep never uses success-only
