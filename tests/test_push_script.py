import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "push_distance_to_hub.py"
spec = importlib.util.spec_from_file_location("push_distance_to_hub", SCRIPT)
push = importlib.util.module_from_spec(spec)
spec.loader.exec_module(push)


def test_the_patterns_select_the_three_files_of_the_three_families_and_nothing_else():
    patterns = push.patterns()
    assert len(patterns) == 9 and all(p.startswith("*/") for p in patterns)
    assert {p.split("/")[-1] for p in patterns} == {"scores.npz", "distances.npz", "history.json"}
    assert {p.split("/")[1] for p in patterns} == {"euclidean__*", "mahalanobis__*", "pca_kmeans__*"}  # no weights, no probes, no SAFE models


def test_the_readme_and_card_section_render_and_the_card_section_starts_with_its_marker():
    text = push.README.format()
    assert "paper_seed{0,1,2}" in text and "{{" not in text and "distances.npz" in text
    assert push.CARD_SECTION.format(marker=push.CARD_MARKER).startswith(push.CARD_MARKER)


def test_the_readme_snippet_rebuilds_the_scores_it_describes():
    """The snippet in the README: step = column 0 - column 1, cumulative = running sum; checked on arrays of the saved shape."""
    snippet = text = push.README.split("```python")[1].split("```")[0]
    d = {"ep": np.array([[3.0, 1.0], [2.0, 2.5], [4.0, 1.0]])}
    scope = {"np": np, "d": d}
    exec(snippet.replace('d = np.load("distances.npz")\n', '').replace('d["<episode key>"]', 'd["ep"]'), scope)
    assert scope["step"].tolist() == [2.0, -0.5, 3.0] and scope["cumulative"].tolist() == [2.0, 1.5, 4.5] and scope["success_distance_only"].tolist() == [3.0, 2.0, 4.0]
