import math
from fractions import Fraction

import numpy as np

from fp.calibration.ranks import conformal_rank


def test_float_ceil_would_be_one_rank_too_high_here_and_conformal_rank_is_not():
    assert int(np.ceil((149 + 1) * (1.0 - 0.18))) == 124  # the floating-point pitfall: 150 * 0.82 = 123.00000000000001
    assert conformal_rank(149, 0.18) == 123


def test_matches_exact_arithmetic_on_the_alpha_grid_for_all_realistic_sample_sizes():
    for a in np.round(np.arange(0.005, 1.0, 0.005), 3):
        exact_alpha = Fraction(str(float(a)))
        for n in range(2, 400):
            assert conformal_rank(n, float(a)) == math.ceil((n + 1) * (1 - exact_alpha)), (n, a)


def test_rank_can_exceed_n_for_small_alpha():
    assert conformal_rank(10, 0.05) == 11  # alpha < 1/(n+1): no finite threshold
