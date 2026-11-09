"""The split-conformal rank, computed once.

k = ceil((n + 1)(1 - alpha)) is an integer rank. In floating point the product can land a hair above an exact integer
(150 * 0.82 = 123.00000000000001), and ceil then returns 124 instead of 123, silently one rank too conservative.
The product is rounded to 9 decimals before the ceil; an alpha given with up to 6 decimals and n up to 10^3 is then exact.
"""

import numpy as np


def conformal_rank(n: int, alpha: float) -> int:
    """ceil((n + 1)(1 - alpha)), robust to floating-point error. May exceed n (then no finite threshold is valid)."""
    return int(np.ceil(round((n + 1) * (1.0 - alpha), 9)))
