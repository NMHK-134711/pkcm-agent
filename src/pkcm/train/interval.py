"""A confidence interval that does not lie when the sample is lopsided.

The Wald interval -- ``1.96 * sqrt(p(1-p)/n)`` -- collapses to zero when every
game goes the same way. Six losses out of six comes out as "0.0% +/- 0.0%",
which reads as certainty and means the opposite: six games is nothing, and the
true rate could easily be a third.

Wilson's score interval is barely more code and behaves at the ends, which is
exactly where a young training run lives.
"""

from __future__ import annotations

import math

#: Two-sided 95%.
Z = 1.959963985


def wilson(wins: int, total: int, z: float = Z) -> tuple[float, float, float]:
    """``(rate, low, high)``. An empty sample is total ignorance, not 50%."""
    if total <= 0:
        return 0.5, 0.0, 1.0
    rate = wins / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(rate * (1 - rate) / total
                           + z * z / (4 * total * total)) / denominator
    return rate, max(0.0, centre - spread), min(1.0, centre + spread)


def separable(wins: int, total: int, z: float = Z) -> bool:
    """Is this distinguishable from a coin flip?"""
    _, low, high = wilson(wins, total, z)
    return low > 0.5 or high < 0.5


def describe(wins: int, total: int) -> str:
    rate, low, high = wilson(wins, total)
    verdict = "" if separable(wins, total) else "   (not separable)"
    return f"{rate:.1%}  [{low:.1%}, {high:.1%}]{verdict}"
