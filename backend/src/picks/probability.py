"""Phase 5: projection-to-probability conversion.

Maps the NB(mean=E[K], dispersion=alpha) distribution to per-line over/under
probabilities. alpha comes from Phase 4d (currently 0.001 — effectively
Poisson, but kept as NB to preserve the option of recalibration later).

Math:
    NB parameterized by (mean, alpha):
        Var(K) = mean * (1 + alpha * mean)
        scipy.stats.nbinom (n, p) form: n = 1/alpha, p = 1/(1 + alpha*mean)

For an alt line L (typically a half-integer like 5.5):
    P(K >= L) = P(K >= ceil(L)) = 1 - CDF(ceil(L) - 1)
    P(K <= L) = P(K <= floor(L)) = CDF(floor(L))

For half-integer lines, p_over + p_under = 1 (no probability mass at the line).
For integer lines, p_over + p_under + P(K == L) = 1 (push case — picks
engine treats integer lines as half-line + 0.5 epsilon for the over side,
i.e. K >= ceil(L) for over and K <= L - 1 for under).
"""
from __future__ import annotations

import math
from typing import Iterable

from scipy.stats import nbinom


def _nb_params(mean: float, alpha: float) -> tuple[float, float]:
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if mean <= 0:
        raise ValueError(f"mean must be > 0, got {mean}")
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mean)
    return n, p


def p_k_geq(line: float, e_k: float, alpha: float) -> float:
    """P(over) under NB(mean=e_k, dispersion=alpha).

    For half-integer lines (5.5): over wins iff K >= 6 → 1 - CDF(5).
    For integer lines (5.0): over wins iff K >= 6 (strictly greater than
    the line; K=5 is a push, treated as neither over nor under) → 1 - CDF(5).
    """
    n, p = _nb_params(e_k, alpha)
    if float(line).is_integer():
        threshold = int(line) + 1
    else:
        threshold = int(math.ceil(line))
    return float(1.0 - nbinom.cdf(threshold - 1, n, p))


def p_k_leq(line: float, e_k: float, alpha: float) -> float:
    """P(under) under NB(mean=e_k, dispersion=alpha).

    For half-integer lines (5.5): under wins iff K <= 5 → CDF(5).
    For integer lines (5.0): under wins iff K <= 4 (strictly less than the
    line; K=5 is a push) → CDF(4).
    """
    n, p = _nb_params(e_k, alpha)
    if float(line).is_integer():
        threshold = int(line) - 1
    else:
        threshold = int(math.floor(line))
    return float(nbinom.cdf(threshold, n, p))


def model_probabilities_for_pitcher(
    e_k: float, alpha: float, lines: Iterable[float],
) -> dict[float, tuple[float, float]]:
    """For each line, return (p_over, p_under) under NB(e_k, alpha).

    For half-integer lines, p_over + p_under == 1 by construction.
    For integer lines, p_over + p_under == 1 - P(K == line) < 1 (the
    missing mass is the push probability).
    """
    out: dict[float, tuple[float, float]] = {}
    for line in lines:
        out[line] = (p_k_geq(line, e_k, alpha), p_k_leq(line, e_k, alpha))
    return out
