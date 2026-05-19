"""Phase 5: devig book odds to recover the book's true probability estimate.

A book's posted odds carry vig (the book's edge). To compare model probabilities
to the book's *belief*, we strip the vig out:

Multiplicative devig (two-sided):
    p_over_raw = implied_from_american(over_odds)
    p_under_raw = implied_from_american(under_odds)
    true_p_over = p_over_raw / (p_over_raw + p_under_raw)
    true_p_under = p_under_raw / (p_over_raw + p_under_raw)

Nearest-paired-line imputation (one-sided):
    Pitcher has line X with both sides posted (paired) and line Y with only
    one side. Vig from line X is applied to Y to impute the missing side.

For both flows, the line must have at least one side posted at one of the
two books we trade (FanDuel, DraftKings).
"""
from __future__ import annotations

# Maximum line-distance for nearest-pair imputation. Beyond 1.5 K, vig is
# assumed to differ enough that imputation is too speculative.
MAX_IMPUTATION_DISTANCE = 1.5


def american_to_decimal(american: int) -> float:
    """Standard conversion: -110 -> 1.909..., +150 -> 2.500."""
    if american == 0:
        raise ValueError("american odds cannot be 0")
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def implied_from_american(american: int) -> float:
    """Raw (with-vig) implied probability."""
    if american == 0:
        raise ValueError("american odds cannot be 0")
    if american > 0:
        return 100.0 / (american + 100.0)
    return abs(american) / (abs(american) + 100.0)


def multiplicative_devig(
    p_over_raw: float, p_under_raw: float,
) -> tuple[float, float]:
    """Multiplicative devig: divide both raw probs by their sum."""
    total = p_over_raw + p_under_raw
    if total <= 0:
        raise ValueError(
            f"sum of raw probabilities must be positive, got {total}"
        )
    return p_over_raw / total, p_under_raw / total


def _opposite_side(side: str) -> str:
    if side == "Over":
        return "Under"
    if side == "Under":
        return "Over"
    raise ValueError(f"side must be 'Over' or 'Under', got {side!r}")


def _has_both_sides(book_lines: dict) -> bool:
    """A line entry is paired iff both 'Over' and 'Under' keys are present."""
    return "Over" in book_lines and "Under" in book_lines


def _vig_from_paired(book_lines: dict) -> float:
    """Vig = (p_over_raw + p_under_raw) - 1.0. Always >= 0 in a real market."""
    over_p = implied_from_american(int(book_lines["Over"]))
    under_p = implied_from_american(int(book_lines["Under"]))
    return (over_p + under_p) - 1.0


def nearest_paired_line_imputation(
    pitcher_book_lines: dict[float, dict[str, int]],
    target_line: float,
    target_side: str,
) -> float | None:
    """For a one-sided ``target_line`` (and the side we're missing), find the
    nearest line within :data:`MAX_IMPUTATION_DISTANCE` that has BOTH sides
    posted, compute its vig, and impute the missing raw probability.

    Returns the imputed raw probability for the missing side, or None if no
    paired line is reachable within the distance limit.
    """
    if target_side not in ("Over", "Under"):
        raise ValueError(
            f"target_side must be 'Over' or 'Under', got {target_side!r}"
        )
    if target_line not in pitcher_book_lines:
        return None
    target_entry = pitcher_book_lines[target_line]
    existing_side = _opposite_side(target_side)
    if existing_side not in target_entry:
        return None
    existing_p_raw = implied_from_american(int(target_entry[existing_side]))

    candidates = sorted(
        (
            (abs(line - target_line), line, entry)
            for line, entry in pitcher_book_lines.items()
            if line != target_line
            and _has_both_sides(entry)
            and abs(line - target_line) <= MAX_IMPUTATION_DISTANCE
        ),
        key=lambda x: x[0],
    )
    if not candidates:
        return None
    _, _, paired = candidates[0]
    vig = _vig_from_paired(paired)
    return max(0.0, min(1.0, (1.0 + vig) - existing_p_raw))


def devig_with_imputation(
    pitcher_book_lines: dict[float, dict[str, int]],
    target_line: float,
    target_side: str,
) -> tuple[float, str] | None:
    """Return ``(true_probability, source)`` for the requested side.

    ``source`` is one of:
    - ``"two_sided"``: both sides posted at ``target_line``; direct devig
    - ``"imputed_nearest_pair"``: target side imputed from nearest paired line
    - ``"no_paired_line"``: returns None — no imputation possible
    """
    entry = pitcher_book_lines.get(target_line)
    if entry is None:
        return None
    if _has_both_sides(entry):
        over_p_raw = implied_from_american(int(entry["Over"]))
        under_p_raw = implied_from_american(int(entry["Under"]))
        true_over, true_under = multiplicative_devig(over_p_raw, under_p_raw)
        return (true_over if target_side == "Over" else true_under, "two_sided")

    # One-sided. Need imputation.
    posted_side = next(iter(entry))
    if posted_side == target_side:
        # The side we want is posted, but its counterpart is missing — impute
        # the counterpart so we can devig.
        imputed_other = nearest_paired_line_imputation(
            pitcher_book_lines, target_line, _opposite_side(target_side),
        )
        if imputed_other is None:
            return None
        posted_p_raw = implied_from_american(int(entry[posted_side]))
        true_posted, _ = multiplicative_devig(posted_p_raw, imputed_other)
        return (true_posted, "imputed_nearest_pair")

    # The target side is the missing one. Impute its raw probability.
    imputed = nearest_paired_line_imputation(
        pitcher_book_lines, target_line, target_side,
    )
    if imputed is None:
        return None
    other_p_raw = implied_from_american(int(entry[posted_side]))
    if target_side == "Over":
        true_over, _ = multiplicative_devig(imputed, other_p_raw)
        return (true_over, "imputed_nearest_pair")
    _, true_under = multiplicative_devig(other_p_raw, imputed)
    return (true_under, "imputed_nearest_pair")
