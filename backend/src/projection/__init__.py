"""Projection layer: E[K], NB CDF, P(K >= line).

BOOK-AGNOSTIC. Must not import from src.picks. Enforced by
tests/test_scaffolding.py::test_projection_does_not_import_picks.

Phase 3a (typed input contract): :mod:`inputs` exposes the dataclasses that
mirror the Phase 2c bundle shape plus loaders for the static reference
tables.

Phase 3b (in :mod:`features_bf`) and 3c (in :mod:`features_kpa` +
:mod:`projector`) build on top.
"""

from .inputs import (
    BatterInputs,
    BookLine,
    BookMarket,
    BundleMetadata,
    ContractViolation,
    GameContext,
    HandednessAverages,
    LeagueAverages,
    Market,
    MLB_TEAM_ABBRS,
    OpposingLineup,
    ParkFactors,
    PitcherInputs,
    ProjectionBundle,
    UmpireKFactors,
    WeatherInputs,
)

__all__ = [
    "BatterInputs",
    "BookLine",
    "BookMarket",
    "BundleMetadata",
    "ContractViolation",
    "GameContext",
    "HandednessAverages",
    "LeagueAverages",
    "Market",
    "MLB_TEAM_ABBRS",
    "OpposingLineup",
    "ParkFactors",
    "PitcherInputs",
    "ProjectionBundle",
    "UmpireKFactors",
    "WeatherInputs",
]
