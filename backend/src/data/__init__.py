"""External-data clients and the AsOfContext primitive.

Built in Phase 2:
- as_of.AsOfContext         — every client method accepts cutoff_date.
- probables.ProbablesClient — FanGraphs primary, StatsAPI fallback,
                              opener detection, cross-source disagreement log.
- statsapi.StatsAPIClient   — lineups, park, weather, umpire, catcher.
- statcast.StatcastClient   — pitch-by-pitch via pybaseball, cached by game_pk.
- odds.OddsAPIClient        — pitcher_strikeouts_alternate, FD/DK, request-header
                              quota tracking. Snapshots committed to data/odds/.
"""
