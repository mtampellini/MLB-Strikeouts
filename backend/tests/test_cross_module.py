"""Cross-module tests: ProbablesClient + OddsAPIClient feeding opener_detection."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.data.odds_client import OddsAPIClient, parse_event_odds_response
from src.data.opener_detection import (
    BulkPitcherCandidate,
    BulkPitcherResult,
    NoOverride,
    PitcherSeasonStats,
    check_opener,
)
from src.data.probables_client import ProbablesClient


_FG_HTML_WITH_LISTED_PROBABLE = '''
<html><body><script id="__NEXT_DATA__" type="application/json">{
  "props": {"pageProps": {"dehydratedState": {"mutations": [], "queries": [
    {"queryKey": ["roster-resource/probables-grid/data"],
     "state": {"data": {"games": [
       {"gameDate": "2026-05-17", "abbName": "TBR", "isHome": true, "teamId": 1,
        "team":     {"opener": null, "primaryPitcher": null,
                     "sp": {"playerId": "999", "name": "Tampa Opener", "throws": "R"}},
        "opponent": {"abbName": "TOR", "opener": null, "primaryPitcher": null, "sp": null}}
     ]}}}
  ]}}}
}</script></body></html>
'''


_ODDS_RESPONSE = {
    "id": "evt-tbr",
    "commence_time": "2026-05-17T23:05:00Z",
    "home_team": "Tampa Bay Rays",
    "away_team": "Toronto Blue Jays",
    "bookmakers": [
        {
            "key": "fanduel",
            "markets": [
                {
                    "key": "pitcher_strikeouts_alternate",
                    "outcomes": [
                        # Only Bulk Reliever has a posted prop — Tampa Opener does NOT.
                        {"name": "Over",  "description": "Bulk Reliever", "point": 4.5, "price": -150},
                        {"name": "Under", "description": "Bulk Reliever", "point": 4.5, "price": 120},
                    ],
                }
            ],
        }
    ],
}


def _fake_lookup(first, last):
    table = {
        ("tampa", "opener"): 9001,
        ("bulk", "reliever"): 8002,
    }
    key = (first.lower(), last.lower())
    if key in table:
        return pd.DataFrame([{"key_mlbam": table[key]}])
    return pd.DataFrame()


def test_market_signal_trigger_fires_from_real_pipeline_data(tmp_path):
    """Probables + Odds together produce the 'no posted prop' opener trigger.

    Listed probable (Tampa Opener) has no K prop. A teammate (Bulk Reliever)
    has one. opener_detection should classify this as opener day and route
    picks to the bulk reliever.
    """
    probables_client = ProbablesClient(
        fetch_fangraphs=lambda d: _FG_HTML_WITH_LISTED_PROBABLE,
        fetch_statsapi_schedule=lambda d: {"dates": []},  # no SA cross-check
        player_lookup_fn=_fake_lookup,
        cache_dir=tmp_path,
    )
    probables = probables_client.fetch(cutoff_date=date(2026, 5, 17))
    assert len(probables) == 1
    listed = probables[0]
    assert listed.pitcher_mlbam_id == 9001

    odds_client = OddsAPIClient(
        api_key="x",
        snapshot_dir=tmp_path / "odds",
        player_lookup_fn=_fake_lookup,
    )
    # Parse the odds response directly (the live-fetch wrapper isn't needed here).
    event = parse_event_odds_response(_ODDS_RESPONSE, resolve_id=odds_client._resolve_id)
    posted_ids = odds_client.posted_k_prop_pitchers([event])
    assert 8002 in posted_ids and 9001 not in posted_ids

    # Now feed both into the opener-detection logic. Use neutral season stats
    # so trigger #1 (low-volume) doesn't fire — we want to isolate the market
    # signal trigger.
    decision = check_opener(
        listed_probable_id=listed.pitcher_mlbam_id,
        listed_probable_stats=PitcherSeasonStats(
            pitcher_mlbam_id=listed.pitcher_mlbam_id, season_ip=120.0, starts=20
        ),
        team_rotation_slot=None,
        posted_k_prop_pitchers=posted_ids,
        team_pitchers={9001, 8002},
        bulk_candidates=[
            BulkPitcherCandidate(pitcher_mlbam_id=8002, season_ip=45.0, recent_bulk_relief_ip=4.0),
        ],
    )
    assert isinstance(decision, BulkPitcherResult)
    assert decision.use_pitcher == 8002
    assert "market signal" in decision.reason


def test_listed_probable_with_posted_prop_does_not_trigger_market_override(tmp_path):
    """Sanity check: when the listed probable HAS a posted prop, the market
    signal trigger should NOT fire, and a normal-volume starter should pass
    through with NoOverride."""
    odds_with_listed = {
        **_ODDS_RESPONSE,
        "bookmakers": [
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "pitcher_strikeouts_alternate",
                        "outcomes": [
                            {"name": "Over", "description": "Tampa Opener", "point": 4.5, "price": -150},
                            {"name": "Under", "description": "Tampa Opener", "point": 4.5, "price": 120},
                        ],
                    }
                ],
            }
        ],
    }
    odds_client = OddsAPIClient(api_key="x", player_lookup_fn=_fake_lookup)
    event = parse_event_odds_response(odds_with_listed, resolve_id=odds_client._resolve_id)
    posted = odds_client.posted_k_prop_pitchers([event])
    assert posted == {9001}
    decision = check_opener(
        listed_probable_id=9001,
        listed_probable_stats=PitcherSeasonStats(
            pitcher_mlbam_id=9001, season_ip=120.0, starts=20
        ),
        team_rotation_slot=None,
        posted_k_prop_pitchers=posted,
        team_pitchers={9001, 8002},
        bulk_candidates=[],
    )
    assert isinstance(decision, NoOverride)
