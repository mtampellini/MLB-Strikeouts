"""Shared pytest fixtures for the data-layer tests."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def isolate_player_cache(tmp_path, monkeypatch):
    """Redirect the shared player-id cache to a tmp path for every test.

    Otherwise tests pollute the committed cache (and inherit pollution from
    previous runs). Tests that *want* to exercise the on-disk cache can pass
    an explicit `path=` to load_cache/save_cache.
    """
    from src.data import player_resolver

    isolated = tmp_path / "player_id_cache.json"
    monkeypatch.setattr(player_resolver, "CACHE_PATH", isolated)


@pytest.fixture(autouse=True)
def isolate_disagreement_log(tmp_path, monkeypatch):
    """Redirect the cross-source disagreement log to tmp.

    Tests that exercise the ProbablesClient against the real FG fixture
    naturally trigger legitimate FG-vs-StatsAPI disagreements (FG returns the
    bulk pitcher on opener days, StatsAPI returns the listed opener), and
    those entries would otherwise pile up in the committed log.
    """
    from src.data import probables_client

    isolated_dir = tmp_path / "raw"
    isolated_log = isolated_dir / "probables_disagreements.jsonl"
    monkeypatch.setattr(probables_client, "DATA_RAW_DIR", isolated_dir)
    monkeypatch.setattr(probables_client, "DISAGREEMENT_LOG", isolated_log)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def fangraphs_html() -> str:
    return (FIXTURES_DIR / "fangraphs" / "probables_2026-05-17.html").read_text(
        encoding="utf-8"
    )


@pytest.fixture
def fangraphs_target_date() -> date:
    return date(2026, 5, 17)


@pytest.fixture
def statsapi_schedule() -> dict:
    return json.loads(
        (FIXTURES_DIR / "statsapi" / "schedule_2026-05-17.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture
def as_of_yesterday(fangraphs_target_date) -> date:
    """One day before the fixture data's "today"."""
    from datetime import timedelta

    return fangraphs_target_date - timedelta(days=1)


@pytest.fixture
def sample_event_odds_payload() -> dict:
    """Synthetic event odds payload mimicking the real Odds API shape.

    Includes:
    - FanDuel + DraftKings (allowed)
    - Caesars (disallowed → should be dropped by the parser)
    - Two pitchers, each with several alt lines, both Over and Under
    """
    return {
        "id": "evt-test-1",
        "commence_time": "2026-05-17T23:05:00Z",
        "home_team": "Tampa Bay Rays",
        "away_team": "Miami Marlins",
        "bookmakers": [
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [
                    {
                        "key": "pitcher_strikeouts_alternate",
                        "outcomes": [
                            {"name": "Over", "description": "Drew Rasmussen", "point": 4.5, "price": -180},
                            {"name": "Under", "description": "Drew Rasmussen", "point": 4.5, "price": 145},
                            {"name": "Over", "description": "Drew Rasmussen", "point": 5.5, "price": 120},
                            {"name": "Under", "description": "Drew Rasmussen", "point": 5.5, "price": -150},
                            {"name": "Over", "description": "Drew Rasmussen", "point": 6.5, "price": 280},
                            {"name": "Over", "description": "Eury Pérez", "point": 5.5, "price": -110},
                            {"name": "Under", "description": "Eury Pérez", "point": 5.5, "price": -110},
                        ],
                    }
                ],
            },
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "pitcher_strikeouts_alternate",
                        "outcomes": [
                            {"name": "Over", "description": "Drew Rasmussen", "point": 5.5, "price": 115},
                            {"name": "Under", "description": "Drew Rasmussen", "point": 5.5, "price": -145},
                        ],
                    }
                ],
            },
            {
                "key": "caesars",
                "title": "Caesars",
                "markets": [
                    {
                        "key": "pitcher_strikeouts_alternate",
                        "outcomes": [
                            {"name": "Over", "description": "Drew Rasmussen", "point": 5.5, "price": 110},
                        ],
                    }
                ],
            },
        ],
    }
