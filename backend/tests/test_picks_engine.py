"""Phase 5: integration tests for the picks engine."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.picks.engine import PickResult, generate_picks
from src.picks.output import write_picks
from src.projection.inputs import ProjectionBundle, ProjectionContext


PHASE3_V2C_I_SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "data" / "phase3_v2c_i_sample_bundle_2026-05-17_656876.json"
)
STEP4_SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "data" / "phase4c_v2_step4_sample_bundle_2026-05-17_656876.json"
)


def _ctx() -> ProjectionContext:
    return ProjectionContext.from_default_paths()


def _bundle_with_synthetic_market(
    fanduel_lines: list[dict] | None = None,
    draftkings_lines: list[dict] | None = None,
) -> ProjectionBundle:
    """Load the Step 4 sample bundle and overlay synthetic market data."""
    if not STEP4_SAMPLE.exists():
        pytest.skip(f"{STEP4_SAMPLE} not present")
    blob = json.loads(STEP4_SAMPLE.read_text(encoding="utf-8"))
    blob = copy.deepcopy(blob)
    blob["market"] = {
        "fanduel": {
            "available": bool(fanduel_lines),
            "lines": fanduel_lines or [],
        },
        "draftkings": {
            "available": bool(draftkings_lines),
            "lines": draftkings_lines or [],
        },
        "snapshot_timestamp": "2026-05-17T15:30:00+00:00",
        "snapshot_source": "live",
    }
    return ProjectionBundle.from_dict(blob)


# ---- Happy path -----------------------------------------------------------


def test_engine_produces_picks_with_edge_above_shadow():
    """Build a bundle with deliberately-bad book odds so the model finds edge."""
    # Rasmussen e_k ~5.7 with fitted coefs. Book setting Over 5.5 at +200
    # implies market_p ~33% raw, devigged ~33% — but our model would say
    # P(K>=6) ~ 0.50, giving big edge.
    lines_fd = [
        {"line": 5.5, "side": "Over", "price": +200},
        {"line": 5.5, "side": "Under", "price": -250},
    ]
    bundle = _bundle_with_synthetic_market(fanduel_lines=lines_fd)
    result = generate_picks([bundle], ctx=_ctx())
    assert isinstance(result, PickResult)
    # We expect at least one pick somewhere (probably primary or secondary)
    total = result.primary + result.secondary + result.shadow
    assert len(total) > 0, "expected at least one pick"
    for p in total:
        assert "pitcher_mlbam_id" in p
        assert "edge_pct" in p
        assert "calibration_note" in p
        assert "pick_id" in p


def test_engine_skips_pitchers_with_no_market_data():
    bundle = _bundle_with_synthetic_market()  # no lines either side
    result = generate_picks([bundle], ctx=_ctx())
    assert result.primary == []
    assert result.secondary == []
    assert result.shadow == []
    assert len(result.skipped) == 1
    assert "no_market_data" in result.skipped[0]["reason"]


def test_engine_evaluates_both_books_independently():
    """Pitcher with edges on both FD and DK should generate picks on both."""
    lines_fd = [
        {"line": 5.5, "side": "Over", "price": +200},
        {"line": 5.5, "side": "Under", "price": -250},
    ]
    lines_dk = [
        {"line": 5.5, "side": "Over", "price": +180},
        {"line": 5.5, "side": "Under", "price": -220},
    ]
    bundle = _bundle_with_synthetic_market(
        fanduel_lines=lines_fd, draftkings_lines=lines_dk,
    )
    result = generate_picks([bundle], ctx=_ctx())
    books = {p["book"] for p in result.primary + result.secondary + result.shadow}
    assert "fanduel" in books and "draftkings" in books, (
        f"expected picks at both books, got {books}"
    )


def test_engine_one_sided_line_uses_imputation():
    """A one-sided line gets devigged via nearest-pair imputation. The
    resulting pick records devig_source='imputed_nearest_pair'."""
    lines_fd = [
        {"line": 5.5, "side": "Over", "price": -110},
        {"line": 5.5, "side": "Under", "price": -110},  # paired
        {"line": 6.5, "side": "Over", "price": +200},   # one-sided, imputable
    ]
    bundle = _bundle_with_synthetic_market(fanduel_lines=lines_fd)
    result = generate_picks([bundle], ctx=_ctx())
    all_picks = result.primary + result.secondary + result.shadow
    sources = {p["devig_source"] for p in all_picks if p["line"] == 6.5}
    # If the 6.5 line generated a pick (depending on edge), source should be imputed
    if sources:
        assert "imputed_nearest_pair" in sources


def test_engine_skips_orphan_one_sided_line():
    """A one-sided line with no paired line within 1.5 K is skipped."""
    lines_fd = [
        {"line": 6.5, "side": "Over", "price": +200},  # one-sided, no paired
    ]
    bundle = _bundle_with_synthetic_market(fanduel_lines=lines_fd)
    result = generate_picks([bundle], ctx=_ctx())
    # No paired line -> devig returns None -> no picks generated. Pitcher
    # also isn't "skipped" in the bundle-level sense (market WAS available),
    # but no picks land for it.
    all_picks = result.primary + result.secondary + result.shadow
    assert len(all_picks) == 0


def test_engine_transient_skip_when_lineup_not_posted():
    """Bundle with opposing_lineup.lineup_posted=False -> transient skip,
    no projection attempted, no picks generated."""
    if not STEP4_SAMPLE.exists():
        pytest.skip(f"{STEP4_SAMPLE} not present")
    blob = json.loads(STEP4_SAMPLE.read_text(encoding="utf-8"))
    blob = copy.deepcopy(blob)
    # Posted lineup requires batters; un-posting requires empty batters
    blob["opposing_lineup"]["lineup_posted"] = False
    blob["opposing_lineup"]["batters"] = []
    blob["market"] = {
        "fanduel": {"available": True, "lines": [
            {"line": 5.5, "side": "Over", "price": +200},
            {"line": 5.5, "side": "Under", "price": -250},
        ]},
        "draftkings": {"available": False, "lines": []},
        "snapshot_timestamp": "2026-05-17T15:30:00+00:00",
        "snapshot_source": "live",
    }
    bundle = ProjectionBundle.from_dict(blob)
    result = generate_picks([bundle], ctx=_ctx())
    assert result.primary == []
    assert result.secondary == []
    assert result.shadow == []
    assert len(result.skipped) == 1
    skip = result.skipped[0]
    assert skip["reason"] == "lineup_not_posted"
    assert skip["is_transient"] is True
    assert "re-evaluate" in skip["detail"]


def test_engine_lineup_posted_generates_picks_normally():
    """Bundle with lineup_posted=True (the loaded sample's default) and
    real features generates picks like the happy-path test."""
    lines_fd = [
        {"line": 5.5, "side": "Over", "price": +200},
        {"line": 5.5, "side": "Under", "price": -250},
    ]
    bundle = _bundle_with_synthetic_market(fanduel_lines=lines_fd)
    assert bundle.opposing_lineup.lineup_posted is True  # baseline guard
    result = generate_picks([bundle], ctx=_ctx())
    total = result.primary + result.secondary + result.shadow
    assert len(total) > 0
    # No transient skips when lineup is posted
    assert all(not s.get("is_transient") for s in result.skipped)


def test_engine_idempotent_re_eval_when_lineup_posts_later():
    """Same pitcher: 11am run has lineup_posted=False (transient skip,
    no picks); 12pm run has lineup_posted=True (picks generated). The
    11am run's skip record doesn't bleed into the 12pm picks — each
    call to generate_picks is stateless w.r.t. prior runs, so the
    12pm result's primary/secondary/shadow are populated solely by the
    current bundle. Idempotency at the ledger level is the orchestrator's
    job (covered in test_pipeline_orchestration); here we verify the
    engine itself doesn't carry state."""
    if not STEP4_SAMPLE.exists():
        pytest.skip(f"{STEP4_SAMPLE} not present")
    blob = json.loads(STEP4_SAMPLE.read_text(encoding="utf-8"))

    # 11am run: lineup not yet posted
    blob_11 = copy.deepcopy(blob)
    blob_11["opposing_lineup"]["lineup_posted"] = False
    blob_11["opposing_lineup"]["batters"] = []
    blob_11["market"] = {
        "fanduel": {"available": True, "lines": [
            {"line": 5.5, "side": "Over", "price": +200},
            {"line": 5.5, "side": "Under", "price": -250},
        ]},
        "draftkings": {"available": False, "lines": []},
        "snapshot_timestamp": "2026-05-17T15:00:00+00:00",
        "snapshot_source": "live",
    }
    r_11 = generate_picks([ProjectionBundle.from_dict(blob_11)], ctx=_ctx())
    assert len(r_11.skipped) == 1
    assert r_11.skipped[0]["reason"] == "lineup_not_posted"
    assert (r_11.primary, r_11.secondary, r_11.shadow) == ([], [], [])

    # 12pm run: same pitcher, lineup posted, picks generated
    blob_12 = copy.deepcopy(blob)  # original sample has lineup_posted=True
    blob_12["market"] = blob_11["market"]
    r_12 = generate_picks([ProjectionBundle.from_dict(blob_12)], ctx=_ctx())
    total_12 = r_12.primary + r_12.secondary + r_12.shadow
    assert len(total_12) > 0
    # 12pm result carries no transient skip
    assert all(s.get("reason") != "lineup_not_posted" for s in r_12.skipped)

    # Across runs, the SAME pick_id appears in 12pm (idempotency hook):
    # if 11am's lineup-pending skip ever did emit picks, those pick_ids
    # would conflict. They don't, so 12pm's pick set stands alone.
    pick_ids_12 = {p["pick_id"] for p in total_12}
    # 11am didn't emit picks for this pitcher, so no overlap to worry about
    assert all(pid not in {} for pid in pick_ids_12)  # trivially true; documents intent


def test_engine_skips_projector_failure():
    """If project() returns skipped, the bundle is added to skipped[]."""
    if not STEP4_SAMPLE.exists():
        pytest.skip(f"{STEP4_SAMPLE} not present")
    blob = json.loads(STEP4_SAMPLE.read_text(encoding="utf-8"))
    blob = copy.deepcopy(blob)
    # Force projector skip by emptying career data (career IP filter trips)
    blob["pitcher"]["statcast_pitches_30d"] = []
    blob["pitcher"]["statcast_pitches_season"] = []
    blob["pitcher"]["statcast_pitches_prior_year"] = []
    bundle = ProjectionBundle.from_dict(blob)
    result = generate_picks([bundle], ctx=_ctx())
    assert len(result.skipped) == 1
    assert "projector_skipped" in result.skipped[0]["reason"]


# ---- Schema invariants ----------------------------------------------------


def test_each_pick_has_canonical_schema():
    lines_fd = [
        {"line": 5.5, "side": "Over", "price": +200},
        {"line": 5.5, "side": "Under", "price": -250},
    ]
    bundle = _bundle_with_synthetic_market(fanduel_lines=lines_fd)
    result = generate_picks([bundle], ctx=_ctx())
    required = {
        "pitcher_mlbam_id", "pitcher_name", "game_pk", "game_date",
        "line", "side", "book", "american_odds", "model_p", "market_p",
        "edge_pct", "ev_pct", "tier", "rank_in_tier", "devig_source",
        "model_e_k", "model_e_bf", "pitcher_archetype",
        "park_k_factor_by_hand", "calibration_note", "pick_id",
        "generated_at",
    }
    for pick in result.primary + result.secondary + result.shadow:
        missing = required - set(pick.keys())
        assert not missing, f"pick missing fields: {missing}"


def test_pick_id_is_stable():
    """Same (pitcher, game, line, side, book) tuple -> same pick_id across runs."""
    lines_fd = [
        {"line": 5.5, "side": "Over", "price": +200},
        {"line": 5.5, "side": "Under", "price": -250},
    ]
    bundle = _bundle_with_synthetic_market(fanduel_lines=lines_fd)
    r1 = generate_picks([bundle], ctx=_ctx())
    r2 = generate_picks([bundle], ctx=_ctx())
    ids1 = {p["pick_id"] for p in r1.primary + r1.secondary + r1.shadow}
    ids2 = {p["pick_id"] for p in r2.primary + r2.secondary + r2.shadow}
    assert ids1 == ids2


def test_metadata_counts_match():
    lines_fd = [
        {"line": 5.5, "side": "Over", "price": +200},
        {"line": 5.5, "side": "Under", "price": -250},
    ]
    bundle = _bundle_with_synthetic_market(fanduel_lines=lines_fd)
    result = generate_picks([bundle], ctx=_ctx())
    md = result.metadata
    assert md["n_primary"] == len(result.primary)
    assert md["n_secondary"] == len(result.secondary)
    assert md["n_shadow"] == len(result.shadow)
    assert md["n_skipped"] == len(result.skipped)
    assert md["n_bundles"] == 1


# ---- Output writer --------------------------------------------------------


def test_write_picks_writes_four_files(tmp_path):
    lines_fd = [
        {"line": 5.5, "side": "Over", "price": +200},
        {"line": 5.5, "side": "Under", "price": -250},
    ]
    bundle = _bundle_with_synthetic_market(fanduel_lines=lines_fd)
    result = generate_picks([bundle], ctx=_ctx())
    paths = write_picks(result, tmp_path)
    assert set(paths.keys()) == {"primary", "secondary", "shadow", "debug"}
    for name, p in paths.items():
        assert p.exists(), f"{name} file not written"
        blob = json.loads(p.read_text(encoding="utf-8"))
        if name == "debug":
            for k in ("metadata", "primary", "secondary", "shadow", "skipped"):
                assert k in blob
        else:
            assert "metadata" in blob and "picks" in blob


# ---- Decoupling test ------------------------------------------------------


def test_projection_does_not_import_picks_static_check():
    """The AST-walk test in test_scaffolding enforces this; restate here
    for human readers of the picks-test file."""
    import ast
    import pathlib
    projection_dir = (
        pathlib.Path(__file__).resolve().parents[1] / "src" / "projection"
    )
    offending = []
    for py in projection_dir.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("src.picks"):
                        offending.append((py, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").startswith("src.picks"):
                    offending.append((py, node.module))
    assert not offending, f"projection imports picks: {offending}"
