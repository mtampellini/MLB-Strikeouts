"""Picks layer: edge, EV, three-tier selection. FD/DK-specific.

Consumes projections from src.projection. The only place book-specific logic
(devig, edge thresholds, EV calc, tier ranking) lives.

Built in Phase 5:
- devig.devig_multiplicative   — same method as HR repo.
- edge.compute_edge            — (model_p - market_p) / market_p.
- ev.compute_ev_pct            — using book price.
- tiers.classify               — primary / secondary / shadow with thresholds:
    Primary:   edge_pct >= 20% AND book_price >= -180, rank by edge_pct.
    Secondary: edge_pct >= 20% AND (book_price < -180 OR rank 11+ in primary).
    Shadow:    10% <= edge_pct < 20%.
"""
