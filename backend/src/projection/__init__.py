"""Projection layer: E[K], NB CDF, P(K >= line).

BOOK-AGNOSTIC. Must not import from src.picks. Enforced by
tests/test_scaffolding.py::test_projection_does_not_import_picks.

Built in Phases 3 + 5:
- combine.project_bf      — additive E[BF] with clip [12, 32].
- combine.project_k_rate  — multiplicative P(K|PA) with clip [0.10, 0.45].
- combine.project_k       — E[K] = E[BF] x P(K|PA).
- distribution.nb_cdf     — NB CDF using locked dispersion alpha from
                            data/processed/nb_dispersion.json.
- distribution.p_at_least — P(K >= line) for every alt line.
"""
