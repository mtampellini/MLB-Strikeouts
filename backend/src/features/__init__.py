"""Feature builders for E[BF] (additive) and P(K|PA) (multiplicative).

Built in Phase 3. Every feature accepts an AsOfContext and returns NaN when
data is missing — no median fill, ever. Skip-logic in src.projection turns a
missing feature into a hard pitcher skip.
"""
