"""Phase 4c-v2 Step 2/6: effective K rate blend.

Blends a pitcher's observed K% (per-PA terminal-event rate) with the
CSW-implied K% (linear function of CSW% derived in Step 1) using
sample-size weighting:

    csw_implied_k = csw_to_k_intercept + csw_to_k_slope * csw_pct
    w_observed = n_pa / (n_pa + k_prior_pa)
    effective_k = w_observed * observed_k + (1 - w_observed) * csw_implied_k

The motivation is structural: prior P(K|PA) fits had wrong-signed
coefficients on standalone K% / CSW% regressors because they competed
with the log5 offset for the same K-rate signal. Absorbing CSW%
INTO the K rate that enters log5 fixes the collinearity at the source.

This module exposes:

- :func:`compute_effective_k_rate` — the helper consumed by Step 3
  (projector wiring).
- :func:`load_csw_to_k_relationship` — convenience loader for the
  intercept/slope pair from Step 1's output file.

No model code is wired yet. The projector still uses observed K% directly
in its log5 offset until Step 3 lands.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_K_PRIOR_PA = 150
P_K_FLOOR = 0.05
P_K_CEIL = 0.50

CONFIDENCE_OBSERVED_DOMINANT_THRESHOLD = 0.75
CONFIDENCE_CSW_DOMINANT_THRESHOLD = 0.25

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
DEFAULT_CSW_TO_K_PATH = PROCESSED_DIR / "csw_to_k_relationship.json"


@dataclass(frozen=True)
class EffectiveKRateResult:
    """Blend output. ``effective_k_rate`` is the value the projector's
    log5 offset will consume. The other fields are diagnostic / debugging."""

    effective_k_rate: float
    observed_k_rate: float | None
    observed_n_pa: int
    csw_implied_k_rate: float | None
    blend_weight_observed: float
    confidence: str


def _classify_confidence(
    w_observed: float, observed_available: bool, csw_available: bool,
) -> str:
    if not observed_available and csw_available:
        return "csw_only_fallback"
    if not csw_available and observed_available:
        return "observed_only_no_csw"
    if w_observed >= CONFIDENCE_OBSERVED_DOMINANT_THRESHOLD:
        return "observed_dominant"
    if w_observed <= CONFIDENCE_CSW_DOMINANT_THRESHOLD:
        return "csw_dominant"
    return "balanced"


def compute_effective_k_rate(
    observed_k_rate: float | None,
    observed_n_pa: int,
    csw_pct: float | None,
    csw_to_k_intercept: float,
    csw_to_k_slope: float,
    k_prior_pa: int = DEFAULT_K_PRIOR_PA,
) -> EffectiveKRateResult | None:
    """Blend observed K% with CSW-implied K% using sample-size weighting.

    Returns ``None`` only when both ``observed_k_rate`` is None AND ``csw_pct``
    is None — the caller has no information to combine.

    See module docstring for the blend formula and confidence categories.
    """
    if observed_k_rate is None and csw_pct is None:
        return None

    csw_implied: float | None
    if csw_pct is not None:
        csw_implied = csw_to_k_intercept + csw_to_k_slope * csw_pct
    else:
        csw_implied = None

    if observed_k_rate is None:
        # CSW-only fallback
        effective = csw_implied if csw_implied is not None else 0.0
        effective = max(P_K_FLOOR, min(P_K_CEIL, effective))
        return EffectiveKRateResult(
            effective_k_rate=effective,
            observed_k_rate=None,
            observed_n_pa=int(observed_n_pa),
            csw_implied_k_rate=csw_implied,
            blend_weight_observed=0.0,
            confidence=_classify_confidence(0.0, False, True),
        )

    if csw_implied is None:
        # Observed only — no CSW signal to blend in.
        effective = max(P_K_FLOOR, min(P_K_CEIL, float(observed_k_rate)))
        return EffectiveKRateResult(
            effective_k_rate=effective,
            observed_k_rate=float(observed_k_rate),
            observed_n_pa=int(observed_n_pa),
            csw_implied_k_rate=None,
            blend_weight_observed=1.0,
            confidence=_classify_confidence(1.0, True, False),
        )

    # Both signals available — sample-size-weighted blend.
    denom = float(observed_n_pa + k_prior_pa)
    w_observed = float(observed_n_pa) / denom if denom > 0 else 0.0
    raw = w_observed * float(observed_k_rate) + (1.0 - w_observed) * csw_implied
    effective = max(P_K_FLOOR, min(P_K_CEIL, raw))
    return EffectiveKRateResult(
        effective_k_rate=effective,
        observed_k_rate=float(observed_k_rate),
        observed_n_pa=int(observed_n_pa),
        csw_implied_k_rate=csw_implied,
        blend_weight_observed=w_observed,
        confidence=_classify_confidence(w_observed, True, True),
    )


def load_csw_to_k_relationship(path: Path | str | None = None) -> tuple[float, float]:
    """Load the CSW-to-K intercept and slope produced by
    :mod:`scripts.derive_csw_to_k_relationship`. Returns
    ``(intercept, slope)``.
    """
    p = Path(path) if path is not None else DEFAULT_CSW_TO_K_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"CSW-to-K relationship file not found at {p}. "
            f"Run `python -m scripts.derive_csw_to_k_relationship` to "
            f"produce it (Phase 4c-v2 Step 1)."
        )
    blob = json.loads(p.read_text(encoding="utf-8"))
    model = blob.get("model") or {}
    if "intercept" not in model or "slope" not in model:
        raise ValueError(
            f"{p}: malformed CSW-to-K relationship — expected "
            f"`model.intercept` and `model.slope`, got keys {sorted(model)}"
        )
    return float(model["intercept"]), float(model["slope"])
