"""Phase 4c-v2 Step 6/6: loader for fitted P(K|PA) regression coefficients.

Reads :file:`data/processed/p_k_pa_coefficients.json` (produced by
:mod:`scripts.fit_p_k_pa_v2` --full) into a typed structure the projector
consumes to apply learned log-odds adjustments per (batter, TTO) cell.

The fit's offset already absorbs the bulk of K-skill (CSW-blended effective
K rate in log5). These coefficients are therefore small log-odds shifts on
features orthogonal to skill:

- pitcher_velocity_trend_z: form-vs-baseline
- log_park_k_factor_by_hand: venue effect
- 19 archetype-TTO interaction shifts: matchup decay

(Balanced, TTO=1) is the absolute reference cell; its coefficient is
implicitly 0 (not stored in the fit's coefficients dict).

The loader REFUSES to load any fit whose gates_failed array is non-empty —
shipping a broken fit silently is the failure mode this guards against.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
DEFAULT_PATH = PROCESSED_DIR / "p_k_pa_coefficients.json"


@dataclass(frozen=True)
class FittedKPACoefficients:
    """Typed view of a fit_p_k_pa_v2 --full output that passed all gates."""

    intercept: float
    pitcher_velocity_trend_z: float
    log_park_k_factor_by_hand: float
    # archetype name -> tto bucket (1..4) -> coefficient. (Balanced, TTO=1)
    # is omitted (coefficient is 0 by construction — reference cell).
    archetype_tto: dict[str, dict[int, float]]

    # Provenance + summary metadata
    model_type: str
    rate_space_r2_out: float
    leakage_shuffle_delta: float
    generated_at: str

    def get_archetype_tto_shift(self, archetype: str, tto: int) -> float:
        """Lookup the additive log-odds shift for (archetype, tto).

        Returns 0.0 for the reference cell (Balanced, TTO=1) and for any
        (archetype, tto) pair missing from the fit (defensive).
        """
        return self.archetype_tto.get(archetype, {}).get(int(tto), 0.0)


def load_fitted_kpa_coefficients(
    path: Path | str | None = None,
) -> FittedKPACoefficients:
    """Load fitted coefficients from p_k_pa_coefficients.json.

    Raises :class:`FileNotFoundError` when the file is missing.
    Raises :class:`ValueError` when the fit's ``gates_failed`` array is
    non-empty (we refuse to ship a fit that didn't pass gates).
    """
    p = Path(path) if path is not None else DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Fitted KPA coefficients not found at {p}. "
            f"Run `python -m scripts.fit_p_k_pa_v2 --full` (Phase 4c-v2 Step 5) "
            f"to produce it."
        )
    blob = json.loads(p.read_text(encoding="utf-8"))

    gates_failed = blob.get("gates_failed") or []
    if gates_failed:
        raise ValueError(
            f"Refusing to load {p}: gates_failed is non-empty: {gates_failed}"
        )

    coefs = blob.get("coefficients") or {}
    if "pitcher_velocity_trend_z" not in coefs or "log_park_k_factor_by_hand" not in coefs:
        raise ValueError(
            f"{p}: malformed coefficients — missing one of "
            f"pitcher_velocity_trend_z / log_park_k_factor_by_hand"
        )

    velocity_z = float(coefs["pitcher_velocity_trend_z"]["value"])
    log_park = float(coefs["log_park_k_factor_by_hand"]["value"])

    archetype_tto: dict[str, dict[int, float]] = {}
    for key, entry in coefs.items():
        if not key.startswith("arch_") or "_tto_" not in key:
            continue
        # arch_<archetype>_tto_<n> where <archetype> may contain hyphens.
        head, tail = key.split("_tto_", 1)
        archetype = head[len("arch_"):]
        try:
            tto = int(tail)
        except ValueError:
            continue
        archetype_tto.setdefault(archetype, {})[tto] = float(entry["value"])

    return FittedKPACoefficients(
        intercept=float(blob.get("intercept") or 0.0),
        pitcher_velocity_trend_z=velocity_z,
        log_park_k_factor_by_hand=log_park,
        archetype_tto=archetype_tto,
        model_type=str(blob.get("model_type") or ""),
        rate_space_r2_out=float(blob.get("rate_space_r2_weighted_out") or 0.0),
        leakage_shuffle_delta=float(blob.get("leakage_shuffle_delta") or 0.0),
        generated_at=str(blob.get("generated_at") or ""),
    )
