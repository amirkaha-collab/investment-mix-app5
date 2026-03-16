"""
Analysis module: US Exposure

Rule 3 – Two methodologies:
  conservative_us_exposure : only holdings that are clearly US-domiciled
  broad_us_exposure        : adds global USD-denominated bond funds
                             if the user opted in via UserAnalysisPreferences
"""

from __future__ import annotations

from collections import defaultdict

from src.config.constants import (
    ASSET_CLASS_BOND,
    BOND_LINKAGE_USD,
    REGION_US,
)
from src.domain.models import (
    HoldingNormalized,
    USExposureSummary,
    UserAnalysisPreferences,
)

_GLOBAL_USD_BOND_HINTS = [
    "global", "גלובל", "international", "בינלאומי", "world",
    "usd bond", "dollar bond", "אגח דולר",
]


def _is_global_usd_bond(h: HoldingNormalized) -> bool:
    """Heuristic: does this look like a global USD-denominated bond fund?"""
    if h.asset_class.lower() != ASSET_CLASS_BOND:
        return False
    if h.bond_linkage_type != BOND_LINKAGE_USD:
        return False
    name_lower = h.normalized_name.lower()
    return any(hint in name_lower for hint in _GLOBAL_USD_BOND_HINTS)


def compute_us_exposure(
    holdings: list[HoldingNormalized],
    prefs: UserAnalysisPreferences,
) -> USExposureSummary:
    """
    Compute both US exposure metrics.

    Conservative:
      Sum of holdings whose primary region is USA (direct equities, pure US ETFs).

    Broad:
      Conservative + global USD bond funds (if user opted in).
    """
    total_portfolio = sum(h.market_value_ils for h in holdings)
    if total_portfolio <= 0:
        return USExposureSummary(
            conservative_us_value_ils=0,
            conservative_us_weight=0,
            broad_us_value_ils=0,
            broad_us_weight=0,
            methodology_note="Portfolio value is zero.",
        )

    conservative_value = 0.0
    broad_extra = 0.0

    for h in holdings:
        # Check geography_breakdown first (most accurate)
        if h.geography_breakdown:
            us_frac = h.geography_breakdown.get("USA", 0.0) + h.geography_breakdown.get("United States", 0.0)
            conservative_value += h.market_value_ils * us_frac
        elif h.region == REGION_US:
            conservative_value += h.market_value_ils

        # Broad extra: global USD bond funds
        if prefs.classify_global_usd_bond_as_us_exposure and _is_global_usd_bond(h):
            if h.region != REGION_US:  # not already counted
                broad_extra += h.market_value_ils

    broad_value = conservative_value + broad_extra

    note_parts = [
        "Conservative: direct US holdings + US-weighted ETF fractions.",
    ]
    if prefs.classify_global_usd_bond_as_us_exposure:
        note_parts.append(
            "Broad: adds global USD bond funds per user preference (classify_global_usd_bond_as_us_exposure=True)."
        )
    else:
        note_parts.append(
            "Broad exposure equals conservative (user did not opt in to counting global USD bond funds as US exposure)."
        )

    return USExposureSummary(
        conservative_us_value_ils=round(conservative_value, 2),
        conservative_us_weight=round(conservative_value / total_portfolio, 6),
        broad_us_value_ils=round(broad_value, 2),
        broad_us_weight=round(broad_value / total_portfolio, 6),
        methodology_note=" ".join(note_parts),
        is_estimated=any(
            "region" in h.estimated_fields or "geography_breakdown" in " ".join(h.estimated_fields)
            for h in holdings
        ),
    )
