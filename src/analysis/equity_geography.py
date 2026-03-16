"""
Analysis module: Equity Geography

Rule 2 – Break down equities by region (Israel / USA / Europe / …).
For funds/ETFs with geography_breakdown populated, apply the breakdown
proportionally to the fund's market value.
"""

from __future__ import annotations

from collections import defaultdict

from src.config.constants import (
    ASSET_CLASS_EQUITY,
    REGION_UNKNOWN,
)
from src.domain.models import (
    EquityGeographyRow,
    HoldingNormalized,
    UserAnalysisPreferences,
)


def compute_equity_geography(
    holdings: list[HoldingNormalized],
    prefs: UserAnalysisPreferences,
) -> list[EquityGeographyRow]:
    """
    Returns per-region breakdown of equity holdings.

    Methodology
    -----------
    - Single equities: value attributed to their `region` field.
    - Funds with geography_breakdown: value is proportionally distributed.
    - Funds without geography_breakdown: entire value attributed to their `region`.
      This is flagged as estimated when region == Unknown.
    """
    # Total portfolio (denominator)
    total_portfolio = sum(h.market_value_ils for h in holdings)
    equities = [h for h in holdings if h.asset_class.lower() == ASSET_CLASS_EQUITY]
    total_equities = sum(h.market_value_ils for h in equities)

    if total_equities <= 0:
        return []

    region_values: dict[str, float] = defaultdict(float)

    for h in equities:
        if h.geography_breakdown:
            # Distribute according to known geography
            for region, fraction in h.geography_breakdown.items():
                region_values[region] += h.market_value_ils * fraction
        else:
            # Use the holding's region directly
            region = h.region or REGION_UNKNOWN
            region_values[region] += h.market_value_ils

    rows: list[EquityGeographyRow] = []
    for region, value in sorted(region_values.items(), key=lambda x: -x[1]):
        is_estimated = region == REGION_UNKNOWN
        rows.append(
            EquityGeographyRow(
                region=region,
                market_value_ils=round(value, 2),
                weight_in_equities=round(value / total_equities, 6),
                weight_in_portfolio=round(value / total_portfolio, 6) if total_portfolio > 0 else 0.0,
                is_estimated=is_estimated,
            )
        )
    return rows
