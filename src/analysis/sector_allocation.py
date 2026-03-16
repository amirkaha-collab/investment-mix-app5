"""
Analysis module: Sector Allocation

Rule 4 – Break equities by GICS sector.

For single equities: use `sector` field directly.
For ETFs/funds with sector_breakdown: distribute proportionally.
For broad market ETFs with no breakdown: use known index weights from constants.
"""

from __future__ import annotations

from collections import defaultdict

from src.config.constants import (
    ASSET_CLASS_EQUITY,
    SECTOR_UNKNOWN,
)
from src.domain.models import (
    HoldingNormalized,
    SectorAllocationRow,
    UserAnalysisPreferences,
)


def compute_sector_allocation(
    holdings: list[HoldingNormalized],
    prefs: UserAnalysisPreferences,
) -> list[SectorAllocationRow]:
    """
    Returns sector allocation across all equity holdings.
    Funds are broken down via their sector_breakdown map if available.
    """
    equities = [h for h in holdings if h.asset_class.lower() == ASSET_CLASS_EQUITY]
    total_equities = sum(h.market_value_ils for h in equities)
    if total_equities <= 0:
        return []

    # sector -> (value, is_estimated)
    sector_values: dict[str, float] = defaultdict(float)
    sector_estimated: dict[str, bool] = defaultdict(bool)
    sector_notes: dict[str, list[str]] = defaultdict(list)

    for h in equities:
        if h.sector_breakdown:
            # High-quality: fund has a known sector breakdown
            for sector, fraction in h.sector_breakdown.items():
                sector_values[sector] += h.market_value_ils * fraction
                sector_notes[sector].append(f"{h.normalized_name} ({fraction:.1%})")
        elif h.sector and h.sector != SECTOR_UNKNOWN:
            # Single equity or fund with known sector
            sector_values[h.sector] += h.market_value_ils
            sector_notes[h.sector].append(h.normalized_name)
        else:
            # No sector information – bucket as Unknown
            sector_values[SECTOR_UNKNOWN] += h.market_value_ils
            sector_estimated[SECTOR_UNKNOWN] = True
            sector_notes[SECTOR_UNKNOWN].append(
                f"{h.normalized_name} [no sector data]"
            )

    rows: list[SectorAllocationRow] = []
    for sector, value in sorted(sector_values.items(), key=lambda x: -x[1]):
        rows.append(
            SectorAllocationRow(
                sector=sector,
                market_value_ils=round(value, 2),
                weight_in_equities=round(value / total_equities, 6),
                source_note="; ".join(sector_notes.get(sector, [])),
                is_estimated=sector_estimated.get(sector, False),
            )
        )
    return rows
