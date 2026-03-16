"""
Analysis module: Asset Allocation

Rule 1 – Split holdings into equity / bond / cash / other.
Reports value (ILS) and percentage of total portfolio.
"""

from __future__ import annotations

from src.config.constants import (
    ASSET_CLASS_BOND,
    ASSET_CLASS_CASH,
    ASSET_CLASS_EQUITY,
    ASSET_CLASS_OTHER,
    ASSET_CLASSES,
)
from src.domain.models import (
    AssetAllocationRow,
    HoldingNormalized,
    UserAnalysisPreferences,
)


def compute_asset_allocation(
    holdings: list[HoldingNormalized],
    prefs: UserAnalysisPreferences,
) -> list[AssetAllocationRow]:
    """
    Compute top-level asset allocation.

    If `prefs.include_cash_in_allocation` is False, cash is excluded from
    the denominator and not shown as a separate row.

    Returns
    -------
    list[AssetAllocationRow]  – one row per asset class, sorted by value desc.
    """
    totals: dict[str, float] = {ac: 0.0 for ac in ASSET_CLASSES}

    for h in holdings:
        ac = h.asset_class.lower()
        if ac not in totals:
            ac = ASSET_CLASS_OTHER
        totals[ac] += h.market_value_ils

    # Apply cash exclusion if requested
    if not prefs.include_cash_in_allocation:
        totals.pop(ASSET_CLASS_CASH, None)

    grand_total = sum(totals.values())
    if grand_total <= 0:
        return []

    rows: list[AssetAllocationRow] = []
    for ac, value in sorted(totals.items(), key=lambda x: -x[1]):
        if value == 0:
            continue
        rows.append(
            AssetAllocationRow(
                asset_class=ac,
                market_value_ils=round(value, 2),
                weight=round(value / grand_total, 6),
            )
        )
    return rows
