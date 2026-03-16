"""
Analysis module: Bond Breakdown + Weighted Duration

Rule 5 – Break bonds by linkage type (CPI, nominal ILS, USD, global, etc.)
Rule 6 – Compute conservative and extended weighted duration.

Duration is only included in the conservative metric if:
  - It came from an official source (not estimated)
  - OR the holding's duration_source is non-empty and confidence is high

Extended duration includes estimated durations when
  prefs.compute_extended_duration_with_estimates is True.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from src.config.constants import (
    ASSET_CLASS_BOND,
    BOND_LINKAGE_UNKNOWN,
)
from src.domain.models import (
    BondBreakdownRow,
    DurationRow,
    HoldingNormalized,
    UserAnalysisPreferences,
)


def compute_bond_breakdown(
    holdings: list[HoldingNormalized],
    prefs: UserAnalysisPreferences,
) -> list[BondBreakdownRow]:
    """Break bonds into linkage-type buckets."""
    bonds = [h for h in holdings if h.asset_class.lower() == ASSET_CLASS_BOND]
    total_bonds = sum(h.market_value_ils for h in bonds)
    total_portfolio = sum(h.market_value_ils for h in holdings)

    if total_bonds <= 0:
        return []

    linkage_values: dict[str, float] = defaultdict(float)
    for h in bonds:
        lt = h.bond_linkage_type or BOND_LINKAGE_UNKNOWN
        linkage_values[lt] += h.market_value_ils

    rows: list[BondBreakdownRow] = []
    for lt, value in sorted(linkage_values.items(), key=lambda x: -x[1]):
        rows.append(
            BondBreakdownRow(
                linkage_type=lt,
                market_value_ils=round(value, 2),
                weight_in_bonds=round(value / total_bonds, 6),
                weight_in_portfolio=round(value / total_portfolio, 6) if total_portfolio > 0 else 0.0,
                is_estimated=False,
            )
        )
    return rows


def compute_duration_table(
    holdings: list[HoldingNormalized],
    prefs: UserAnalysisPreferences,
) -> tuple[list[DurationRow], Optional[float], Optional[float]]:
    """
    Build the duration table and compute conservative & extended WAD.

    Returns
    -------
    (duration_rows, conservative_wad, extended_wad)
    """
    bonds = [h for h in holdings if h.asset_class.lower() == ASSET_CLASS_BOND]
    total_bonds = sum(h.market_value_ils for h in bonds)
    if total_bonds <= 0:
        return [], None, None

    rows: list[DurationRow] = []
    conservative_numerator = 0.0
    conservative_denominator = 0.0
    extended_numerator = 0.0
    extended_denominator = 0.0

    for h in bonds:
        is_estimated = any("duration" in ef for ef in h.estimated_fields)
        weight = h.market_value_ils / total_bonds if total_bonds > 0 else 0.0
        weighted_contribution = None

        if h.duration is not None:
            weighted_contribution = h.duration * weight

            # Conservative: only official sources
            if not is_estimated:
                conservative_numerator += h.duration * h.market_value_ils
                conservative_denominator += h.market_value_ils

            # Extended: include estimated if user opted in
            if not is_estimated or prefs.compute_extended_duration_with_estimates:
                extended_numerator += h.duration * h.market_value_ils
                extended_denominator += h.market_value_ils

        rows.append(
            DurationRow(
                row_id=h.row_id,
                name=h.normalized_name,
                bond_linkage_type=h.bond_linkage_type or BOND_LINKAGE_UNKNOWN,
                market_value_ils=round(h.market_value_ils, 2),
                duration=h.duration,
                duration_source=h.duration_source,
                is_estimated=is_estimated,
                weighted_contribution=round(weighted_contribution, 4)
                if weighted_contribution is not None else None,
            )
        )

    conservative_wad = (
        round(conservative_numerator / conservative_denominator, 2)
        if conservative_denominator > 0 else None
    )
    extended_wad = (
        round(extended_numerator / extended_denominator, 2)
        if extended_denominator > 0 else None
    )

    return rows, conservative_wad, extended_wad
