"""
QA Engine – validates internal consistency of computed analysis.

All rules emit either warnings (non-blocking) or errors (block PPTX generation).
Nothing here crashes the app; instead warnings and errors accumulate
in AnalysisOutputs.qa_warnings / qa_errors.
"""

from __future__ import annotations

import math
from typing import Optional

from src.config.constants import QA_WEIGHT_SUM_TOLERANCE
from src.domain.models import (
    AnalysisOutputs,
    AssetAllocationRow,
    BondBreakdownRow,
    DurationRow,
    EquityGeographyRow,
    FundCostRow,
    HoldingNormalized,
    SectorAllocationRow,
    UserAnalysisPreferences,
)


class QAEngine:
    """
    Run all QA checks against an AnalysisOutputs object.
    Populates qa_warnings and qa_errors in-place.
    """

    def run(
        self,
        outputs: AnalysisOutputs,
        holdings: list[HoldingNormalized],
        prefs: UserAnalysisPreferences,
    ) -> None:
        self._check_holdings_total(outputs, holdings)
        self._check_asset_allocation_sum(outputs)
        self._check_equity_geography_sum(outputs)
        self._check_sector_sum(outputs)
        self._check_bond_breakdown_sum(outputs)
        self._check_duration_denominator(outputs)
        self._check_fund_cost_calculation(outputs)
        self._check_total_cost_policy(outputs, prefs)
        self._check_estimated_fields_labeled(outputs, holdings)
        self._check_mandatory_pptx_fields(outputs)

    # ──────────────────────────────────────────────────────────────────────────

    def _check_holdings_total(
        self, outputs: AnalysisOutputs, holdings: list[HoldingNormalized]
    ) -> None:
        computed = sum(h.market_value_ils for h in holdings)
        reported = outputs.total_portfolio_value_ils
        if reported > 0 and abs(computed - reported) / reported > 0.02:
            outputs.qa_warnings.append(
                f"Holdings sum ({computed:,.0f} ILS) differs from reported total "
                f"({reported:,.0f} ILS) by more than 2%."
            )

    def _check_asset_allocation_sum(self, outputs: AnalysisOutputs) -> None:
        total_w = sum(r.weight for r in outputs.asset_allocation)
        if outputs.asset_allocation and abs(total_w - 1.0) > QA_WEIGHT_SUM_TOLERANCE:
            outputs.qa_warnings.append(
                f"Asset allocation weights sum to {total_w:.4f} (expected 1.000 ± {QA_WEIGHT_SUM_TOLERANCE})."
            )

    def _check_equity_geography_sum(self, outputs: AnalysisOutputs) -> None:
        total_w = sum(r.weight_in_equities for r in outputs.equity_geography)
        if outputs.equity_geography and abs(total_w - 1.0) > QA_WEIGHT_SUM_TOLERANCE:
            outputs.qa_warnings.append(
                f"Equity geography weights sum to {total_w:.4f} (expected ~1.000)."
            )

    def _check_sector_sum(self, outputs: AnalysisOutputs) -> None:
        total_w = sum(r.weight_in_equities for r in outputs.sector_allocation)
        if outputs.sector_allocation and abs(total_w - 1.0) > QA_WEIGHT_SUM_TOLERANCE:
            outputs.qa_warnings.append(
                f"Sector allocation weights sum to {total_w:.4f} (expected ~1.000)."
            )

    def _check_bond_breakdown_sum(self, outputs: AnalysisOutputs) -> None:
        total_w = sum(r.weight_in_bonds for r in outputs.bond_breakdown)
        if outputs.bond_breakdown and abs(total_w - 1.0) > QA_WEIGHT_SUM_TOLERANCE:
            outputs.qa_warnings.append(
                f"Bond breakdown weights sum to {total_w:.4f} (expected ~1.000)."
            )

    def _check_duration_denominator(self, outputs: AnalysisOutputs) -> None:
        """Verify that duration contributions sum correctly to the reported WAD."""
        rows_with_dur = [r for r in outputs.duration_table if r.weighted_contribution is not None and not r.is_estimated]
        if not rows_with_dur or outputs.conservative_weighted_duration is None:
            return

        total_bond_value = sum(r.market_value_ils for r in rows_with_dur)
        if total_bond_value <= 0:
            return

        recomputed = sum(
            r.duration * r.market_value_ils
            for r in rows_with_dur
            if r.duration is not None
        ) / total_bond_value

        if abs(recomputed - outputs.conservative_weighted_duration) > 0.05:
            outputs.qa_warnings.append(
                f"Conservative WAD QA mismatch: stored={outputs.conservative_weighted_duration:.2f}, "
                f"recomputed={recomputed:.2f}."
            )

    def _check_fund_cost_calculation(self, outputs: AnalysisOutputs) -> None:
        """Verify effective_fund_cost_on_total_portfolio is consistent with rows."""
        if not outputs.fund_cost_table or outputs.effective_fund_cost_on_total_portfolio is None:
            return
        total_portfolio = outputs.total_portfolio_value_ils
        if total_portfolio <= 0:
            return
        recomputed = sum(
            (r.fee_percent or 0) * r.market_value_ils
            for r in outputs.fund_cost_table
        ) / total_portfolio
        if abs(recomputed - outputs.effective_fund_cost_on_total_portfolio) > 0.001:
            outputs.qa_warnings.append(
                f"Effective fund cost QA mismatch: stored={outputs.effective_fund_cost_on_total_portfolio:.4f}%, "
                f"recomputed={recomputed:.4f}%."
            )

    def _check_total_cost_policy(
        self, outputs: AnalysisOutputs, prefs: UserAnalysisPreferences
    ) -> None:
        """Total cost must not be shown if PM fee was not provided."""
        if outputs.total_cost_percent is not None and prefs.portfolio_manager_fee_percent is None:
            outputs.qa_errors.append(
                "total_cost_percent was set but portfolio_manager_fee_percent is None. "
                "Total cost must not be computed without a PM fee input."
            )

    def _check_estimated_fields_labeled(
        self, outputs: AnalysisOutputs, holdings: list[HoldingNormalized]
    ) -> None:
        unlabeled = [
            h.normalized_name for h in holdings
            if h.estimated_fields and not any(
                a.holding_id == h.row_id for a in outputs.assumptions
            )
        ]
        if unlabeled:
            outputs.qa_warnings.append(
                f"{len(unlabeled)} holding(s) have estimated fields not reflected in assumptions table: "
                + ", ".join(unlabeled[:5])
                + (" ..." if len(unlabeled) > 5 else "")
            )

    def _check_mandatory_pptx_fields(self, outputs: AnalysisOutputs) -> None:
        """Block PPTX generation if critical data is entirely missing."""
        if not outputs.asset_allocation:
            outputs.qa_errors.append(
                "Asset allocation table is empty – cannot generate Cover slide."
            )
        if outputs.total_portfolio_value_ils <= 0:
            outputs.qa_errors.append(
                "Total portfolio value is zero or negative – presentation cannot be generated."
            )
