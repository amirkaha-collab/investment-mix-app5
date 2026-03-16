"""
Analysis Engine – orchestrates all sub-analysis modules into AnalysisOutputs.

This is the single entry-point for running portfolio analysis.
The caller provides:
  - holdings        : list[HoldingNormalized]  (post-research-enrichment)
  - prefs           : UserAnalysisPreferences
  - reported_total  : float | None  (total from original statement, if available)

The engine:
  1. Computes all analysis tables
  2. Builds the assumptions and data-quality tables
  3. Runs QA checks
  4. Returns a fully populated AnalysisOutputs
"""

from __future__ import annotations

import datetime
import logging
from typing import Optional

from src.config.constants import CONFIDENCE_ESTIMATED, CONFIDENCE_HIGH
from src.domain.models import (
    AnalysisOutputs,
    AssumptionRow,
    ConfidenceLevel,
    DataQualityNote,
    HoldingNormalized,
    UserAnalysisPreferences,
)

from .asset_allocation import compute_asset_allocation
from .equity_geography import compute_equity_geography
from .us_exposure import compute_us_exposure
from .sector_allocation import compute_sector_allocation
from .bond_analysis import compute_bond_breakdown, compute_duration_table
from .fund_costs import compute_fund_costs
from .fx_exposure_concentration import compute_fx_exposure, compute_concentration
from .qa_engine import QAEngine

logger = logging.getLogger(__name__)


def run_analysis(
    holdings: list[HoldingNormalized],
    prefs: UserAnalysisPreferences,
    reported_total: Optional[float] = None,
) -> AnalysisOutputs:
    """
    Run full portfolio analysis.

    Parameters
    ----------
    holdings       : Enriched, user-confirmed holdings list
    prefs          : User preferences collected via clarification Q&A
    reported_total : Total portfolio value from the original statement (if available)

    Returns
    -------
    AnalysisOutputs – fully populated, QA-checked.
    """
    logger.info(f"Running analysis on {len(holdings)} holdings")

    # ── Total portfolio value ──────────────────────────────────────────────────
    computed_total = sum(h.market_value_ils for h in holdings)
    total = reported_total if reported_total and reported_total > 0 else computed_total

    # ── Report date ────────────────────────────────────────────────────────────
    if not prefs.report_date:
        prefs.report_date = datetime.date.today().isoformat()

    outputs = AnalysisOutputs(
        total_portfolio_value_ils=round(total, 2),
        holdings_count=len(holdings),
        preferences=prefs,
    )

    # ── 1. Asset Allocation ────────────────────────────────────────────────────
    logger.info("Computing asset allocation")
    outputs.asset_allocation = compute_asset_allocation(holdings, prefs)

    # ── 2. Equity Geography ────────────────────────────────────────────────────
    logger.info("Computing equity geography")
    outputs.equity_geography = compute_equity_geography(holdings, prefs)

    # ── 3. US Exposure ────────────────────────────────────────────────────────
    logger.info("Computing US exposure")
    outputs.us_exposure = compute_us_exposure(holdings, prefs)

    # ── 4. Sector Allocation ──────────────────────────────────────────────────
    logger.info("Computing sector allocation")
    outputs.sector_allocation = compute_sector_allocation(holdings, prefs)

    # ── 5 + 6. Bond Breakdown + Duration ──────────────────────────────────────
    logger.info("Computing bond breakdown and duration")
    outputs.bond_breakdown = compute_bond_breakdown(holdings, prefs)
    dur_rows, con_wad, ext_wad = compute_duration_table(holdings, prefs)
    outputs.duration_table = dur_rows
    outputs.conservative_weighted_duration = con_wad
    outputs.extended_weighted_duration = ext_wad

    # ── 7 + 8. Fund Costs ─────────────────────────────────────────────────────
    logger.info("Computing fund costs")
    cost_rows, wt_on_funds, eff_on_total, total_cost, cost_is_assumption = compute_fund_costs(
        holdings, prefs
    )
    outputs.fund_cost_table = cost_rows
    outputs.weighted_fund_cost_on_funds = wt_on_funds
    outputs.effective_fund_cost_on_total_portfolio = eff_on_total
    outputs.total_cost_percent = total_cost
    outputs.total_cost_is_assumption = cost_is_assumption

    # ── 9. FX Exposure ────────────────────────────────────────────────────────
    logger.info("Computing FX exposure")
    outputs.fx_exposure = compute_fx_exposure(holdings, prefs)

    # ── 10. Concentration ─────────────────────────────────────────────────────
    logger.info("Computing concentration")
    top_rows, top5, top10, max_single, conc_warnings = compute_concentration(holdings)
    outputs.top_holdings = top_rows
    outputs.top5_concentration = top5
    outputs.top10_concentration = top10
    outputs.max_single_holding_weight = max_single
    outputs.concentration_warnings = conc_warnings

    # ── 11. Assumptions & data quality ────────────────────────────────────────
    logger.info("Building assumptions table")
    outputs.assumptions = _build_assumptions(holdings)
    outputs.data_quality_notes = _build_data_quality_notes(holdings)
    outputs.source_urls = list(
        {url for h in holdings for url in h.source_urls}
    )

    # ── Methodology notes ─────────────────────────────────────────────────────
    outputs.methodology_notes = _build_methodology_notes(prefs)

    # ── QA ────────────────────────────────────────────────────────────────────
    logger.info("Running QA checks")
    QAEngine().run(outputs, holdings, prefs)
    if outputs.qa_warnings:
        logger.warning(f"{len(outputs.qa_warnings)} QA warnings generated")
    if outputs.qa_errors:
        logger.error(f"{len(outputs.qa_errors)} QA errors – PPTX generation will be blocked")

    logger.info("Analysis complete")
    return outputs


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_assumptions(holdings: list[HoldingNormalized]) -> list[AssumptionRow]:
    rows: list[AssumptionRow] = []
    for h in holdings:
        for ef in h.estimated_fields:
            # ef format: "field_name: reason" or just "field_name"
            if ":" in ef:
                field_name, reason = ef.split(":", 1)
            else:
                field_name, reason = ef, "rule-based estimate"

            field_name = field_name.strip()
            reason = reason.strip()
            assumed_value = _get_field_value(h, field_name)

            rows.append(
                AssumptionRow(
                    field=field_name,
                    holding_id=h.row_id,
                    holding_name=h.normalized_name,
                    assumed_value=str(assumed_value),
                    reason=reason,
                    confidence=ConfidenceLevel.ESTIMATED,
                    source="; ".join(h.source_urls[:1]),
                )
            )
    return rows


def _get_field_value(h: HoldingNormalized, field: str) -> object:
    return getattr(h, field, "N/A")


def _build_data_quality_notes(holdings: list[HoldingNormalized]) -> list[DataQualityNote]:
    notes: list[DataQualityNote] = []
    for h in holdings:
        if h.confidence_score < 0.5:
            notes.append(
                DataQualityNote(
                    category="weak_source",
                    holding_id=h.row_id,
                    holding_name=h.normalized_name,
                    description=f"Confidence score = {h.confidence_score:.2f} – data may be unreliable.",
                    recommendation="Verify against official fund factsheet.",
                )
            )
        if not h.isin and not h.ticker:
            notes.append(
                DataQualityNote(
                    category="missing_data",
                    holding_id=h.row_id,
                    holding_name=h.normalized_name,
                    description="No ISIN or ticker found – research matching may be inaccurate.",
                    recommendation="Add ISIN or ticker in the holdings review step.",
                )
            )
        if h.is_bond and not h.bond_linkage_type:
            notes.append(
                DataQualityNote(
                    category="classification_ambiguity",
                    holding_id=h.row_id,
                    holding_name=h.normalized_name,
                    description="Bond linkage type could not be determined.",
                    recommendation="Specify CPI-linked / nominal ILS / USD in the review step.",
                )
            )
    return notes


def _build_methodology_notes(prefs: UserAnalysisPreferences) -> list[str]:
    notes = [
        "All values are converted to ILS using the exchange rate specified in settings.",
        "Asset allocation uses market values as at the statement date.",
        "Sector breakdown for broad-market ETFs uses published index composition data.",
        "US exposure uses two methodologies: conservative (direct US holdings) "
        "and broad (adds qualifying USD bond funds per user preference).",
    ]
    if prefs.compute_extended_duration_with_estimates:
        notes.append(
            "Extended weighted average duration includes estimated durations "
            "(marked as ESTIMATED) per user preference."
        )
    else:
        notes.append(
            "Conservative weighted average duration excludes estimated durations. "
            "Holdings with no official duration source are omitted from this metric."
        )
    if prefs.portfolio_manager_fee_percent is not None:
        flag = " (ASSUMPTION)" if prefs.manager_fee_is_assumption else " (confirmed)"
        notes.append(
            f"Total cost includes portfolio manager fee of "
            f"{prefs.portfolio_manager_fee_percent:.2f}%{flag}."
        )
    return notes
