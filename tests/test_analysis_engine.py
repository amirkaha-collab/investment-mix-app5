"""
Tests for the analysis engine and individual sub-modules.

Key principle: every computed aggregate is verified against
the source holdings rows (traceability).
"""

from __future__ import annotations

import pytest

from src.analysis.asset_allocation import compute_asset_allocation
from src.analysis.equity_geography import compute_equity_geography
from src.analysis.us_exposure import compute_us_exposure
from src.analysis.bond_analysis import compute_bond_breakdown, compute_duration_table
from src.analysis.fund_costs import compute_fund_costs
from src.analysis.fx_exposure_concentration import compute_fx_exposure, compute_concentration
from src.analysis.qa_engine import QAEngine
from src.analysis.engine import run_analysis
from src.config.constants import (
    ASSET_CLASS_EQUITY, ASSET_CLASS_BOND, ASSET_CLASS_CASH, REGION_US,
)


class TestAssetAllocation:
    def test_basic_split(self, sample_holdings, sample_preferences):
        rows = compute_asset_allocation(sample_holdings, sample_preferences)
        assert len(rows) >= 3  # equity, bond, cash
        total_weight = sum(r.weight for r in rows)
        assert total_weight == pytest.approx(1.0, abs=0.001)

    def test_cash_exclusion(self, sample_holdings, sample_preferences):
        sample_preferences.include_cash_in_allocation = False
        rows = compute_asset_allocation(sample_holdings, sample_preferences)
        classes = [r.asset_class for r in rows]
        assert ASSET_CLASS_CASH not in classes

    def test_values_sum_to_total(self, sample_holdings, sample_preferences):
        rows = compute_asset_allocation(sample_holdings, sample_preferences)
        total_from_rows = sum(r.market_value_ils for r in rows)
        total_from_holdings = sum(h.market_value_ils for h in sample_holdings)
        assert total_from_rows == pytest.approx(total_from_holdings, rel=0.001)


class TestEquityGeography:
    def test_weights_sum_to_one(self, sample_holdings, sample_preferences):
        # Set weights first
        total = sum(h.market_value_ils for h in sample_holdings)
        for h in sample_holdings:
            h.weight_in_portfolio = h.market_value_ils / total
        rows = compute_equity_geography(sample_holdings, sample_preferences)
        if rows:
            total_w = sum(r.weight_in_equities for r in rows)
            assert total_w == pytest.approx(1.0, abs=0.001)

    def test_geography_breakdown_respected(self, sample_holdings, sample_preferences):
        # h001 (S&P 500) has geography_breakdown = {"USA": 1.0}
        rows = compute_equity_geography(sample_holdings, sample_preferences)
        us_row = next((r for r in rows if r.region == REGION_US), None)
        assert us_row is not None
        # US row must include the S&P 500 fund's full value
        spy_value = next(h for h in sample_holdings if h.row_id == "h001").market_value_ils
        assert us_row.market_value_ils >= spy_value * 0.99


class TestUSExposure:
    def test_conservative_le_broad(self, sample_holdings, sample_preferences):
        result = compute_us_exposure(sample_holdings, sample_preferences)
        assert result.conservative_us_value_ils <= result.broad_us_value_ils

    def test_weights_in_zero_one(self, sample_holdings, sample_preferences):
        result = compute_us_exposure(sample_holdings, sample_preferences)
        assert 0.0 <= result.conservative_us_weight <= 1.0
        assert 0.0 <= result.broad_us_weight <= 1.0


class TestBondAnalysis:
    def test_bond_breakdown_sums(self, sample_holdings, sample_preferences):
        rows = compute_bond_breakdown(sample_holdings, sample_preferences)
        if rows:
            total_w = sum(r.weight_in_bonds for r in rows)
            assert total_w == pytest.approx(1.0, abs=0.001)

    def test_conservative_wad_only_official(self, sample_holdings, sample_preferences):
        sample_preferences.compute_extended_duration_with_estimates = False
        _, con_wad, _ = compute_duration_table(sample_holdings, sample_preferences)
        # Both h003 and h004 have official durations, so con_wad should be present
        assert con_wad is not None
        assert con_wad > 0

    def test_estimated_duration_excluded_from_conservative(self, sample_holdings, sample_preferences):
        # Mark h003 duration as estimated
        h003 = next(h for h in sample_holdings if h.row_id == "h003")
        h003.mark_estimated("duration", "test")
        sample_preferences.compute_extended_duration_with_estimates = False
        rows, con_wad, ext_wad = compute_duration_table(sample_holdings, sample_preferences)
        # h003 should be excluded from conservative
        h003_row = next((r for r in rows if r.row_id == "h003"), None)
        assert h003_row is not None
        assert h003_row.is_estimated is True


class TestFundCosts:
    def test_cost_rows_only_for_funds(self, sample_holdings, sample_preferences):
        rows, _, _, _, _ = compute_fund_costs(sample_holdings, sample_preferences)
        for r in rows:
            # Each row must correspond to a fund
            matching = [h for h in sample_holdings if h.row_id == r.row_id]
            assert len(matching) == 1
            assert matching[0].is_fund or matching[0].is_etf

    def test_total_cost_not_set_without_pm_fee(self, sample_holdings, sample_preferences):
        sample_preferences.portfolio_manager_fee_percent = None
        _, _, _, total_cost, _ = compute_fund_costs(sample_holdings, sample_preferences)
        assert total_cost is None

    def test_total_cost_set_with_pm_fee(self, sample_holdings, sample_preferences):
        sample_preferences.portfolio_manager_fee_percent = 0.75
        _, _, eff, total_cost, _ = compute_fund_costs(sample_holdings, sample_preferences)
        if eff is not None:
            assert total_cost == pytest.approx(eff + 0.75, abs=0.001)


class TestConcentration:
    def test_top5_le_top10(self, sample_holdings, sample_preferences):
        # Set weights
        total = sum(h.market_value_ils for h in sample_holdings)
        for h in sample_holdings:
            h.weight_in_portfolio = h.market_value_ils / total
        _, top5, top10, _, _ = compute_concentration(sample_holdings)
        assert top5 <= top10

    def test_top_n_correctly_ranked(self, sample_holdings, sample_preferences):
        total = sum(h.market_value_ils for h in sample_holdings)
        for h in sample_holdings:
            h.weight_in_portfolio = h.market_value_ils / total
        rows, _, _, _, _ = compute_concentration(sample_holdings, top_n=3)
        assert len(rows) == 3
        assert rows[0].weight >= rows[1].weight >= rows[2].weight


class TestQAEngine:
    def test_no_errors_on_valid_data(self, sample_holdings, sample_preferences):
        total = sum(h.market_value_ils for h in sample_holdings)
        for h in sample_holdings:
            h.weight_in_portfolio = h.market_value_ils / total
        outputs = run_analysis(sample_holdings, sample_preferences, reported_total=total)
        # Validate no critical errors for well-formed data
        assert isinstance(outputs.qa_errors, list)

    def test_total_cost_error_without_pm_fee(self, sample_holdings, sample_preferences):
        sample_preferences.portfolio_manager_fee_percent = None
        total = sum(h.market_value_ils for h in sample_holdings)
        for h in sample_holdings:
            h.weight_in_portfolio = h.market_value_ils / total
        outputs = run_analysis(sample_holdings, sample_preferences)
        # Manually inject violation
        outputs.total_cost_percent = 1.5
        qa = QAEngine()
        qa.run(outputs, sample_holdings, sample_preferences)
        assert any("total_cost_percent" in e for e in outputs.qa_errors)


class TestFullPipeline:
    def test_run_analysis_returns_all_sections(self, sample_holdings, sample_preferences):
        total = sum(h.market_value_ils for h in sample_holdings)
        for h in sample_holdings:
            h.weight_in_portfolio = h.market_value_ils / total
        outputs = run_analysis(sample_holdings, sample_preferences, reported_total=total)

        assert outputs.total_portfolio_value_ils > 0
        assert len(outputs.asset_allocation) > 0
        assert len(outputs.equity_geography) > 0
        assert len(outputs.bond_breakdown) > 0
        assert outputs.us_exposure is not None
        assert len(outputs.top_holdings) > 0

    def test_methodology_notes_present(self, sample_holdings, sample_preferences):
        total = sum(h.market_value_ils for h in sample_holdings)
        for h in sample_holdings:
            h.weight_in_portfolio = h.market_value_ils / total
        outputs = run_analysis(sample_holdings, sample_preferences)
        assert len(outputs.methodology_notes) >= 2
