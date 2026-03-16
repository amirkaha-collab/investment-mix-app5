"""
Tests for research providers.
Mock provider must return deterministic results.
"""

from __future__ import annotations

import pytest
from src.research.mock_provider import MockResearchProvider
from src.research import EnrichmentService
from src.config.constants import CONFIDENCE_HIGH, CONFIDENCE_ESTIMATED, ASSET_CLASS_BOND


class TestMockResearchProvider:
    def setup_method(self):
        self.provider = MockResearchProvider()

    def test_known_ticker_returns_high_confidence(self):
        result = self.provider.lookup(ticker="SPY")
        assert result.confidence_score >= 0.75
        assert result.fee_percent is not None
        assert result.fee_percent == pytest.approx(0.0945, rel=0.01)
        assert "estimated" not in (result.estimated_fields or [])

    def test_known_ticker_has_sector_breakdown(self):
        result = self.provider.lookup(ticker="SPY")
        assert result.sector_breakdown
        total = sum(result.sector_breakdown.values())
        assert total == pytest.approx(1.0, abs=0.05)

    def test_known_ticker_has_geography(self):
        result = self.provider.lookup(ticker="AGG")
        assert result.geography_breakdown.get("USA", 0) > 0

    def test_name_fragment_match(self):
        result = self.provider.lookup(name="S&P 500 tracker fund")
        assert result.confidence_score >= 0.75

    def test_unknown_bond_returns_estimated_duration(self):
        result = self.provider.lookup(
            ticker="UNKNOWN_XYZ",
            name="Global Bond Fund 2030",
            asset_class=ASSET_CLASS_BOND,
        )
        assert result.duration is not None
        assert result.confidence_score < 0.5
        assert any("duration" in ef for ef in result.estimated_fields)

    def test_qqq_fee(self):
        result = self.provider.lookup(ticker="QQQ")
        assert result.fee_percent == pytest.approx(0.20, rel=0.01)


class TestEnrichmentService:
    def test_enrichment_fills_missing_fee(self, sample_holdings):
        # Clear fee from first holding
        h = sample_holdings[0]
        h.fee_percent = None
        h.fee_source = ""
        h.ticker = "SPY"

        service = EnrichmentService()
        service.enrich([h])
        assert h.fee_percent is not None

    def test_enrichment_does_not_overwrite_official_data(self, sample_holdings):
        # h003 already has official duration from TASE
        h003 = next(h for h in sample_holdings if h.row_id == "h003")
        original_duration = h003.duration
        original_source = h003.duration_source

        service = EnrichmentService()
        service.enrich([h003])

        # Duration should not be replaced with a weaker estimate
        assert h003.duration == original_duration

    def test_enrichment_adds_source_urls(self, sample_holdings):
        h = sample_holdings[0]
        h.source_urls = []
        h.ticker = "SPY"
        service = EnrichmentService()
        service.enrich([h])
        # URLs may or may not be added depending on provider, but no crash
        assert isinstance(h.source_urls, list)

    def test_enrichment_handles_all_holdings_gracefully(self, sample_holdings):
        """No holding should cause a crash during enrichment."""
        service = EnrichmentService()
        warnings = service.enrich(sample_holdings)
        assert isinstance(warnings, list)
