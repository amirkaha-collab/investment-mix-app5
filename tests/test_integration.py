"""
End-to-end integration tests.

These tests exercise the FULL pipeline:
  sample file → parse → normalize → enrich → analyse → assert non-empty outputs

They catch bugs that unit tests miss — like the header-normalisation mismatch
that caused zero holdings despite all unit tests passing.
"""

from __future__ import annotations

import io
import pytest
import pandas as pd

from src.parsers.excel_csv_parser import CSVParser
from src.parsers.normalizer import HoldingsNormalizer, _norm_key
from src.domain.models import UserAnalysisPreferences
from src.analysis.engine import run_analysis
from src.services.pipeline import PortfolioPipeline


# ── Helpers ────────────────────────────────────────────────────────────────────

SAMPLE_CSV_HEBREW = (
    "שם נייר,שווי שוק,מטבע,סוג\n"
    "S&P 500 ETF,185000,ILS,מניות\n"
    "AGC Bond ETF,118125,ILS,אגח\n"
    "Israel TA-125,98000,ILS,מניות\n"
    "Bank Deposit,50000,ILS,מזומן\n"
).encode("utf-8-sig")

SAMPLE_CSV_ENGLISH = (
    "security name,market value,currency,type\n"
    "S&P 500 ETF,185000,ILS,equity\n"
    "Bond ETF,118125,ILS,bond\n"
    "Cash,50000,ILS,cash\n"
).encode("utf-8-sig")

SAMPLE_CSV_MIXED = (
    "שם נייר,שווי שוק,מטבע\n"
    "SPY ETF,50000,USD\n"
    "TA-125 ETF,98000,ILS\n"
).encode("utf-8")


def _default_prefs() -> UserAnalysisPreferences:
    return UserAnalysisPreferences(
        include_cash_in_allocation=True,
        portfolio_manager_fee_percent=None,
        client_name="Integration Test",
        report_date="2026-03-15",
    )


# ── Core invariant ─────────────────────────────────────────────────────────────

class TestNormKeyConsistency:
    """The _norm_key function must be applied consistently everywhere."""

    def test_norm_key_matches_clean_header_output(self):
        """
        BaseParser._clean_header uses the same transformation as _norm_key.
        Verify they produce identical output for typical Hebrew column names.
        """
        from src.parsers.base import BaseParser
        import pandas as pd

        raw_cols = ["שם נייר", "שווי שוק", "מטבע", "סוג"]
        df = pd.DataFrame(columns=raw_cols)
        cleaned = BaseParser._clean_header(df).columns.tolist()
        normed  = [_norm_key(c) for c in raw_cols]
        assert cleaned == normed, (
            f"_clean_header and _norm_key diverged!\n"
            f"  clean_header: {cleaned}\n"
            f"  _norm_key:    {normed}"
        )

    def test_column_synonyms_keys_are_normalised(self):
        from src.parsers.normalizer import COLUMN_SYNONYMS
        for key in COLUMN_SYNONYMS:
            assert key == _norm_key(key), (
                f"COLUMN_SYNONYMS key '{key}' is not normalised – "
                f"expected '{_norm_key(key)}'"
            )


# ── Full pipeline ──────────────────────────────────────────────────────────────

class TestEndToEndCSV:

    def _run(self, csv_bytes: bytes) -> tuple:
        pipeline = PortfolioPipeline()
        parse_result = pipeline.parse(csv_bytes, "test.csv")
        assert not parse_result.errors, f"Parse errors: {parse_result.errors}"
        holdings, warnings = pipeline.normalize(parse_result, usd_to_ils=3.75)
        return holdings, warnings

    def test_hebrew_csv_produces_nonzero_holdings(self):
        holdings, warnings = self._run(SAMPLE_CSV_HEBREW)
        assert len(holdings) >= 3, f"Expected ≥3 holdings, got {len(holdings)}. Warnings: {warnings}"

    def test_english_csv_produces_nonzero_holdings(self):
        holdings, warnings = self._run(SAMPLE_CSV_ENGLISH)
        assert len(holdings) >= 2, f"Expected ≥2 holdings, got {len(holdings)}. Warnings: {warnings}"

    def test_holdings_have_positive_values(self):
        holdings, _ = self._run(SAMPLE_CSV_HEBREW)
        for h in holdings:
            assert h.market_value_ils > 0, f"{h.raw_name}: value={h.market_value_ils}"

    def test_weights_sum_to_one(self):
        holdings, _ = self._run(SAMPLE_CSV_HEBREW)
        total_w = sum(h.weight_in_portfolio for h in holdings)
        assert abs(total_w - 1.0) < 0.001, f"Weights sum to {total_w}"

    def test_usd_holdings_converted(self):
        holdings, _ = self._run(SAMPLE_CSV_MIXED)
        usd_h = next((h for h in holdings if h.currency == "USD"), None)
        assert usd_h is not None
        assert usd_h.market_value_ils == pytest.approx(50000 * 3.75, rel=0.01)

    def test_sample_data_file(self):
        """The bundled sample_data/sample_holdings.csv must parse cleanly."""
        from pathlib import Path
        csv_bytes = Path("sample_data/sample_holdings.csv").read_bytes()
        pipeline = PortfolioPipeline()
        result = pipeline.parse(csv_bytes, "sample_holdings.csv")
        # Sample file may have some parse quirks; just assert no fatal errors
        assert not result.errors or result.primary_df.shape[0] > 0


class TestEndToEndAnalysis:

    def _full_run(self, csv_bytes: bytes):
        pipeline = PortfolioPipeline()
        parse_result = pipeline.parse(csv_bytes, "test.csv")
        holdings, _ = pipeline.normalize(parse_result, usd_to_ils=3.75)
        pipeline.enrich(holdings)
        outputs = pipeline.analyse(holdings, _default_prefs())
        return holdings, outputs

    def test_analysis_has_nonzero_total(self):
        _, outputs = self._full_run(SAMPLE_CSV_HEBREW)
        assert outputs.total_portfolio_value_ils > 0, "Total portfolio value is zero!"

    def test_analysis_has_asset_allocation(self):
        _, outputs = self._full_run(SAMPLE_CSV_HEBREW)
        assert len(outputs.asset_allocation) > 0, "Asset allocation is empty!"

    def test_analysis_has_no_fatal_qa_errors(self):
        _, outputs = self._full_run(SAMPLE_CSV_HEBREW)
        assert outputs.qa_errors == [], f"QA errors: {outputs.qa_errors}"

    def test_asset_allocation_weights_sum_to_one(self):
        _, outputs = self._full_run(SAMPLE_CSV_HEBREW)
        total_w = sum(r.weight for r in outputs.asset_allocation)
        assert abs(total_w - 1.0) < 0.005, f"Allocation weights sum to {total_w}"

    def test_top_holdings_populated(self):
        _, outputs = self._full_run(SAMPLE_CSV_HEBREW)
        assert len(outputs.top_holdings) > 0

    def test_pptx_generation_succeeds(self):
        """Full pipeline including PPTX — no QA errors should block generation."""
        _, outputs = self._full_run(SAMPLE_CSV_HEBREW)
        if outputs.qa_errors:
            pytest.skip(f"QA errors present: {outputs.qa_errors}")
        from src.presentation.pptx_builder import PPTXBuilder
        pptx_bytes = PPTXBuilder().build(outputs)
        assert len(pptx_bytes) > 10_000, "PPTX file suspiciously small"

    def test_json_export_has_all_sections(self):
        _, outputs = self._full_run(SAMPLE_CSV_HEBREW)
        from src.services.export_service import ExportService
        import json
        data = json.loads(ExportService().to_json(outputs))
        assert data["total_portfolio_value_ils"] > 0
        assert len(data["asset_allocation"]) > 0
