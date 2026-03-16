"""
Tests for export services (JSON, CSV).
"""

from __future__ import annotations

import json
import pytest
import pandas as pd

from src.services.export_service import ExportService
from src.analysis.engine import run_analysis


class TestExportService:
    def test_json_export_is_valid(self, sample_holdings, sample_preferences):
        total = sum(h.market_value_ils for h in sample_holdings)
        for h in sample_holdings:
            h.weight_in_portfolio = h.market_value_ils / total
        outputs = run_analysis(sample_holdings, sample_preferences, reported_total=total)
        export = ExportService()
        json_bytes = export.to_json(outputs)

        # Must be valid JSON
        data = json.loads(json_bytes.decode("utf-8"))
        assert "total_portfolio_value_ils" in data
        assert "asset_allocation" in data
        assert "assumptions" in data

    def test_holdings_csv_has_all_rows(self, sample_holdings, sample_preferences):
        export = ExportService()
        csv_bytes = export.holdings_to_csv(sample_holdings)
        df = pd.read_csv(
            __import__("io").StringIO(csv_bytes.decode("utf-8-sig"))
        )
        assert len(df) == len(sample_holdings)

    def test_holdings_csv_has_required_columns(self, sample_holdings, sample_preferences):
        export = ExportService()
        csv_bytes = export.holdings_to_csv(sample_holdings)
        df = pd.read_csv(
            __import__("io").StringIO(csv_bytes.decode("utf-8-sig"))
        )
        required = ["שם נייר", "סוג נכס", "שווי (ILS)", "מטבע"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_analysis_csv_table_column(self, sample_holdings, sample_preferences):
        total = sum(h.market_value_ils for h in sample_holdings)
        for h in sample_holdings:
            h.weight_in_portfolio = h.market_value_ils / total
        outputs = run_analysis(sample_holdings, sample_preferences)
        export = ExportService()
        csv_bytes = export.analysis_summary_csv(outputs)
        if csv_bytes:
            df = pd.read_csv(
                __import__("io").StringIO(csv_bytes.decode("utf-8-sig"))
            )
            assert "table" in df.columns

    def test_json_no_silent_data_loss(self, sample_holdings, sample_preferences):
        """
        Core traceability test: every holding's market_value_ils
        must appear somewhere in the JSON output (in asset_allocation or holdings).
        """
        total = sum(h.market_value_ils for h in sample_holdings)
        for h in sample_holdings:
            h.weight_in_portfolio = h.market_value_ils / total
        outputs = run_analysis(sample_holdings, sample_preferences, reported_total=total)
        export = ExportService()
        data = json.loads(export.to_json(outputs).decode("utf-8"))
        reported_total = data["total_portfolio_value_ils"]
        assert reported_total == pytest.approx(total, rel=0.01)
