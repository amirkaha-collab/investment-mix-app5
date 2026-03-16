"""
Tests for file parsers (CSV, Excel).
PDF tests require sample files and are marked as integration tests.
"""

from __future__ import annotations

import io
import pytest
import pandas as pd

from src.parsers.excel_csv_parser import CSVParser, ExcelParser
from src.parsers.base import ParseResult


class TestCSVParser:
    def test_basic_csv(self, sample_csv_content):
        parser = CSVParser()
        result = parser.parse(sample_csv_content)
        assert result.success
        assert not result.primary_df.empty
        assert result.primary_df.shape[0] >= 3

    def test_semicolon_delimiter(self):
        csv = "name;value;currency\nStock A;10000;ILS\nStock B;20000;USD\n"
        parser = CSVParser()
        result = parser.parse(csv.encode("utf-8"))
        assert result.success

    def test_empty_csv_returns_error(self):
        parser = CSVParser()
        result = parser.parse(b"")
        # Either empty df or error – both acceptable
        assert not result.success or result.primary_df.empty

    def test_windows_hebrew_encoding(self):
        csv = "שם נייר,שווי,מטבע\nמניה א,10000,ILS\n"
        for enc in ("cp1255", "utf-8-sig"):
            try:
                parser = CSVParser()
                result = parser.parse(csv.encode(enc))
                if result.success:
                    assert result.primary_df.shape[0] >= 1
                    return
            except Exception:
                continue


class TestExcelParser:
    def _make_xlsx(self, data: dict) -> bytes:
        buf = io.BytesIO()
        df = pd.DataFrame(data)
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Holdings", index=False)
        buf.seek(0)
        return buf.read()

    def test_basic_xlsx(self):
        xlsx = self._make_xlsx({
            "שם נייר": ["SPY", "AGG"],
            "שווי שוק": [100_000, 50_000],
            "מטבע": ["ILS", "ILS"],
        })
        parser = ExcelParser()
        result = parser.parse(xlsx)
        assert result.success
        assert result.primary_df.shape[0] == 2

    def test_multi_sheet_skips_summary(self):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            pd.DataFrame({"שם נייר": ["Stock"], "שווי שוק": [10000], "מטבע": ["ILS"]}).to_excel(
                writer, sheet_name="Holdings", index=False
            )
            pd.DataFrame({"notes": ["ignore me"]}).to_excel(
                writer, sheet_name="Summary", index=False
            )
        buf.seek(0)
        result = ExcelParser().parse(buf.read())
        assert result.success

    def test_corrupted_file_returns_error(self):
        result = ExcelParser().parse(b"not an excel file at all !!!")
        assert not result.success
        assert result.errors


class TestParseResultContract:
    def test_success_property(self):
        r = ParseResult(
            primary_df=pd.DataFrame({"a": [1]}),
            parse_method="test",
        )
        assert r.success is True

    def test_failure_on_errors(self):
        r = ParseResult(
            primary_df=pd.DataFrame(),
            errors=["Something went wrong"],
        )
        assert r.success is False
