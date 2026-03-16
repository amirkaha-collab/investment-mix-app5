"""
Export Service – serialises analysis outputs to JSON and CSV.

No data is persisted to disk by default; all outputs are returned as bytes
so the caller (Streamlit UI) controls where they go.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import pandas as pd

from src.domain.models import AnalysisOutputs, HoldingNormalized
from src.utils.formatters import holdings_to_dataframe

logger = logging.getLogger(__name__)


class ExportService:

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def to_json(outputs: AnalysisOutputs) -> bytes:
        """Serialise the full AnalysisOutputs to JSON bytes (UTF-8)."""
        data = outputs.model_dump(mode="json")
        return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

    @staticmethod
    def holdings_to_csv(holdings: list[HoldingNormalized]) -> bytes:
        """Serialise normalised holdings to UTF-8 CSV with BOM (for Excel)."""
        df = holdings_to_dataframe(holdings)
        buf = io.StringIO()
        df.to_csv(buf, index=False, encoding="utf-8-sig")
        return buf.getvalue().encode("utf-8-sig")

    @staticmethod
    def analysis_summary_csv(outputs: AnalysisOutputs) -> bytes:
        """
        Export all analysis tables as a multi-sheet-friendly CSV.
        Returns a single CSV with a 'table' column indicating which
        analysis section each row belongs to.
        """
        frames: list[pd.DataFrame] = []

        def _add(label: str, df: pd.DataFrame) -> None:
            if not df.empty:
                df.insert(0, "table", label)
                frames.append(df)

        if outputs.asset_allocation:
            _add("asset_allocation", pd.DataFrame([r.model_dump() for r in outputs.asset_allocation]))
        if outputs.equity_geography:
            _add("equity_geography", pd.DataFrame([r.model_dump() for r in outputs.equity_geography]))
        if outputs.sector_allocation:
            _add("sector_allocation", pd.DataFrame([r.model_dump() for r in outputs.sector_allocation]))
        if outputs.bond_breakdown:
            _add("bond_breakdown", pd.DataFrame([r.model_dump() for r in outputs.bond_breakdown]))
        if outputs.duration_table:
            _add("duration", pd.DataFrame([r.model_dump() for r in outputs.duration_table]))
        if outputs.fund_cost_table:
            _add("fund_costs", pd.DataFrame([r.model_dump() for r in outputs.fund_cost_table]))
        if outputs.fx_exposure:
            _add("fx_exposure", pd.DataFrame([r.model_dump() for r in outputs.fx_exposure]))
        if outputs.top_holdings:
            _add("concentration", pd.DataFrame([r.model_dump() for r in outputs.top_holdings]))
        if outputs.assumptions:
            _add("assumptions", pd.DataFrame([r.model_dump() for r in outputs.assumptions]))

        if not frames:
            return b""

        combined = pd.concat(frames, ignore_index=True)
        buf = io.StringIO()
        combined.to_csv(buf, index=False, encoding="utf-8-sig")
        return buf.getvalue().encode("utf-8-sig")
