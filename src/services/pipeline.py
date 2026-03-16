"""
Pipeline Service – coordinates the full analysis workflow.

This module is the single entry-point for the business logic layer.
The Streamlit UI calls methods here; no UI code appears in this file.

Pipeline stages:
  1. parse(file_bytes, filename)          → ParseResult
  2. normalize(parse_result)              → list[HoldingNormalized] + warnings
  3. enrich(holdings)                     → list[HoldingNormalized] (mutated)
  4. run_analysis(holdings, prefs)        → AnalysisOutputs
  5. build_presentation(outputs)          → bytes (PPTX)
  6. export_*(holdings, outputs)          → bytes (CSV / JSON)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

from src.config.settings import get_settings
from src.domain.models import AnalysisOutputs, HoldingNormalized, UserAnalysisPreferences
from src.parsers import get_parser, ParseResult
from src.parsers.normalizer import HoldingsNormalizer
from src.research import EnrichmentService
from src.analysis import run_analysis
from src.presentation import PPTXBuilder
from .export_service import ExportService

logger = logging.getLogger(__name__)
settings = get_settings()


class PortfolioPipeline:
    """
    Stateless pipeline helper.  All state (holdings, outputs) is passed
    explicitly between stages so the UI can store them in st.session_state.
    """

    def __init__(self) -> None:
        self._enricher = EnrichmentService()
        self._pptx_builder = PPTXBuilder()
        self._export = ExportService()

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 1: Parse
    # ──────────────────────────────────────────────────────────────────────────

    def parse(self, file_bytes: bytes, filename: str) -> ParseResult:
        """
        Parse uploaded file bytes.  Returns ParseResult with raw DataFrame.
        """
        logger.info(f"Parsing file: {filename} ({len(file_bytes):,} bytes)")
        parser = get_parser(filename)
        result = parser.parse(file_bytes)
        logger.info(
            f"Parse complete: method={result.parse_method}, "
            f"rows={len(result.primary_df)}, errors={result.errors}"
        )
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 2: Normalize
    # ──────────────────────────────────────────────────────────────────────────

    def normalize(
        self, parse_result: ParseResult, usd_to_ils: float | None = None
    ) -> tuple[list[HoldingNormalized], list[str]]:
        """
        Convert raw DataFrame to validated HoldingNormalized list.
        Returns (holdings, warnings).
        """
        rate = usd_to_ils or settings.usd_to_ils_rate
        normalizer = HoldingsNormalizer(usd_to_ils=rate)
        holdings, warnings = normalizer.normalize(parse_result.primary_df)
        logger.info(f"Normalized {len(holdings)} holdings, {len(warnings)} warnings")
        return holdings, warnings

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 3: Enrich (research)
    # ──────────────────────────────────────────────────────────────────────────

    def enrich(self, holdings: list[HoldingNormalized]) -> list[str]:
        """
        Enrich holdings with research data.  Holdings are mutated in-place.
        Returns list of warning strings.
        """
        logger.info(f"Enriching {len(holdings)} holdings")
        warnings = self._enricher.enrich(holdings)
        return warnings

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 4: Analyse
    # ──────────────────────────────────────────────────────────────────────────

    def analyse(
        self,
        holdings: list[HoldingNormalized],
        prefs: UserAnalysisPreferences,
        reported_total: Optional[float] = None,
    ) -> AnalysisOutputs:
        """
        Run full portfolio analysis.  Returns AnalysisOutputs.
        """
        return run_analysis(holdings, prefs, reported_total)

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 5: Build presentation
    # ──────────────────────────────────────────────────────────────────────────

    def build_presentation(self, outputs: AnalysisOutputs) -> bytes:
        """
        Generate PPTX bytes.  Raises ValueError if QA errors are present.
        """
        return self._pptx_builder.build(outputs)

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 6: Export
    # ──────────────────────────────────────────────────────────────────────────

    def export_json(self, outputs: AnalysisOutputs) -> bytes:
        return self._export.to_json(outputs)

    def export_holdings_csv(self, holdings: list[HoldingNormalized]) -> bytes:
        return self._export.holdings_to_csv(holdings)

    def export_analysis_csv(self, outputs: AnalysisOutputs) -> bytes:
        return self._export.analysis_summary_csv(outputs)
