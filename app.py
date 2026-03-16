"""
Portfolio Analyzer – Main Streamlit Application
================================================

Entry point: `streamlit run app.py`

Architecture notes:
  - All state lives in st.session_state (managed via src/ui/state.py)
  - Business logic lives in src/services/pipeline.py
  - UI rendering lives in src/ui/components.py
  - No computation happens in this file; only routing and state transitions
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ── Ensure src/ is on the Python path ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd

from src.config.constants import LEGAL_DISCLAIMER
from src.config.settings import get_settings
from src.domain.models import UserAnalysisPreferences
from src.services.pipeline import PortfolioPipeline
from src.ui import state as S
from src.ui.components import (
    render_sidebar_progress,
    render_disclaimer,
    render_kpi_row,
    render_asset_allocation,
    render_equity_geography,
    render_us_exposure,
    render_sector_allocation,
    render_bond_breakdown,
    render_duration,
    render_fund_costs,
    render_fx_exposure,
    render_concentration,
    render_assumptions,
    render_qa_status,
    render_holdings_editor,
    render_clarification_form,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ─────────────────────────────────────────────────────────
S.init_session_state()

# ── Pipeline (cached per session) ─────────────────────────────────────────────
@st.cache_resource
def get_pipeline() -> PortfolioPipeline:
    return PortfolioPipeline()

pipeline = get_pipeline()

# ── Sidebar ────────────────────────────────────────────────────────────────────
render_sidebar_progress(S.current_stage())

with st.sidebar:
    if st.button("🔄 התחל מחדש", use_container_width=True):
        S.reset_pipeline()
        st.rerun()

    st.divider()
    st.markdown("**הגדרות שע\"ח**")
    usd_rate = st.number_input(
        "USD → ILS",
        min_value=1.0, max_value=10.0,
        value=float(st.session_state[S.KEY_USD_TO_ILS]),
        step=0.01, format="%.2f",
        key="usd_rate_input"
    )
    st.session_state[S.KEY_USD_TO_ILS] = usd_rate


def _render_debug_panel(parse_result) -> None:
    """Render the technical debug panel for the parse result."""
    with st.expander("🔍 Debug: פרטים טכניים על חילוץ הקובץ", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**שיטת חילוץ:** `{parse_result.parse_method}`")
            st.markdown(f"**שגיאות:** {parse_result.errors or 'אין'}")
            st.markdown(f"**טבלאות גולמיות:** {len(parse_result.raw_tables)}")
        with col2:
            st.markdown(f"**אזהרות טכניות:** {len(parse_result.warnings)}")
            for w in parse_result.warnings:
                st.caption(f"• {w}")
        if not parse_result.primary_df.empty:
            st.markdown(f"**עמודות שחולצו:** `{list(parse_result.primary_df.columns)}`")
            st.markdown("**תצוגה מקדימה (20 שורות):**")
            st.dataframe(parse_result.primary_df.head(20), use_container_width=True)
        else:
            st.warning("DataFrame ריק — לא חולצו עמודות")

# ── Main header ────────────────────────────────────────────────────────────────
st.title("📊 מנתח תיק השקעות")
render_disclaimer()
st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1: Upload
# ──────────────────────────────────────────────────────────────────────────────
if S.current_stage() == S.STAGE_UPLOAD:
    st.header("1️⃣ העלאת קובץ")
    st.markdown(
        "העלה קובץ של תדפיס תיק השקעות. "
        "הפורמטים הנתמכים: **PDF**, **XLSX/XLS**, **CSV**."
    )

    uploaded = st.file_uploader(
        "בחר קובץ תדפיס תיק",
        type=["pdf", "xlsx", "xls", "csv"],
        key="file_uploader",
    )

    if uploaded:
        file_bytes = uploaded.read()
        st.session_state[S.KEY_UPLOADED_BYTES] = file_bytes
        st.session_state[S.KEY_UPLOADED_FILENAME] = uploaded.name
        st.success(f"✅ הקובץ '{uploaded.name}' הועלה ({len(file_bytes):,} bytes)")

        if st.button("▶ התחל חילוץ נתונים", type="primary", use_container_width=True):
            S.advance_to(S.STAGE_PARSING)
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2: Parsing
# ──────────────────────────────────────────────────────────────────────────────
elif S.current_stage() == S.STAGE_PARSING:
    st.header("2️⃣ חילוץ וניתוח נתונים")

    file_bytes = st.session_state[S.KEY_UPLOADED_BYTES]
    filename   = st.session_state[S.KEY_UPLOADED_FILENAME]

    # ── Sidebar debug toggle (always visible in parsing stage) ────────────────
    show_debug = st.sidebar.checkbox("🔧 הצג debug מידע", value=False, key="show_debug")

    with st.spinner(f"מחלץ טבלת אחזקות מ-{filename}…"):
        try:
            parse_result = pipeline.parse(file_bytes, filename)
            st.session_state[S.KEY_PARSE_RESULT] = parse_result
        except Exception as exc:
            st.error(f"❌ שגיאה בלתי צפויה בחילוץ: {exc}")
            logger.exception("Parse error")
            st.stop()

    # ── Parse errors → clear user message ─────────────────────────────────────
    if parse_result.errors:
        for e in parse_result.errors:
            st.error(e)
        st.info(
            "💡 **המלצה:** הורד את הדוח מהבנק בפורמט **XLSX** או **CSV** ישירות, "
            "או ייצא את טבלת האחזקות ל-Excel והעלה אותה כאן."
        )
        # Show debug even on failure so user can investigate
        if show_debug:
            _render_debug_panel(parse_result)
        if st.button("← חזור להעלאה"):
            S.advance_to(S.STAGE_UPLOAD)
            st.rerun()
        st.stop()

    # ── Parse succeeded ────────────────────────────────────────────────────────
    method = parse_result.parse_method or "לא ידוע"

    # User-facing status banner (no internal module names)
    if "ocr" in method.lower():
        st.info("🔍 זוהה PDF תמונתי — הופעל OCR לשחזור הטבלה.")
    elif "text_line" in method.lower():
        st.warning("⚠️ לא זוהתה טבלה מובנית — בוצע חילוץ מטקסט חופשי. בדוק את הנתונים בשלב הבא.")
    elif "pdfplumber" in method.lower() or "pymupdf" in method.lower() or "structured" in method.lower():
        pass  # clean extraction, no extra message needed

    # Show only meaningful user warnings (strip internal tech messages)
    _INTERNAL = ("No module named", "camelot", "pdfplumber:", "pymupdf text extraction used",
                 "table structure may not", "strategy")
    for w in parse_result.warnings:
        if not any(skip in w for skip in _INTERNAL):
            st.warning(w)

    st.session_state[S.KEY_PARSE_WARNINGS] = parse_result.warnings

    # ── Normalize ──────────────────────────────────────────────────────────────
    try:
        holdings, norm_warnings = pipeline.normalize(
            parse_result, usd_to_ils=st.session_state[S.KEY_USD_TO_ILS]
        )
    except Exception as exc:
        st.error(f"❌ שגיאה בנרמול נתונים: {exc}")
        logger.exception("Normalize error")
        st.stop()

    st.session_state[S.KEY_HOLDINGS]      = holdings
    st.session_state[S.KEY_NORM_WARNINGS] = norm_warnings

    # Normalization warnings — only meaningful ones
    for w in norm_warnings:
        if w and "first column" not in w:   # suppress the generic fallback hint
            st.warning(w)

    if len(holdings) == 0:
        st.error(
            "❌ לא חולצו אחזקות מהקובץ. "
            "ייתכן שמבנה העמודות שונה מהמצופה."
        )
        st.info(
            "💡 נסה אחד מהבאים:\n"
            "1. הפעל את ה-Debug panel (בסרגל הצד) כדי לראות מה חולץ בפועל\n"
            "2. הורד את הדוח כ-XLSX/CSV מהבנק\n"
            "3. המשך לשלב אישור אחזקות והוסף שורות ידנית"
        )
        if show_debug:
            _render_debug_panel(parse_result)
        # Allow continuing to manual entry
        if st.button("▶ המשך לעריכה ידנית של אחזקות", use_container_width=True):
            S.advance_to(S.STAGE_REVIEW)
            st.rerun()
        st.stop()

    st.success(f"✅ חולצו **{len(holdings)}** אחזקות | שיטה: `{method}`")

    # ── Debug panel ────────────────────────────────────────────────────────────
    if show_debug:
        _render_debug_panel(parse_result)

    if st.button("▶ המשך לאישור אחזקות", type="primary", use_container_width=True):
        S.advance_to(S.STAGE_REVIEW)
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3: Review holdings
# ──────────────────────────────────────────────────────────────────────────────
elif S.current_stage() == S.STAGE_REVIEW:
    st.header("3️⃣ אישור ועריכת אחזקות")
    st.info(
        "בדוק את הטבלה ותקן כל שדה שנראה שגוי. "
        "ניתן להוסיף שורות חדשות, למחוק, ולשנות ערכים."
    )

    holdings = st.session_state[S.KEY_HOLDINGS]
    if not holdings:
        st.error("לא נמצאו אחזקות. חזור לשלב העלאה.")
        if st.button("← חזור"):
            S.advance_to(S.STAGE_UPLOAD)
            st.rerun()
        st.stop()

    from src.utils.formatters import holdings_to_dataframe
    holdings_df = holdings_to_dataframe(holdings)
    edited_df = render_holdings_editor(holdings_df)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← חזור לחילוץ"):
            S.advance_to(S.STAGE_PARSING)
            st.rerun()
    with col2:
        if st.button("✅ אשר אחזקות והמשך", type="primary", use_container_width=True):
            # Re-normalize from the edited DataFrame
            from src.parsers.normalizer import HoldingsNormalizer
            from src.parsers.base import ParseResult as PR
            try:
                # Map edited columns back to parseable form
                edit_df = edited_df.rename(columns={
                    "שם נייר": "raw_name",
                    "סוג נכס": "asset_class",
                    "מטבע": "currency",
                    "שווי (ILS)": "market_value_ils",
                    "ISIN": "isin",
                    "Ticker": "ticker",
                    "אזור": "region",
                    "ענף": "sector",
                    "דמי ניהול %": "fee_percent",
                    "מח\"מ": "duration",
                })
                normalizer = HoldingsNormalizer(usd_to_ils=st.session_state[S.KEY_USD_TO_ILS])
                new_holdings, _ = normalizer.normalize(edit_df)
                st.session_state[S.KEY_HOLDINGS] = new_holdings
            except Exception as exc:
                st.error(f"שגיאה בעיבוד העריכה: {exc}")
                logger.exception("Review re-normalize error")
                st.stop()

            S.advance_to(S.STAGE_CLARIFICATION)
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 4: Clarification questions
# ──────────────────────────────────────────────────────────────────────────────
elif S.current_stage() == S.STAGE_CLARIFICATION:
    st.header("4️⃣ שאלות השלמה")

    answers = render_clarification_form()

    if answers.get("submitted"):
        pm_fee = float(answers["pm_fee"]) if answers["pm_fee"] > 0 else None
        prefs = UserAnalysisPreferences(
            include_cash_in_allocation=answers["include_cash"],
            portfolio_manager_fee_percent=pm_fee,
            manager_fee_is_assumption=answers["pm_fee_is_assumption"],
            classify_global_usd_bond_as_us_exposure=answers["global_usd_bond_as_us"],
            compute_extended_duration_with_estimates=answers["extended_duration"],
            client_name=answers["client_name"],
            report_title=answers["report_title"],
        )
        st.session_state[S.KEY_PREFS] = prefs
        S.advance_to(S.STAGE_RESEARCH)
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 5: Research
# ──────────────────────────────────────────────────────────────────────────────
elif S.current_stage() == S.STAGE_RESEARCH:
    st.header("5️⃣ מחקר אינטרנטי")

    holdings = st.session_state[S.KEY_HOLDINGS]
    progress = st.progress(0, text="מתחיל מחקר…")

    with st.spinner("מחפש נתונים לניירות…"):
        try:
            enrich_warnings = pipeline.enrich(holdings)
            st.session_state[S.KEY_ENRICH_WARNINGS] = enrich_warnings
            st.session_state[S.KEY_HOLDINGS] = holdings  # mutated in-place
        except Exception as exc:
            st.error(f"שגיאה במחקר: {exc}")
            logger.exception("Enrich error")
            st.stop()

    progress.progress(100, text="מחקר הושלם")
    st.success(f"✅ מחקר הושלם – {len(holdings)} ניירות נבדקו")

    for w in enrich_warnings:
        st.warning(w)

    if st.button("▶ המשך לניתוח", type="primary", use_container_width=True):
        S.advance_to(S.STAGE_ANALYSIS)
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 6: Analysis
# ──────────────────────────────────────────────────────────────────────────────
elif S.current_stage() == S.STAGE_ANALYSIS:
    st.header("6️⃣ חישוב ניתוח")

    holdings = st.session_state[S.KEY_HOLDINGS]
    prefs = st.session_state[S.KEY_PREFS]

    with st.spinner("מחשב…"):
        try:
            outputs = pipeline.analyse(holdings, prefs)
            st.session_state[S.KEY_OUTPUTS] = outputs
        except Exception as exc:
            st.error(f"שגיאה בניתוח: {exc}")
            logger.exception("Analysis error")
            st.stop()

    st.success("✅ ניתוח הושלם")

    if st.button("▶ הצג דוח", type="primary", use_container_width=True):
        S.advance_to(S.STAGE_REPORT)
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# STAGES 7+8: QA + Report
# ──────────────────────────────────────────────────────────────────────────────
elif S.current_stage() in (S.STAGE_QA, S.STAGE_REPORT):
    # Jump straight to report (QA results embedded)
    st.header("📋 דוח ניתוח תיק")

    outputs = st.session_state.get(S.KEY_OUTPUTS)
    if not outputs:
        st.error("אין נתוני ניתוח. חזור לתחילה.")
        st.stop()

    # ── QA status banner ──────────────────────────────────────────────────────
    with st.expander("🔍 סטטוס בדיקות QA", expanded=bool(outputs.qa_errors)):
        render_qa_status(outputs)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    render_kpi_row(outputs)
    st.divider()

    # ── Analysis tabs ─────────────────────────────────────────────────────────
    tab_names = [
        "הקצאת נכסים",
        "גיאוגרפיה",
        "חשיפה לארה\"ב",
        "סקטורים",
        "אגח",
        "מח\"מ",
        "עלויות",
        "מט\"ח",
        "ריכוזיות",
        "הנחות",
        "מתודולוגיה",
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_asset_allocation(outputs)
    with tabs[1]:
        render_equity_geography(outputs)
    with tabs[2]:
        render_us_exposure(outputs)
    with tabs[3]:
        render_sector_allocation(outputs)
    with tabs[4]:
        render_bond_breakdown(outputs)
    with tabs[5]:
        render_duration(outputs)
    with tabs[6]:
        render_fund_costs(outputs)
    with tabs[7]:
        render_fx_exposure(outputs)
    with tabs[8]:
        render_concentration(outputs)
    with tabs[9]:
        render_assumptions(outputs)
    with tabs[10]:
        st.markdown("### הערות מתודולוגיות")
        for note in outputs.methodology_notes:
            st.markdown(f"- {note}")
        if outputs.data_quality_notes:
            st.markdown("### הערות איכות נתונים")
            for n in outputs.data_quality_notes:
                st.markdown(f"- **{n.category}** | {n.holding_name}: {n.description}")

    st.divider()
    if st.button("▶ עבור לייצוא תוצרים", type="primary", use_container_width=True):
        S.advance_to(S.STAGE_DOWNLOAD)
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 9: Download exports
# ──────────────────────────────────────────────────────────────────────────────
elif S.current_stage() == S.STAGE_DOWNLOAD:
    st.header("9️⃣ הורדת תוצרים")

    outputs = st.session_state.get(S.KEY_OUTPUTS)
    holdings = st.session_state.get(S.KEY_HOLDINGS)

    if not outputs or not holdings:
        st.error("אין נתונים להורדה.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)

    # ── JSON ─────────────────────────────────────────────────────────────────
    with col1:
        if st.session_state[S.KEY_JSON_BYTES] is None:
            st.session_state[S.KEY_JSON_BYTES] = pipeline.export_json(outputs)
        st.download_button(
            "⬇️ JSON מלא",
            data=st.session_state[S.KEY_JSON_BYTES],
            file_name="portfolio_analysis.json",
            mime="application/json",
            use_container_width=True,
        )

    # ── Holdings CSV ──────────────────────────────────────────────────────────
    with col2:
        if st.session_state[S.KEY_HOLDINGS_CSV] is None:
            st.session_state[S.KEY_HOLDINGS_CSV] = pipeline.export_holdings_csv(holdings)
        st.download_button(
            "⬇️ CSV אחזקות",
            data=st.session_state[S.KEY_HOLDINGS_CSV],
            file_name="holdings_normalized.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Analysis CSV ──────────────────────────────────────────────────────────
    with col3:
        if st.session_state[S.KEY_ANALYSIS_CSV] is None:
            st.session_state[S.KEY_ANALYSIS_CSV] = pipeline.export_analysis_csv(outputs)
        st.download_button(
            "⬇️ CSV ניתוח",
            data=st.session_state[S.KEY_ANALYSIS_CSV],
            file_name="analysis_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── PPTX ─────────────────────────────────────────────────────────────────
    with col4:
        if outputs.qa_errors:
            st.error("לא ניתן ליצור מצגת – יש שגיאות QA:")
            for e in outputs.qa_errors:
                st.error(e)
        else:
            if st.session_state[S.KEY_PPTX_BYTES] is None:
                with st.spinner("בונה מצגת PowerPoint…"):
                    try:
                        pptx_bytes = pipeline.build_presentation(outputs)
                        st.session_state[S.KEY_PPTX_BYTES] = pptx_bytes
                    except Exception as exc:
                        st.error(f"שגיאה ביצירת מצגת: {exc}")
                        logger.exception("PPTX build error")

            if st.session_state[S.KEY_PPTX_BYTES]:
                st.download_button(
                    "⬇️ מצגת PPTX",
                    data=st.session_state[S.KEY_PPTX_BYTES],
                    file_name="portfolio_presentation.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    use_container_width=True,
                )

    st.divider()
    st.success("✅ כל התוצרים מוכנים להורדה.")
    st.caption(LEGAL_DISCLAIMER)

    if st.button("← חזור לדוח"):
        S.advance_to(S.STAGE_REPORT)
        st.rerun()

else:
    st.error(f"שלב לא מוכר: {S.current_stage()}")
    S.reset_pipeline()
    st.rerun()
