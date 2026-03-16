"""
Streamlit session state management.

All session_state keys are defined here as constants.
Call `init_session_state()` once at the top of each page.
"""

from __future__ import annotations

import streamlit as st


# ─── Stage constants ────────────────────────────────────────────────────────────
STAGE_UPLOAD = "upload"
STAGE_PARSING = "parsing"
STAGE_REVIEW = "review"
STAGE_CLARIFICATION = "clarification"
STAGE_RESEARCH = "research"
STAGE_ANALYSIS = "analysis"
STAGE_QA = "qa"
STAGE_REPORT = "report"
STAGE_DOWNLOAD = "download"

STAGES_ORDERED = [
    STAGE_UPLOAD,
    STAGE_PARSING,
    STAGE_REVIEW,
    STAGE_CLARIFICATION,
    STAGE_RESEARCH,
    STAGE_ANALYSIS,
    STAGE_QA,
    STAGE_REPORT,
    STAGE_DOWNLOAD,
]

STAGE_LABELS_HE = {
    STAGE_UPLOAD: "1. העלאת קובץ",
    STAGE_PARSING: "2. חילוץ נתונים",
    STAGE_REVIEW: "3. אישור אחזקות",
    STAGE_CLARIFICATION: "4. שאלות השלמה",
    STAGE_RESEARCH: "5. מחקר אינטרנטי",
    STAGE_ANALYSIS: "6. חישוב ניתוח",
    STAGE_QA: "7. בדיקת עקביות",
    STAGE_REPORT: "8. הצגת דוח",
    STAGE_DOWNLOAD: "9. הורדת תוצרים",
}

# ─── State keys ────────────────────────────────────────────────────────────────
KEY_STAGE = "stage"
KEY_UPLOADED_BYTES = "uploaded_bytes"
KEY_UPLOADED_FILENAME = "uploaded_filename"
KEY_PARSE_RESULT = "parse_result"
KEY_PARSE_WARNINGS = "parse_warnings"
KEY_HOLDINGS = "holdings"
KEY_NORM_WARNINGS = "norm_warnings"
KEY_ENRICH_WARNINGS = "enrich_warnings"
KEY_PREFS = "prefs"
KEY_OUTPUTS = "outputs"
KEY_PPTX_BYTES = "pptx_bytes"
KEY_JSON_BYTES = "json_bytes"
KEY_HOLDINGS_CSV = "holdings_csv"
KEY_ANALYSIS_CSV = "analysis_csv"
KEY_USD_TO_ILS = "usd_to_ils"


def init_session_state() -> None:
    """Initialise all session state keys to their defaults."""
    defaults = {
        KEY_STAGE: STAGE_UPLOAD,
        KEY_UPLOADED_BYTES: None,
        KEY_UPLOADED_FILENAME: None,
        KEY_PARSE_RESULT: None,
        KEY_PARSE_WARNINGS: [],
        KEY_HOLDINGS: None,
        KEY_NORM_WARNINGS: [],
        KEY_ENRICH_WARNINGS: [],
        KEY_PREFS: None,
        KEY_OUTPUTS: None,
        KEY_PPTX_BYTES: None,
        KEY_JSON_BYTES: None,
        KEY_HOLDINGS_CSV: None,
        KEY_ANALYSIS_CSV: None,
        KEY_USD_TO_ILS: 3.75,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def advance_to(stage: str) -> None:
    st.session_state[KEY_STAGE] = stage


def current_stage() -> str:
    return st.session_state.get(KEY_STAGE, STAGE_UPLOAD)


def reset_pipeline() -> None:
    """Clear all analysis state (keep uploaded file)."""
    for key in [
        KEY_PARSE_RESULT, KEY_PARSE_WARNINGS,
        KEY_HOLDINGS, KEY_NORM_WARNINGS, KEY_ENRICH_WARNINGS,
        KEY_PREFS, KEY_OUTPUTS, KEY_PPTX_BYTES,
        KEY_JSON_BYTES, KEY_HOLDINGS_CSV, KEY_ANALYSIS_CSV,
    ]:
        st.session_state[key] = None
    st.session_state[KEY_STAGE] = STAGE_UPLOAD
