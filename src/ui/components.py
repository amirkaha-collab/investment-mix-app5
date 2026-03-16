"""
Reusable Streamlit UI components.

Each function renders a self-contained UI block.
No business logic here – only display and input collection.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from src.config.constants import LEGAL_DISCLAIMER_EN, LEGAL_DISCLAIMER
from src.domain.models import AnalysisOutputs, HoldingNormalized
from src.utils.formatters import (
    analysis_table_asset_alloc,
    analysis_table_geo,
    analysis_table_sectors,
    analysis_table_bonds,
    analysis_table_duration,
    analysis_table_fund_costs,
    analysis_table_fx,
    analysis_table_top_holdings,
    analysis_table_assumptions,
    fmt_ils,
    fmt_pct,
    fmt_pct_raw,
    fmt_duration,
)
from src.utils.chart_utils import pie_chart, bar_chart


def render_sidebar_progress(current_stage: str) -> None:
    """Render stage progress in the sidebar."""
    from .state import STAGES_ORDERED, STAGE_LABELS_HE
    st.sidebar.markdown("## 📊 שלבי העבודה")
    for stage in STAGES_ORDERED:
        label = STAGE_LABELS_HE[stage]
        if stage == current_stage:
            st.sidebar.markdown(f"**▶ {label}**")
        elif STAGES_ORDERED.index(stage) < STAGES_ORDERED.index(current_stage):
            st.sidebar.markdown(f"✅ ~~{label}~~")
        else:
            st.sidebar.markdown(f"⬜ {label}")
    st.sidebar.divider()
    st.sidebar.caption(LEGAL_DISCLAIMER_EN)


def render_disclaimer() -> None:
    st.info(f"⚖️ {LEGAL_DISCLAIMER}", icon="ℹ️")


def render_kpi_row(outputs: AnalysisOutputs) -> None:
    """Render top KPI cards."""
    cols = st.columns(4)
    with cols[0]:
        st.metric("שווי תיק כולל", fmt_ils(outputs.total_portfolio_value_ils))
    with cols[1]:
        st.metric("מספר אחזקות", str(outputs.holdings_count))
    with cols[2]:
        wad = outputs.conservative_weighted_duration
        st.metric("מח\"מ שמרן", fmt_duration(wad) if wad else "—")
    with cols[3]:
        tc = outputs.total_cost_percent
        st.metric("עלות כוללת", fmt_pct_raw(tc) if tc else "—")


def render_asset_allocation(outputs: AnalysisOutputs) -> None:
    if not outputs.asset_allocation:
        st.warning("אין נתוני הקצאת נכסים.")
        return
    col1, col2 = st.columns([1, 1])
    with col1:
        chart = pie_chart(
            [r.asset_class for r in outputs.asset_allocation],
            [r.weight * 100 for r in outputs.asset_allocation],
            title="הקצאת נכסים",
        )
        st.image(chart, use_column_width=True)
    with col2:
        st.dataframe(analysis_table_asset_alloc(outputs.asset_allocation), hide_index=True)


def render_equity_geography(outputs: AnalysisOutputs) -> None:
    if not outputs.equity_geography:
        st.info("אין אחזקות מניות להצגה.")
        return
    col1, col2 = st.columns([1, 1])
    with col1:
        chart = pie_chart(
            [r.region for r in outputs.equity_geography],
            [r.weight_in_equities * 100 for r in outputs.equity_geography],
            title="גיאוגרפיה – מניות",
        )
        st.image(chart, use_column_width=True)
    with col2:
        st.dataframe(analysis_table_geo(outputs.equity_geography), hide_index=True)


def render_us_exposure(outputs: AnalysisOutputs) -> None:
    us = outputs.us_exposure
    if not us:
        st.info("אין נתוני חשיפה לארה\"ב.")
        return
    col1, col2 = st.columns(2)
    with col1:
        st.metric("חשיפה שמרנית (ILS)", fmt_ils(us.conservative_us_value_ils))
        st.caption(fmt_pct(us.conservative_us_weight) + " מהתיק")
    with col2:
        st.metric("חשיפה רחבה (ILS)", fmt_ils(us.broad_us_value_ils))
        st.caption(fmt_pct(us.broad_us_weight) + " מהתיק")
    st.caption(f"מתודולוגיה: {us.methodology_note}")


def render_sector_allocation(outputs: AnalysisOutputs) -> None:
    if not outputs.sector_allocation:
        st.info("אין נתוני חלוקה סקטוריאלית.")
        return
    col1, col2 = st.columns([1, 1])
    with col1:
        chart = bar_chart(
            [r.sector for r in outputs.sector_allocation[:8]],
            [r.weight_in_equities * 100 for r in outputs.sector_allocation[:8]],
            title="סקטורים",
            horizontal=True,
        )
        st.image(chart, use_column_width=True)
    with col2:
        st.dataframe(analysis_table_sectors(outputs.sector_allocation), hide_index=True)


def render_bond_breakdown(outputs: AnalysisOutputs) -> None:
    if not outputs.bond_breakdown:
        st.info("אין אחזקות אגרות חוב.")
        return
    col1, col2 = st.columns([1, 1])
    with col1:
        chart = pie_chart(
            [r.linkage_type for r in outputs.bond_breakdown],
            [r.weight_in_bonds * 100 for r in outputs.bond_breakdown],
            title="פירוק אגח",
        )
        st.image(chart, use_column_width=True)
    with col2:
        st.dataframe(analysis_table_bonds(outputs.bond_breakdown), hide_index=True)


def render_duration(outputs: AnalysisOutputs) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("מח\"מ שמרן (שנים)", fmt_duration(outputs.conservative_weighted_duration))
    with col2:
        st.metric("מח\"מ מורחב (שנים)", fmt_duration(outputs.extended_weighted_duration))
    if outputs.duration_table:
        st.dataframe(analysis_table_duration(outputs.duration_table), hide_index=True)


def render_fund_costs(outputs: AnalysisOutputs) -> None:
    if not outputs.fund_cost_table:
        st.info("אין קרנות/ETF בתיק.")
        return
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("עלות קרנות (על קרנות)", fmt_pct_raw(outputs.weighted_fund_cost_on_funds))
    with col2:
        st.metric("עלות קרנות (על כלל תיק)", fmt_pct_raw(outputs.effective_fund_cost_on_total_portfolio))
    with col3:
        tc = outputs.total_cost_percent
        label = "עלות כוללת" + (" [ASSUMPTION]" if outputs.total_cost_is_assumption else "")
        st.metric(label, fmt_pct_raw(tc) if tc else "לא חושב (חסר דמי מנהל)")
    st.dataframe(analysis_table_fund_costs(outputs.fund_cost_table), hide_index=True)


def render_fx_exposure(outputs: AnalysisOutputs) -> None:
    if not outputs.fx_exposure:
        st.info("אין נתוני מט\"ח.")
        return
    col1, col2 = st.columns([1, 1])
    with col1:
        chart = pie_chart(
            [r.currency for r in outputs.fx_exposure],
            [r.weight * 100 for r in outputs.fx_exposure],
            title="חשיפת מטבע",
        )
        st.image(chart, use_column_width=True)
    with col2:
        st.dataframe(analysis_table_fx(outputs.fx_exposure), hide_index=True)


def render_concentration(outputs: AnalysisOutputs) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Top-5 ריכוזיות", fmt_pct(outputs.top5_concentration))
    with col2:
        st.metric("Top-10 ריכוזיות", fmt_pct(outputs.top10_concentration))
    with col3:
        st.metric("אחזקה בודדת מקס.", fmt_pct(outputs.max_single_holding_weight))

    for w in outputs.concentration_warnings:
        st.warning(w)

    if outputs.top_holdings:
        st.dataframe(analysis_table_top_holdings(outputs.top_holdings), hide_index=True)


def render_assumptions(outputs: AnalysisOutputs) -> None:
    if not outputs.assumptions:
        st.success("אין הנחות – כל הנתונים ממקורות מאומתים.")
        return
    st.warning(f"⚠️ {len(outputs.assumptions)} שדות מוערכים בתיק זה. בדוק לפני שימוש.")
    st.dataframe(analysis_table_assumptions(outputs.assumptions), hide_index=True)


def render_qa_status(outputs: AnalysisOutputs) -> None:
    if outputs.qa_errors:
        for e in outputs.qa_errors:
            st.error(f"🚫 {e}")
    if outputs.qa_warnings:
        for w in outputs.qa_warnings:
            st.warning(f"⚠️ {w}")
    if not outputs.qa_errors and not outputs.qa_warnings:
        st.success("✅ כל בדיקות ה-QA עברו בהצלחה.")


def render_holdings_editor(holdings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Render an editable holdings table.
    Returns the edited DataFrame.
    """
    st.markdown("### ✏️ סקירת ואישור אחזקות")
    st.caption(
        "ניתן לערוך, להוסיף ולמחוק שורות. "
        "לחץ על שורה ושנה ערכים בטבלה. "
        "**יש לאשר לפני המעבר לניתוח.**"
    )
    edited = st.data_editor(
        holdings_df,
        use_container_width=True,
        num_rows="dynamic",
        key="holdings_editor",
    )
    return edited


def render_clarification_form() -> dict:
    """
    Render clarification questions form.
    Returns dict of user answers.
    """
    st.markdown("### ❓ שאלות השלמה")
    st.caption("ענה על השאלות הבאות כדי לדייק את הניתוח.")

    answers: dict = {}

    with st.form("clarification_form"):
        answers["include_cash"] = st.checkbox(
            "לכלול מזומן / פיקדונות בחישוב הקצאת הנכסים?", value=True
        )
        answers["pm_fee"] = st.number_input(
            "דמי ניהול מנהל תיקים (% שנתי) – השאר 0 אם לא ידוע",
            min_value=0.0, max_value=5.0, value=0.0, step=0.05, format="%.2f"
        )
        answers["pm_fee_is_assumption"] = st.checkbox(
            "דמי הניהול לעיל הם הערכה (לא ידועים בוודאות)?", value=False
        )
        answers["global_usd_bond_as_us"] = st.checkbox(
            "לסווג קרנות אג\"ח דולריות גלובליות כחשיפה לארה\"ב (חשיפה רחבה)?",
            value=False
        )
        answers["extended_duration"] = st.checkbox(
            "לחשב מח\"מ מורחב עם הערכות (כולל מח\"מ מוערך)?",
            value=False
        )
        answers["client_name"] = st.text_input("שם הלקוח (לכותרת המצגת)", value="")
        answers["report_title"] = st.text_input(
            "כותרת הדוח", value="ניתוח תיק השקעות"
        )

        submitted = st.form_submit_button("✅ אשר והמשך לניתוח", type="primary")
        answers["submitted"] = submitted

    return answers
