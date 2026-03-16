"""
Chart utilities – generates matplotlib charts as in-memory PNG images.

All charts are returned as bytes (PNG) so they can be:
  - Displayed in Streamlit via st.image()
  - Injected into PPTX slides as pictures
"""

from __future__ import annotations

import io
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.config.constants import PPTX_CHART_IMAGE_DPI

# ─── Colour palette ──────────────────────────────────────────────────────────────
PALETTE = [
    "#2C5F8A", "#3D8EB9", "#5ABCD8", "#87D4E8",
    "#1A3A5C", "#4A90D9", "#7AB8E0", "#A8D5F0",
    "#C0392B", "#E74C3C", "#F39C12", "#2ECC71",
]


def _fig_to_bytes(fig: plt.Figure, dpi: int = PPTX_CHART_IMAGE_DPI) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", transparent=False)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def pie_chart(
    labels: list[str],
    values: list[float],
    title: str = "",
    figsize: tuple = (6, 5),
) -> bytes:
    """Generate a pie chart and return PNG bytes."""
    fig, ax = plt.subplots(figsize=figsize)
    colors = PALETTE[: len(labels)]
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.82,
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.legend(
        wedges,
        [f"{l} ({v:.1f}%)" for l, v in zip(labels, values)],
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
    )
    if title:
        ax.set_title(title, fontsize=11, pad=12)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def bar_chart(
    labels: list[str],
    values: list[float],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    horizontal: bool = True,
    figsize: tuple = (7, 4),
    value_format: str = "{:.1f}%",
) -> bytes:
    """Generate a bar chart and return PNG bytes."""
    fig, ax = plt.subplots(figsize=figsize)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]

    if horizontal:
        bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)
        ax.set_xlabel(ylabel or "Value")
        ax.invert_yaxis()
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                value_format.format(val),
                va="center",
                fontsize=8,
            )
    else:
        bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.6)
        ax.set_ylabel(ylabel or "Value")
        plt.xticks(rotation=30, ha="right", fontsize=8)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                value_format.format(val),
                ha="center",
                fontsize=8,
            )

    if title:
        ax.set_title(title, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def stacked_bar_chart(
    categories: list[str],
    series: dict[str, list[float]],
    title: str = "",
    figsize: tuple = (8, 4),
) -> bytes:
    """Generate a stacked horizontal bar chart."""
    fig, ax = plt.subplots(figsize=figsize)
    lefts = [0.0] * len(categories)
    for i, (label, vals) in enumerate(series.items()):
        color = PALETTE[i % len(PALETTE)]
        ax.barh(categories, vals, left=lefts, color=color, label=label, height=0.5)
        lefts = [l + v for l, v in zip(lefts, vals)]

    ax.set_xlim(0, 100)
    ax.set_xlabel("% of Portfolio")
    ax.legend(loc="lower right", fontsize=8)
    if title:
        ax.set_title(title, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    fig.tight_layout()
    return _fig_to_bytes(fig)


def kpi_card_chart(
    metrics: dict[str, str],
    title: str = "",
    figsize: tuple = (8, 2.5),
) -> bytes:
    """Generate a simple KPI card image (text boxes)."""
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, (label, value) in zip(axes, metrics.items()):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.65, value, ha="center", va="center", fontsize=16, fontweight="bold", color=PALETTE[0])
        ax.text(0.5, 0.25, label, ha="center", va="center", fontsize=9, color="#555")
        ax.set_facecolor("#F4F8FB")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    return _fig_to_bytes(fig)
