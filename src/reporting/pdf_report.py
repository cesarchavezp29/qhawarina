"""Qhawarina PDF instability report generator.

Produces professional PDF reports (daily, weekly, monthly):
  Page 1: Cover (logo + title + date)
  Page 2: Resumen Ejecutivo (score gauges + narrative/explanation)
  Page 3: Analisis (trend chart + key events that happened)
  Page 4: (monthly only) Weekly breakdown table

Reports focus on EVENTS THAT OCCURRED — not forecasts or preventive measures.
"""

import calendar
import logging
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from src.reporting.daily_report import (
    SEVERITY_LABELS,
    _level_label,
)
from src.visualization.style import (
    FONT_FAMILY,
    NEXUS_COLORS,
    apply_nexus_style,
)

logger = logging.getLogger("nexus.reporting.pdf")

# ── Brand constants ──────────────────────────────────────────────────────────

POLITICAL_COLOR = "#C0392B"
ECONOMIC_COLOR = "#2E5090"
GOLD = "#D4A03C"
DARK = "#1B2A4A"
BG = "#FAFBFC"
TEAL = "#1ABC9C"
PAGE_W, PAGE_H = 8.5, 11

MONTHS_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}

LOGO_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "logo.png"


# ── Page primitives ──────────────────────────────────────────────────────────

def _new_page(pdf):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=BG)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.06)
    return fig


def _add_footer(fig):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(
        0.5, 0.015,
        f"Qhawarina \u00b7 Observatorio Econ\u00f3mico del Per\u00fa  |  {ts}",
        ha="center", va="bottom", fontsize=7,
        color=NEXUS_COLORS["text_secondary"], fontstyle="italic",
    )


def _save_page(pdf, fig):
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


# ── PAGE 1: Cover ────────────────────────────────────────────────────────────

def _cover_page(pdf, title, date_label, logo_path=None):
    fig = _new_page(pdf)

    lp = Path(logo_path) if logo_path else LOGO_PATH
    if lp.exists():
        logo_ax = fig.add_axes([0.25, 0.48, 0.5, 0.4])
        logo_ax.imshow(mpimg.imread(str(lp)))
        logo_ax.axis("off")

    fig.text(0.5, 0.40, title, ha="center", va="top", fontsize=22,
             fontweight="bold", color=DARK, fontfamily=FONT_FAMILY)
    fig.text(0.5, 0.34, date_label, ha="center", va="top", fontsize=16,
             color=NEXUS_COLORS["text_secondary"], fontfamily=FONT_FAMILY)
    fig.text(0.5, 0.28, "Qhawarina \u2014 Observatorio Econ\u00f3mico del Per\u00fa",
             ha="center", va="top", fontsize=11, color=GOLD,
             fontfamily=FONT_FAMILY, fontstyle="italic")

    line_ax = fig.add_axes([0.15, 0.25, 0.7, 0.002])
    line_ax.axhline(0, color=GOLD, linewidth=2)
    line_ax.axis("off")

    _add_footer(fig)
    _save_page(pdf, fig)


# ── Score gauge helper ───────────────────────────────────────────────────────

def _draw_gauge(ax, v2, label, color, n_articles):
    ax.barh(0, 1.0, height=0.4, color="#E8ECF0", edgecolor="none")

    n_seg = 50
    for i in range(int(v2 * n_seg)):
        frac = i / n_seg
        c = TEAL if frac < 0.33 else (GOLD if frac < 0.66 else color)
        ax.barh(0, 1 / n_seg, left=frac, height=0.4, color=c, edgecolor="none")

    ax.plot(v2, 0, marker="v", markersize=10, color=DARK, zorder=5)

    level = _level_label(v2)
    ax.text(-0.02, 0, label, ha="right", va="center", fontsize=11,
            fontweight="bold", color=color, fontfamily=FONT_FAMILY)
    ax.text(1.02, 0.12, level, ha="left", va="center", fontsize=9,
            fontweight="bold", color=color, fontfamily=FONT_FAMILY)
    ax.text(1.02, -0.12, f"v2={v2:.2f}  |  {n_articles} art.", ha="left",
            va="center", fontsize=8, color=NEXUS_COLORS["text_secondary"])

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.4, 0.4)
    ax.axis("off")


# ── PAGE 2: Resumen Ejecutivo ────────────────────────────────────────────────

def _summary_page(pdf, row_data, pol_articles, econ_articles, narrative=None):
    fig = _new_page(pdf)

    fig.text(0.5, 0.94, "RESUMEN EJECUTIVO", ha="center", va="top",
             fontsize=16, fontweight="bold", color=DARK, fontfamily=FONT_FAMILY)

    # Gauges
    ax_pol = fig.add_axes([0.22, 0.82, 0.55, 0.06])
    _draw_gauge(ax_pol, row_data["political_v2"], "Pol\u00edtico",
                POLITICAL_COLOR, int(row_data["n_articles_political"]))

    ax_econ = fig.add_axes([0.22, 0.72, 0.55, 0.06])
    _draw_gauge(ax_econ, row_data["economic_v2"], "Econ\u00f3mico",
                ECONOMIC_COLOR, int(row_data["n_articles_economic"]))

    # Score table
    fig.text(0.08, 0.66, "Detalle", fontsize=11, fontweight="bold",
             color=DARK, fontfamily=FONT_FAMILY)

    table_data = [
        ["Pol\u00edtico",
         f"{row_data['political_score']:.2f}",
         f"{row_data['political_zscore']:.2f}",
         _level_label(row_data["political_v2"]),
         f"{row_data['political_v2']:.2f}",
         str(int(row_data["n_articles_political"]))],
        ["Econ\u00f3mico",
         f"{row_data['economic_score']:.2f}",
         f"{row_data['economic_zscore']:.2f}",
         _level_label(row_data["economic_v2"]),
         f"{row_data['economic_v2']:.2f}",
         str(int(row_data["n_articles_economic"]))],
    ]

    ax_t = fig.add_axes([0.08, 0.55, 0.84, 0.10])
    ax_t.axis("off")
    tbl = ax_t.table(
        cellText=table_data,
        colLabels=["", "Score", "Z-score", "Nivel", "V2", "Art."],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for j in range(6):
        tbl[0, j].set_facecolor(DARK)
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for j in range(6):
        tbl[1, j].set_facecolor("#FDF2F2")
        tbl[2, j].set_facecolor("#F0F4FA")

    # Narrative or auto-explanation
    y = 0.50
    if narrative:
        fig.text(0.08, y, "Resumen", fontsize=11, fontweight="bold",
                 color=DARK, fontfamily=FONT_FAMILY)
        y -= 0.025
        fig.text(0.08, y, narrative, fontsize=9,
                 color=NEXUS_COLORS["text_primary"], fontfamily=FONT_FAMILY,
                 va="top", wrap=True, transform=fig.transFigure,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F6F8",
                           edgecolor=NEXUS_COLORS["border"], alpha=0.8))
    else:
        explanation = _build_explanation(pol_articles, econ_articles)
        if explanation:
            fig.text(0.08, y, "Eventos Clave", fontsize=11, fontweight="bold",
                     color=DARK, fontfamily=FONT_FAMILY)
            y -= 0.025
            fig.text(0.08, y, explanation, fontsize=9,
                     color=NEXUS_COLORS["text_primary"], fontfamily=FONT_FAMILY,
                     va="top", wrap=True, transform=fig.transFigure)

    _add_footer(fig)
    _save_page(pdf, fig)


def _build_explanation(pol_articles, econ_articles):
    """Build concise explanation from top articles — what happened."""
    parts = []
    for cat_name, arts, color in [
        ("Pol\u00edtico", pol_articles, POLITICAL_COLOR),
        ("Econ\u00f3mico", econ_articles, ECONOMIC_COLOR),
    ]:
        if arts.empty:
            continue
        sev = arts["article_severity"].value_counts()
        high = int(sev.get(3, 0))
        med = int(sev.get(2, 0))

        top = arts.head(3)["title"].tolist()
        titles = "; ".join(t[:65] for t in top)

        count_str = f"{len(arts)} art."
        if high:
            count_str += f" ({high} alta sev.)"
        elif med:
            count_str += f" ({med} media sev.)"

        parts.append(f"{cat_name} [{count_str}]: {titles}.")

    return "\n".join(parts)


# ── PAGE 3: Analysis (chart + key events) ────────────────────────────────────

def _analysis_page(pdf, index_df, start_date, end_date, pol_articles,
                   econ_articles, chart_title="Tendencia"):
    fig = _new_page(pdf)

    fig.text(0.5, 0.94, "AN\u00c1LISIS", ha="center", va="top",
             fontsize=16, fontweight="bold", color=DARK, fontfamily=FONT_FAMILY)

    # ── Chart (top half) ──
    mask = (index_df["date"] >= pd.Timestamp(start_date)) & \
           (index_df["date"] <= pd.Timestamp(end_date))
    df = index_df[mask].sort_values("date")

    if not df.empty:
        ax = fig.add_axes([0.10, 0.55, 0.80, 0.33])
        ax.plot(df["date"], df["political_score"], color=POLITICAL_COLOR,
                linewidth=1.5, label="Pol\u00edtico", alpha=0.9)
        ax.plot(df["date"], df["economic_score"], color=ECONOMIC_COLOR,
                linewidth=1.5, label="Econ\u00f3mico", alpha=0.9)

        if len(df) > 10:
            ax.plot(df["date"], df["political_score"].rolling(7, min_periods=3).mean(),
                    color=POLITICAL_COLOR, linewidth=2, linestyle="--", alpha=0.5)
            ax.plot(df["date"], df["economic_score"].rolling(7, min_periods=3).mean(),
                    color=ECONOMIC_COLOR, linewidth=2, linestyle="--", alpha=0.5)

        ax.set_ylabel("Score (avg. severidad)", fontsize=9, color=DARK)
        ax.set_title(chart_title, fontsize=11, color=DARK, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=25, labelsize=7)

        ax2 = ax.twinx()
        ax2.fill_between(df["date"], df["political_v2"], alpha=0.06, color=POLITICAL_COLOR)
        ax2.fill_between(df["date"], df["economic_v2"], alpha=0.06, color=ECONOMIC_COLOR)
        ax2.set_ylabel("V2", fontsize=9, color=NEXUS_COLORS["text_secondary"])
        ax2.set_ylim(0, 1)
    else:
        fig.text(0.5, 0.65, "Sin datos para gr\u00e1fico", ha="center",
                 fontsize=12, color=NEXUS_COLORS["text_secondary"])

    # ── Key events (bottom half) ──
    y = 0.48
    sev_colors = {3: "#C0392B", 2: "#E67E22", 1: "#27AE60"}
    line_h = 0.030

    # Merge political + economic, sort by severity, take top events
    all_arts = pd.concat([pol_articles, econ_articles]).drop_duplicates(subset=["url"])
    all_arts = all_arts.sort_values("article_severity", ascending=False)

    fig.text(0.08, y, "Eventos Registrados", fontsize=11, fontweight="bold",
             color=DARK, fontfamily=FONT_FAMILY)
    y -= 0.025

    if all_arts.empty:
        fig.text(0.08, y, "Sin eventos relevantes en este per\u00edodo.",
                 fontsize=9, color=NEXUS_COLORS["text_secondary"])
    else:
        for _, row in all_arts.head(12).iterrows():
            if y < 0.04:
                break

            sev = int(row.get("article_severity", 1))
            sev_lbl = SEVERITY_LABELS.get(sev, "?")
            cat = row.get("article_category", "?")
            cat_lbl = "P" if cat in ("political", "both") else "E"
            source = row.get("source", "?")
            title = str(row.get("title", ""))
            if len(title) > 90:
                title = title[:87] + "..."

            fig.text(0.08, y, f"[{sev_lbl}]", fontsize=7.5, fontweight="bold",
                     color=sev_colors.get(sev, "#666"), fontfamily=FONT_FAMILY, va="top")
            fig.text(0.155, y, f"[{cat_lbl}]", fontsize=7.5, fontweight="bold",
                     color=POLITICAL_COLOR if cat_lbl == "P" else ECONOMIC_COLOR,
                     fontfamily=FONT_FAMILY, va="top")
            fig.text(0.19, y, f"({source}) {title}", fontsize=7.5,
                     color=NEXUS_COLORS["text_primary"], fontfamily=FONT_FAMILY, va="top")
            y -= line_h

        remaining = len(all_arts) - 12
        if remaining > 0:
            fig.text(0.08, y, f"... y {remaining} eventos m\u00e1s",
                     fontsize=7.5, color=NEXUS_COLORS["text_secondary"],
                     fontstyle="italic")

    _add_footer(fig)
    _save_page(pdf, fig)


# ── PAGE 4 (monthly only): Weekly breakdown ──────────────────────────────────

def _weekly_breakdown_page(pdf, index_df, year, month):
    fig = _new_page(pdf)

    fig.text(0.5, 0.93, "DESGLOSE SEMANAL", ha="center", va="top",
             fontsize=14, fontweight="bold", color=DARK, fontfamily=FONT_FAMILY)

    mask = (index_df["date"].dt.year == year) & (index_df["date"].dt.month == month)
    df = index_df[mask].copy()

    if df.empty:
        fig.text(0.5, 0.5, "Sin datos", ha="center", fontsize=12,
                 color=NEXUS_COLORS["text_secondary"])
        _add_footer(fig)
        _save_page(pdf, fig)
        return

    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    weekly = df.groupby("week").agg(
        pol_avg=("political_score", "mean"),
        pol_max=("political_score", "max"),
        econ_avg=("economic_score", "mean"),
        econ_max=("economic_score", "max"),
        pol_v2=("political_v2", "mean"),
        econ_v2=("economic_v2", "mean"),
        n_days=("date", "count"),
        total_articles=("n_articles_total", "sum"),
    ).reset_index()

    col_labels = ["Sem.", "D\u00edas", "Pol. Avg", "Pol. Max", "Pol. V2",
                   "Econ. Avg", "Econ. Max", "Econ. V2", "Art."]
    cell_data = []
    for _, r in weekly.iterrows():
        cell_data.append([
            f"S{int(r['week'])}", str(int(r["n_days"])),
            f"{r['pol_avg']:.2f}", f"{r['pol_max']:.2f}", f"{r['pol_v2']:.2f}",
            f"{r['econ_avg']:.2f}", f"{r['econ_max']:.2f}", f"{r['econ_v2']:.2f}",
            str(int(r["total_articles"])),
        ])

    ax = fig.add_axes([0.06, 0.55, 0.88, 0.30])
    ax.axis("off")
    tbl = ax.table(cellText=cell_data, colLabels=col_labels,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor(DARK)
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(len(cell_data)):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                tbl[i + 1, j].set_facecolor("#F5F6F8")

    _add_footer(fig)
    _save_page(pdf, fig)


# ── Data loading helpers ─────────────────────────────────────────────────────

def _load_index(index_path):
    idx = pd.read_parquet(index_path)
    idx["date"] = pd.to_datetime(idx["date"])
    return idx.sort_values("date")


def _load_articles(classified_path):
    articles = pd.read_parquet(classified_path)
    articles["published"] = pd.to_datetime(articles["published"], utc=True)
    articles["_date"] = articles["published"].dt.date
    return articles


def _get_period_articles(articles, start_date, end_date):
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    mask = (articles["_date"] >= start) & (articles["_date"] <= end)
    period = articles[mask]

    pol = period[period["article_category"].isin(["political", "both"])]
    econ = period[period["article_category"].isin(["economic", "both"])]

    return (pol.sort_values("article_severity", ascending=False),
            econ.sort_values("article_severity", ascending=False))


def _aggregate_row(index_df, start_date, end_date):
    mask = (index_df["date"] >= pd.Timestamp(start_date)) & \
           (index_df["date"] <= pd.Timestamp(end_date))
    df = index_df[mask]
    if df.empty:
        return None

    return {
        "political_score": df["political_score"].mean(),
        "political_zscore": df["political_zscore"].mean(),
        "political_v2": df["political_v2"].mean(),
        "economic_score": df["economic_score"].mean(),
        "economic_zscore": df["economic_zscore"].mean(),
        "economic_v2": df["economic_v2"].mean(),
        "n_articles_political": df["n_articles_political"].sum(),
        "n_articles_economic": df["n_articles_economic"].sum(),
        "n_articles_total": df["n_articles_total"].sum(),
        "political_score_max": df["political_score"].max(),
        "political_score_min": df["political_score"].min(),
        "economic_score_max": df["economic_score"].max(),
        "economic_score_min": df["economic_score"].min(),
        "political_v2_max": df["political_v2"].max(),
        "economic_v2_max": df["economic_v2"].max(),
        "n_days": len(df),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def generate_daily_pdf(date, index_path, classified_path, output_path,
                       logo_path=None, narrative=None):
    """3-page daily PDF: Cover, Executive Summary, Analysis."""
    apply_nexus_style()

    index_df = _load_index(index_path)
    articles = _load_articles(classified_path)

    target = pd.Timestamp(date)
    row = index_df[index_df["date"] == target]
    if row.empty:
        logger.warning("No data for date %s", date)
        return output_path

    row_data = row.iloc[0].to_dict()
    pol, econ = _get_period_articles(articles, date, date)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # For daily chart: show surrounding 14 days for context
    ctx_start = (target - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    ctx_end = date

    with PdfPages(str(output_path)) as pdf:
        _cover_page(pdf, "REPORTE DIARIO DE INESTABILIDAD", date, logo_path)
        _summary_page(pdf, row_data, pol, econ, narrative)
        _analysis_page(pdf, index_df, ctx_start, ctx_end, pol, econ,
                       chart_title=f"Tendencia \u2014 \u00daltimos 14 d\u00edas")

    logger.info("Daily PDF: %s (3 pages)", output_path)
    return output_path


def generate_weekly_pdf(year, week, index_path, classified_path, output_path,
                        logo_path=None, narrative=None):
    """3-page weekly PDF: Cover, Executive Summary, Analysis."""
    apply_nexus_style()

    index_df = _load_index(index_path)
    articles = _load_articles(classified_path)

    start = datetime.strptime(f"{year}-W{week:02d}-1", "%G-W%V-%u").date()
    end = start + timedelta(days=6)

    row_data = _aggregate_row(index_df, str(start), str(end))
    if row_data is None:
        logger.warning("No data for week %d of %d", week, year)
        return Path(output_path)

    pol, econ = _get_period_articles(articles, str(start), str(end))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    date_label = f"Semana {week} \u2014 {start.strftime('%d %b')} al {end.strftime('%d %b %Y')}"

    with PdfPages(str(output_path)) as pdf:
        _cover_page(pdf, "REPORTE SEMANAL DE INESTABILIDAD", date_label, logo_path)
        _summary_page(pdf, row_data, pol, econ, narrative)
        _analysis_page(pdf, index_df, str(start), str(end), pol, econ,
                       chart_title=f"Tendencia Semanal \u2014 S{week}")

    logger.info("Weekly PDF: %s (3 pages)", output_path)
    return output_path


def generate_monthly_pdf(year, month, index_path, classified_path, output_path,
                         logo_path=None, narrative=None):
    """4-page monthly PDF: Cover, Executive Summary, Analysis, Weekly Breakdown."""
    apply_nexus_style()

    index_df = _load_index(index_path)
    articles = _load_articles(classified_path)

    _, last_day = calendar.monthrange(year, month)
    start = f"{year}-{month:02d}-01"
    end = f"{year}-{month:02d}-{last_day:02d}"

    row_data = _aggregate_row(index_df, start, end)
    if row_data is None:
        logger.warning("No data for %d-%02d", year, month)
        return Path(output_path)

    pol, econ = _get_period_articles(articles, start, end)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    month_name = MONTHS_ES.get(month, str(month))

    with PdfPages(str(output_path)) as pdf:
        _cover_page(pdf, "REPORTE MENSUAL DE INESTABILIDAD",
                    f"{month_name} {year}", logo_path)
        _summary_page(pdf, row_data, pol, econ, narrative)
        _analysis_page(pdf, index_df, start, end, pol, econ,
                       chart_title=f"Tendencia \u2014 {month_name} {year}")
        _weekly_breakdown_page(pdf, index_df, year, month)

    logger.info("Monthly PDF: %s (4 pages)", output_path)
    return output_path
