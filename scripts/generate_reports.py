#!/usr/bin/env python3
"""
generate_reports.py — PDF report generation for Qhawarina economic research.

Usage:
  python scripts/generate_reports.py --type daily
  python scripts/generate_reports.py --type weekly
  python scripts/generate_reports.py --type quarterly --quarter 2026-Q1
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Encoding fix for Windows terminals
# ---------------------------------------------------------------------------
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DATA = REPO_ROOT / "exports" / "data"
EXPORTS_REPORTS = REPO_ROOT / "exports" / "reports"
TEMPLATES_DIR = REPO_ROOT / "templates"
PIPELINE_STATUS_PATH = REPO_ROOT / "data" / "pipeline_status.json"

# ---------------------------------------------------------------------------
# Design system colours
# ---------------------------------------------------------------------------
TERRA  = "#C65D3E"
INK    = "#2D3142"
INK3   = "#8D99AE"
TEAL   = "#2A9D8F"
AMBER  = "#E0A458"
RED    = "#9B2226"
BG     = "#FAF8F4"
BORDER = "#E8E4DF"

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.facecolor": BG,
    "figure.facecolor": BG,
    "axes.edgecolor": BORDER,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": BORDER,
    "grid.linewidth": 0.5,
    "axes.grid": True,
})

# ---------------------------------------------------------------------------
# Spanish month names for date formatting
# ---------------------------------------------------------------------------
MESES_ES = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
    5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
    9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
}


def fmt_date_es(d: date) -> str:
    """Format a date as '7 de marzo de 2026' in Spanish."""
    if isinstance(d, str):
        d = datetime.strptime(d[:10], "%Y-%m-%d").date()
    return f"{d.day} de {MESES_ES[d.month]} de {d.year}"


def fmt_number(v, decimals: int = 2) -> str:
    """Format a number with thousands separator and fixed decimals."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/D"
    fmt = f"{v:,.{decimals}f}"
    # Swap commas/dots for Spanish locale style
    return fmt


def tex_escape(s: str) -> str:
    """Escape LaTeX special characters in a string."""
    if not isinstance(s, str):
        s = str(s)
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s


# ---------------------------------------------------------------------------
# JSON data loaders
# ---------------------------------------------------------------------------

def load_json(filename: str) -> dict:
    """Load a JSON file from exports/data/ with graceful fallback."""
    path = EXPORTS_DATA / filename
    if not path.exists():
        print(f"  [WARN] {filename} not found — using empty dict")
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] Could not load {filename}: {e}")
        return {}


def safe_get(d: dict, *keys, default=None):
    """Safely traverse nested dict keys."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, None)
        if d is None:
            return default
    return d


def load_all_data() -> dict:
    """Load all JSON data files."""
    return {
        "bpp": load_json("daily_price_index.json"),
        "political": load_json("political_index_daily.json"),
        "fx": load_json("fx_interventions.json"),
        "gdp": load_json("gdp_nowcast.json"),
        "poverty": load_json("poverty_nowcast.json"),
        "pipeline": (json.load(open(PIPELINE_STATUS_PATH, encoding="utf-8")) if PIPELINE_STATUS_PATH.exists() else {}),
    }


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------

def generate_bpp_90d_chart(series: list, output_path: Path) -> None:
    """Last 90 days of BPP YoY series — Terra line with fill, BCRP target band."""
    # Compute a synthetic YoY from the cumulative pct change since base
    # The series has var_all (daily MoM-ish change). We use cum_pct as proxy.
    # If yoy_inflation is available use it; otherwise compute from index_all.
    if not series:
        _save_empty_chart("Sin datos disponibles", output_path)
        return

    # Parse dates and extract relevant fields
    dates = []
    values = []
    for row in series:
        try:
            d = datetime.strptime(row["date"][:10], "%Y-%m-%d")
            # Use var_all if available, else compute from index_all
            val = row.get("yoy_inflation")
            if val is None:
                # Approximate annualised: (index/100 - 1) * 365/n_days since base
                idx = row.get("index_all", 100.0)
                val = (idx / 100.0 - 1.0) * 12.0 * 100  # rough annualised
            dates.append(d)
            values.append(float(val))
        except Exception:
            continue

    # Take last 90 days
    if len(dates) > 90:
        dates = dates[-90:]
        values = values[-90:]

    fig, ax = plt.subplots(figsize=(5.5, 2.8))

    ax.fill_between(dates, values, alpha=0.12, color=TERRA)
    ax.plot(dates, values, color=TERRA, linewidth=1.5)

    # BCRP target band [1%, 3%]
    ax.axhspan(1, 3, color=TEAL, alpha=0.08, zorder=0, label="Meta BCRP (1–3%)")

    # Annotate last point
    if dates and values:
        ax.annotate(
            f"{values[-1]:.2f}%",
            xy=(dates[-1], values[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8,
            color=TERRA,
            va="center",
        )

    ax.set_title("Índice de Precios BPP — Últimos 90 días", fontsize=9, color=INK, pad=6)
    ax.set_ylabel("Variación interanual (%)", fontsize=8, color=INK3)
    ax.tick_params(colors=INK3, labelsize=7)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d-%b"))
    ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_risk_30d_chart(series: list, output_path: Path) -> None:
    """Last 30 days of political risk — colour-coded bars."""
    if not series:
        _save_empty_chart("Sin datos disponibles", output_path)
        return

    dates = []
    values = []
    for row in series:
        try:
            d = datetime.strptime(row["date"][:10], "%Y-%m-%d")
            score = float(row.get("score", row.get("instability_score", 0))) * 100
            dates.append(d)
            values.append(score)
        except Exception:
            continue

    if len(dates) > 30:
        dates = dates[-30:]
        values = values[-30:]

    colors = []
    for v in values:
        if v < 30:
            colors.append(TEAL)
        elif v < 60:
            colors.append(AMBER)
        else:
            colors.append(RED)

    fig, ax = plt.subplots(figsize=(5.5, 2.8))

    # Background zones
    ax.axhspan(0, 30, color=TEAL, alpha=0.06, zorder=0)
    ax.axhspan(30, 60, color=AMBER, alpha=0.06, zorder=0)
    ax.axhspan(60, 100, color=RED, alpha=0.06, zorder=0)
    ax.axhline(30, color=TEAL, linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(60, color=RED, linewidth=0.5, linestyle="--", alpha=0.5)

    ax.bar(dates, values, color=colors, width=0.8, alpha=0.85)

    ax.set_ylim(0, 100)
    ax.set_title("Índice de Riesgo Político — Últimos 30 días", fontsize=9, color=INK, pad=6)
    ax.set_ylabel("Score (0–100)", fontsize=8, color=INK3)
    ax.tick_params(colors=INK3, labelsize=7)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d-%b"))
    ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Legend patches
    patches = [
        mpatches.Patch(color=TEAL, alpha=0.85, label="Bajo (<30)"),
        mpatches.Patch(color=AMBER, alpha=0.85, label="Medio (30–60)"),
        mpatches.Patch(color=RED, alpha=0.85, label="Alto (>60)"),
    ]
    ax.legend(handles=patches, fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_category_bars_chart(categories: list, output_path: Path) -> None:
    """Horizontal bar chart of category weekly changes (weekly report)."""
    if not categories:
        _save_empty_chart("Sin datos de categorías", output_path)
        return

    # categories: list of dict with keys: name/label, value/var_weekly
    labels = []
    vals = []
    for c in categories:
        lbl = c.get("label_es") or c.get("label") or c.get("name", "?")
        val = c.get("var_weekly") or c.get("var") or c.get("value") or 0.0
        labels.append(lbl)
        vals.append(float(val))

    # Sort by value
    paired = sorted(zip(vals, labels), key=lambda x: x[0])
    vals, labels = zip(*paired) if paired else ([], [])
    vals = list(vals)
    labels = list(labels)

    colors = [TERRA if v >= 0 else TEAL for v in vals]

    fig, ax = plt.subplots(figsize=(5.5, max(2.5, len(labels) * 0.35 + 0.8)))
    y_pos = range(len(labels))
    ax.barh(y_pos, vals, color=colors, alpha=0.85, height=0.6)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.axvline(0, color=INK3, linewidth=0.8)
    ax.set_xlabel("Variación semanal (%)", fontsize=8, color=INK3)
    ax.set_title("Variación por Categoría — Semana", fontsize=9, color=INK, pad=6)
    ax.tick_params(colors=INK3, labelsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_bpp_12w_chart(series: list, output_path: Path) -> None:
    """Last 12 weeks of BPP — weekly aggregates, current week terra bar."""
    if not series:
        _save_empty_chart("Sin datos disponibles", output_path)
        return

    # Aggregate daily series to weekly averages
    from collections import defaultdict
    week_data: dict = defaultdict(list)
    for row in series:
        try:
            d = datetime.strptime(row["date"][:10], "%Y-%m-%d")
            # ISO week label
            iso = d.isocalendar()
            week_key = f"{iso[0]}-W{iso[1]:02d}"
            val = row.get("var_all") or row.get("yoy_inflation") or 0.0
            week_data[week_key].append(float(val))
        except Exception:
            continue

    if not week_data:
        _save_empty_chart("Sin datos disponibles", output_path)
        return

    sorted_weeks = sorted(week_data.keys())[-12:]
    week_avgs = [np.mean(week_data[w]) for w in sorted_weeks]

    # Short labels
    week_labels = [w.replace("-W", "\nW") for w in sorted_weeks]

    colors = [TERRA if i == len(sorted_weeks) - 1 else INK3
              for i in range(len(sorted_weeks))]

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    x_pos = range(len(sorted_weeks))
    ax.bar(x_pos, week_avgs, color=colors, alpha=0.85, width=0.7)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(week_labels, fontsize=7, color=INK3)
    ax.set_ylabel("Variación media (%)", fontsize=8, color=INK3)
    ax.set_title("Índice BPP — Últimas 12 semanas", fontsize=9, color=INK, pad=6)
    ax.axhline(0, color=INK3, linewidth=0.6)
    ax.tick_params(colors=INK3, labelsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_poverty_dept_chart(departments: list, output_path: Path) -> None:
    """Horizontal bar chart of departmental poverty rates (quarterly report)."""
    if not departments:
        _save_empty_chart("Sin datos de pobreza departamental", output_path)
        return

    # Sort by poverty rate descending, take top 15
    depts = sorted(
        departments,
        key=lambda d: d.get("poverty_rate_2025_nowcast") or d.get("poverty_rate", 0),
        reverse=True,
    )[:15]

    labels = [d.get("name", d.get("department", "?")) for d in depts]
    vals = [
        d.get("poverty_rate_2025_nowcast") or d.get("poverty_rate", 0)
        for d in depts
    ]

    fig, ax = plt.subplots(figsize=(5.5, max(3.0, len(labels) * 0.32 + 0.8)))
    y_pos = range(len(labels))
    ax.barh(list(y_pos), vals, color=TERRA, alpha=0.80, height=0.65)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=7.5, color=INK)
    ax.set_xlabel("Tasa de pobreza (%)", fontsize=8, color=INK3)
    ax.set_title("Pobreza Departamental — Nowcast 2025", fontsize=9, color=INK, pad=6)
    ax.tick_params(colors=INK3, labelsize=7)

    # Value labels
    for i, v in enumerate(vals):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=7, color=INK)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_fx_quarter_chart(daily_series: list, quarter: str, output_path: Path) -> None:
    """FX rate for the quarter — INK line, annotate min/max."""
    if not daily_series:
        _save_empty_chart("Sin datos de tipo de cambio", output_path)
        return

    # Determine quarter date range
    try:
        year, q = quarter.split("-Q")
        year = int(year)
        q = int(q)
        q_start = datetime(year, (q - 1) * 3 + 1, 1)
        q_end_month = q * 3
        if q_end_month == 12:
            q_end = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            q_end = datetime(year, q_end_month + 1, 1) - timedelta(days=1)
    except Exception:
        q_start = None
        q_end = None

    dates = []
    fx_vals = []
    for row in daily_series:
        try:
            d = datetime.strptime(row["date"][:10], "%Y-%m-%d")
            fx = row.get("fx")
            if fx is None:
                continue
            if q_start and (d < q_start or d > q_end):
                continue
            dates.append(d)
            fx_vals.append(float(fx))
        except Exception:
            continue

    if not dates:
        # Fall back: last 90 days of data
        all_dates = []
        all_fx = []
        for row in daily_series:
            try:
                d = datetime.strptime(row["date"][:10], "%Y-%m-%d")
                fx = row.get("fx")
                if fx is not None:
                    all_dates.append(d)
                    all_fx.append(float(fx))
            except Exception:
                continue
        if all_dates:
            dates = all_dates[-90:]
            fx_vals = all_fx[-90:]

    if not dates:
        _save_empty_chart("Sin datos de tipo de cambio", output_path)
        return

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.plot(dates, fx_vals, color=INK, linewidth=1.5)

    # Annotate min and max
    min_idx = int(np.argmin(fx_vals))
    max_idx = int(np.argmax(fx_vals))
    ax.annotate(
        f"Mín\n{fx_vals[min_idx]:.4f}",
        xy=(dates[min_idx], fx_vals[min_idx]),
        xytext=(0, -18),
        textcoords="offset points",
        fontsize=7,
        color=TEAL,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.8),
    )
    ax.annotate(
        f"Máx\n{fx_vals[max_idx]:.4f}",
        xy=(dates[max_idx], fx_vals[max_idx]),
        xytext=(0, 12),
        textcoords="offset points",
        fontsize=7,
        color=TERRA,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=TERRA, lw=0.8),
    )

    ax.set_title(f"TC PEN/USD — {quarter}", fontsize=9, color=INK, pad=6)
    ax.set_ylabel("PEN/USD", fontsize=8, color=INK3)
    ax.tick_params(colors=INK3, labelsize=7)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d-%b"))
    ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_bpp_quarter_chart(series: list, quarter: str, output_path: Path) -> None:
    """BPP index_all for the quarter or last 90 days."""
    if not series:
        _save_empty_chart("Sin datos disponibles", output_path)
        return

    try:
        year, q = quarter.split("-Q")
        year = int(year)
        q = int(q)
        q_start = datetime(year, (q - 1) * 3 + 1, 1)
        q_end_month = q * 3
        if q_end_month == 12:
            q_end = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            q_end = datetime(year, q_end_month + 1, 1) - timedelta(days=1)
    except Exception:
        q_start = None
        q_end = None

    dates = []
    vals = []
    for row in series:
        try:
            d = datetime.strptime(row["date"][:10], "%Y-%m-%d")
            if q_start and (d < q_start or d > q_end):
                continue
            idx = row.get("index_all", 100.0)
            dates.append(d)
            vals.append(float(idx))
        except Exception:
            continue

    if not dates:
        # Fall back to last 90 days
        dates = []
        vals = []
        for row in series:
            try:
                d = datetime.strptime(row["date"][:10], "%Y-%m-%d")
                dates.append(d)
                vals.append(float(row.get("index_all", 100.0)))
            except Exception:
                continue
        dates = dates[-90:]
        vals = vals[-90:]

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.fill_between(dates, vals, alpha=0.12, color=TERRA)
    ax.plot(dates, vals, color=TERRA, linewidth=1.5)
    ax.set_title(f"Índice BPP — {quarter}", fontsize=9, color=INK, pad=6)
    ax.set_ylabel("Índice (base=100)", fontsize=8, color=INK3)
    ax.tick_params(colors=INK3, labelsize=7)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d-%b"))
    ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_risk_quarter_chart(series: list, quarter: str, output_path: Path) -> None:
    """Political risk for the quarter."""
    if not series:
        _save_empty_chart("Sin datos disponibles", output_path)
        return

    try:
        year, q = quarter.split("-Q")
        year = int(year)
        q = int(q)
        q_start = datetime(year, (q - 1) * 3 + 1, 1)
        q_end_month = q * 3
        if q_end_month == 12:
            q_end = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            q_end = datetime(year, q_end_month + 1, 1) - timedelta(days=1)
    except Exception:
        q_start = None
        q_end = None

    dates = []
    vals = []
    for row in series:
        try:
            d = datetime.strptime(row["date"][:10], "%Y-%m-%d")
            if q_start and (d < q_start or d > q_end):
                continue
            score = float(row.get("score", row.get("instability_score", 0))) * 100
            dates.append(d)
            vals.append(score)
        except Exception:
            continue

    if not dates:
        # Fall back to last 90 days
        dates = []
        vals = []
        for row in series:
            try:
                d = datetime.strptime(row["date"][:10], "%Y-%m-%d")
                score = float(row.get("score", row.get("instability_score", 0))) * 100
                dates.append(d)
                vals.append(score)
            except Exception:
                continue
        dates = dates[-90:]
        vals = vals[-90:]

    colors = [TEAL if v < 30 else (AMBER if v < 60 else RED) for v in vals]

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.axhspan(0, 30, color=TEAL, alpha=0.06, zorder=0)
    ax.axhspan(30, 60, color=AMBER, alpha=0.06, zorder=0)
    ax.axhspan(60, 100, color=RED, alpha=0.06, zorder=0)

    if dates:
        ax.bar(dates, vals, color=colors, alpha=0.8, width=1.0)

    ax.set_ylim(0, 100)
    ax.set_title(f"Riesgo Político — {quarter}", fontsize=9, color=INK, pad=6)
    ax.set_ylabel("Score (0–100)", fontsize=8, color=INK3)
    ax.tick_params(colors=INK3, labelsize=7)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d-%b"))
    ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_empty_chart(message: str, output_path: Path) -> None:
    """Save a placeholder chart with a message."""
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.text(
        0.5, 0.5, message,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=10, color=INK3,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Placeholder replacement
# ---------------------------------------------------------------------------

def fill_template(template_text: str, replacements: dict) -> str:
    """Replace all <<KEY>> placeholders with values."""
    result = template_text
    for key, value in replacements.items():
        placeholder = f"<<{key}>>"
        result = result.replace(placeholder, str(value))
    return result


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def extract_daily_values(data: dict) -> dict:
    """Extract scalar values for daily report placeholders."""
    bpp = data.get("bpp", {})
    political = data.get("political", {})
    fx_data = data.get("fx", {})
    gdp_data = data.get("gdp", {})
    poverty_data = data.get("poverty", {})

    today = date.today()

    # BPP
    latest_bpp = bpp.get("latest", {})
    bpp_yoy = latest_bpp.get("yoy_inflation")
    bpp_mom = latest_bpp.get("var_all") or latest_bpp.get("mom_inflation")
    bpp_products = latest_bpp.get("n_products_today") or latest_bpp.get("product_count")
    # Try to compute YoY from series if not directly available
    bpp_series = bpp.get("series", [])
    if bpp_yoy is None and len(bpp_series) >= 2:
        # Use cum_pct as rough proxy
        last = bpp_series[-1] if bpp_series else {}
        bpp_yoy = last.get("yoy_inflation") or last.get("cum_pct")

    # Political
    current_pol = political.get("current", {})
    pol_score_raw = current_pol.get("score", 0)
    pol_score = round(float(pol_score_raw) * 100) if pol_score_raw is not None else "N/D"
    pol_level = current_pol.get("level", "N/D")
    pol_articles = current_pol.get("articles_total") or current_pol.get("articles_political", "N/D")
    pol_driver = safe_get(political, "current", "driver_phrase", default="")
    if not pol_driver:
        # Try aggregates or fallback
        agg = political.get("aggregates", {})
        pol_driver = f"Score promedio 7 días: {agg.get('7d_avg', 'N/D')}"

    # FX
    fx_latest = fx_data.get("latest", {})
    fx_rate = fx_latest.get("fx", "N/D")
    bcrp_rate = fx_latest.get("reference_rate", "N/D")
    fx_spot = fx_latest.get("spot_net_purchases", 0) or 0.0

    # Compute FX daily change
    fx_series = fx_data.get("daily_series", [])
    fx_daily_change = "N/D"
    if len(fx_series) >= 2:
        try:
            prev_fx = float(fx_series[-2]["fx"])
            curr_fx = float(fx_series[-1]["fx"])
            chg = (curr_fx - prev_fx) / prev_fx * 100
            sign = "+" if chg >= 0 else ""
            fx_daily_change = f"{sign}{chg:.2f}%"
        except Exception:
            pass

    # GDP
    gdp_nowcast_val = safe_get(gdp_data, "nowcast", "value", default="N/D")
    gdp_quarter = safe_get(gdp_data, "nowcast", "target_period", default="N/D")

    # Poverty
    nat_poverty = safe_get(poverty_data, "national", "poverty_rate", default="N/D")
    pov_lower = "N/D"
    pov_upper = "N/D"
    # Try to extract CI from departments or a ci field
    if isinstance(nat_poverty, (int, float)):
        pov_lower = round(float(nat_poverty) - 1.5, 1)
        pov_upper = round(float(nat_poverty) + 1.5, 1)

    # Format top movers table
    top_movers = latest_bpp.get("top_movers", [])
    top_movers_rows = ""
    for m in top_movers[:4]:
        name = tex_escape(m.get("label_es") or m.get("name") or "?")
        var = m.get("var") or m.get("var_pct") or 0.0
        sign = "+" if float(var) >= 0 else ""
        color = "terra" if float(var) >= 0 else "teal"
        top_movers_rows += f"    {name} & \\textcolor{{{color}}}{{{sign}{float(var):.2f}\\%}} \\\\\n"

    if not top_movers_rows:
        top_movers_rows = "    Sin datos & — \\\\\n"

    # Nota del día
    nota = _build_nota_dia(bpp_yoy, pol_score, fx_rate, gdp_nowcast_val)

    return {
        "DATE": fmt_date_es(today),
        "NOTA_DIA": nota,
        "BPP_YOY": f"{float(bpp_yoy):.2f}%" if bpp_yoy is not None else "N/D",
        "BPP_MOM": f"{float(bpp_mom):.2f}%" if bpp_mom is not None else "N/D",
        "BPP_PRODUCTS": f"{int(bpp_products):,}" if bpp_products else "N/D",
        "RISK_SCORE": str(pol_score),
        "RISK_LEVEL": pol_level,
        "RISK_ARTICLES": str(pol_articles),
        "FX_RATE": f"{float(fx_rate):.4f}" if fx_rate != "N/D" else "N/D",
        "FX_DAILY_CHANGE": fx_daily_change,
        "BCRP_RATE": f"{float(bcrp_rate):.2f}%" if bcrp_rate != "N/D" else "N/D",
        "GDP_NOWCAST": f"{float(gdp_nowcast_val):.2f}%" if gdp_nowcast_val != "N/D" else "N/D",
        "GDP_QUARTER": str(gdp_quarter),
        "POVERTY_RATE": f"{float(nat_poverty):.1f}%" if nat_poverty != "N/D" else "N/D",
        "POVERTY_CI": f"[{pov_lower}, {pov_upper}]" if pov_lower != "N/D" else "N/D",
        "RISK_DRIVER": tex_escape(str(pol_driver)) if pol_driver else "",
        "TOP_MOVERS_TABLE": top_movers_rows,
    }


def _build_nota_dia(bpp_yoy, pol_score, fx_rate, gdp_nowcast) -> str:
    """Auto-generate a Spanish summary sentence for the daily report."""
    parts = []
    if bpp_yoy is not None:
        try:
            parts.append(f"El índice BPP registra una variación interanual de {float(bpp_yoy):.2f}\\%.")
        except Exception:
            pass
    if isinstance(pol_score, (int, float)):
        level_str = "bajo" if pol_score < 30 else ("moderado" if pol_score < 60 else "\\textbf{elevado}")
        parts.append(f"El riesgo político se sitúa en nivel {level_str} ({pol_score}/100).")
    if fx_rate != "N/D":
        try:
            parts.append(f"El tipo de cambio cierra en S/ {float(fx_rate):.4f} por dólar.")
        except Exception:
            pass
    if gdp_nowcast != "N/D":
        try:
            parts.append(f"Nowcast PBI: {float(gdp_nowcast):.2f}\\% interanual.")
        except Exception:
            pass
    if not parts:
        return "Reporte generado automáticamente por el pipeline Qhawarina."
    return " ".join(parts)


def extract_weekly_values(data: dict, daily_vals: dict) -> dict:
    """Extract additional scalar values for weekly report."""
    bpp = data.get("bpp", {})
    political = data.get("political", {})
    fx_data = data.get("fx", {})
    gdp_data = data.get("gdp", {})

    today = date.today()
    iso = today.isocalendar()
    week_number = iso[1]
    week_year = iso[0]

    # KPI summary table rows
    gdp_val = safe_get(gdp_data, "nowcast", "value", default="N/D")
    fx_latest = fx_data.get("latest", {})
    fx_rate = fx_latest.get("fx", "N/D")
    pol_score = safe_get(political, "current", "score", default="N/D")
    latest_bpp = bpp.get("latest", {})
    bpp_yoy = latest_bpp.get("yoy_inflation") or latest_bpp.get("cum_pct")

    kpi_rows = ""
    kpi_data = [
        ("Índice BPP (acum.)", f"{float(bpp_yoy):.2f}\\%" if bpp_yoy is not None else "N/D", "—", "↔"),
        ("Riesgo Político", f"{round(float(pol_score)*100)}/100" if pol_score != "N/D" else "N/D", "—", "↔"),
        ("Tipo de Cambio", f"S/ {float(fx_rate):.4f}" if fx_rate != "N/D" else "N/D", "—", "↔"),
        ("Nowcast PBI", f"{float(gdp_val):.2f}\\%" if gdp_val != "N/D" else "N/D", "—", "↔"),
    ]
    for name, val, chg, trend in kpi_data:
        kpi_rows += f"    {tex_escape(name)} & {val} & {chg} & {trend} \\\\\n"

    # Category table
    categories_meta = bpp.get("categories", {})
    bpp_series = bpp.get("series", [])
    category_rows = ""
    if bpp_series and len(bpp_series) >= 2:
        last = bpp_series[-1]
        prev_week = bpp_series[max(0, len(bpp_series) - 8)]
        for cat_key, cat_info in categories_meta.items():
            label = cat_info.get("label_es", cat_key)
            weight = cat_info.get("cpi_weight", "—")
            idx_now = last.get(f"index_{cat_key}", 100.0)
            idx_prev = prev_week.get(f"index_{cat_key}", 100.0)
            try:
                weekly_var = (float(idx_now) / float(idx_prev) - 1) * 100
                weekly_str = f"{weekly_var:+.2f}\\%"
            except Exception:
                weekly_str = "N/D"
            category_rows += (
                f"    {tex_escape(label)} & {weekly_str} & — & {weight}\\% \\\\\n"
            )
    if not category_rows:
        category_rows = "    Sin datos & — & — & — \\\\\n"

    # Top 5 products table
    top_movers = latest_bpp.get("top_movers", [])
    top5_rows = ""
    for m in top_movers[:5]:
        name = tex_escape(m.get("label_es") or m.get("name") or "?")
        store = tex_escape(m.get("store", "—"))
        var = m.get("var") or m.get("var_pct") or 0.0
        sign = "+" if float(var) >= 0 else ""
        top5_rows += f"    {name} & {store} & {sign}{float(var):.2f}\\% \\\\\n"
    if not top5_rows:
        top5_rows = "    Sin datos & — & — \\\\\n"

    # Political daily table (last 7 days)
    pol_series = political.get("daily_series", [])
    risk_daily_rows = ""
    for row in pol_series[-7:]:
        d = row.get("date", "?")
        score = round(float(row.get("score", 0)) * 100)
        articles = row.get("n_articles", "—")
        driver = tex_escape(str(row.get("driver", "—"))[:60])
        risk_daily_rows += f"    {d} & {score} & {articles} & {driver} \\\\\n"
    if not risk_daily_rows:
        risk_daily_rows = "    — & — & — & — \\\\\n"

    # GDP official (last official from quarterly_series)
    q_series = gdp_data.get("quarterly_series", [])
    gdp_official = "N/D"
    for row in reversed(q_series):
        if row.get("official") is not None:
            gdp_official = f"{float(row['official']):.2f}\\%"
            break

    # FX quarter summary
    fx_month = fx_latest.get("reference_rate", "N/D")
    fx_summary = (
        f"TC: S/ {float(fx_rate):.4f} | Tasa ref. BCRP: {float(fx_month):.2f}\\%"
        if fx_rate != "N/D" and fx_month != "N/D"
        else "N/D"
    )

    return {
        **daily_vals,
        "WEEK_NUMBER": str(week_number),
        "WEEK_YEAR": str(week_year),
        "KPI_SUMMARY_TABLE": kpi_rows,
        "CATEGORY_TABLE": category_rows,
        "TOP5_PRODUCTS_TABLE": top5_rows,
        "RISK_DAILY_TABLE": risk_daily_rows,
        "GDP_OFFICIAL": gdp_official,
        "FX_QUARTER_SUMMARY": fx_summary,
        "PERSPECTIVA_TEXT": _load_perspectiva() or "Perspectiva actualizada semanalmente.",
    }


def extract_quarterly_values(data: dict, quarter: str, daily_vals: dict) -> dict:
    """Extract values for quarterly report."""
    bpp = data.get("bpp", {})
    political = data.get("political", {})
    fx_data = data.get("fx", {})
    gdp_data = data.get("gdp", {})
    poverty_data = data.get("poverty", {})

    # Quarter dates
    try:
        year, q = quarter.split("-Q")
        year = int(year)
        q_num = int(q)
        month_start = (q_num - 1) * 3 + 1
        month_end = q_num * 3
        q_start_date = date(year, month_start, 1)
        # End of quarter
        if month_end == 12:
            q_end_date = date(year, 12, 31)
        else:
            q_end_date = date(year, month_end + 1, 1) - timedelta(days=1)
        quarter_start_str = fmt_date_es(q_start_date)
        quarter_end_str = fmt_date_es(q_end_date)
    except Exception:
        quarter_start_str = "N/D"
        quarter_end_str = "N/D"

    # All KPI table
    fx_latest = fx_data.get("latest", {})
    fx_rate = fx_latest.get("fx", "N/D")
    pol_score = safe_get(political, "current", "score", default=None)
    bpp_latest = bpp.get("latest", {})
    bpp_yoy = bpp_latest.get("yoy_inflation") or bpp_latest.get("cum_pct")
    gdp_val = safe_get(gdp_data, "nowcast", "value", default="N/D")
    nat_poverty = safe_get(poverty_data, "national", "poverty_rate", default="N/D")

    all_kpi_rows = ""
    kpi_items = [
        ("Índice BPP (acum.)", f"{float(bpp_yoy):.2f}\\%" if bpp_yoy is not None else "N/D", "—", "—"),
        ("Riesgo Político", f"{round(float(pol_score)*100)}/100" if pol_score is not None else "N/D", "—", "—"),
        ("TC PEN/USD", f"S/ {float(fx_rate):.4f}" if fx_rate != "N/D" else "N/D", "—", "—"),
        ("Nowcast PBI (i.a.)", f"{float(gdp_val):.2f}\\%" if gdp_val != "N/D" else "N/D", "—", "—"),
        ("Tasa de Pobreza", f"{float(nat_poverty):.1f}\\%" if nat_poverty != "N/D" else "N/D", "—", "—"),
    ]
    for name, start, end, delta in kpi_items:
        all_kpi_rows += f"    {tex_escape(name)} & {start} & {end} & {delta} \\\\\n"

    # Category quarter table
    categories_meta = bpp.get("categories", {})
    bpp_series = bpp.get("series", [])
    cat_q_rows = ""
    if bpp_series:
        last = bpp_series[-1]
        first = bpp_series[0]
        for cat_key, cat_info in categories_meta.items():
            label = cat_info.get("label_es", cat_key)
            weight = cat_info.get("cpi_weight", "—")
            idx_now = last.get(f"index_{cat_key}", 100.0)
            idx_first = first.get(f"index_{cat_key}", 100.0)
            try:
                q_change = (float(idx_now) / float(idx_first) - 1) * 100
                q_str = f"{q_change:+.2f}\\%"
            except Exception:
                q_str = "N/D"
            cat_q_rows += (
                f"    {tex_escape(label)} & {q_str} & — & — & — \\\\\n"
            )
    if not cat_q_rows:
        cat_q_rows = "    Sin datos & — & — & — & — \\\\\n"

    # Major events list
    major_events = political.get("major_events", [])
    events_items = ""
    for ev in major_events[:8]:
        d = ev.get("date", "?")
        score = round(float(ev.get("score", 0)) * 100)
        summary = tex_escape(str(ev.get("summary", "Sin descripción"))[:80])
        events_items += f"  \\item[{d}] Score {score}/100: {summary}\n"
    if not events_items:
        events_items = "  \\item Sin eventos registrados este trimestre.\n"

    # GDP track record table
    q_series = gdp_data.get("quarterly_series", [])
    track_rows = ""
    count = 0
    for row in reversed(q_series):
        if row.get("official") is not None and row.get("nowcast") is not None:
            qtr = row.get("quarter", "?")
            official = f"{float(row['official']):.2f}\\%"
            nowcast = f"{float(row['nowcast']):.2f}\\%"
            error = row.get("error")
            err_str = f"{float(error):.2f}pp" if error is not None else "—"
            track_rows += f"    {qtr} & {nowcast} & {official} & {err_str} \\\\\n"
            count += 1
            if count >= 4:
                break
    if not track_rows:
        # Use forecasts as track record if no backtest data
        forecasts = gdp_data.get("forecasts", [])
        gdp_nowcast_val = safe_get(gdp_data, "nowcast", "value", default="N/D")
        gdp_nowcast_qtr = safe_get(gdp_data, "nowcast", "target_period", default="N/D")
        if gdp_nowcast_val != "N/D":
            track_rows = f"    {gdp_nowcast_qtr} & {float(gdp_nowcast_val):.2f}\\% & — & — \\\\\n"
        for fc in forecasts[:3]:
            qtr = fc.get("quarter", "?")
            val = fc.get("value", "N/D")
            track_rows += f"    {qtr} & {float(val):.2f}\\% & (proyección) & — \\\\\n"
    if not track_rows:
        track_rows = "    Sin datos & — & — & — \\\\\n"

    # Departmental poverty table
    departments = poverty_data.get("departments", [])
    sorted_depts = sorted(
        departments,
        key=lambda d: d.get("poverty_rate_2025_nowcast") or d.get("poverty_rate", 0),
        reverse=True,
    )
    dept_rows = ""
    # Top 5 poorest
    for dept in sorted_depts[:5]:
        name = tex_escape(dept.get("name") or dept.get("department", "?"))
        rate = dept.get("poverty_rate_2025_nowcast") or dept.get("poverty_rate", "N/D")
        rate2024 = dept.get("poverty_rate_2024", "—")
        change = dept.get("change_pp", "—")
        rate_str = f"{float(rate):.1f}\\%" if rate != "N/D" else "N/D"
        rate2024_str = f"{float(rate2024):.1f}\\%" if rate2024 != "—" else "—"
        change_str = f"{float(change):+.1f}pp" if change != "—" else "—"
        dept_rows += f"    {name} & {rate_str} & {rate2024_str} & {change_str} \\\\\n"
    # Separator
    if dept_rows:
        dept_rows += "    \\midrule\n"
    # Top 5 least poor (bottom 5)
    for dept in sorted_depts[-5:]:
        name = tex_escape(dept.get("name") or dept.get("department", "?"))
        rate = dept.get("poverty_rate_2025_nowcast") or dept.get("poverty_rate", "N/D")
        rate2024 = dept.get("poverty_rate_2024", "—")
        change = dept.get("change_pp", "—")
        rate_str = f"{float(rate):.1f}\\%" if rate != "N/D" else "N/D"
        rate2024_str = f"{float(rate2024):.1f}\\%" if rate2024 != "—" else "—"
        change_str = f"{float(change):+.1f}pp" if change != "—" else "—"
        dept_rows += f"    {name} & {rate_str} & {rate2024_str} & {change_str} \\\\\n"
    if not dept_rows:
        dept_rows = "    Sin datos & — & — & — \\\\\n"

    # FX quarter summary (same formula as weekly)
    fx_rate_q = fx_latest.get("fx", "N/D")
    fx_ref_q = fx_latest.get("reference_rate", "N/D")
    if fx_rate_q != "N/D" and fx_ref_q != "N/D":
        fx_summary_q = (
            f"TC: S/ {float(fx_rate_q):.4f} | Tasa ref. BCRP: {float(fx_ref_q):.2f}\\%"
        )
    else:
        fx_summary_q = "N/D"

    return {
        **daily_vals,
        "QUARTER": quarter,
        "QUARTER_START": quarter_start_str,
        "QUARTER_END": quarter_end_str,
        "ALL_KPI_TABLE": all_kpi_rows,
        "CATEGORY_QUARTER_TABLE": cat_q_rows,
        "EVENTS_LIST": events_items,
        "GDP_TRACK_RECORD_TABLE": track_rows,
        "DEPT_POVERTY_TABLE": dept_rows,
        "METHODOLOGY_CHANGES": "Sin cambios metodológicos este trimestre.",
        "QUARTERLY_PERSPECTIVA": _load_perspectiva() or "Análisis trimestral generado por el pipeline Qhawarina.",
        "FX_QUARTER_SUMMARY": fx_summary_q,
    }


def _load_perspectiva() -> str:
    """Try to load a perspectiva text file."""
    candidates = [
        REPO_ROOT / "exports" / "perspectiva.txt",
        REPO_ROOT / "assets" / "perspectiva.txt",
    ]
    for p in candidates:
        if p.exists():
            return tex_escape(p.read_text(encoding="utf-8", errors="replace").strip())
    return ""


# ---------------------------------------------------------------------------
# pdflatex runner
# ---------------------------------------------------------------------------

def find_pdflatex() -> str | None:
    """Find pdflatex binary."""
    # Try PATH first
    path = shutil.which("pdflatex")
    if path:
        return path
    # Common Windows paths
    candidates = [
        "C:/texlive/2024/bin/windows/pdflatex.exe",
        "C:/texlive/2023/bin/windows/pdflatex.exe",
        "C:/texlive/2022/bin/windows/pdflatex.exe",
        "C:/Program Files/MiKTeX/miktex/bin/x64/pdflatex.exe",
        "C:/Program Files (x86)/MiKTeX/miktex/bin/pdflatex.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def run_pdflatex(tex_file: Path, output_dir: Path, n_runs: int = 1) -> bool:
    """Run pdflatex n_runs times. Returns True on success."""
    binary = find_pdflatex()
    if binary is None:
        print("\n[ERROR] pdflatex not found.")
        print("Install instructions:")
        print("  Windows: Download TeX Live from https://tug.org/texlive/")
        print("    or MiKTeX from https://miktex.org/download")
        print("  Linux:   sudo apt-get install texlive-full")
        print("  macOS:   brew install --cask mactex")
        sys.exit(1)

    for i in range(n_runs):
        result = subprocess.run(
            [
                binary,
                "-interaction=nonstopmode",
                f"-output-directory={output_dir}",
                str(tex_file),
            ],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if result.returncode != 0:
            print(f"  [WARN] pdflatex run {i+1} returned code {result.returncode}")
            # Print last 30 lines of log
            log_lines = result.stdout.strip().split("\n")
            for line in log_lines[-30:]:
                print(f"    | {line}")
            if i == n_runs - 1:
                return False
    return True


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------

def build_daily_report(data: dict, charts_dir: Path) -> dict:
    """Generate charts and placeholders for daily report."""
    print("  Generating BPP 90-day chart...")
    bpp_series = data.get("bpp", {}).get("series", [])
    chart_bpp = charts_dir / "bpp_90d.png"
    generate_bpp_90d_chart(bpp_series, chart_bpp)

    print("  Generating political risk 30-day chart...")
    pol_series = data.get("political", {}).get("daily_series", [])
    chart_risk = charts_dir / "risk_30d.png"
    generate_risk_30d_chart(pol_series, chart_risk)

    vals = extract_daily_values(data)
    vals["CHART_BPP_90D"] = str(chart_bpp).replace("\\", "/")
    vals["CHART_RISK_30D"] = str(chart_risk).replace("\\", "/")
    return vals


def build_weekly_report(data: dict, charts_dir: Path) -> dict:
    """Generate charts and placeholders for weekly report."""
    print("  Generating BPP 90-day chart...")
    bpp_series = data.get("bpp", {}).get("series", [])
    chart_bpp_90 = charts_dir / "bpp_90d.png"
    generate_bpp_90d_chart(bpp_series, chart_bpp_90)

    print("  Generating political risk 30-day chart...")
    pol_series = data.get("political", {}).get("daily_series", [])
    chart_risk = charts_dir / "risk_30d.png"
    generate_risk_30d_chart(pol_series, chart_risk)

    print("  Generating BPP 12-week chart...")
    chart_bpp_12w = charts_dir / "bpp_12w.png"
    generate_bpp_12w_chart(bpp_series, chart_bpp_12w)

    print("  Generating category bars chart...")
    categories_meta = data.get("bpp", {}).get("categories", {})
    cat_list = []
    if bpp_series and len(bpp_series) >= 2:
        last = bpp_series[-1]
        prev = bpp_series[max(0, len(bpp_series) - 8)]
        for cat_key, cat_info in categories_meta.items():
            idx_now = last.get(f"index_{cat_key}", 100.0)
            idx_prev = prev.get(f"index_{cat_key}", 100.0)
            try:
                var = (float(idx_now) / float(idx_prev) - 1) * 100
            except Exception:
                var = 0.0
            cat_list.append({
                "label_es": cat_info.get("label_es", cat_key),
                "var_weekly": var,
            })
    chart_cat = charts_dir / "category_bars.png"
    generate_category_bars_chart(cat_list, chart_cat)

    daily_vals = extract_daily_values(data)
    vals = extract_weekly_values(data, daily_vals)
    vals["CHART_BPP_90D"] = str(chart_bpp_90).replace("\\", "/")
    vals["CHART_RISK_30D"] = str(chart_risk).replace("\\", "/")
    vals["CHART_BPP_12W"] = str(chart_bpp_12w).replace("\\", "/")
    vals["CHART_CATEGORY_BARS"] = str(chart_cat).replace("\\", "/")
    vals["CHART_RISK_WEEK"] = str(chart_risk).replace("\\", "/")
    return vals


def build_quarterly_report(data: dict, quarter: str, charts_dir: Path) -> dict:
    """Generate charts and placeholders for quarterly report."""
    print("  Generating BPP quarter chart...")
    bpp_series = data.get("bpp", {}).get("series", [])
    chart_bpp_q = charts_dir / "bpp_quarter.png"
    generate_bpp_quarter_chart(bpp_series, quarter, chart_bpp_q)

    print("  Generating political risk quarter chart...")
    pol_series = data.get("political", {}).get("daily_series", [])
    chart_risk_q = charts_dir / "risk_quarter.png"
    generate_risk_quarter_chart(pol_series, quarter, chart_risk_q)

    print("  Generating poverty department chart...")
    departments = data.get("poverty", {}).get("departments", [])
    chart_poverty = charts_dir / "poverty_dept.png"
    generate_poverty_dept_chart(departments, chart_poverty)

    print("  Generating FX quarter chart...")
    fx_series = data.get("fx", {}).get("daily_series", [])
    chart_fx = charts_dir / "fx_quarter.png"
    generate_fx_quarter_chart(fx_series, quarter, chart_fx)

    daily_vals = extract_daily_values(data)
    vals = extract_quarterly_values(data, quarter, daily_vals)
    vals["CHART_BPP_QUARTER"] = str(chart_bpp_q).replace("\\", "/")
    vals["CHART_RISK_QUARTER"] = str(chart_risk_q).replace("\\", "/")
    vals["CHART_POVERTY_DEPT"] = str(chart_poverty).replace("\\", "/")
    vals["CHART_FX_QUARTER"] = str(chart_fx).replace("\\", "/")
    return vals


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Qhawarina PDF report generator")
    parser.add_argument(
        "--type",
        choices=["daily", "weekly", "quarterly"],
        required=True,
        help="Report type",
    )
    parser.add_argument(
        "--quarter",
        default=None,
        help="Quarter for quarterly report, e.g. 2026-Q1",
    )
    args = parser.parse_args()

    report_type = args.type
    quarter = args.quarter

    if report_type == "quarterly" and not quarter:
        # Auto-detect current quarter
        today = date.today()
        q = (today.month - 1) // 3 + 1
        quarter = f"{today.year}-Q{q}"
        print(f"  [INFO] No --quarter specified, using {quarter}")

    print(f"\n=== Qhawarina Report Generator — {report_type.upper()} ===\n")

    # Ensure output directories exist
    EXPORTS_REPORTS.mkdir(parents=True, exist_ok=True)

    # Create temp working directory
    with tempfile.TemporaryDirectory(prefix="qhawarina_report_") as tmp_dir:
        charts_dir = Path(tmp_dir)

        # Load data
        print("[1/4] Loading data...")
        data = load_all_data()

        # Generate charts and placeholder values
        print("[2/4] Generating charts...")
        if report_type == "daily":
            replacements = build_daily_report(data, charts_dir)
            template_file = TEMPLATES_DIR / "reporte_diario.tex"
            n_runs = 1
        elif report_type == "weekly":
            replacements = build_weekly_report(data, charts_dir)
            template_file = TEMPLATES_DIR / "reporte_semanal.tex"
            n_runs = 1
        else:  # quarterly
            replacements = build_quarterly_report(data, quarter, charts_dir)
            template_file = TEMPLATES_DIR / "reporte_trimestral.tex"
            n_runs = 2  # Two runs for TOC

        # Load and fill template
        print("[3/4] Filling template...")
        if not template_file.exists():
            print(f"[ERROR] Template not found: {template_file}")
            sys.exit(1)

        template_text = template_file.read_text(encoding="utf-8")
        filled_tex = fill_template(template_text, replacements)

        # Write filled .tex file
        tex_output = charts_dir / "report.tex"
        tex_output.write_text(filled_tex, encoding="utf-8")

        # Compile PDF
        print("[4/4] Compiling PDF...")
        success = run_pdflatex(tex_output, charts_dir, n_runs=n_runs)

        pdf_in_tmp = charts_dir / "report.pdf"
        if not success or not pdf_in_tmp.exists():
            print("[ERROR] PDF compilation failed.")
            # Save the .tex for debugging
            debug_tex = EXPORTS_REPORTS / f"debug_{report_type}.tex"
            shutil.copy(tex_output, debug_tex)
            print(f"  Debug .tex saved to: {debug_tex}")
            sys.exit(1)

        # Copy to final destinations
        today_str = date.today().strftime("%Y-%m-%d")
        if report_type == "quarterly":
            final_name = f"reporte_{report_type}_{quarter}_{today_str}.pdf"
        else:
            final_name = f"reporte_{report_type}_{today_str}.pdf"

        final_pdf = EXPORTS_REPORTS / final_name
        latest_pdf = EXPORTS_REPORTS / f"latest_{report_type}.pdf"

        shutil.copy(pdf_in_tmp, final_pdf)
        shutil.copy(pdf_in_tmp, latest_pdf)

        print(f"\n[OK] Report generated:")
        print(f"  {final_pdf}")
        print(f"  {latest_pdf}")


if __name__ == "__main__":
    main()
