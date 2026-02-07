"""Generate all Political Instability Index charts for Qhawarina.

Produces:
1. Monthly composite timeline (v2) with crisis annotations
2. Component decomposition (events z-score + cabinet z-score)
3. Severity distribution pie/bar chart
4. Top 20 crisis months ranking
5. Weekly index timeline
6. Dual-component comparison (z-score vs level vs v2)

Usage:
    python scripts/generate_political_charts.py
    python scripts/generate_political_charts.py --only timeline
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Brand identity: Qhawarina ────────────────────────────────────────────────

BRAND = "Qhawarina"
COLORS = {
    "bg": "#FAFBFC",
    "text": "#1B2A4A",
    "text2": "#5A6B8A",
    "border": "#E2E6EC",
    "accent": "#D4A03C",
    "watermark": "#C0C8D4",
    "crisis_high": "#C0392B",
    "crisis_med": "#E67E22",
    "crisis_low": "#2980B9",
    "events": "#8E44AD",
    "cabinet": "#27AE60",
    "v2_fill": "#D4A03C",
    "zscore_line": "#2C3E50",
    "level_line": "#E74C3C",
}
FONT = "Segoe UI"
DPI = 200
OUTPUT_DIR = PROJECT_ROOT / "data" / "targets" / "charts" / "political"


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "Calibri", "Arial"],
        "font.size": 10,
        "figure.facecolor": COLORS["bg"],
        "figure.edgecolor": "none",
        "figure.dpi": DPI,
        "axes.facecolor": COLORS["bg"],
        "axes.edgecolor": COLORS["border"],
        "axes.labelcolor": COLORS["text"],
        "axes.grid": True,
        "grid.color": COLORS["border"],
        "grid.alpha": 0.5,
        "grid.linewidth": 0.5,
        "text.color": COLORS["text"],
        "savefig.facecolor": COLORS["bg"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    })


def watermark(ax):
    ax.text(0.98, 0.02, f"{BRAND}", transform=ax.transAxes,
            fontsize=7, color=COLORS["watermark"], ha="right", va="bottom",
            fontstyle="italic", alpha=0.6)


def source_line(fig, text=None):
    if text is None:
        text = f"Fuente: Wikipedia, BCRP, ENAHO. Elaboracion: {BRAND}."
    fig.text(0.5, 0.01, text, ha="center", va="bottom",
             fontsize=7, color=COLORS["text2"], fontstyle="italic")


# ── Chart 1: Monthly timeline with crisis annotations ─────────────────────────

def chart_timeline(monthly, events):
    fig, ax = plt.subplots(figsize=(16, 6))

    dates = monthly["date"]
    v2 = monthly["composite_v2"]

    # Fill area
    ax.fill_between(dates, 0, v2, alpha=0.3, color=COLORS["v2_fill"])
    ax.plot(dates, v2, color=COLORS["accent"], linewidth=1.2, label="Composite v2")

    # Crisis annotations
    crises = {
        "2000-11": "Fujimori\nresigna",
        "2005-01": "Andahuaylazo",
        "2009-06": "Baguazo",
        "2017-12": "PPK\nvacancia",
        "2018-03": "PPK\nresigna",
        "2019-09": "Disolucion\nCongreso",
        "2020-11": "Vizcarra\nremovido",
        "2022-12": "Castillo\nautogolpe",
        "2023-01": "Protestas\nBoluarte",
        "2025-10": "Boluarte\nremovida",
    }

    monthly_ym = monthly.set_index(dates.dt.strftime("%Y-%m"))
    for ym, label in crises.items():
        if ym in monthly_ym.index:
            row = monthly_ym.loc[ym]
            d = row["date"]
            y = row["composite_v2"]
            ax.annotate(label, xy=(d, y), xytext=(0, 20),
                        textcoords="offset points", fontsize=6.5,
                        color=COLORS["crisis_high"], ha="center", va="bottom",
                        arrowprops=dict(arrowstyle="-", color=COLORS["crisis_high"],
                                        lw=0.7))

    ax.axhline(0.5, color=COLORS["border"], linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Indice de Inestabilidad Politica (v2)", fontsize=10)
    ax.set_title(f"{BRAND} — Indice de Inestabilidad Politica Mensual (2000-2025)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    watermark(ax)
    source_line(fig)

    fig.savefig(OUTPUT_DIR / "political_index_timeline.png", dpi=DPI)
    plt.close(fig)
    print("  [1/6] political_index_timeline.png")


# ── Chart 2: Component decomposition ──────────────────────────────────────────

def chart_components(monthly):
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    dates = monthly["date"]

    # Events z-score
    ax = axes[0]
    ez = monthly["events_zscore"].fillna(0)
    ax.fill_between(dates, 0, ez, where=ez > 0, alpha=0.4, color=COLORS["events"])
    ax.fill_between(dates, 0, ez, where=ez <= 0, alpha=0.2, color=COLORS["crisis_low"])
    ax.plot(dates, ez, color=COLORS["events"], linewidth=0.8)
    ax.set_ylabel("Events Z-score", fontsize=9)
    ax.set_title("Componente 1: Eventos Institucionales (peso 35%)", fontsize=11, fontweight="bold")
    ax.axhline(0, color=COLORS["border"], linewidth=0.5)
    watermark(ax)

    # Cabinet z-score
    ax = axes[1]
    cz = monthly["cabinet_zscore"].fillna(0)
    ax.fill_between(dates, 0, cz, where=cz > 0, alpha=0.4, color=COLORS["cabinet"])
    ax.fill_between(dates, 0, cz, where=cz <= 0, alpha=0.2, color=COLORS["crisis_low"])
    ax.plot(dates, cz, color=COLORS["cabinet"], linewidth=0.8)
    ax.set_ylabel("Cabinet Z-score", fontsize=9)
    ax.set_title("Componente 2: Inestabilidad del Gabinete (peso 20%)", fontsize=11, fontweight="bold")
    ax.axhline(0, color=COLORS["border"], linewidth=0.5)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    watermark(ax)

    fig.suptitle(f"{BRAND} — Descomposicion del Indice Politico",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    source_line(fig)

    fig.savefig(OUTPUT_DIR / "political_components.png", dpi=DPI)
    plt.close(fig)
    print("  [2/6] political_components.png")


# ── Chart 3: Severity distribution ────────────────────────────────────────────

def chart_severity(events):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    dist = events["severity_nlp_bin3"].value_counts().sort_index()
    labels = ["LOW (1)", "MEDIUM (2)", "HIGH (3)"]
    colors_pie = [COLORS["crisis_low"], COLORS["crisis_med"], COLORS["crisis_high"]]
    values = [dist.get(i, 0) for i in [1, 2, 3]]

    # Pie
    ax = axes[0]
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors_pie, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 9}
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight("bold")
    ax.set_title("Distribucion de Severidad\n(898 eventos)", fontsize=11, fontweight="bold")

    # Bar
    ax = axes[1]
    bars = ax.bar(labels, values, color=colors_pie, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Numero de eventos", fontsize=9)
    ax.set_title("Clasificacion Claude API\n(contexto enriquecido)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.15)

    fig.suptitle(f"{BRAND} — Clasificacion de Eventos Politicos",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    source_line(fig)

    fig.savefig(OUTPUT_DIR / "severity_distribution.png", dpi=DPI)
    plt.close(fig)
    print("  [3/6] severity_distribution.png")


# ── Chart 4: Top 20 crisis months ─────────────────────────────────────────────

def chart_top_crises(monthly):
    top20 = monthly.nlargest(20, "composite_v2").sort_values("composite_v2")
    fig, ax = plt.subplots(figsize=(10, 8))

    labels = top20["date"].dt.strftime("%Y-%m")
    vals = top20["composite_v2"]

    # Color by v2 intensity
    norm = plt.Normalize(vals.min(), vals.max())
    cmap = plt.cm.YlOrRd
    colors_bar = cmap(norm(vals))

    bars = ax.barh(range(len(top20)), vals, color=colors_bar, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Composite v2", fontsize=10)
    ax.set_title(f"{BRAND} — Top 20 Meses de Crisis Politica",
                 fontsize=13, fontweight="bold", pad=12)

    # Add value labels
    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7.5, color=COLORS["text2"])

    ax.set_xlim(0, vals.max() * 1.12)
    watermark(ax)
    source_line(fig)

    fig.savefig(OUTPUT_DIR / "top20_crisis_months.png", dpi=DPI)
    plt.close(fig)
    print("  [4/6] top20_crisis_months.png")


# ── Chart 5: Weekly index ─────────────────────────────────────────────────────

def chart_weekly(weekly):
    fig, ax = plt.subplots(figsize=(16, 5))

    dates = weekly["date"]
    wi = weekly["weekly_v2"]

    ax.fill_between(dates, 0, wi, alpha=0.25, color=COLORS["v2_fill"])
    ax.plot(dates, wi, color=COLORS["accent"], linewidth=0.6, alpha=0.9)

    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Indice Semanal v2", fontsize=10)
    ax.set_title(f"{BRAND} — Indice de Inestabilidad Politica Semanal (2000-2025)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    watermark(ax)
    source_line(fig)

    fig.savefig(OUTPUT_DIR / "political_index_weekly.png", dpi=DPI)
    plt.close(fig)
    print("  [5/6] political_index_weekly.png")


# ── Chart 6: Dual-component comparison ────────────────────────────────────────

def chart_dual_component(monthly):
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    dates = monthly["date"]

    # Z-score composite
    ax = axes[0]
    ci = monthly["composite_index"]
    ax.fill_between(dates, 0, ci, where=ci > 0, alpha=0.3, color=COLORS["zscore_line"])
    ax.plot(dates, ci, color=COLORS["zscore_line"], linewidth=0.9)
    ax.set_ylabel("Z-score", fontsize=9)
    ax.set_title("composite_index (z-score) — sensible a ventana rolling", fontsize=10, fontweight="bold")
    ax.axhline(0, color=COLORS["border"], linewidth=0.5)
    watermark(ax)

    # Level (percentile)
    ax = axes[1]
    lv = monthly["composite_level"]
    ax.fill_between(dates, 0, lv, alpha=0.3, color=COLORS["level_line"])
    ax.plot(dates, lv, color=COLORS["level_line"], linewidth=0.9)
    ax.set_ylabel("Percentil", fontsize=9)
    ax.set_title("composite_level (percentil) — inmune a amortiguacion", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    watermark(ax)

    # V2 blend
    ax = axes[2]
    v2 = monthly["composite_v2"]
    ax.fill_between(dates, 0, v2, alpha=0.3, color=COLORS["v2_fill"])
    ax.plot(dates, v2, color=COLORS["accent"], linewidth=1.0)
    ax.set_ylabel("v2 (blend)", fontsize=9)
    ax.set_title("composite_v2 = 0.5 * level + 0.5 * zscore_norm — balance", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    watermark(ax)

    fig.suptitle(f"{BRAND} — Comparacion de Componentes del Indice",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    source_line(fig)

    fig.savefig(OUTPUT_DIR / "dual_component_comparison.png", dpi=DPI)
    plt.close(fig)
    print("  [6/6] dual_component_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate political index charts")
    parser.add_argument("--only", type=str, help="Generate only this chart")
    args = parser.parse_args()

    setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    proc = PROJECT_ROOT / "data" / "processed" / "political_instability"
    monthly = pd.read_parquet(proc / "political_index_monthly.parquet")
    weekly = pd.read_parquet(proc / "political_index_weekly.parquet")
    events = pd.read_parquet(proc / "events.parquet")

    print(f"{BRAND} — Generating political instability charts...")
    print(f"  Output: {OUTPUT_DIR}")

    charts = {
        "timeline": lambda: chart_timeline(monthly, events),
        "components": lambda: chart_components(monthly),
        "severity": lambda: chart_severity(events),
        "top20": lambda: chart_top_crises(monthly),
        "weekly": lambda: chart_weekly(weekly),
        "dual": lambda: chart_dual_component(monthly),
    }

    if args.only:
        if args.only in charts:
            charts[args.only]()
        else:
            print(f"Unknown chart: {args.only}. Options: {list(charts.keys())}")
            return 1
    else:
        for fn in charts.values():
            fn()

    print(f"\nDone! {len(charts) if not args.only else 1} charts saved to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
