"""Generate daily chart images for social media posting.

Creates a dated folder: exports/charts/YYYY-MM-DD/ with 4 PNG charts:
  1. political_risk.png   — 30-day political risk index
  2. financial_stress.png — 60-day TC PEN/USD + BVL stock index
  3. daily_prices.png     — 30-day supermarket price index
  4. summary.png          — 2x2 combined summary card

Run after export_web_data.py in the daily pipeline.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_daily_charts")

# ── Paths ──────────────────────────────────────────────────────────────────────
POLITICAL_PATH = project_root / "exports/data/political_index_daily.json"
FX_PATH        = project_root / "exports/data/fx_interventions.json"
PRICES_PATH    = project_root / "exports/data/daily_price_index.json"
CHARTS_BASE    = project_root / "exports/charts"

# ── Style ──────────────────────────────────────────────────────────────────────
COLORS = {
    "political": "#DC2626",   # red
    "financial": "#2563EB",   # blue
    "tc":        "#059669",   # green
    "bvl":       "#7C3AED",   # purple
    "prices":    "#7C3AED",   # purple
    "food":      "#D97706",   # amber
    "grid":      "#F3F4F6",
    "text":      "#111827",
    "subtext":   "#6B7280",
    "bg":        "#FFFFFF",
}

SEVERITY_COLORS = {
    "CRITICAL": "#DC2626",
    "ALTO":     "#EA580C",
    "MEDIO":    "#CA8A04",
    "BAJO":     "#16A34A",
    "MINIMO":   "#6B7280",
}

WATERMARK = "qhawarina.pe"


def setup_ax(ax, title: str, ylabel: str = ""):
    ax.set_facecolor(COLORS["bg"])
    ax.set_title(title, fontsize=11, fontweight="bold", color=COLORS["text"], pad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=COLORS["subtext"])
    ax.tick_params(colors=COLORS["subtext"], labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#E5E7EB")
    ax.yaxis.grid(True, color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


def add_watermark(fig):
    """Place qhawarina.pe outside the chart area, at figure bottom-right."""
    fig.text(0.99, 0.005, WATERMARK, ha="right", va="bottom",
             fontsize=7, color=COLORS["subtext"], alpha=0.8)


def chart_political(out_path: Path, days: int = 30):
    data = json.loads(POLITICAL_PATH.read_text(encoding="utf-8"))
    series  = data.get("daily_series", [])[-days:]
    current = data.get("current", {})

    dates  = [pd.Timestamp(r["date"]) for r in series]
    scores = [r["score"] for r in series]

    level    = current.get("level", "BAJO")
    score    = current.get("score", 0.0)
    articles = current.get("articles_total", 0)
    color    = SEVERITY_COLORS.get(level, COLORS["political"])

    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=COLORS["bg"])
    fig.subplots_adjust(bottom=0.18)

    ax.fill_between(dates, scores, alpha=0.15, color=color)
    ax.plot(dates, scores, color=color, linewidth=2)
    ax.axhline(0.5, color="#9CA3AF", linewidth=0.8, linestyle="--")

    setup_ax(ax, f"Índice de Riesgo Político | {current.get('date', '')}", "Score (0–1)")

    ax.annotate(
        f"{level}  {score:.2f}  |  {articles} artículos",
        xy=(0.02, 0.92), xycoords="axes fraction",
        fontsize=9, color=color, fontweight="bold",
    )

    add_watermark(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def chart_financial(out_path: Path, days: int = 60):
    """TC PEN/USD + bond spread (sol_10y minus usd_10y) as financial stress proxy."""
    data  = json.loads(FX_PATH.read_text(encoding="utf-8"))
    daily = data.get("daily_series", [])[-days:]

    dates  = [pd.Timestamp(r["date"]) for r in daily]
    tc     = [r.get("fx") for r in daily]
    sol10  = [r.get("bond_sol_10y") for r in daily]
    usd10  = [r.get("bond_usd_10y") for r in daily]
    spread = [
        (s - u) if (s is not None and u is not None) else None
        for s, u in zip(sol10, usd10)
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), facecolor=COLORS["bg"],
                                    sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    fig.subplots_adjust(bottom=0.14, hspace=0.35)

    # ── Panel 1: TC PEN/USD ──
    ax1.plot(dates, tc, color=COLORS["tc"], linewidth=2)
    setup_ax(ax1, f"Tipo de Cambio PEN/USD | Últimos {days} días", "S/ por USD")

    # ── Panel 2: Bond spread (spread sol-USD = prima de riesgo PEN) ──
    spread_color = "#DC2626"  # red — higher spread = more stress
    ax2.fill_between(dates, [v if v is not None else 0 for v in spread],
                     alpha=0.15, color=spread_color)
    ax2.plot(dates, spread, color=spread_color, linewidth=2)
    setup_ax(ax2, "Diferencial Bonos Soberanos Sol−USD (10a)", "pp")
    ax2.annotate("↑ mayor diferencial = mayor estrés cambiario",
                 xy=(0.02, 0.88), xycoords="axes fraction",
                 fontsize=7, color=COLORS["subtext"])

    add_watermark(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def chart_prices(out_path: Path, days: int = 30):
    data   = json.loads(PRICES_PATH.read_text(encoding="utf-8"))
    series = data.get("series", [])[-days:]
    latest = data.get("latest", {})

    dates      = [pd.Timestamp(r["date"]) for r in series]
    index_all  = [r.get("index_all") for r in series]
    index_food = [r.get("index_food") for r in series]

    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=COLORS["bg"])
    fig.subplots_adjust(bottom=0.18)

    ax.plot(dates, index_all,  color=COLORS["prices"], linewidth=2, label="General")
    ax.plot(dates, index_food, color=COLORS["food"],   linewidth=2, label="Alimentos", linestyle="--")
    ax.axhline(100, color="#9CA3AF", linewidth=0.8, linestyle=":")

    setup_ax(ax, f"Índice Precios Supermercados | Últimos {days} días", "Índice (base=100)")
    ax.legend(fontsize=8, framealpha=0.5)

    cum = latest.get("cum_pct", 0) or 0
    var = latest.get("var_all", 0) or 0
    ax.annotate(
        f"Var. diaria: {var:+.2f}%  |  Acum. mes: {cum:+.2f}%",
        xy=(0.02, 0.92), xycoords="axes fraction",
        fontsize=9, color=COLORS["prices"], fontweight="bold",
    )

    fig.text(0.99, 0.005, f"{WATERMARK}  |  Plaza Vea · Metro · Wong",
             ha="right", va="bottom", fontsize=7, color=COLORS["subtext"], alpha=0.8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def chart_summary(out_path: Path):
    """2x2 combined summary card."""
    political_data = json.loads(POLITICAL_PATH.read_text(encoding="utf-8"))
    fx_data        = json.loads(FX_PATH.read_text(encoding="utf-8"))
    prices_data    = json.loads(PRICES_PATH.read_text(encoding="utf-8"))

    today = datetime.today().strftime("%d %b %Y")

    fig = plt.figure(figsize=(10, 6.5), facecolor=COLORS["bg"])
    fig.suptitle(f"Economía Peruana — {today}",
                 fontsize=13, fontweight="bold", color=COLORS["text"], y=0.98)
    fig.subplots_adjust(bottom=0.06)

    gs = gridspec.GridSpec(2, 2, hspace=0.50, wspace=0.35,
                           top=0.90, bottom=0.10, left=0.08, right=0.97)
    days = 30

    # ── Panel 1: Political ──
    ax1 = fig.add_subplot(gs[0, 0])
    series  = political_data.get("daily_series", [])[-days:]
    dates   = [pd.Timestamp(r["date"]) for r in series]
    scores  = [r["score"] for r in series]
    current = political_data.get("current", {})
    level   = current.get("level", "BAJO")
    color   = SEVERITY_COLORS.get(level, COLORS["political"])
    ax1.fill_between(dates, scores, alpha=0.15, color=color)
    ax1.plot(dates, scores, color=color, linewidth=1.5)
    setup_ax(ax1, "Riesgo Político", "Score")
    ax1.annotate(f"{level}  {current.get('score', 0):.2f}", xy=(0.05, 0.88),
                 xycoords="axes fraction", fontsize=8, color=color, fontweight="bold")

    # ── Panel 2: TC ──
    ax2 = fig.add_subplot(gs[0, 1])
    daily_fx = fx_data.get("daily_series", [])[-days:]
    d2 = [pd.Timestamp(r["date"]) for r in daily_fx]
    tc = [r.get("fx") for r in daily_fx]
    ax2.plot(d2, tc, color=COLORS["tc"], linewidth=1.5)
    setup_ax(ax2, "Tipo de Cambio PEN/USD", "S/ por USD")
    latest_tc = next((v for v in reversed(tc) if v is not None), None)
    if latest_tc:
        ax2.annotate(f"S/ {latest_tc:.4f}", xy=(0.05, 0.88),
                     xycoords="axes fraction", fontsize=8, color=COLORS["tc"], fontweight="bold")

    # ── Panel 3: Prices ──
    ax3 = fig.add_subplot(gs[1, 0])
    pseries = prices_data.get("series", [])[-days:]
    d3 = [pd.Timestamp(r["date"]) for r in pseries]
    ax3.plot(d3, [r.get("index_all") for r in pseries],  color=COLORS["prices"], linewidth=1.5, label="General")
    ax3.plot(d3, [r.get("index_food") for r in pseries], color=COLORS["food"],   linewidth=1.5, linestyle="--", label="Alimentos")
    ax3.axhline(100, color="#9CA3AF", linewidth=0.6, linestyle=":")
    setup_ax(ax3, "Precios Supermercados", "Índice")
    ax3.legend(fontsize=7, framealpha=0.4)

    # ── Panel 4: BVL ──
    ax4 = fig.add_subplot(gs[1, 1])
    bvl = [r.get("bvl") for r in daily_fx]
    bvl_k = [v / 1000 if v is not None else None for v in bvl]
    ax4.fill_between(d2, [v if v is not None else 0 for v in bvl_k],
                     alpha=0.12, color=COLORS["bvl"])
    ax4.plot(d2, bvl_k, color=COLORS["bvl"], linewidth=1.5)
    setup_ax(ax4, "BVL", "Miles puntos")
    latest_bvl = next((v for v in reversed(bvl) if v is not None), None)
    if latest_bvl:
        ax4.annotate(f"{latest_bvl:,.0f}", xy=(0.05, 0.88),
                     xycoords="axes fraction", fontsize=8, color=COLORS["bvl"], fontweight="bold")

    fig.text(0.99, 0.005, WATERMARK, ha="right", va="bottom",
             fontsize=8, color=COLORS["subtext"], alpha=0.8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main() -> int:
    today   = datetime.today().strftime("%Y-%m-%d")
    out_dir = CHARTS_BASE / today
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output folder: %s", out_dir)

    errors = 0
    for name, fn, path in [
        ("political_risk",   chart_political, out_dir / "political_risk.png"),
        ("financial_stress", chart_financial, out_dir / "financial_stress.png"),
        ("daily_prices",     chart_prices,    out_dir / "daily_prices.png"),
        ("summary",          chart_summary,   out_dir / "summary.png"),
    ]:
        try:
            fn(path)
        except Exception as e:
            logger.error("Failed to generate %s: %s", name, e)
            errors += 1

    logger.info("Done — %d charts saved to %s", 4 - errors, out_dir)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
