"""Chart generation for Qhawarina PDF reports.

Produces matplotlib charts as base64-encoded PNGs for embedding in HTML templates.
All charts follow NEXUS brand identity.
"""

import base64
import io
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logger = logging.getLogger("nexus.reporting.charts")

# ── Brand colors ─────────────────────────────────────────────────────────────
POL_COLOR = "#C0392B"
ECON_COLOR = "#2E5090"
POL_LIGHT = "#F2D7D5"
ECON_LIGHT = "#D6E4F0"
GOLD = "#D4A03C"
DARK = "#1B2A4A"
BG = "#FAFBFC"
GRID = "#E2E6EC"
TEAL = "#1ABC9C"
FONT = "Segoe UI"


def _to_base64(fig, dpi=150) -> str:
    """Render figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _style_ax(ax, title=None, ylabel=None):
    """Apply consistent styling to an axis."""
    ax.set_facecolor(BG)
    ax.grid(True, alpha=0.3, color=GRID, linewidth=0.5)
    ax.tick_params(labelsize=8, colors=DARK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color=DARK,
                      fontfamily=FONT, pad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=DARK, fontfamily=FONT)


# ── Public chart functions ───────────────────────────────────────────────────

def weekly_trend_chart(index_df: pd.DataFrame, end_date: str,
                       lookback_days: int = 60) -> str:
    """13-week trend chart with current week highlighted.

    Returns base64-encoded PNG.
    """
    idx = index_df.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    end = pd.Timestamp(end_date)
    start = end - pd.Timedelta(days=lookback_days)
    week_start = end - pd.Timedelta(days=6)

    mask = (idx["date"] >= start) & (idx["date"] <= end)
    df = idx[mask].sort_values("date")

    if df.empty:
        return ""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4), facecolor="white",
                                    sharex=True, gridspec_kw={"hspace": 0.15})

    # Political
    ax1.fill_between(df["date"], 0, df["political_index"],
                     alpha=0.15, color=POL_COLOR)
    ax1.plot(df["date"], df["political_smooth"], color=POL_COLOR,
             linewidth=2, label="Político (7d MA)")
    ax1.axhline(100, color=DARK, linewidth=0.5, linestyle="--", alpha=0.4)
    ax1.axvspan(week_start, end, alpha=0.08, color=GOLD)
    _style_ax(ax1, "Índice Político", "índice")
    ax1.legend(fontsize=7, loc="upper left", framealpha=0.8)

    # Economic
    ax2.fill_between(df["date"], 0, df["economic_index"],
                     alpha=0.15, color=ECON_COLOR)
    ax2.plot(df["date"], df["economic_smooth"], color=ECON_COLOR,
             linewidth=2, label="Económico (7d MA)")
    ax2.axhline(100, color=DARK, linewidth=0.5, linestyle="--", alpha=0.4)
    ax2.axvspan(week_start, end, alpha=0.08, color=GOLD)
    _style_ax(ax2, "Índice Económico", "índice")
    ax2.legend(fontsize=7, loc="upper left", framealpha=0.8)

    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d\n%b"))

    return _to_base64(fig)


def weekly_bar_chart(week_daily: pd.DataFrame) -> str:
    """Daily bar chart for the current week.

    Returns base64-encoded PNG.
    """
    df = week_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    if df.empty:
        return ""

    fig, ax = plt.subplots(figsize=(7, 2.5), facecolor="white")

    days_es = {0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie", 5: "Sáb", 6: "Dom"}
    labels = [f"{days_es.get(d.dayofweek, '?')}\n{d.day}" for d in df["date"]]
    x = np.arange(len(df))
    w = 0.35

    bars_pol = ax.bar(x - w/2, df["political_index"], w, color=POL_COLOR,
                      alpha=0.7, label="Político")
    bars_econ = ax.bar(x + w/2, df["economic_index"], w, color=ECON_COLOR,
                       alpha=0.7, label="Económico")

    ax.axhline(100, color=DARK, linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, fontfamily=FONT)
    _style_ax(ax, "Índice Diario de la Semana", "índice")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.8)

    # Value labels on all bars
    for bars in [bars_pol, bars_econ]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 2,
                        f"{h:.0f}", ha="center", va="bottom",
                        fontsize=6, color=DARK, fontfamily=FONT)

    return _to_base64(fig)


def article_count_chart(week_daily: pd.DataFrame) -> str:
    """Stacked bar chart of article counts per day.

    Returns base64-encoded PNG.
    """
    df = week_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    if df.empty:
        return ""

    fig, ax = plt.subplots(figsize=(7, 2), facecolor="white")

    days_es = {0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie", 5: "Sáb", 6: "Dom"}
    labels = [f"{days_es.get(d.dayofweek, '?')}\n{d.day}" for d in df["date"]]
    x = np.arange(len(df))

    other = df["n_articles_total"] - df["n_articles_political"] - df["n_articles_economic"]
    ax.bar(x, df["n_articles_political"], color=POL_COLOR, alpha=0.6, label="Políticos")
    ax.bar(x, df["n_articles_economic"], bottom=df["n_articles_political"],
           color=ECON_COLOR, alpha=0.6, label="Económicos")
    ax.bar(x, other, bottom=df["n_articles_political"] + df["n_articles_economic"],
           color="#BDC3C7", alpha=0.4, label="Otros")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, fontfamily=FONT)
    _style_ax(ax, "Artículos Analizados por Día", "artículos")
    ax.legend(fontsize=6, loc="upper right", framealpha=0.8, ncol=3)

    return _to_base64(fig)


def sparkline(values: list, color: str = POL_COLOR, width: float = 2.5,
              height: float = 0.6) -> str:
    """Tiny sparkline chart for inline use.

    Returns base64-encoded PNG.
    """
    if not values or len(values) < 2:
        return ""

    fig, ax = plt.subplots(figsize=(width, height), facecolor="white")
    ax.plot(values, color=color, linewidth=1.5)
    ax.fill_between(range(len(values)), values, alpha=0.1, color=color)
    ax.axhline(100, color=DARK, linewidth=0.3, linestyle="--", alpha=0.3)
    ax.set_xlim(0, len(values) - 1)
    ax.axis("off")

    return _to_base64(fig, dpi=100)


def gauge_svg(value: float, max_val: float = 300, color: str = POL_COLOR) -> str:
    """Generate a simple SVG gauge arc.

    Returns SVG string (not base64, directly embeddable).
    """
    # Normalize to 0-180 degrees
    pct = min(value / max_val, 1.0)
    angle = 180 * pct

    import math
    # Arc endpoint
    rad = math.radians(180 - angle)
    r = 40
    cx, cy = 50, 55
    ex = cx + r * math.cos(rad)
    ey = cy - r * math.sin(rad)
    large_arc = 1 if angle > 90 else 0

    return f"""<svg viewBox="0 0 100 65" width="120" height="78">
  <path d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}"
        fill="none" stroke="#E8ECF0" stroke-width="6" stroke-linecap="round"/>
  <path d="M {cx-r} {cy} A {r} {r} 0 {large_arc} 1 {ex:.1f} {ey:.1f}"
        fill="none" stroke="{color}" stroke-width="6" stroke-linecap="round"/>
  <text x="{cx}" y="{cy-8}" text-anchor="middle" font-size="16"
        font-weight="bold" fill="{DARK}" font-family="{FONT}">{value:.0f}</text>
  <text x="{cx}" y="{cy+8}" text-anchor="middle" font-size="7"
        fill="#5A6B8A" font-family="{FONT}">media=100</text>
</svg>"""
