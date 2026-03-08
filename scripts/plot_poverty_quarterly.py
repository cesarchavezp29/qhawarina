"""Plot Peru quarterly poverty rate nowcast with historical annual INEI data.

Produces a two-panel chart:
  - Top: annual official rates (dots) + quarterly model estimates (line) + confidence band
  - Bottom: quarter-over-quarter change (delta)

Output: assets/poverty_rolling_quarterly.png
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# ── Project imports ──────────────────────────────────────────────────────────
from src.visualization.style import (
    apply_nexus_style,
    NEXUS_COLORS,
    TITLE_SIZE,
    SUBTITLE_SIZE,
    WATERMARK_SIZE,
    SOURCE_SIZE,
    add_watermark,
    add_source_line,
)

apply_nexus_style()

# ── Qhawarina branding colors ───────────────────────────────────────────────
TEAL = "#00897B"
CORAL = "#E53935"
TEAL_LIGHT = "#B2DFDB"
CORAL_LIGHT = "#FFCDD2"
GRAY = "#78909C"

# ── Confidence band width (pp) from backtested RMSE ─────────────────────────
BAND_PP = 1.5

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
TARGETS_DIR = ROOT / "data" / "targets"
RESULTS_DIR = ROOT / "data" / "results"
ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = ASSETS_DIR / "poverty_rolling_quarterly.png"


def load_annual_national() -> pd.DataFrame:
    """Load departmental poverty data and compute national annual average.

    Returns DataFrame with columns: year, poverty_rate_pct (percentage).
    """
    df = pd.read_parquet(TARGETS_DIR / "poverty_departmental.parquet")
    national = df.groupby("year")["poverty_rate"].mean().reset_index()
    # poverty_rate is stored as fraction (0-1), convert to percentage
    national["poverty_rate_pct"] = national["poverty_rate"] * 100
    # Create a date at mid-year (July 1) for plotting
    national["date"] = pd.to_datetime(national["year"].astype(str) + "-07-01")
    return national


def load_quarterly_nowcast() -> pd.DataFrame:
    """Load quarterly nowcast results.

    Returns DataFrame with columns: date, poverty_rate_pct, quarter_label.
    """
    df = pd.read_excel(RESULTS_DIR / "poverty_quarterly_nowcast.xlsx")
    # Parse quarter_label like "2023-Q1" into a date at quarter midpoint
    dates = []
    for ql in df["quarter_label"]:
        year, q = ql.split("-Q")
        year = int(year)
        q = int(q)
        # Use middle of quarter: month 2, 5, 8, 11
        mid_month = 3 * q - 1
        dates.append(pd.Timestamp(year, mid_month, 15))
    df["date"] = dates
    df["poverty_rate_pct"] = df["national_poverty_rate"]
    # Quarter-over-quarter change
    df["delta_qoq"] = df["poverty_rate_pct"].diff()
    return df


def plot_poverty_rolling_quarterly():
    """Create the two-panel poverty chart and save to assets/."""

    annual = load_annual_national()
    quarterly = load_quarterly_nowcast()

    # ── Figure setup: 2 rows, shared x-axis ──────────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(12, 7.5),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        sharex=False,
    )

    # ── TOP PANEL: poverty rate level ────────────────────────────────────────

    # Historical annual dots
    ax_top.scatter(
        annual["date"],
        annual["poverty_rate_pct"],
        color=CORAL,
        s=50,
        zorder=5,
        label="INEI oficial (anual)",
        edgecolors="white",
        linewidths=0.6,
    )

    # Quarterly nowcast line
    ax_top.plot(
        quarterly["date"],
        quarterly["poverty_rate_pct"],
        color=TEAL,
        linewidth=2.2,
        zorder=4,
        label="Nowcast trimestral (modelo)",
    )

    # Quarterly nowcast dots
    ax_top.scatter(
        quarterly["date"],
        quarterly["poverty_rate_pct"],
        color=TEAL,
        s=30,
        zorder=5,
        edgecolors="white",
        linewidths=0.4,
    )

    # Confidence band
    ax_top.fill_between(
        quarterly["date"],
        quarterly["poverty_rate_pct"] - BAND_PP,
        quarterly["poverty_rate_pct"] + BAND_PP,
        color=TEAL_LIGHT,
        alpha=0.35,
        zorder=2,
        label=f"Banda +/-{BAND_PP}pp (RMSE backtest)",
    )

    # Annotate latest quarter
    latest = quarterly.iloc[-1]
    ax_top.annotate(
        f"{latest['poverty_rate_pct']:.1f}%\n{latest['quarter_label']}",
        xy=(latest["date"], latest["poverty_rate_pct"]),
        xytext=(30, 20),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        color=TEAL,
        arrowprops=dict(
            arrowstyle="-|>",
            color=TEAL,
            lw=1.2,
        ),
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=TEAL,
            alpha=0.9,
        ),
        zorder=6,
    )

    # Axis formatting — top panel
    ax_top.set_title(
        "Peru: Quarterly Poverty Rate Nowcast",
        fontsize=TITLE_SIZE + 1,
        fontweight="bold",
        color=NEXUS_COLORS["text_primary"],
        pad=12,
    )
    ax_top.set_ylabel("Tasa de pobreza (%)", fontsize=10)
    ax_top.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax_top.set_ylim(
        max(0, annual["poverty_rate_pct"].min() - 5),
        annual["poverty_rate_pct"].max() + 5,
    )

    # Light horizontal grid
    ax_top.grid(axis="y", alpha=0.3, linewidth=0.5, color=NEXUS_COLORS["border"])
    ax_top.set_axisbelow(True)

    # Legend
    ax_top.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor=NEXUS_COLORS["border"],
        fontsize=9,
    )

    # Spine cleanup
    for spine in ["top", "right"]:
        ax_top.spines[spine].set_visible(False)

    add_watermark(ax_top)

    # X-axis range: show from 2004 to a bit past latest quarter
    x_min = pd.Timestamp("2003-06-01")
    x_max = latest["date"] + pd.DateOffset(months=6)
    ax_top.set_xlim(x_min, x_max)

    # ── BOTTOM PANEL: quarter-over-quarter delta ─────────────────────────────

    qoq = quarterly.dropna(subset=["delta_qoq"])

    colors_bar = [TEAL if d <= 0 else CORAL for d in qoq["delta_qoq"]]
    ax_bot.bar(
        qoq["date"],
        qoq["delta_qoq"],
        width=60,
        color=colors_bar,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
        zorder=3,
    )

    # Zero line
    ax_bot.axhline(0, color=GRAY, linewidth=0.8, zorder=2)

    # Annotate each bar
    for _, row in qoq.iterrows():
        d = row["delta_qoq"]
        va = "bottom" if d >= 0 else "top"
        offset = 0.03 if d >= 0 else -0.03
        ax_bot.text(
            row["date"],
            d + offset,
            f"{d:+.2f}",
            ha="center",
            va=va,
            fontsize=7.5,
            color=NEXUS_COLORS["text_secondary"],
            fontweight="bold",
        )

    ax_bot.set_ylabel("Cambio t/t (pp)", fontsize=10)
    ax_bot.set_xlabel("")
    ax_bot.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.1f"))

    # Symmetric y-limits
    abs_max = max(abs(qoq["delta_qoq"].min()), abs(qoq["delta_qoq"].max()), 0.5) + 0.2
    ax_bot.set_ylim(-abs_max, abs_max)

    ax_bot.grid(axis="y", alpha=0.3, linewidth=0.5, color=NEXUS_COLORS["border"])
    ax_bot.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax_bot.spines[spine].set_visible(False)

    # X-axis: show only the quarterly nowcast range
    q_x_min = quarterly["date"].min() - pd.DateOffset(months=2)
    q_x_max = latest["date"] + pd.DateOffset(months=3)
    ax_bot.set_xlim(q_x_min, q_x_max)
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y-Q"))
    # Use quarter labels
    ax_bot.set_xticks(quarterly["date"])
    ax_bot.set_xticklabels(quarterly["quarter_label"], rotation=45, ha="right", fontsize=7.5)

    # ── Source line ──────────────────────────────────────────────────────────
    add_source_line(
        fig,
        text=(
            "Fuente: INEI/ENAHO (anual), modelo Qhawarina (trimestral). "
            "Banda basada en RMSE de backtest (~1.5pp)."
        ),
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved to {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    plot_poverty_rolling_quarterly()
