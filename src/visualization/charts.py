"""NEXUS chart generation — time series, sparklines, and ranking bar charts.

All charts follow the NEXUS brand identity defined in ``style.py``.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from config.settings import TARGETS_DIR, DEPARTMENTS
from src.visualization.style import (
    NEXUS_COLORS,
    NEXUS_CMAPS,
    MAP_DPI,
    TITLE_SIZE,
    SUBTITLE_SIZE,
    add_watermark,
    add_source_line,
    fmt_pct,
)

logger = logging.getLogger(__name__)

MAPS_DIR = TARGETS_DIR / "maps"

# ── Colors for multi-line charts ─────────────────────────────────────────────

SERIES_COLORS = {
    "poverty_rate": "#D4A03C",
    "extreme_poverty_rate": "#E04040",
    "poverty_gap": "#E07030",
    "poverty_severity": "#6B2000",
    "gini": "#7B4FA0",
    "mean_consumption": "#2D8B57",
    "mean_income": "#1B6B3A",
}

SERIES_LABELS = {
    "poverty_rate": "Pobreza total",
    "extreme_poverty_rate": "Pobreza extrema",
    "poverty_gap": "Brecha (FGT1)",
    "poverty_severity": "Severidad (FGT2)",
    "gini": "Gini",
    "mean_consumption": "Consumo p/c",
    "mean_income": "Ingreso p/c",
}


def _load_national_series() -> pd.DataFrame:
    """Compute national weighted averages from departmental data.

    For poverty rates / FGT: uses simple mean across departments
    (the departmental figures are already population-weighted).
    For Gini: national Gini must be computed from microdata; here we
    use the simple average as an approximation for visualization only.
    """
    df = pd.read_parquet(TARGETS_DIR / "poverty_departmental.parquet")
    national = df.groupby("year").agg({
        "poverty_rate": "mean",
        "extreme_poverty_rate": "mean",
        "poverty_gap": "mean",
        "poverty_severity": "mean",
        "mean_consumption": "mean",
        "mean_income": "mean",
        "gini": "mean",
    }).reset_index()
    return national


def plot_national_trends(output_dir: Path = MAPS_DIR) -> Path:
    """Plot national poverty trends over time (2004-2024).

    Four subplots: poverty rates, FGT indices, consumption/income, Gini.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    nat = _load_national_series()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Poverty rates
    ax = axes[0, 0]
    ax.plot(nat["year"], nat["poverty_rate"] * 100,
            color=SERIES_COLORS["poverty_rate"], linewidth=2.2, marker="o",
            markersize=4, label=SERIES_LABELS["poverty_rate"])
    ax.plot(nat["year"], nat["extreme_poverty_rate"] * 100,
            color=SERIES_COLORS["extreme_poverty_rate"], linewidth=2.2, marker="s",
            markersize=4, label=SERIES_LABELS["extreme_poverty_rate"])
    ax.set_ylabel("Porcentaje (%)", fontsize=9, color=NEXUS_COLORS["text_secondary"])
    ax.set_title("Tasas de Pobreza", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, color=NEXUS_COLORS["border"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # Panel 2: FGT indices
    ax = axes[0, 1]
    ax.plot(nat["year"], nat["poverty_gap"] * 100,
            color=SERIES_COLORS["poverty_gap"], linewidth=2.2, marker="^",
            markersize=4, label=SERIES_LABELS["poverty_gap"])
    ax.plot(nat["year"], nat["poverty_severity"] * 100,
            color=SERIES_COLORS["poverty_severity"], linewidth=2.2, marker="D",
            markersize=4, label=SERIES_LABELS["poverty_severity"])
    ax.set_ylabel("Porcentaje (%)", fontsize=9, color=NEXUS_COLORS["text_secondary"])
    ax.set_title("\u00cdndices FGT", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, color=NEXUS_COLORS["border"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    # Panel 3: Consumption & Income
    ax = axes[1, 0]
    ax.plot(nat["year"], nat["mean_consumption"],
            color=SERIES_COLORS["mean_consumption"], linewidth=2.2, marker="o",
            markersize=4, label=SERIES_LABELS["mean_consumption"])
    ax.plot(nat["year"], nat["mean_income"],
            color=SERIES_COLORS["mean_income"], linewidth=2.2, marker="s",
            markersize=4, label=SERIES_LABELS["mean_income"])
    ax.set_ylabel("Soles mensuales p/c", fontsize=9,
                   color=NEXUS_COLORS["text_secondary"])
    ax.set_title("Consumo e Ingreso Real", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, color=NEXUS_COLORS["border"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"S/{x:,.0f}"))

    # Panel 4: Gini
    ax = axes[1, 1]
    ax.plot(nat["year"], nat["gini"],
            color=SERIES_COLORS["gini"], linewidth=2.2, marker="o",
            markersize=4)
    ax.set_ylabel("Coeficiente de Gini", fontsize=9,
                   color=NEXUS_COLORS["text_secondary"])
    ax.set_title("Desigualdad (Gini)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, color=NEXUS_COLORS["border"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.3f}"))

    # Common formatting
    for ax in axes.flat:
        ax.set_xlim(nat["year"].min() - 0.5, nat["year"].max() + 0.5)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
        ax.tick_params(axis="x", rotation=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(NEXUS_COLORS["border"])
        ax.spines["bottom"].set_color(NEXUS_COLORS["border"])

    fig.suptitle(
        "Per\u00fa: Tendencias Nacionales de Pobreza y Desigualdad, 2004\u20132024",
        fontsize=15, fontweight="bold",
        color=NEXUS_COLORS["text_primary"],
        y=0.98,
    )
    add_source_line(fig, "Fuente: ENAHO/INEI. Promedios departamentales. Elaboraci\u00f3n: NEXUS.")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out = output_dir / "national_trends.png"
    fig.savefig(out, dpi=MAP_DPI)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def plot_ranking_bars(year: int = 2024, output_dir: Path = MAPS_DIR) -> Path:
    """Horizontal bar chart of departments sorted by poverty rate."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(TARGETS_DIR / "poverty_departmental.parquet")
    data = df[df["year"] == year].sort_values("poverty_rate", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(10, 10))

    # Color bars using poverty colormap
    cmap = NEXUS_CMAPS["poverty"]
    vmin, vmax = data["poverty_rate"].min(), data["poverty_rate"].max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = [cmap(norm(v)) for v in data["poverty_rate"]]

    bars = ax.barh(
        data["department_name"], data["poverty_rate"] * 100,
        color=colors, edgecolor=NEXUS_COLORS["polygon_edge"],
        linewidth=0.3, height=0.7,
    )

    # Value labels on bars
    for bar, val in zip(bars, data["poverty_rate"]):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            fmt_pct(val),
            va="center", ha="left",
            fontsize=8, color=NEXUS_COLORS["text_secondary"],
        )

    # National average line
    nat_avg = data["poverty_rate"].mean()
    ax.axvline(
        nat_avg * 100, color=NEXUS_COLORS["accent"],
        linestyle="--", linewidth=1.5, alpha=0.8,
    )
    ax.text(
        nat_avg * 100 + 0.5, len(data) - 0.5,
        f"Promedio: {fmt_pct(nat_avg)}",
        fontsize=8, color=NEXUS_COLORS["accent"],
        fontweight="bold",
    )

    ax.set_xlabel("Tasa de Pobreza (%)", fontsize=10,
                   color=NEXUS_COLORS["text_secondary"])
    ax.set_title(
        f"Per\u00fa: Ranking de Pobreza por Departamento, {year}",
        fontsize=14, fontweight="bold",
        color=NEXUS_COLORS["text_primary"], pad=12,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(NEXUS_COLORS["border"])
    ax.spines["bottom"].set_color(NEXUS_COLORS["border"])
    ax.grid(True, axis="x", alpha=0.3, color=NEXUS_COLORS["border"])
    ax.tick_params(axis="y", labelsize=9)

    add_watermark(ax)
    add_source_line(fig)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    out = output_dir / f"dept_ranking_{year}.png"
    fig.savefig(out, dpi=MAP_DPI)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def plot_department_sparklines(output_dir: Path = MAPS_DIR) -> Path:
    """Small multiples grid: 24 sparkline panels (2004-2024 poverty trend)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(TARGETS_DIR / "poverty_departmental.parquet")

    # Sort departments by 2024 poverty rate (highest first)
    dept_order = (
        df[df["year"] == df["year"].max()]
        .sort_values("poverty_rate", ascending=False)["department_code"]
        .tolist()
    )

    n_depts = len(dept_order)
    ncols = 6
    nrows = (n_depts + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 2.5))
    axes_flat = axes.flatten()

    global_ymax = df["poverty_rate"].max() * 100

    for i, dept_code in enumerate(dept_order):
        ax = axes_flat[i]
        dept_data = df[df["department_code"] == dept_code].sort_values("year")
        dept_name = dept_data["department_name"].iloc[0]

        # Poverty rate line
        ax.plot(
            dept_data["year"], dept_data["poverty_rate"] * 100,
            color=SERIES_COLORS["poverty_rate"],
            linewidth=1.5, marker="", zorder=3,
        )
        ax.fill_between(
            dept_data["year"], dept_data["poverty_rate"] * 100,
            alpha=0.15, color=SERIES_COLORS["poverty_rate"],
        )

        # Extreme poverty line
        ax.plot(
            dept_data["year"], dept_data["extreme_poverty_rate"] * 100,
            color=SERIES_COLORS["extreme_poverty_rate"],
            linewidth=1.0, linestyle="--", alpha=0.7,
        )

        # Title and value annotation
        latest = dept_data.iloc[-1]
        ax.set_title(
            f"{dept_name}",
            fontsize=8, fontweight="bold",
            color=NEXUS_COLORS["text_primary"],
            pad=3,
        )
        ax.text(
            0.97, 0.90, fmt_pct(latest["poverty_rate"]),
            transform=ax.transAxes,
            fontsize=7, fontweight="bold",
            color=SERIES_COLORS["poverty_rate"],
            ha="right", va="top",
        )

        # Formatting
        ax.set_ylim(0, global_ymax * 1.05)
        ax.set_xlim(df["year"].min(), df["year"].max())
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.tick_params(axis="both", labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(NEXUS_COLORS["border"])
        ax.spines["bottom"].set_color(NEXUS_COLORS["border"])
        ax.grid(True, alpha=0.2, color=NEXUS_COLORS["border"])

    # Hide unused axes
    for j in range(n_depts, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Per\u00fa: Evoluci\u00f3n de la Pobreza por Departamento, 2004\u20132024",
        fontsize=15, fontweight="bold",
        color=NEXUS_COLORS["text_primary"],
        y=0.99,
    )

    # Shared legend
    fig.text(
        0.5, 0.01,
        "\u2014\u2014 Pobreza total   - - Pobreza extrema   |   "
        "Fuente: ENAHO/INEI. Elaboraci\u00f3n: NEXUS.",
        ha="center", fontsize=8,
        color=NEXUS_COLORS["text_secondary"],
    )

    plt.tight_layout(rect=[0, 0.025, 1, 0.97])

    out = output_dir / "dept_sparklines.png"
    fig.savefig(out, dpi=MAP_DPI)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out
