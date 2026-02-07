"""NEXUS brand identity — colors, colormaps, typography, and styling helpers.

Provides a consistent visual identity across all NEXUS maps and charts.
Call ``apply_nexus_style()`` once at startup to set matplotlib rcParams.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# ── Brand colors ─────────────────────────────────────────────────────────────

NEXUS_COLORS = {
    # Core palette
    "background": "#FAFBFC",
    "text_primary": "#1B2A4A",
    "text_secondary": "#5A6B8A",
    "border": "#E2E6EC",
    "accent": "#D4A03C",
    # Watermark / faint elements
    "watermark": "#C0C8D4",
    # Map border
    "polygon_edge": "#000000",
    # Label backgrounds
    "label_bg": "#FFFFFF",
}

# ── Custom sequential colormaps ──────────────────────────────────────────────

def _make_cmap(name: str, colors: list[str]) -> LinearSegmentedColormap:
    """Build a LinearSegmentedColormap from a list of hex colors."""
    return LinearSegmentedColormap.from_list(name, colors, N=256)


NEXUS_CMAPS = {
    "poverty": _make_cmap("nexus_poverty", ["#F7F4E9", "#D4A03C", "#8B2500"]),
    "consumption": _make_cmap("nexus_consumption", ["#F0F7F0", "#2D8B57", "#0B3D20"]),
    "income": _make_cmap("nexus_consumption", ["#F0F7F0", "#2D8B57", "#0B3D20"]),
    "gini": _make_cmap("nexus_gini", ["#F5F0F8", "#7B4FA0", "#2D0A4E"]),
    "extreme_poverty": _make_cmap("nexus_extreme", ["#FFF5F0", "#E04040", "#5C0000"]),
    "fgt": _make_cmap("nexus_fgt", ["#FFF8F0", "#E07030", "#6B2000"]),
}

# ── Typography constants ─────────────────────────────────────────────────────

FONT_FAMILY = "Segoe UI"
TITLE_SIZE = 15
SUBTITLE_SIZE = 10
LABEL_SIZE = 6
LEGEND_SIZE = 9
WATERMARK_SIZE = 8
SOURCE_SIZE = 7

# ── Layout constants ─────────────────────────────────────────────────────────

SINGLE_FIGSIZE = (12, 10)
TRIPLE_FIGSIZE = (20, 8)
MAP_DPI = 200
POLYGON_LINEWIDTH = 0.3


def apply_nexus_style():
    """Set matplotlib rcParams to NEXUS brand defaults."""
    matplotlib.use("Agg")
    plt.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "Calibri", "Arial", "Helvetica"],
        "font.size": 10,
        # Figure
        "figure.facecolor": NEXUS_COLORS["background"],
        "figure.edgecolor": "none",
        "figure.dpi": MAP_DPI,
        # Axes
        "axes.facecolor": NEXUS_COLORS["background"],
        "axes.edgecolor": NEXUS_COLORS["border"],
        "axes.labelcolor": NEXUS_COLORS["text_primary"],
        "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "bold",
        "axes.titlecolor": NEXUS_COLORS["text_primary"],
        "axes.labelsize": 9,
        # Grid
        "axes.grid": False,
        "grid.color": NEXUS_COLORS["border"],
        "grid.alpha": 0.5,
        "grid.linewidth": 0.5,
        # Ticks
        "xtick.color": NEXUS_COLORS["text_secondary"],
        "ytick.color": NEXUS_COLORS["text_secondary"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        # Legend
        "legend.fontsize": LEGEND_SIZE,
        "legend.framealpha": 0.9,
        "legend.edgecolor": NEXUS_COLORS["border"],
        # Text
        "text.color": NEXUS_COLORS["text_primary"],
        # Savefig
        "savefig.facecolor": NEXUS_COLORS["background"],
        "savefig.edgecolor": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    })


def add_watermark(ax):
    """Add NEXUS branding watermark to bottom-right of axes."""
    ax.text(
        0.98, 0.02, "NEXUS \u00b7 ENAHO/INEI",
        transform=ax.transAxes,
        fontsize=WATERMARK_SIZE,
        color=NEXUS_COLORS["watermark"],
        ha="right", va="bottom",
        fontstyle="italic",
        alpha=0.7,
    )


def add_source_line(fig, text="Fuente: ENAHO/INEI. Elaboraci\u00f3n: NEXUS."):
    """Add source attribution at the bottom of a figure."""
    fig.text(
        0.5, 0.01, text,
        ha="center", va="bottom",
        fontsize=SOURCE_SIZE,
        color=NEXUS_COLORS["text_secondary"],
        fontstyle="italic",
    )


def fmt_pct(v, decimals=1):
    """Format a 0-1 proportion as a percentage string."""
    return f"{v * 100:.{decimals}f}%"


def fmt_soles(v, decimals=0):
    """Format a value as Peruvian soles."""
    return f"S/ {v:,.{decimals}f}"
