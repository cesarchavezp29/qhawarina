"""
paper_style.py — Shared style configuration for all paper figures.
Apply with: from paper_style import apply_style, C, STYLE, zero_line, legend_below, legend_outside
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Dimensions ─────────────────────────────────────────────────────────────────
STYLE = {
    "font.family":  "serif",
    "font.serif":   ["Palatino Linotype", "Palatino", "Times New Roman", "DejaVu Serif"],
    "font.size":    10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
    "legend.borderpad": 0.4,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
}

# ── Figure sizes ───────────────────────────────────────────────────────────────
SZ = {
    "single":      (5.5, 3.8),
    "single_tall": (5.5, 4.5),
    "single_sq":   (5.5, 4.0),
    "wide":        (7.5, 4.0),
    "wide_short":  (7.5, 3.5),
    "wide_tall":   (7.5, 4.5),
    "two_panel_h": (7.5, 3.5),
    "two_panel_v": (5.5, 6.5),
    "three_panel": (7.5, 3.0),
    "forest":      (5.5, 6.5),
}

# ── Color palette (colorblind-safe, print-friendly) ────────────────────────────
C = {
    "main":      "#2C3E50",   # dark navy — primary lines
    "accent1":   "#C0392B",   # red — highlights, GIRF, OLS fit
    "accent2":   "#2980B9",   # blue — secondary, literature
    "accent3":   "#27AE60",   # green — tertiary, FX channel
    "gray_line": "#7F8C8D",   # gray — alternatives, robustness
    "ci_light":  "#BDC3C7",   # light gray — 90% CI fill
    "ci_dark":   "#95A5A6",   # medium gray — 68% CI fill
    "zero":      "#95A5A6",   # zero reference line color
    "hiking":    "#E74C3C",   # hiking episode shading
    "cutting":   "#2980B9",   # cutting episode shading
}


def apply_style():
    """Call once at top of each script."""
    mpl.rcParams.update(STYLE)


def zero_line(ax, **kwargs):
    """Consistent zero reference line."""
    ax.axhline(0, color=C["zero"], ls="--", lw=0.7, zorder=0, **kwargs)


def legend_below(ax, ncol=3, **kwargs):
    """Place legend below the plot."""
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=ncol, frameon=True, **kwargs
    )


def legend_outside(ax, **kwargs):
    """Place legend outside right."""
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0, frameon=True, **kwargs
    )


def stat_box(ax, text, loc="lower left", fontsize=9, **kwargs):
    """Place a stats annotation with white background box."""
    xy_map = {
        "lower left":  (0.03, 0.03),
        "lower right": (0.97, 0.03),
        "upper left":  (0.03, 0.97),
        "upper right": (0.97, 0.97),
    }
    ha_map = {
        "lower left": "left", "lower right": "right",
        "upper left": "left", "upper right": "right",
    }
    va_map = {
        "lower left": "bottom", "lower right": "bottom",
        "upper left": "top",    "upper right": "top",
    }
    ax.annotate(
        text,
        xy=xy_map[loc], xycoords="axes fraction",
        ha=ha_map[loc], va=va_map[loc],
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", lw=0.7),
        **kwargs,
    )
