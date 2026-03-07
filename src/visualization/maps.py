"""NEXUS map generation — choropleth maps at department, province, and district levels.

All maps follow the NEXUS brand identity defined in ``style.py``.
"""

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config.settings import TARGETS_DIR, DATA_DIR, RAW_ENAHO_DIR, DEPARTMENTS
from src.visualization.style import (
    apply_nexus_style,
    NEXUS_COLORS,
    NEXUS_CMAPS,
    POLYGON_LINEWIDTH,
    SINGLE_FIGSIZE,
    TRIPLE_FIGSIZE,
    MAP_DPI,
    TITLE_SIZE,
    SUBTITLE_SIZE,
    LABEL_SIZE,
    WATERMARK_SIZE,
    add_watermark,
    add_source_line,
    fmt_pct,
    fmt_soles,
)

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

GEO_DIR = DATA_DIR / "raw" / "geo" / "peru_boundaries"
MAPS_DIR = TARGETS_DIR / "maps"

# ── Department label positions (manual tweaks for readability) ───────────────
# Format: {dept_code: (x_offset, y_offset)} — offsets in degrees from centroid

DEPT_LABEL_OFFSETS = {
    "15": (0.8, 0.3),    # Lima — push right to avoid overlap
    "07": (0.0, 0.0),    # Callao (merged into Lima)
    "11": (0.3, -0.2),   # Ica
    "23": (0.3, -0.2),   # Tacna
    "18": (0.3, 0.0),    # Moquegua
    "24": (0.3, 0.0),    # Tumbes
}


def load_geodata(level: str = "department") -> gpd.GeoDataFrame:
    """Load GeoDataFrame for a given administrative level.

    Parameters
    ----------
    level : str
        One of 'department', 'province', 'district'.

    Returns
    -------
    GeoDataFrame with standardized columns: ``code``, ``name``, ``geometry``.
    """
    if level == "department":
        gdf = gpd.read_file(GEO_DIR / "departamentos.geojson")
        gdf = gdf.rename(columns={"FIRST_IDDP": "code", "NOMBDEP": "name"})
    elif level == "province":
        gdf = gpd.read_file(GEO_DIR / "provincias.geojson")
        gdf = gdf.rename(columns={"FIRST_IDPR": "code", "NOMBPROV": "name"})
    elif level == "district":
        gdf = gpd.read_file(GEO_DIR / "distritos.geojson")
        gdf = gdf.rename(columns={"IDDIST": "code", "NOMBDIST": "name"})
        gdf["dept_code"] = gdf["IDDPTO"]
    else:
        raise ValueError(f"Unknown level: {level!r}")

    return gdf


def load_poverty_data(year: int | None = None) -> pd.DataFrame:
    """Load departmental poverty data, optionally filtered to a year."""
    df = pd.read_parquet(TARGETS_DIR / "poverty_departmental.parquet")
    if year is not None:
        df = df[df["year"] == year].copy()
    return df


def _merge_dept_data(gdf: gpd.GeoDataFrame, data: pd.DataFrame) -> gpd.GeoDataFrame:
    """Merge poverty data onto department GeoDataFrame.

    Handles Callao (code '07') being merged into Lima ('15') in poverty data.
    """
    merged = gdf.merge(data, left_on="code", right_on="department_code", how="left")
    # Callao gets Lima's values
    if "07" in gdf["code"].values and "07" not in data["department_code"].values:
        lima_row = data[data["department_code"] == "15"]
        if not lima_row.empty:
            for col in data.columns:
                if col not in ("department_code", "department_name", "year"):
                    merged.loc[merged["code"] == "07", col] = lima_row[col].values[0]
    return merged


def _add_dept_labels(ax, gdf: gpd.GeoDataFrame, column: str, formatter=None):
    """Add department name + value labels to a map."""
    if formatter is None:
        formatter = fmt_pct

    for _, row in gdf.iterrows():
        if pd.isna(row.get(column)):
            continue
        centroid = row.geometry.centroid
        code = row.get("code", "")
        dx, dy = DEPT_LABEL_OFFSETS.get(code, (0, 0))
        x, y = centroid.x + dx, centroid.y + dy

        # Short department name
        name = row.get("name", "")
        if isinstance(name, str):
            name = name.title()
            # Abbreviate long names
            if len(name) > 12:
                name = name[:11] + "."

        label = f"{name}\n{formatter(row[column])}"
        ax.text(
            x, y, label,
            fontsize=LABEL_SIZE,
            fontweight="bold",
            ha="center", va="center",
            color=NEXUS_COLORS["text_primary"],
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=NEXUS_COLORS["label_bg"],
                edgecolor="none",
                alpha=0.80,
            ),
        )


def plot_choropleth(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    cmap_name: str = "poverty",
    ax=None,
    subtitle: str = "",
    show_labels: bool = True,
    formatter=None,
    vmin: float | None = None,
    vmax: float | None = None,
    legend_label: str = "",
    level: str = "department",
):
    """Plot a single choropleth panel.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must have ``column`` and ``geometry``.
    column : str
        Column to visualize.
    title : str
        Panel title.
    cmap_name : str
        Key into ``NEXUS_CMAPS``.
    ax : matplotlib Axes, optional
        If None, creates a new figure.
    subtitle : str
        Subtitle text below title.
    show_labels : bool
        Whether to add department labels.
    formatter : callable
        Value formatter (default: fmt_pct).
    vmin, vmax : float
        Color scale bounds.
    legend_label : str
        Label for the colorbar.
    level : str
        'department', 'province', or 'district'.

    Returns
    -------
    ax : matplotlib Axes
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=SINGLE_FIGSIZE)

    cmap = NEXUS_CMAPS.get(cmap_name, NEXUS_CMAPS["poverty"])
    valid = gdf[gdf[column].notna()]

    if vmin is None:
        vmin = valid[column].min()
    if vmax is None:
        vmax = valid[column].max()

    norm = Normalize(vmin=vmin, vmax=vmax)

    # Plot background (all polygons in light gray)
    gdf.plot(ax=ax, color=NEXUS_COLORS["border"], edgecolor="none", linewidth=0)
    # Plot data
    valid.plot(
        ax=ax,
        column=column,
        cmap=cmap,
        norm=norm,
        edgecolor=NEXUS_COLORS["polygon_edge"],
        linewidth=POLYGON_LINEWIDTH,
        legend=False,
    )
    # Border for missing polygons too
    gdf.plot(ax=ax, facecolor="none",
             edgecolor=NEXUS_COLORS["polygon_edge"],
             linewidth=POLYGON_LINEWIDTH)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.08)

    if formatter is None or formatter is fmt_pct:
        cb = plt.colorbar(sm, cax=cax)
        cb.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x*100:.0f}%")
        )
    elif formatter is fmt_soles:
        cb = plt.colorbar(sm, cax=cax)
        cb.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"S/{x:,.0f}")
        )
    else:
        cb = plt.colorbar(sm, cax=cax)

    cb.ax.tick_params(labelsize=7)
    if legend_label:
        cb.set_label(legend_label, fontsize=8, color=NEXUS_COLORS["text_secondary"])

    # Labels
    if show_labels and level == "department":
        _add_dept_labels(ax, gdf, column, formatter=formatter)

    # Title
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold",
                 color=NEXUS_COLORS["text_primary"], pad=10)
    if subtitle:
        ax.text(
            0.5, 1.01, subtitle,
            transform=ax.transAxes,
            fontsize=SUBTITLE_SIZE,
            color=NEXUS_COLORS["text_secondary"],
            ha="center", va="bottom",
        )

    ax.set_axis_off()
    add_watermark(ax)

    return ax


def plot_department_maps(year: int = 2024, output_dir: Path = MAPS_DIR) -> Path:
    """Generate 3-panel department map: poverty rate, consumption, Gini.

    Returns path to saved PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf = load_geodata("department")
    data = load_poverty_data(year)
    merged = _merge_dept_data(gdf, data)

    fig, axes = plt.subplots(1, 3, figsize=TRIPLE_FIGSIZE)

    # Panel 1: Poverty rate
    plot_choropleth(
        merged, "poverty_rate", f"Tasa de Pobreza {year}",
        cmap_name="poverty", ax=axes[0],
        legend_label="Pobreza (%)",
    )

    # Panel 2: Mean consumption
    plot_choropleth(
        merged, "mean_consumption", f"Consumo Per C\u00e1pita {year}",
        cmap_name="consumption", ax=axes[1],
        formatter=fmt_soles,
        legend_label="S/ mensuales",
    )

    # Panel 3: Gini
    plot_choropleth(
        merged, "gini", f"Coeficiente de Gini {year}",
        cmap_name="gini", ax=axes[2],
        formatter=lambda v: f"{v:.3f}",
        legend_label="Gini",
    )

    fig.suptitle(
        f"Per\u00fa: Indicadores de Pobreza por Departamento, {year}",
        fontsize=17, fontweight="bold",
        color=NEXUS_COLORS["text_primary"],
        y=0.98,
    )
    add_source_line(fig)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out = output_dir / f"dept_poverty_{year}.png"
    fig.savefig(out, dpi=MAP_DPI)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def plot_extreme_poverty_maps(year: int = 2024, output_dir: Path = MAPS_DIR) -> Path:
    """Generate 3-panel: extreme poverty, FGT1 (gap), FGT2 (severity)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf = load_geodata("department")
    data = load_poverty_data(year)
    merged = _merge_dept_data(gdf, data)

    fig, axes = plt.subplots(1, 3, figsize=TRIPLE_FIGSIZE)

    plot_choropleth(
        merged, "extreme_poverty_rate", f"Pobreza Extrema {year}",
        cmap_name="extreme_poverty", ax=axes[0],
        legend_label="Pobreza extrema (%)",
    )
    plot_choropleth(
        merged, "poverty_gap", f"Brecha de Pobreza (FGT1) {year}",
        cmap_name="fgt", ax=axes[1],
        legend_label="FGT1",
    )
    plot_choropleth(
        merged, "poverty_severity", f"Severidad (FGT2) {year}",
        cmap_name="fgt", ax=axes[2],
        legend_label="FGT2",
    )

    fig.suptitle(
        f"Per\u00fa: Pobreza Extrema e \u00cdndices FGT por Departamento, {year}",
        fontsize=17, fontweight="bold",
        color=NEXUS_COLORS["text_primary"],
        y=0.98,
    )
    add_source_line(fig)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out = output_dir / f"dept_extreme_poverty_{year}.png"
    fig.savefig(out, dpi=MAP_DPI)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def plot_evolution_maps(
    years: list[int] | None = None,
    output_dir: Path = MAPS_DIR,
) -> Path:
    """Generate multi-year comparison: poverty rate across selected years."""
    if years is None:
        years = [2004, 2015, 2024]

    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(years)

    gdf = load_geodata("department")
    all_data = load_poverty_data()

    # Consistent color scale across panels
    subset = all_data[all_data["year"].isin(years)]
    vmin = subset["poverty_rate"].min()
    vmax = subset["poverty_rate"].max()

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 8))
    if n == 1:
        axes = [axes]

    for i, year in enumerate(years):
        data = all_data[all_data["year"] == year]
        merged = _merge_dept_data(gdf, data)
        plot_choropleth(
            merged, "poverty_rate", str(year),
            cmap_name="poverty", ax=axes[i],
            vmin=vmin, vmax=vmax,
            legend_label="Pobreza (%)",
        )

    fig.suptitle(
        f"Per\u00fa: Evoluci\u00f3n de la Pobreza Departamental",
        fontsize=17, fontweight="bold",
        color=NEXUS_COLORS["text_primary"],
        y=0.98,
    )
    add_source_line(fig)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out = output_dir / "dept_evolution.png"
    fig.savefig(out, dpi=MAP_DPI)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def plot_province_map(
    province_data: pd.DataFrame,
    column: str = "poverty_rate",
    year: int = 2024,
    cmap_name: str = "poverty",
    title: str | None = None,
    output_dir: Path = MAPS_DIR,
    filename: str | None = None,
    formatter=None,
    legend_label: str = "",
) -> Path:
    """Plot a province-level choropleth.

    Parameters
    ----------
    province_data : DataFrame
        Must have columns ``province_code`` and ``column``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf = load_geodata("province")
    merged = gdf.merge(
        province_data, left_on="code", right_on="province_code", how="left",
    )

    if title is None:
        title = f"Per\u00fa: Pobreza Provincial, {year}"
    if filename is None:
        filename = f"prov_{column}_{year}.png"

    fig, ax = plt.subplots(1, 1, figsize=SINGLE_FIGSIZE)
    plot_choropleth(
        merged, column, title,
        cmap_name=cmap_name, ax=ax,
        show_labels=False,
        formatter=formatter,
        legend_label=legend_label,
        level="province",
    )
    add_source_line(fig)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    out = output_dir / filename
    fig.savefig(out, dpi=MAP_DPI)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def plot_district_map(
    year: int = 2024,
    output_dir: Path = MAPS_DIR,
) -> Path:
    """Plot district-level map colored by department poverty data.

    Since we only have department-level poverty, districts are colored
    by their parent department's poverty rate, showing fine-grained borders.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf = load_geodata("district")
    data = load_poverty_data(year)

    # Merge department-level data onto districts via dept_code
    merged = gdf.merge(
        data, left_on="dept_code", right_on="department_code", how="left",
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    plot_choropleth(
        merged, "poverty_rate",
        f"Per\u00fa: Pobreza por Distrito (datos departamentales), {year}",
        cmap_name="poverty", ax=ax,
        show_labels=False,
        legend_label="Pobreza (%)",
        level="district",
    )
    add_source_line(fig)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    out = output_dir / f"dist_poverty_{year}.png"
    fig.savefig(out, dpi=MAP_DPI)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def compute_province_poverty(year: int = 2024, min_obs: int = 30) -> pd.DataFrame:
    """Compute province-level poverty from ENAHO microdata.

    Parameters
    ----------
    year : int
        Year to compute.
    min_obs : int
        Minimum number of household observations per province.
        Provinces with fewer are excluded as unreliable.

    Returns
    -------
    DataFrame with columns: province_code, poverty_rate, extreme_poverty_rate, n_obs.
    """
    from src.ingestion.inei import ENAHOClient

    client = ENAHOClient()
    df = client.read_sumaria(year)
    if df is None:
        raise FileNotFoundError(f"ENAHO data not found for {year}")

    df.columns = [c.lower() for c in df.columns]
    df["province_code"] = df["ubigeo"].astype(str).str[:4]

    mie = df["mieperho"].astype(float).clip(lower=1)
    df["weight"] = (df["factor07"] * mie).round(0)

    pob = df["pobreza"]
    if pob.dtype in ("int8", "int16", "int32", "int64", "float64"):
        df["is_poor"] = pob.isin([1, 2]).astype(float)
        df["is_extreme_poor"] = (pob == 1).astype(float)
    else:
        pobreza = pob.astype(str).str.lower().str.strip()
        df["is_poor"] = (
            pobreza.str.contains("pobre", na=False)
            & ~pobreza.str.contains("no pobre", na=False)
        ).astype(float)
        df["is_extreme_poor"] = (pobreza == "pobre extremo").astype(float)

    rows = []
    for prov_code, grp in df.groupby("province_code"):
        n = len(grp)
        if n < min_obs:
            continue
        w = grp["weight"].values
        mask = np.isfinite(w) & (w > 0)
        if not mask.any():
            continue
        poverty_rate = float(np.average(grp["is_poor"].values[mask], weights=w[mask]))
        extreme_rate = float(np.average(grp["is_extreme_poor"].values[mask], weights=w[mask]))
        rows.append({
            "province_code": prov_code,
            "poverty_rate": poverty_rate,
            "extreme_poverty_rate": extreme_rate,
            "n_obs": n,
        })

    result = pd.DataFrame(rows)
    logger.info(
        "Province poverty %d: %d provinces (>=%d obs), mean poverty %.1f%%",
        year, len(result), min_obs, result["poverty_rate"].mean() * 100,
    )
    return result


def generate_all_maps(year: int = 2024) -> list[Path]:
    """Generate all NEXUS maps — wrapper for orchestrator.

    Calls each map generator with try/except so one failure
    doesn't block the rest.

    Returns
    -------
    list[Path]
        Paths to all successfully generated PNG files.
    """
    from src.visualization.style import apply_nexus_style
    apply_nexus_style()

    generated: list[Path] = []

    generators = [
        ("dept_poverty", lambda: plot_department_maps(year)),
        ("dept_extreme_poverty", lambda: plot_extreme_poverty_maps(year)),
        ("dept_evolution", lambda: plot_evolution_maps([2004, 2015, year])),
        ("dist_poverty", lambda: plot_district_map(year)),
    ]

    for name, func in generators:
        try:
            path = func()
            generated.append(path)
            logger.info("Generated %s", name)
        except Exception as e:
            logger.error("Failed to generate %s: %s", name, e)

    # Province maps (require ENAHO microdata)
    try:
        prov_data = compute_province_poverty(year)
        p1 = plot_province_map(
            prov_data, column="poverty_rate", year=year,
            cmap_name="poverty",
            title=f"Per\u00fa: Pobreza Provincial, {year}",
            filename=f"prov_poverty_{year}.png",
            legend_label="Pobreza (%)",
        )
        generated.append(p1)

        p2 = plot_province_map(
            prov_data, column="extreme_poverty_rate", year=year,
            cmap_name="extreme_poverty",
            title=f"Per\u00fa: Pobreza Extrema Provincial, {year}",
            filename=f"prov_extreme_poverty_{year}.png",
            legend_label="Pobreza extrema (%)",
        )
        generated.append(p2)
        logger.info("Generated province maps")
    except Exception as e:
        logger.error("Failed to generate province maps: %s", e)

    logger.info("Map generation complete: %d files", len(generated))
    return generated
