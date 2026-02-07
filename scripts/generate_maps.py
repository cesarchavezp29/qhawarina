"""Generate all NEXUS maps and charts.

Usage:
    python scripts/generate_maps.py [--only NAME] [--year YEAR]

Outputs are saved to data/targets/maps/.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.settings import TARGETS_DIR, RAW_ENAHO_DIR
from src.visualization.style import apply_nexus_style, fmt_pct, fmt_soles

apply_nexus_style()

from src.visualization.maps import (
    MAPS_DIR,
    plot_department_maps,
    plot_extreme_poverty_maps,
    plot_evolution_maps,
    plot_province_map,
    plot_district_map,
)
from src.visualization.charts import (
    plot_national_trends,
    plot_ranking_bars,
    plot_department_sparklines,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Province-level poverty computation ───────────────────────────────────────

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
    DataFrame with columns: province_code, province_name, poverty_rate,
    extreme_poverty_rate, n_obs.
    """
    from src.ingestion.inei import ENAHOClient

    client = ENAHOClient()
    df = client.read_sumaria(year)
    if df is None:
        raise FileNotFoundError(f"ENAHO data not found for {year}")

    # Normalize columns
    df.columns = [c.lower() for c in df.columns]

    # Province code = first 4 digits of ubigeo
    df["province_code"] = df["ubigeo"].astype(str).str[:4]

    # Person-level weight
    mie = df["mieperho"].astype(float).clip(lower=1)
    df["weight"] = (df["factor07"] * mie).round(0)

    # Poverty flags from official variable
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

    # Aggregate by province
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


# ── Map generators ───────────────────────────────────────────────────────────

GENERATORS = {
    "dept_poverty": lambda year: plot_department_maps(year),
    "dept_extreme_poverty": lambda year: plot_extreme_poverty_maps(year),
    "dept_evolution": lambda _: plot_evolution_maps([2004, 2015, 2024]),
    "national_trends": lambda _: plot_national_trends(),
    "dept_ranking": lambda year: plot_ranking_bars(year),
    "dept_sparklines": lambda _: plot_department_sparklines(),
    "dist_poverty": lambda year: plot_district_map(year),
}


def generate_province_maps(year: int) -> list[Path]:
    """Generate province-level choropleth maps."""
    prov_data = compute_province_poverty(year)

    paths = []
    # Poverty rate
    p = plot_province_map(
        prov_data, column="poverty_rate", year=year,
        cmap_name="poverty",
        title=f"Per\u00fa: Pobreza Provincial, {year}",
        filename=f"prov_poverty_{year}.png",
        legend_label="Pobreza (%)",
    )
    paths.append(p)

    # Extreme poverty
    p = plot_province_map(
        prov_data, column="extreme_poverty_rate", year=year,
        cmap_name="extreme_poverty",
        title=f"Per\u00fa: Pobreza Extrema Provincial, {year}",
        filename=f"prov_extreme_poverty_{year}.png",
        legend_label="Pobreza extrema (%)",
    )
    paths.append(p)

    return paths


def main():
    parser = argparse.ArgumentParser(description="Generate NEXUS maps and charts")
    parser.add_argument("--only", type=str, default=None,
                        help="Generate only this map (e.g. dept_poverty, prov_poverty)")
    parser.add_argument("--year", type=int, default=2024,
                        help="Target year (default: 2024)")
    args = parser.parse_args()

    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    start = time.time()
    generated = []

    if args.only:
        names = [args.only]
    else:
        names = list(GENERATORS.keys()) + ["prov_poverty"]

    for name in names:
        logger.info("Generating: %s", name)
        t0 = time.time()
        try:
            if name == "prov_poverty":
                paths = generate_province_maps(args.year)
                generated.extend(paths)
            elif name in GENERATORS:
                path = GENERATORS[name](args.year)
                generated.append(path)
            else:
                logger.warning("Unknown generator: %s", name)
        except Exception as e:
            logger.error("Failed to generate %s: %s", name, e, exc_info=True)

        logger.info("  Done in %.1fs", time.time() - t0)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Generated {len(generated)} files in {elapsed:.1f}s:")
    for p in generated:
        if p is not None:
            size_kb = p.stat().st_size / 1024
            print(f"  {p.name:40s}  ({size_kb:,.0f} KB)")
    print(f"\nOutput directory: {MAPS_DIR}")


if __name__ == "__main__":
    main()
