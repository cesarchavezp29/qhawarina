"""Comprehensive data validation and visualization for BCRP data."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import yaml

from config.settings import RAW_BCRP_DIR, SERIES_CATALOG_PATH
from src.visualization.style import (
    NEXUS_COLORS as C,
    apply_nexus_style,
    add_watermark,
    add_source_line,
    MAP_DPI,
)

BRAND = "Qhawarina"


def load_data():
    """Load combined BCRP data."""
    path = RAW_BCRP_DIR / "bcrp_national_all.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_catalog():
    """Load series catalog for friendly names."""
    with open(SERIES_CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)

    code_to_name = {}
    for cat_key, cat_val in catalog.get("national", {}).items():
        if isinstance(cat_val, dict):
            for entry in cat_val.get("series", []):
                code_to_name[entry["code"]] = entry["name"]
    return code_to_name


def print_validation_report(df, code_to_name):
    """Print detailed validation report to console."""
    print("=" * 80)
    print(f"{BRAND} — BCRP Data Validation Report")
    print("=" * 80)

    # Overall stats
    print(f"\nTotal observations: {len(df):,}")
    print(f"Unique series: {df['series_code'].nunique()}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Categories: {sorted(df['category'].unique())}")

    # Per-series stats
    print(f"\n{'Code':<12} {'Obs':>4} {'Non-null':>8} {'Missing%':>8} "
          f"{'Min':>12} {'Max':>12} {'Mean':>12}  Name")
    print("-" * 110)

    total_obs = 0
    total_nonnull = 0
    for code in sorted(df["series_code"].unique()):
        s = df[df["series_code"] == code]
        obs = len(s)
        nonnull = s["value"].notna().sum()
        missing_pct = (1 - nonnull / obs) * 100 if obs > 0 else 0
        vals = s["value"].dropna()
        mn = vals.min() if len(vals) > 0 else float("nan")
        mx = vals.max() if len(vals) > 0 else float("nan")
        mean = vals.mean() if len(vals) > 0 else float("nan")
        name = code_to_name.get(code, "?")[:40]
        print(f"{code:<12} {obs:>4} {nonnull:>8} {missing_pct:>7.1f}% "
              f"{mn:>12.2f} {mx:>12.2f} {mean:>12.2f}  {name}")
        total_obs += obs
        total_nonnull += nonnull

    total_missing = (1 - total_nonnull / total_obs) * 100
    print("-" * 110)
    print(f"{'TOTAL':<12} {total_obs:>4} {total_nonnull:>8} {total_missing:>7.1f}%")

    # Date coverage check
    print("\n--- Date Coverage ---")
    for code in sorted(df["series_code"].unique()):
        s = df[df["series_code"] == code].sort_values("date")
        vals = s.dropna(subset=["value"])
        if len(vals) == 0:
            continue
        first = vals["date"].min().date()
        last = vals["date"].max().date()
        # Check for gaps (missing months)
        all_dates = set(s["date"].dt.date)
        expected = pd.date_range(s["date"].min(), s["date"].max(), freq="MS")
        missing_dates = set(expected.date) - all_dates
        gap_str = f"{len(missing_dates)} gaps" if missing_dates else "continuous"
        name_short = code_to_name.get(code, "?")[:30]
        print(f"  {code}: {first} to {last} ({len(vals)} obs, {gap_str})  {name_short}")

    # Value sanity checks
    print("\n--- Value Sanity Checks ---")
    checks = [
        ("PN01246PM", "Tipo de cambio S//USD", 2.0, 5.0),
        ("PD04722MM", "Tasa referencia %", 0.0, 10.0),
        ("PN01770AM", "PBI global index", 50, 250),
        ("PN01271PM", "IPC var% mensual", -2.0, 3.0),
        ("PN01273PM", "IPC var% 12m", -1.0, 10.0),
        ("PN38714BM", "Exportaciones mill USD", 500, 12000),
        ("PN38718BM", "Importaciones mill USD", 300, 12000),
        ("PN00027MM", "RIN mill USD", 5000, 100000),
        ("PN00518MM", "Credito mill S/", 20000, 600000),
        ("PN07807NM", "TAMN %", 5, 30),
        ("PN07816NM", "TIPMN %", 0.5, 10),
        ("PD37966AM", "Electricidad GWh", 1500, 6000),
        ("PN02204FM", "Ingresos gob mill S/", 1000, 25000),
        ("PN02409FM", "Gastos gob mill S/", 1000, 25000),
        # New series
        ("PN01652XM", "Cobre ctv USD/lb", 50, 600),
        ("PN01654XM", "Oro USD/oz", 250, 3000),
        ("PN01660XM", "Petroleo WTI USD/bbl", 10, 150),
        ("PN01655XM", "Plata USD/oz", 4, 50),
        ("PN01657XM", "Zinc ctv USD/lb", 20, 250),
        ("PN39445PM", "IPC alimentos indice", 50, 200),
        ("PN01383PM", "IPC alimentos var%", -5.0, 20.0),
        ("PN38923BM", "Terminos intercambio idx", 50, 200),
        ("PN01013MM", "Emision primaria mill S/", 5000, 200000),
        ("PN00178MM", "Circulante mill S/", 5000, 150000),
        ("PN38063GM", "Tasa desempleo %", 3.0, 20.0),
        ("PN31879GM", "Empleo formal miles", 1000, 6000),
        ("PD37981AM", "Expectativas empresas", 20, 90),
        ("PD12912AM", "Expect inflacion 12m", 0.5, 8.0),
    ]
    all_pass = True
    for code, desc, lo, hi in checks:
        s = df[df["series_code"] == code]["value"].dropna()
        if len(s) == 0:
            print(f"  SKIP  {code} ({desc}): no data")
            continue
        mn, mx = s.min(), s.max()
        ok = mn >= lo * 0.5 and mx <= hi * 2  # generous margin
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {status}  {code} ({desc}): [{mn:.2f}, {mx:.2f}] expected ~[{lo}, {hi}]")

    print(f"\nAll sanity checks passed: {all_pass}")
    return all_pass


def plot_all_series(df, code_to_name, output_dir):
    """Generate time series plots for all categories."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = {
        "gdp_indicators": {
            "title": "GDP Indicators (var% interanual)",
            "codes": ["PN01728AM", "PN01713AM", "PN01716AM", "PN01717AM",
                       "PN01720AM", "PN01723AM", "PN01724AM", "PN01725AM", "PN01726AM"],
        },
        "gdp_level": {
            "title": "GDP Level & Seasonally Adjusted Growth",
            "codes": ["PN01770AM", "PN01731AM"],
        },
        "leading_indicators": {
            "title": "Leading Indicators",
            "codes": ["PD37966AM", "PD37967GM", "PD04722MM", "PN01246PM"],
        },
        "trade": {
            "title": "Trade Balance (millones USD FOB)",
            "codes": ["PN38714BM", "PN38718BM"],
        },
        "credit_financial": {
            "title": "Credit & Financial Indicators",
            "codes": ["PN00518MM", "PN00027MM", "PN07807NM", "PN07816NM"],
        },
        "inflation": {
            "title": "Inflation Indicators — IPC Lima",
            "codes": ["PN01271PM", "PN01273PM", "PN38706PM"],
        },
        "fiscal": {
            "title": "Fiscal Indicators (millones S/)",
            "codes": ["PN02204FM", "PN02409FM"],
        },
        "commodity_prices": {
            "title": "Commodity Prices",
            "codes": ["PN01652XM", "PN01654XM", "PN01660XM", "PN01655XM", "PN01657XM"],
        },
        "food_monetary": {
            "title": "Food Prices, Monetary & Trade",
            "codes": ["PN39445PM", "PN01383PM", "PN38923BM", "PN01013MM", "PN00178MM"],
        },
        "employment_confidence": {
            "title": "Employment & Confidence Indicators",
            "codes": ["PN38063GM", "PN31879GM", "PD37981AM", "PD12912AM"],
        },
    }

    for cat_key, cat_info in categories.items():
        codes = cat_info["codes"]
        title = cat_info["title"]
        n = len(codes)

        if n <= 2:
            fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
            if n == 1:
                axes = [axes]
        else:
            rows = (n + 1) // 2
            fig, axes = plt.subplots(rows, 2, figsize=(16, 3.5 * rows), sharex=True)
            axes = axes.flatten()

        fig.suptitle(f"{BRAND} — {title}", fontsize=14, fontweight="bold", y=1.02)

        for i, code in enumerate(codes):
            ax = axes[i]
            s = df[df["series_code"] == code].sort_values("date")
            vals = s.dropna(subset=["value"])

            if len(vals) == 0:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                ax.set_title(code)
                continue

            ax.plot(vals["date"], vals["value"], linewidth=0.8, color=C["accent"])
            ax.fill_between(vals["date"], vals["value"], alpha=0.1, color=C["accent"])

            # Add zero line for growth rates
            if any(k in code for k in ["AM", "PM", "GM"]):
                if vals["value"].min() < 0:
                    ax.axhline(y=0, color=C["text_secondary"], linewidth=0.5, linestyle="--")

            name = code_to_name.get(code, code)[:50]
            nonnull = vals["value"].notna().sum()
            ax.set_title(f"{code}: {name}", fontsize=9)
            ax.text(0.98, 0.95, f"n={nonnull}", transform=ax.transAxes,
                    fontsize=7, ha="right", va="top", color=C["text_secondary"])

            ax.xaxis.set_major_locator(mdates.YearLocator(4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.tick_params(axis="both", labelsize=8)
            watermark(ax)

        # Hide unused subplots
        for j in range(len(codes), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        add_source_line(fig, f"Fuente: BCRP. Elaboracion: {BRAND}.")
        out_path = output_dir / f"validation_{cat_key}.png"
        fig.savefig(out_path, dpi=MAP_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    # ── Summary overview plot ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{BRAND} — Data Overview", fontsize=14, fontweight="bold")

    # Panel 1: Observation counts per series
    ax = axes[0, 0]
    counts = df.groupby("series_code")["value"].apply(lambda x: x.notna().sum())
    counts = counts.sort_values()
    colors = ["#C0392B" if c < 200 else "#27AE60" for c in counts]
    ax.barh(range(len(counts)), counts.values, color=colors, height=0.7)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=7)
    ax.set_xlabel("Non-null observations")
    ax.set_title("Observations per Series")
    ax.axvline(x=240, color=C["accent"], linestyle="--", linewidth=0.8, label="20yr monthly")
    ax.legend(fontsize=7)
    watermark(ax)

    # Panel 2: Missing data heatmap
    ax = axes[0, 1]
    pivot = df.pivot_table(index="date", columns="series_code", values="value")
    missing_pct = pivot.isna().resample("YE").mean()
    im = ax.imshow(missing_pct.T.values, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(missing_pct.columns)))
    ax.set_yticklabels(missing_pct.columns, fontsize=6)
    years = missing_pct.index.year
    ax.set_xticks(range(0, len(years), 4))
    ax.set_xticklabels(years[::4], fontsize=8)
    ax.set_title("Missing Data by Year (red=missing)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    watermark(ax)

    # Panel 3: GDP sectoral growth rates
    ax = axes[1, 0]
    gdp_codes = ["PN01728AM", "PN01713AM", "PN01720AM", "PN01724AM"]
    gdp_colors = [C["accent"], "#2980B9", "#8E44AD", "#27AE60"]
    for code, clr in zip(gdp_codes, gdp_colors):
        s = df[df["series_code"] == code].sort_values("date").dropna(subset=["value"])
        name_short = code_to_name.get(code, code)[:25]
        ax.plot(s["date"], s["value"], linewidth=0.7, label=name_short, alpha=0.8, color=clr)
    ax.axhline(y=0, color=C["border"], linewidth=0.5)
    ax.set_title("GDP Sectoral Growth (var% interanual)")
    ax.legend(fontsize=7, loc="lower left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    watermark(ax)

    # Panel 4: Key macro indicators (normalized)
    ax = axes[1, 1]
    macro_codes = [
        ("PN01246PM", "Tipo de cambio", C["accent"]),
        ("PD04722MM", "Tasa referencia", "#2980B9"),
        ("PN01273PM", "Inflacion 12m", "#C0392B"),
    ]
    for code, label, clr in macro_codes:
        s = df[df["series_code"] == code].sort_values("date").dropna(subset=["value"])
        ax.plot(s["date"], s["value"], linewidth=0.8, label=label, color=clr)
    ax.set_title("Key Macro Indicators")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    watermark(ax)

    plt.tight_layout()
    add_source_line(fig, f"Fuente: BCRP. Elaboracion: {BRAND}.")
    out_path = output_dir / "validation_overview.png"
    fig.savefig(out_path, dpi=MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def watermark(ax):
    add_watermark(ax)


def main():
    apply_nexus_style()
    df = load_data()
    code_to_name = load_catalog()

    # Validation report
    all_pass = print_validation_report(df, code_to_name)

    # Generate plots
    output_dir = PROJECT_ROOT / "data" / "validation"
    plot_all_series(df, code_to_name, output_dir)

    print(f"\nAll plots saved to: {output_dir}")
    if all_pass:
        print(f"\nVALIDATION PASSED — Data is ready.")
    else:
        print(f"\nWARNING: Some sanity checks failed. Review before proceeding.")


if __name__ == "__main__":
    main()
