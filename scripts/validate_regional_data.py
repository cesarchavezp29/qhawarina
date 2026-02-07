"""Validate downloaded regional BCRP data and generate summary charts.

Mirrors scripts/validate_data.py for the regional/departmental files.
Per category parquet:
  - Count series, observations, date range, missing %
  - Sanity checks (electricity > 0, credit > 0, etc.)
  - Per-department completeness summary

Usage:
    python scripts/validate_regional_data.py              # all categories
    python scripts/validate_regional_data.py --only credit,electricity
    python scripts/validate_regional_data.py --no-plots   # report only
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
import pandas as pd
import numpy as np
import yaml

from config.settings import RAW_BCRP_DIR
from src.visualization.style import (
    NEXUS_COLORS as C,
    apply_nexus_style,
    add_watermark,
    add_source_line,
    MAP_DPI,
)

BRAND = "Qhawarina"

OUTPUT_DIR = PROJECT_ROOT / "data" / "validation" / "regional"

# Category key → (parquet file, sanity range description)
CATEGORY_INFO = {
    "credit_by_department": {
        "file": "bcrp_regional_credit.parquet",
        "sanity_col": "value",
        "sanity_range": (0, 500_000),
        "description": "Credit (millones S/)",
    },
    "deposits_by_department": {
        "file": "bcrp_regional_deposits.parquet",
        "sanity_col": "value",
        "sanity_range": (0, 500_000),
        "description": "Deposits (millones S/)",
    },
    "pension_affiliates_by_department": {
        "file": "bcrp_regional_pension.parquet",
        "sanity_col": "value",
        "sanity_range": (0, 10_000),
        "description": "AFP Affiliates (miles personas)",
    },
    "electricity_by_department": {
        "file": "bcrp_regional_electricity.parquet",
        "sanity_col": "value",
        "sanity_range": (0, 10_000),
        "description": "Electricity Production (GWh)",
    },
    "mining_copper_by_department": {
        "file": "bcrp_regional_mining_copper.parquet",
        "sanity_col": "value",
        "sanity_range": (0, 500_000),
        "description": "Copper (TMF)",
    },
    "mining_gold_by_department": {
        "file": "bcrp_regional_mining_gold.parquet",
        "sanity_col": "value",
        "sanity_range": (0, 1_000),
        "description": "Gold (miles oz troy)",
    },
    "tax_revenue_by_department": {
        "file": "bcrp_regional_tax.parquet",
        "sanity_col": "value",
        "sanity_range": (0, 50_000),
        "description": "Tax Revenue (millones S/)",
    },
    "government_spending_local": {
        "file": "bcrp_regional_spending_local.parquet",
        "sanity_col": "value",
        "sanity_range": (0, 10_000),
        "description": "Local Gov Spending (millones S/)",
    },
    "government_spending_regional": {
        "file": "bcrp_regional_spending_regional.parquet",
        "sanity_col": "value",
        "sanity_range": (0, 10_000),
        "description": "Regional Gov Spending (millones S/)",
    },
    "exports_by_department": {
        "file": "bcrp_regional_exports.parquet",
        "sanity_col": "value",
        "sanity_range": (0, 20_000),
        "description": "Exports FOB (millones USD)",
    },
    "imports_by_customs": {
        "file": "bcrp_regional_imports.parquet",
        "sanity_col": "value",
        "sanity_range": (0, 20_000),
        "description": "Imports FOB (millones USD)",
    },
}


def validate_category(cat_key: str, info: dict) -> dict:
    """Validate a single regional category parquet file."""
    filepath = RAW_BCRP_DIR / info["file"]
    result = {
        "category": cat_key,
        "description": info["description"],
        "file": info["file"],
        "exists": filepath.exists(),
    }

    if not filepath.exists():
        result["status"] = "MISSING"
        return result

    df = pd.read_parquet(filepath)
    df["date"] = pd.to_datetime(df["date"])

    result["n_rows"] = len(df)
    result["n_series"] = df["series_code"].nunique()
    result["date_min"] = df["date"].min()
    result["date_max"] = df["date"].max()
    result["n_months"] = df["date"].nunique()

    # Missing values
    n_total = len(df)
    n_nonnull = df["value"].notna().sum()
    result["missing_pct"] = round((1 - n_nonnull / max(n_total, 1)) * 100, 1)

    # Per-department coverage
    if "department" in df.columns:
        dept_counts = df.groupby("department")["value"].apply(
            lambda x: x.notna().sum()
        )
        result["departments"] = len(dept_counts)
        result["dept_min_obs"] = int(dept_counts.min()) if len(dept_counts) > 0 else 0
        result["dept_max_obs"] = int(dept_counts.max()) if len(dept_counts) > 0 else 0
    else:
        result["departments"] = 0

    # Sanity check: values within plausible range
    vals = df["value"].dropna()
    lo, hi = info["sanity_range"]
    out_of_range = ((vals < lo * 0.5) | (vals > hi * 2)).sum()
    result["sanity_passed"] = out_of_range == 0
    result["out_of_range"] = int(out_of_range)

    # Negative value check (all regional series should be non-negative)
    n_negative = (vals < 0).sum()
    result["negative_values"] = int(n_negative)

    result["status"] = "OK" if result["sanity_passed"] and n_negative == 0 else "WARN"
    result["df"] = df  # Keep for plotting
    return result


def print_report(results: list[dict]):
    """Print console validation report."""
    print("=" * 90)
    print(f"{BRAND} — Regional Data Validation Report")
    print("=" * 90)

    total_rows = 0
    total_series = 0
    all_ok = True

    fmt = "{:<35s} {:>5s} {:>6s} {:>8s} {:>12s} {:>12s} {:>6s}"
    print(fmt.format("Category", "Srs", "Rows", "Miss%", "From", "To", "Status"))
    print("-" * 90)

    for r in results:
        if not r["exists"]:
            print(f"  {r['category']:<33s}  {'':>5s} {'':>6s} {'':>8s} {'':>12s} {'':>12s} MISSING")
            all_ok = False
            continue

        total_rows += r["n_rows"]
        total_series += r["n_series"]
        if r["status"] != "OK":
            all_ok = False

        print(fmt.format(
            r["category"],
            str(r["n_series"]),
            str(r["n_rows"]),
            f"{r['missing_pct']:.1f}%",
            r["date_min"].strftime("%Y-%m"),
            r["date_max"].strftime("%Y-%m"),
            r["status"],
        ))

    print("-" * 90)
    print(f"TOTAL: {total_series} series, {total_rows:,} rows")
    print(f"\nAll checks passed: {all_ok}")

    # Detailed per-category notes
    print("\n--- Detailed Notes ---")
    for r in results:
        if not r["exists"]:
            continue
        notes = []
        if r.get("departments", 0) > 0:
            notes.append(f"{r['departments']} depts ({r['dept_min_obs']}-{r['dept_max_obs']} obs)")
        if r["out_of_range"] > 0:
            notes.append(f"{r['out_of_range']} out-of-range values")
        if r["negative_values"] > 0:
            notes.append(f"{r['negative_values']} negative values")
        note_str = "; ".join(notes) if notes else "clean"
        print(f"  {r['category']}: {note_str}")

    return all_ok


def plot_category_summary(results: list[dict], output_dir: Path):
    """Generate summary chart for all regional categories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_nexus_style()

    valid_results = [r for r in results if r["exists"] and "df" in r]
    if not valid_results:
        print("No data to plot.")
        return

    n_cats = len(valid_results)
    fig, axes = plt.subplots(
        min(n_cats, 4), 1,
        figsize=(14, 3 * min(n_cats, 4)),
        sharex=True,
    )
    if min(n_cats, 4) == 1:
        axes = [axes]

    fig.suptitle(
        f"{BRAND} — Regional Data Overview",
        fontsize=14, fontweight="bold", y=1.02,
    )

    # Plot first 4 categories as time-series overviews
    colors = [C["accent"], "#2980B9", "#8E44AD", "#27AE60", "#C0392B"]
    for i, r in enumerate(valid_results[:4]):
        ax = axes[i]
        df = r["df"]

        # Plot a few representative departments
        if "department" in df.columns:
            depts = df["department"].dropna().unique()[:5]
            for j, dept in enumerate(depts):
                s = df[df["department"] == dept].sort_values("date")
                ax.plot(
                    s["date"], s["value"],
                    linewidth=0.6, alpha=0.7,
                    color=colors[j % len(colors)],
                    label=dept[:15],
                )
        else:
            codes = df["series_code"].unique()[:5]
            for j, code in enumerate(codes):
                s = df[df["series_code"] == code].sort_values("date")
                name = s["series_name"].iloc[0] if "series_name" in s.columns else code
                ax.plot(
                    s["date"], s["value"],
                    linewidth=0.6, alpha=0.7,
                    color=colors[j % len(colors)],
                    label=name[:15],
                )

        ax.set_title(r["description"], fontsize=10)
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        add_watermark(ax)

    plt.tight_layout()
    add_source_line(fig, f"Fuente: BCRP. Elaboracion: {BRAND}.")
    out_path = output_dir / "regional_validation_overview.png"
    fig.savefig(out_path, dpi=MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Observation counts bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"{BRAND} — Regional Series Counts", fontsize=14, fontweight="bold")

    cat_names = [r["description"][:30] for r in valid_results]
    cat_counts = [r["n_series"] for r in valid_results]
    bars = ax.barh(range(len(cat_names)), cat_counts, color=C["accent"], height=0.6)
    ax.set_yticks(range(len(cat_names)))
    ax.set_yticklabels(cat_names, fontsize=9)
    ax.set_xlabel("Number of series")
    for bar, count in zip(bars, cat_counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=8)
    add_watermark(ax)

    plt.tight_layout()
    add_source_line(fig, f"Fuente: BCRP. Elaboracion: {BRAND}.")
    out_path = output_dir / "regional_series_counts.png"
    fig.savefig(out_path, dpi=MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate regional BCRP data"
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated category keys to validate",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip chart generation",
    )
    args = parser.parse_args()

    # Determine categories
    if args.only:
        requested = [s.strip() for s in args.only.split(",")]
        categories = {k: v for k, v in CATEGORY_INFO.items() if k in requested or
                      any(r in k for r in requested)}
    else:
        categories = CATEGORY_INFO

    results = []
    for cat_key, info in categories.items():
        r = validate_category(cat_key, info)
        results.append(r)

    all_ok = print_report(results)

    if not args.no_plots:
        plot_category_summary(results, OUTPUT_DIR)

    # Clean up dataframes from results before returning
    for r in results:
        r.pop("df", None)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
