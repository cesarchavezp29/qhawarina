"""Generate individual plots for each BCRP variable."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import yaml

from config.settings import RAW_BCRP_DIR, SERIES_CATALOG_PATH


def load_data():
    path = RAW_BCRP_DIR / "bcrp_national_all.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_catalog():
    with open(SERIES_CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)

    code_info = {}
    for cat_key, cat_val in catalog.get("national", {}).items():
        if isinstance(cat_val, dict):
            for entry in cat_val.get("series", []):
                code_info[entry["code"]] = {
                    "name": entry["name"],
                    "category": cat_key,
                }
    return code_info


def plot_series(df, code, info, output_dir):
    s = df[df["series_code"] == code].sort_values("date")
    vals = s.dropna(subset=["value"])

    if len(vals) == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(vals["date"], vals["value"], linewidth=1.0, color="#1f77b4")
    ax.fill_between(vals["date"], vals["value"], alpha=0.08, color="#1f77b4")

    # Zero line for growth rates / percentages
    if vals["value"].min() < 0:
        ax.axhline(y=0, color="gray", linewidth=0.6, linestyle="--")

    # Stats
    v = vals["value"]
    stats_text = (
        f"n = {len(v)}  |  "
        f"min = {v.min():.2f}  |  max = {v.max():.2f}  |  "
        f"mean = {v.mean():.2f}  |  last = {v.iloc[-1]:.2f}"
    )
    date_range = f"{vals['date'].min().strftime('%Y-%m')} to {vals['date'].max().strftime('%Y-%m')}"

    name = info["name"]
    category = info["category"].replace("_", " ").title()

    ax.set_title(f"{code}: {name}", fontsize=13, fontweight="bold", pad=12)
    ax.text(0.5, 1.02, f"[{category}]  {date_range}", transform=ax.transAxes,
            fontsize=9, ha="center", va="bottom", color="gray")
    ax.text(0.5, -0.10, stats_text, transform=ax.transAxes,
            fontsize=9, ha="center", va="top", color="#555555",
            fontfamily="monospace")

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.3, which="major")
    ax.grid(True, alpha=0.1, which="minor")

    # Highlight COVID period
    covid_start = pd.Timestamp("2020-03-01")
    covid_end = pd.Timestamp("2020-12-01")
    if vals["date"].min() < covid_start < vals["date"].max():
        ax.axvspan(covid_start, covid_end, alpha=0.06, color="red", label="COVID-19")

    plt.tight_layout()
    safe_code = code.replace("/", "_")
    out_path = output_dir / f"{safe_code}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    df = load_data()
    code_info = load_catalog()

    output_dir = PROJECT_ROOT / "data" / "validation" / "individual"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by category then code for organized output
    sorted_codes = sorted(code_info.keys(), key=lambda c: (code_info[c]["category"], c))

    current_cat = None
    count = 0
    for code in sorted_codes:
        info = code_info[code]
        cat = info["category"].replace("_", " ").title()
        if cat != current_cat:
            current_cat = cat
            print(f"\n--- {current_cat} ---")

        path = plot_series(df, code, info, output_dir)
        if path:
            count += 1
            print(f"  {code}: {info['name']}")

    print(f"\n{'='*60}")
    print(f"Generated {count} individual plots in:")
    print(f"  {output_dir}")


if __name__ == "__main__":
    main()
