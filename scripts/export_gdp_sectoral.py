#!/usr/bin/env python3
"""
Export GDP sectoral breakdowns to JSON for Qhawarina website.

Generates:
- gdp_sectoral.json: All 8 sectors with time series data
"""

import json
from pathlib import Path

import pandas as pd


def export_gdp_sectoral():
    """Export GDP sectoral data to JSON."""

    # Load quarterly GDP targets with sectors
    gdp_file = Path("data/targets/gdp_quarterly.parquet")
    df = pd.read_parquet(gdp_file)

    # Sort by date
    df = df.sort_values("date")

    # Define sectors with display names
    sectors = {
        "agropecuario_yoy": {
            "id": "agropecuario",
            "name_es": "Agropecuario",
            "name_en": "Agriculture & Livestock",
            "color": "#22c55e"  # green
        },
        "pesca_yoy": {
            "id": "pesca",
            "name_es": "Pesca",
            "name_en": "Fishing",
            "color": "#3b82f6"  # blue
        },
        "mineria_yoy": {
            "id": "mineria",
            "name_es": "Minería",
            "name_en": "Mining",
            "color": "#8b5cf6"  # purple
        },
        "manufactura_yoy": {
            "id": "manufactura",
            "name_es": "Manufactura",
            "name_en": "Manufacturing",
            "color": "#f97316"  # orange
        },
        "electricidad_yoy": {
            "id": "electricidad",
            "name_es": "Electricidad y Agua",
            "name_en": "Electricity & Water",
            "color": "#eab308"  # yellow
        },
        "construccion_yoy": {
            "id": "construccion",
            "name_es": "Construcción",
            "name_en": "Construction",
            "color": "#ef4444"  # red
        },
        "comercio_yoy": {
            "id": "comercio",
            "name_es": "Comercio",
            "name_en": "Commerce",
            "color": "#06b6d4"  # cyan
        },
        "servicios_yoy": {
            "id": "servicios",
            "name_es": "Servicios",
            "name_en": "Services",
            "color": "#ec4899"  # pink
        }
    }

    # Helper function to format quarter dates
    def format_quarter(dt):
        quarter = (dt.month - 1) // 3 + 1
        return f"{dt.year}-Q{quarter}"

    # Build time series for each sector
    sectoral_data = {
        "metadata": {
            "last_update": df["date"].max().isoformat(),
            "frequency": "quarterly",
            "unit": "% YoY",
            "source": "BCRP (INEI GDP by economic activity)",
            "n_observations": len(df),
            "n_sectors": len(sectors)
        },
        "total_gdp": {
            "dates": [format_quarter(d) for d in df["date"]],
            "values": df["gdp_yoy"].round(2).tolist()
        },
        "sectors": []
    }

    # Add each sector's time series
    for col, meta in sectors.items():
        sector_series = {
            "id": meta["id"],
            "name_es": meta["name_es"],
            "name_en": meta["name_en"],
            "color": meta["color"],
            "dates": [format_quarter(d) for d in df["date"]],
            "values": df[col].round(2).tolist(),
            "latest_value": float(df[col].iloc[-1].round(2)),
            "latest_date": format_quarter(df["date"].iloc[-1])
        }
        sectoral_data["sectors"].append(sector_series)

    # Calculate sector weights (contribution to GDP growth)
    # Using last 4 quarters average as proxy
    recent_df = df.tail(4)
    weights = {}
    for col, meta in sectors.items():
        weights[meta["id"]] = float(recent_df[col].mean().round(1))

    sectoral_data["sector_weights"] = weights

    # Save to exports directory
    export_dir = Path("exports/data")
    export_dir.mkdir(parents=True, exist_ok=True)

    output_file = export_dir / "gdp_sectoral.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sectoral_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Exported GDP sectoral data:")
    print(f"   {output_file} ({output_file.stat().st_size / 1024:.1f} KB)")
    print(f"   {len(sectors)} sectors, {len(df)} quarters")
    print(f"   Latest quarter: {format_quarter(df['date'].max())}")

    # Print latest values
    print("\n[STATS] Latest sector growth rates (YoY):")
    for col, meta in sectors.items():
        latest = df[col].iloc[-1]
        print(f"   {meta['name_es']:25} {latest:>6.1f}%")

    return sectoral_data


if __name__ == "__main__":
    export_gdp_sectoral()
