#!/usr/bin/env python3
"""
Export inflation category breakdowns to JSON for Qhawarina website.

Sources all index series from BCRP and computes monthly/12m variations.

Categories:
  - IPC Total
  - Alimentos y Bebidas
  - Sin Alimentos y Bebidas
  - Sin Alimentos y Energia (Core)
  - Subyacente
  - Transables
  - No Transables
"""

import json
from pathlib import Path

import pandas as pd
import requests


BCRP_BASE = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"
START = "2010-1"
END = "2025-12"


CATEGORY_SERIES = {
    "total": {
        "code": "PN01271PM",
        "type": "var_pct",
        "name_es": "IPC Total",
        "name_en": "Total CPI",
        "color": "#1f2937",
        "description_es": "Índice de Precios al Consumidor — Lima Metropolitana",
        "weight_pct": 100.0,
    },
    "alimentos": {
        "code": "PN39445PM",
        "type": "index",
        "name_es": "Alimentos y Bebidas",
        "name_en": "Food & Beverages",
        "color": "#ef4444",
        "description_es": "Incluye todos los alimentos y bebidas no alcohólicas",
        "weight_pct": 26.7,
    },
    "sin_alimentos": {
        "code": "PN38706PM",
        "type": "index",
        "name_es": "Sin Alimentos y Bebidas",
        "name_en": "Excluding Food",
        "color": "#3b82f6",
        "description_es": "CPI excluyendo alimentos y bebidas",
        "weight_pct": 73.3,
    },
    "core": {
        "code": "PN38707PM",
        "type": "index",
        "name_es": "Sin Alimentos ni Energía",
        "name_en": "Core (ex-Food & Energy)",
        "color": "#8b5cf6",
        "description_es": "Inflación subyacente excluyendo alimentos y componente energético",
        "weight_pct": None,
    },
    "subyacente": {
        "code": "PN38708PM",
        "type": "index",
        "name_es": "Subyacente",
        "name_en": "Underlying",
        "color": "#f97316",
        "description_es": "Excluye bienes con precios administrados o alta volatilidad",
        "weight_pct": None,
    },
    "transables": {
        "code": "PN38709PM",
        "type": "index",
        "name_es": "Transables",
        "name_en": "Tradables",
        "color": "#22c55e",
        "description_es": "Bienes cuyo precio se determina en mercados internacionales",
        "weight_pct": None,
    },
    "no_transables": {
        "code": "PN38710PM",
        "type": "index",
        "name_es": "No Transables",
        "name_en": "Non-Tradables",
        "color": "#ec4899",
        "description_es": "Bienes y servicios cuyo precio se determina localmente",
        "weight_pct": None,
    },
}


def fetch_bcrp_series(code: str, start: str = START, end: str = END) -> pd.DataFrame:
    """Download a single BCRP series and return as a DataFrame."""
    url = f"{BCRP_BASE}/{code}/json/{start}/{end}/esp"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()

    records = []
    for period in data.get("periods", []):
        name = period["name"]  # e.g. "Ene.2020"
        val = period["values"][0]
        if val not in ("n.d.", None, ""):
            records.append({"period_label": name, "value": float(val)})

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Parse period labels like "Ene.2020" or "Feb.2021"
    month_map = {
        "Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Ago": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dic": 12,
    }
    months = []
    for label in df["period_label"]:
        parts = label.split(".")
        m = month_map.get(parts[0], None)
        y = int(parts[1])
        months.append(pd.Timestamp(year=y, month=m, day=1))
    df["date"] = months
    df = df.set_index("date").sort_index()
    return df[["value"]]


def compute_var_from_index(df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly variation (%) and 12-month variation from index."""
    df["var_monthly"] = df["value"].pct_change(1) * 100
    df["var_12m"] = df["value"].pct_change(12) * 100
    return df


def format_dates(index: pd.DatetimeIndex) -> list:
    return [d.strftime("%Y-%m") for d in index]


def export_inflation_categories():
    print("Fetching inflation category data from BCRP...")

    category_data = {}
    for cat_id, meta in CATEGORY_SERIES.items():
        code = meta["code"]
        print(f"  Downloading {code}: {meta['name_es']}...")
        try:
            df = fetch_bcrp_series(code)
            if df.empty:
                print(f"    [SKIP] No data returned")
                continue

            if meta["type"] == "index":
                df = compute_var_from_index(df)
                var_col = "var_monthly"
                var12_col = "var_12m"
            else:
                # Already a monthly variation series
                df["var_monthly"] = df["value"]
                df["var_12m"] = df["value"].rolling(12).sum()  # approx
                var_col = "var_monthly"
                var12_col = "var_12m"

            # Drop NaN rows from differencing
            clean = df.dropna(subset=[var_col])

            category_data[cat_id] = {
                "id": cat_id,
                "name_es": meta["name_es"],
                "name_en": meta["name_en"],
                "color": meta["color"],
                "description_es": meta["description_es"],
                "weight_pct": meta["weight_pct"],
                "dates": format_dates(clean.index),
                "values_monthly": [round(v, 3) for v in clean[var_col].tolist()],
                "values_12m": [round(v, 3) for v in clean[var12_col].dropna().tolist()],
                "latest_monthly": round(float(clean[var_col].iloc[-1]), 3),
                "latest_12m": round(float(df.dropna(subset=[var12_col])[var12_col].iloc[-1]), 3),
                "latest_date": clean.index[-1].strftime("%Y-%m"),
                "n_obs": len(clean),
            }
            print(f"    [OK] {len(clean)} months, latest: {clean[var_col].iloc[-1]:.2f}% (m/m)")

        except Exception as e:
            print(f"    [ERR] {e}")

    # Build output JSON
    dates_all = category_data.get("total", {}).get("dates", [])
    output = {
        "metadata": {
            "last_update": max(
                [v["latest_date"] for v in category_data.values()],
                default="",
            ),
            "source": "BCRP / INEI",
            "frequency": "monthly",
            "unit_monthly": "% mensual",
            "unit_12m": "% interanual",
            "base": "Dic.2021 = 100 (para índices)",
            "coverage": "Lima Metropolitana",
            "n_categories": len(category_data),
        },
        "categories": list(category_data.values()),
    }

    # Save to exports
    export_dir = Path("exports/data")
    export_dir.mkdir(parents=True, exist_ok=True)
    output_file = export_dir / "inflation_categories.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Exported inflation categories:")
    print(f"     {output_file} ({output_file.stat().st_size / 1024:.1f} KB)")
    print(f"     {len(category_data)} categories")

    print("\nLatest month-on-month inflation by category:")
    for cat in output["categories"]:
        print(f"  {cat['name_es']:30} {cat['latest_monthly']:>+6.3f}% m/m  | "
              f"{cat['latest_12m']:>+6.2f}% 12m  ({cat['latest_date']})")

    return output


if __name__ == "__main__":
    export_inflation_categories()
