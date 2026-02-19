"""Build daily Jevons price index from supermarket snapshots.

Methodology (Billion Prices Project / Cavallo & Rigobon):
  1. Match products across consecutive day pairs by (store, sku_id)
  2. Compute bilateral Jevons index: geometric mean of (p_t / p_{t-1})
     filtered to 0.5 < ratio < 2.0 to exclude promotions/errors
  3. Chain-link daily indices: Index_t = Index_{t-1} * Jevons_t
  4. Disaggregate by CPI category (food, non-food, etc.)
  5. Export JSON for website consumption

Output: exports/data/daily_price_index.json
         data/processed/national/daily_price_index.parquet
"""

import json
import math
import sys
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SNAPSHOT_DIR = PROJECT_ROOT / "data" / "raw" / "supermarket" / "snapshots"
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "national" / "daily_price_index.parquet"
OUTPUT_JSON = PROJECT_ROOT / "exports" / "data" / "daily_price_index.json"

# CPI-aligned category mapping (same as supermarket.py)
CATEGORY_MAP = {
    "arroz_cereales":    {"label_es": "Arroz y Cereales",    "label_en": "Rice & Cereals",    "color": "#f97316", "cpi_weight": 4.2},
    "pan_harinas":       {"label_es": "Pan y Harinas",       "label_en": "Bread & Flour",     "color": "#eab308", "cpi_weight": 3.1},
    "carnes":            {"label_es": "Carnes",              "label_en": "Meat",              "color": "#ef4444", "cpi_weight": 5.8},
    "pescados_mariscos": {"label_es": "Pescados y Mariscos", "label_en": "Fish & Seafood",    "color": "#3b82f6", "cpi_weight": 1.9},
    "lacteos":           {"label_es": "Lácteos y Huevos",   "label_en": "Dairy & Eggs",      "color": "#a3e635", "cpi_weight": 3.4},
    "frutas":            {"label_es": "Frutas",              "label_en": "Fruits",            "color": "#22c55e", "cpi_weight": 2.1},
    "verduras":          {"label_es": "Verduras",            "label_en": "Vegetables",        "color": "#16a34a", "cpi_weight": 2.3},
    "aceites_grasas":    {"label_es": "Aceites y Grasas",   "label_en": "Oils & Fats",       "color": "#84cc16", "cpi_weight": 1.2},
    "azucar_dulces":     {"label_es": "Azúcar y Dulces",    "label_en": "Sugar & Sweets",    "color": "#f472b6", "cpi_weight": 1.1},
    "bebidas":           {"label_es": "Bebidas",             "label_en": "Beverages",         "color": "#06b6d4", "cpi_weight": 1.8},
    "limpieza":          {"label_es": "Limpieza",            "label_en": "Cleaning",          "color": "#8b5cf6", "cpi_weight": 2.4},
    "cuidado_personal":  {"label_es": "Cuidado Personal",   "label_en": "Personal Care",     "color": "#ec4899", "cpi_weight": 2.9},
    "other":             {"label_es": "Otros",               "label_en": "Other",             "color": "#6b7280", "cpi_weight": 0.0},
}

FOOD_CATEGORIES = {
    "arroz_cereales", "pan_harinas", "carnes", "pescados_mariscos",
    "lacteos", "frutas", "verduras", "aceites_grasas", "azucar_dulces", "bebidas",
}


def load_snapshots() -> dict[str, pd.DataFrame]:
    """Load all available daily snapshots sorted by date."""
    snapshots = {}
    for f in sorted(SNAPSHOT_DIR.glob("*.parquet")):
        day = f.stem  # "2026-02-10"
        df = pd.read_parquet(f)
        # Keep only available products with valid prices
        df = df[df["available"] == True].copy()
        df = df[df["price"] > 0].copy()
        snapshots[day] = df
    return snapshots


def map_category(row: pd.Series) -> str:
    """Map raw category columns to CPI category key."""
    cat1 = str(row.get("category_l1", "")).lower()
    cat2 = str(row.get("category_l2", "")).lower()
    cat3 = str(row.get("category_l3", "")).lower()
    name = str(row.get("product_name", "")).lower()
    combined = f"{cat1} {cat2} {cat3} {name}"

    if any(w in combined for w in ["arroz", "cereal", "avena", "quinua", "kiwicha"]):
        return "arroz_cereales"
    if any(w in combined for w in ["pan", "harina", "galleta", "fideos", "pasta", "maca"]):
        return "pan_harinas"
    if any(w in combined for w in ["carne", "pollo", "cerdo", "res", "pavo", "chorizo", "jamon", "salchicha", "embutido"]):
        return "carnes"
    if any(w in combined for w in ["pescado", "mariscos", "atun", "salmon", "sardina", "anchoveta", "camaron"]):
        return "pescados_mariscos"
    if any(w in combined for w in ["leche", "yogurt", "queso", "mantequilla", "crema", "lacteo", "huevo"]):
        return "lacteos"
    if any(w in combined for w in ["fruta", "manzana", "platano", "naranja", "uva", "pera", "durazno", "fresa", "mango"]):
        return "frutas"
    if any(w in combined for w in ["verdura", "vegetal", "zanahoria", "papa", "cebolla", "tomate", "lechuga", "brocoli", "espinaca"]):
        return "verduras"
    if any(w in combined for w in ["aceite", "manteca", "margarina", "grasa"]):
        return "aceites_grasas"
    if any(w in combined for w in ["azucar", "miel", "chocolate", "caramelo", "mermelada", "dulce"]):
        return "azucar_dulces"
    if any(w in combined for w in ["gaseosa", "jugo", "agua", "bebida", "cerveza", "vino", "néctar", "nectar"]):
        return "bebidas"
    if any(w in combined for w in ["limpieza", "detergente", "jabon", "jabón", "lejia", "desinfectante", "suavizante"]):
        return "limpieza"
    if any(w in combined for w in ["shampoo", "cuidado personal", "desodorante", "crema corporal", "pasta dental", "afeitadora"]):
        return "cuidado_personal"
    return "other"


def jevons_ratio(df_t0: pd.DataFrame, df_t1: pd.DataFrame,
                 category: str | None = None) -> float | None:
    """
    Bilateral Jevons price index between two snapshots.

    Matches products by (store, sku_id), filters extreme ratios,
    returns geometric mean of price ratios.
    """
    # Add category if not present
    for df in [df_t0, df_t1]:
        if "cpi_cat" not in df.columns:
            df["cpi_cat"] = df.apply(map_category, axis=1)

    if category:
        df_t0 = df_t0[df_t0["cpi_cat"] == category]
        df_t1 = df_t1[df_t1["cpi_cat"] == category]

    # Merge on (store, sku_id) — same product, same store
    merged = df_t0[["store", "sku_id", "price"]].merge(
        df_t1[["store", "sku_id", "price"]],
        on=["store", "sku_id"],
        suffixes=("_t0", "_t1"),
    )

    if len(merged) < 10:
        return None

    # Compute price ratios
    merged["ratio"] = merged["price_t1"] / merged["price_t0"]

    # Filter extreme ratios (promotions, data errors)
    merged = merged[(merged["ratio"] >= 0.5) & (merged["ratio"] <= 2.0)]

    if len(merged) < 10:
        return None

    # Jevons: geometric mean of ratios
    return float(np.exp(np.log(merged["ratio"]).mean()))


def build_chain_index(snapshots: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build chain-linked daily price index from all snapshots.

    Returns DataFrame with columns:
        date, index_all, index_food, index_nonfood,
        <per-category indexes>, n_matched, jevons_ratio
    """
    dates = sorted(snapshots.keys())
    if len(dates) < 2:
        print(f"[WARN] Only {len(dates)} snapshot(s) — need at least 2 to compute index")
        return pd.DataFrame()

    # Categorize all snapshots once (speed up)
    print("Categorizing products...")
    for day, df in snapshots.items():
        if "cpi_cat" not in df.columns:
            snapshots[day]["cpi_cat"] = df.apply(map_category, axis=1)
        print(f"  {day}: {len(df)} products")

    # Initialize index at 100 on first date
    rows = []
    index_all = 100.0
    index_food = 100.0
    index_nonfood = 100.0
    cat_indexes = {cat: 100.0 for cat in CATEGORY_MAP}

    rows.append({
        "date": dates[0],
        "index_all": index_all,
        "index_food": index_food,
        "index_nonfood": index_nonfood,
        **{f"index_{cat}": 100.0 for cat in CATEGORY_MAP},
        "n_matched": 0,
        "jevons_ratio_all": None,
    })

    # Chain-link across consecutive day pairs
    for i in range(len(dates) - 1):
        d0, d1 = dates[i], dates[i + 1]
        df0, df1 = snapshots[d0], snapshots[d1]

        print(f"  {d0} -> {d1}: computing Jevons...")

        # Overall index
        r_all = jevons_ratio(df0, df1)
        if r_all:
            index_all = index_all * r_all

        # Food vs non-food
        food_cats = list(FOOD_CATEGORIES)
        df0_food = df0[df0["cpi_cat"].isin(food_cats)]
        df1_food = df1[df1["cpi_cat"].isin(food_cats)]
        r_food = jevons_ratio(df0_food, df1_food)
        if r_food:
            index_food = index_food * r_food

        df0_nonfood = df0[~df0["cpi_cat"].isin(food_cats)]
        df1_nonfood = df1[~df1["cpi_cat"].isin(food_cats)]
        r_nonfood = jevons_ratio(df0_nonfood, df1_nonfood)
        if r_nonfood:
            index_nonfood = index_nonfood * r_nonfood

        # Per-category indexes
        cat_ratios = {}
        for cat in CATEGORY_MAP:
            r = jevons_ratio(df0, df1, category=cat)
            if r:
                cat_indexes[cat] = cat_indexes[cat] * r
            cat_ratios[f"index_{cat}"] = round(cat_indexes[cat], 4)

        # Count matched products
        merged = df0[["store", "sku_id"]].merge(
            df1[["store", "sku_id"]], on=["store", "sku_id"]
        )
        n_matched = len(merged)
        print(f"    Matched: {n_matched}, All ratio: {r_all:.5f}" if r_all else
              f"    Matched: {n_matched}, All ratio: None")

        rows.append({
            "date": d1,
            "index_all": round(index_all, 4),
            "index_food": round(index_food, 4),
            "index_nonfood": round(index_nonfood, 4),
            **cat_ratios,
            "n_matched": n_matched,
            "jevons_ratio_all": round(r_all, 6) if r_all else None,
        })

    return pd.DataFrame(rows)


def compute_variations(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily and cumulative variation columns."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    for col in ["index_all", "index_food", "index_nonfood"]:
        base_col = col.replace("index_", "")
        df[f"var_{base_col}"] = (df[col] / df[col].shift(1) - 1) * 100
        df[f"cum_{base_col}_pct"] = (df[col] / 100 - 1) * 100  # vs day 1

    return df


def export_json(df: pd.DataFrame, output_path: Path) -> None:
    """Export index to JSON for website consumption."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for _, row in df.iterrows():
        record = {
            "date": row["date"].strftime("%Y-%m-%d"),
            "index_all": row["index_all"],
            "index_food": row["index_food"],
            "index_nonfood": row["index_nonfood"],
            "var_all": None if (v := row.get("var_all")) is None or (isinstance(v, float) and math.isnan(v)) else round(v, 4),
            "var_food": None if (v := row.get("var_food")) is None or (isinstance(v, float) and math.isnan(v)) else round(v, 4),
            "cum_pct": round(row.get("cum_all_pct", 0) or 0, 4),
            "interpolated": bool(row.get("interpolated", False)),
        }
        # Add per-category indexes
        for cat in CATEGORY_MAP:
            record[f"index_{cat}"] = row.get(f"index_{cat}", 100.0)
        records.append(record)

    # Compute summary stats
    latest = records[-1] if records else {}
    first = records[0] if records else {}

    output = {
        "metadata": {
            "methodology": "Jevons bilateral price index, chain-linked daily",
            "base_date": first.get("date", ""),
            "base_value": 100,
            "last_date": latest.get("date", ""),
            "n_days": len(records),
            "stores": ["Plaza Vea", "Metro", "Wong"],
            "n_products_approx": 42000,
            "ratio_filter": "0.5 < ratio < 2.0",
            "reference": "Cavallo & Rigobon (2016), Billion Prices Project, MIT",
            "updated": date.today().isoformat(),
        },
        "categories": {
            cat: {
                "label_es": meta["label_es"],
                "label_en": meta["label_en"],
                "color": meta["color"],
                "cpi_weight": meta["cpi_weight"],
            }
            for cat, meta in CATEGORY_MAP.items()
            if cat != "other"
        },
        "series": records,
        "latest": {
            "date": latest.get("date"),
            "index_all": latest.get("index_all"),
            "index_food": latest.get("index_food"),
            "cum_pct": latest.get("cum_pct"),
            "var_all": latest.get("var_all"),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[OK] Exported {len(records)} days to {output_path}")
    size_kb = output_path.stat().st_size / 1024
    print(f"     Size: {size_kb:.1f} KB")


def main():
    print("=" * 60)
    print("QHAWARINA — Daily Price Index (Cavallo/BPP Methodology)")
    print("=" * 60)

    # Load snapshots
    print("\nLoading snapshots...")
    snapshots = load_snapshots()
    print(f"Found {len(snapshots)} snapshots: {list(snapshots.keys())}")

    if len(snapshots) < 2:
        print("[ERROR] Need at least 2 snapshots to compute index")
        return 1

    # Build chain index
    print("\nBuilding chain-linked Jevons index...")
    df = build_chain_index(snapshots)

    if df.empty:
        print("[ERROR] Index computation failed")
        return 1

    # Add variation columns
    df = compute_variations(df)

    # ── Fill missing business days with linear interpolation ─────────────────
    df["date"] = pd.to_datetime(df["date"])
    all_bdays = pd.bdate_range(df["date"].min(), df["date"].max())
    df_full = df.set_index("date").reindex(all_bdays)
    n_missing = df_full["index_all"].isna().sum()
    if n_missing > 0:
        index_cols = [c for c in df_full.columns if c.startswith("index_") or c in ("n_matched", "jevons_ratio_all")]
        df_full[index_cols] = df_full[index_cols].interpolate(method="linear")
        df_full["interpolated"] = False
        # Mark as interpolated: missing n_matched OR zero BUT not the base date (first row has n_matched=0 by design)
        base_date = df_full.index[0]
        df_full.loc[
            (df_full.index != base_date) & (df_full["n_matched"].isna() | (df_full["n_matched"] == 0)),
            "interpolated"
        ] = True
        df_full = df_full.reset_index().rename(columns={"index": "date"})
        df_full = compute_variations(df_full)
        print(f"[OK] Interpolated {n_missing} missing business day(s): "
              f"{[d.strftime('%Y-%m-%d') for d in all_bdays if d not in df['date'].values]}")
    else:
        df_full["interpolated"] = False
        df_full = df_full.reset_index().rename(columns={"index": "date"})
    df = df_full
    # ─────────────────────────────────────────────────────────────────────────

    print(f"\n[OK] Index computed for {len(df)} days")
    print(df[["date", "index_all", "index_food", "index_nonfood", "var_all"]].to_string())

    # Save parquet
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\n[OK] Saved parquet: {OUTPUT_PARQUET}")

    # Export JSON for website
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    export_json(df, OUTPUT_JSON)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
