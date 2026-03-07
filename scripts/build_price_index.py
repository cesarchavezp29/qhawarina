"""Build daily Jevons price index from supermarket snapshots.

Methodology (Billion Prices Project / Cavallo & Rigobon):
  1. Match products across consecutive day pairs by (store, sku_id)
  2. Compute bilateral Jevons index: geometric mean of (p_t / p_{t-1})
     filtered to 0.5 < ratio < 2.0 to exclude promotions/errors
  3. Chain-link daily indices: Index_t = Index_{t-1} * Jevons_t
  4. Disaggregate by CPI category (food, non-food, etc.)
  5. Aggregate with CPI expenditure weights (weighted Jevons)
  6. Export JSON for website consumption

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


def _weighted_index(cat_indexes: dict) -> float:
    """
    Compute CPI-weighted geometric mean of category-level indexes.

    Uses expenditure weights from CATEGORY_MAP (INEI IPC basket).
    Returns 100.0 if no valid categories available.
    """
    valid = [
        (meta["cpi_weight"], cat_indexes[cat])
        for cat, meta in CATEGORY_MAP.items()
        if cat != "other" and meta["cpi_weight"] > 0 and cat_indexes.get(cat, 0) > 0
    ]
    if not valid:
        return 100.0
    total_w = sum(w for w, _ in valid)
    log_sum = sum(w * math.log(v) for w, v in valid)
    return math.exp(log_sum / total_w)


def build_chain_index(snapshots: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build chain-linked daily price index from all snapshots.

    Returns DataFrame with columns:
        date, index_all (CPI-weighted), index_all_unweighted, index_food,
        index_nonfood, <per-category indexes>, n_matched, jevons_ratio_all
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
    index_all_unweighted = 100.0
    index_food = 100.0
    index_nonfood = 100.0
    cat_indexes = {cat: 100.0 for cat in CATEGORY_MAP}

    rows.append({
        "date": dates[0],
        "index_all": 100.0,
        "index_all_unweighted": 100.0,
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

        # Unweighted overall index (reference / backward compat)
        r_all = jevons_ratio(df0, df1)
        if r_all:
            index_all_unweighted = index_all_unweighted * r_all

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

        # Weighted aggregate: CPI expenditure-weighted geometric mean of category indexes
        index_all_weighted = _weighted_index(cat_indexes)

        # Count matched products
        merged = df0[["store", "sku_id"]].merge(
            df1[["store", "sku_id"]], on=["store", "sku_id"]
        )
        n_matched = len(merged)
        print(f"    Matched: {n_matched}, Weighted idx: {index_all_weighted:.4f}, "
              f"Unweighted: {index_all_unweighted:.4f}")

        rows.append({
            "date": d1,
            "index_all": round(index_all_weighted, 4),
            "index_all_unweighted": round(index_all_unweighted, 4),
            "index_food": round(index_food, 4),
            "index_nonfood": round(index_nonfood, 4),
            **cat_ratios,
            "n_matched": n_matched,
            "jevons_ratio_all": round(r_all, 6) if r_all else None,
        })

    return pd.DataFrame(rows)


def compute_variations(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily, cumulative, and rolling inflation variation columns."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    for col in ["index_all", "index_food", "index_nonfood"]:
        base_col = col.replace("index_", "")
        df[f"var_{base_col}"] = (df[col] / df[col].shift(1) - 1) * 100
        df[f"cum_{base_col}_pct"] = (df[col] / 100 - 1) * 100  # vs day 1

    # 30-day rolling average for smoother inflation estimates (Cavallo & Rigobon 2016)
    df["rolling30_all"] = df["index_all"].rolling(30, min_periods=5).mean()

    # YoY: (30d rolling avg today) / (30d rolling avg 365 days ago) - 1
    df["yoy_inflation"] = (df["rolling30_all"] / df["rolling30_all"].shift(365) - 1) * 100
    # MoM: (30d rolling avg today) / (30d rolling avg 30 days ago) - 1
    df["mom_inflation"] = (df["rolling30_all"] / df["rolling30_all"].shift(30) - 1) * 100

    return df


def _compute_top_movers(records: list, n: int = 4) -> list:
    """Return top N CPI categories by absolute daily price change."""
    if len(records) < 2:
        return []
    prev, curr = records[-2], records[-1]
    movers = []
    for cat, meta in CATEGORY_MAP.items():
        if cat == "other":
            continue
        v0 = prev.get(f"index_{cat}")
        v1 = curr.get(f"index_{cat}")
        if not v0 or not v1 or v0 == 0:
            continue
        var = round((v1 / v0 - 1) * 100, 2)
        movers.append({
            "category": cat,
            "label_es": meta["label_es"],
            "label_en": meta["label_en"],
            "var": var,
        })
    movers.sort(key=lambda x: abs(x["var"]), reverse=True)
    return movers[:n]


def _compute_product_movers(df_t0: pd.DataFrame | None,
                            df_t1: pd.DataFrame | None, n: int = 4) -> list:
    """Return top N individual products by absolute daily price change."""
    if df_t0 is None or df_t1 is None:
        return []

    for df in [df_t0, df_t1]:
        if "cpi_cat" not in df.columns:
            df["cpi_cat"] = df.apply(map_category, axis=1)

    keep_t0 = [c for c in ["store", "sku_id", "price", "product_name", "cpi_cat"] if c in df_t0.columns]
    keep_t1 = [c for c in ["store", "sku_id", "price", "product_name"] if c in df_t1.columns]

    merged = df_t0[keep_t0].merge(
        df_t1[keep_t1], on=["store", "sku_id"], suffixes=("_t0", "_t1"),
    )
    if merged.empty:
        return []

    merged["ratio"] = merged["price_t1"] / merged["price_t0"]
    merged = merged[(merged["ratio"] >= 0.5) & (merged["ratio"] <= 2.0)]
    merged["var_pct"] = (merged["ratio"] - 1) * 100
    merged = merged.reindex(merged["var_pct"].abs().sort_values(ascending=False).index).head(n)

    result = []
    for _, row in merged.iterrows():
        name = str(row.get("product_name_t0", row.get("product_name", "")))
        if len(name) > 40:
            name = name[:37] + "..."
        result.append({
            "name": name,
            "store": str(row.get("store", "")),
            "category": str(row.get("cpi_cat", "")),
            "var": round(float(row["var_pct"]), 2),
        })
    return result


def _compute_coverage(df_latest: pd.DataFrame | None) -> tuple[dict, dict]:
    """
    Compute stores_coverage and categories_coverage from latest snapshot.
    Returns (stores_coverage, categories_coverage).
    """
    if df_latest is None or df_latest.empty:
        return {}, {}

    df = df_latest.copy()
    if "cpi_cat" not in df.columns:
        df["cpi_cat"] = df.apply(map_category, axis=1)

    stores_coverage = {}
    if "store" in df.columns:
        for store, grp in df.groupby("store"):
            stores_coverage[str(store)] = int(len(grp))

    categories_coverage = {}
    for cat, grp in df.groupby("cpi_cat"):
        if cat != "other":
            categories_coverage[str(cat)] = int(len(grp))

    return stores_coverage, categories_coverage


def export_json(df: pd.DataFrame, output_path: Path,
                df_prev: pd.DataFrame | None = None,
                df_latest: pd.DataFrame | None = None) -> None:
    """Export index to JSON for website consumption."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def safe(v):
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(v, 4)

    records = []
    for _, row in df.iterrows():
        record = {
            "date": row["date"].strftime("%Y-%m-%d"),
            "index_all": row["index_all"],
            "index_all_unweighted": row.get("index_all_unweighted"),
            "index_food": row["index_food"],
            "index_nonfood": row["index_nonfood"],
            "var_all": safe(row.get("var_all")),
            "var_food": safe(row.get("var_food")),
            "cum_pct": round(row.get("cum_all_pct", 0) or 0, 4),
            "yoy_inflation": safe(row.get("yoy_inflation")),
            "mom_inflation": safe(row.get("mom_inflation")),
            "interpolated": bool(row.get("interpolated", False)),
            "n_matched": int(row.get("n_matched", 0) or 0),
        }
        for cat in CATEGORY_MAP:
            record[f"index_{cat}"] = row.get(f"index_{cat}", 100.0)
        records.append(record)

    latest = records[-1] if records else {}
    first = records[0] if records else {}

    stores_coverage, categories_coverage = _compute_coverage(df_latest)
    product_movers = _compute_product_movers(df_prev, df_latest)

    output = {
        "metadata": {
            "methodology": "Weighted Jevons bilateral price index (CPI expenditure weights), chain-linked daily",
            "methodology_note": "index_all uses INEI IPC expenditure weights via weighted geometric mean of category-level Jevons ratios. index_all_unweighted is the simple geometric mean across all matched products.",
            "base_date": first.get("date", ""),
            "base_value": 100,
            "last_date": latest.get("date", ""),
            "n_days": len(records),
            "stores": ["Plaza Vea", "Metro", "Wong"],
            "n_products_approx": 42000,
            "ratio_filter": "0.5 < ratio < 2.0",
            "inflation_window": "30-day rolling average (yoy_inflation, mom_inflation)",
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
            "yoy_inflation": latest.get("yoy_inflation"),
            "mom_inflation": latest.get("mom_inflation"),
            "n_products_today": latest.get("n_matched", 0),
            "top_movers": _compute_top_movers(records),
            "product_movers": product_movers,
            "stores_coverage": stores_coverage,
            "categories_coverage": categories_coverage,
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

    # Add variation columns (daily, cumulative, YoY, MoM)
    df = compute_variations(df)

    # ── Fill missing days with linear interpolation ───────────────────────────
    # Note: Supermarkets operate 7 days/week, so use date_range (not bdate_range)
    df["date"] = pd.to_datetime(df["date"])
    all_days = pd.date_range(df["date"].min(), df["date"].max(), freq='D')
    df_full = df.set_index("date").reindex(all_days)
    n_missing = df_full["index_all"].isna().sum()
    if n_missing > 0:
        index_cols = [c for c in df_full.columns if c.startswith("index_") or c in ("n_matched", "jevons_ratio_all")]
        df_full[index_cols] = df_full[index_cols].interpolate(method="linear")
        df_full["interpolated"] = False
        base_date = df_full.index[0]
        df_full.loc[
            (df_full.index != base_date) & (df_full["n_matched"].isna() | (df_full["n_matched"] == 0)),
            "interpolated"
        ] = True
        df_full = df_full.reset_index().rename(columns={"index": "date"})
        df_full = compute_variations(df_full)
        print(f"[OK] Interpolated {n_missing} missing day(s): "
              f"{[d.strftime('%Y-%m-%d') for d in all_days if d not in df['date'].values]}")
    else:
        df_full["interpolated"] = False
        df_full = df_full.reset_index().rename(columns={"index": "date"})
    df = df_full
    # ─────────────────────────────────────────────────────────────────────────

    print(f"\n[OK] Index computed for {len(df)} days")
    print(df[["date", "index_all", "index_food", "index_nonfood",
               "var_all", "yoy_inflation", "mom_inflation"]].to_string())

    # Save parquet
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\n[OK] Saved parquet: {OUTPUT_PARQUET}")

    # Get latest two raw snapshots for product movers + coverage
    sorted_dates = sorted(snapshots.keys())
    df_latest_snap = snapshots[sorted_dates[-1]] if sorted_dates else None
    df_prev_snap = snapshots[sorted_dates[-2]] if len(sorted_dates) >= 2 else None

    # Export JSON for website
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    export_json(df, OUTPUT_JSON, df_prev=df_prev_snap, df_latest=df_latest_snap)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
