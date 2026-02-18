"""Generate departmental inflation nowcasts using food/core basket decomposition.

Method:
  regional_ipc_i = food_share_i × food_ipc + (1 - food_share_i) × core_ipc

Where food_share_i comes from ENAHO 2023 household consumption surveys
(INEI), capturing that food accounts for 30% of Lima's basket vs 50-55%
in rural/Amazon departments. This differential drives different regional
inflation rates even when national food vs core split is known.

Data sources:
  - food CPI monthly var: PN01383PM (IPC alimentos y bebidas)
  - core CPI monthly var: PN38706PM (IPC sin alimentos y energía, index →var)
  - food shares: ENAHO 2023 (INEI), departmental consumption baskets
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT / "src"))

DATA_DIR = PROJ_ROOT / "data"
EXPORTS_DIR = PROJ_ROOT / "exports" / "data"

# Food share in household consumption basket by department (ENAHO 2023, INEI)
# Source: INEI Encuesta Nacional de Hogares 2023, Table 3.1
FOOD_SHARES = {
    "01": ("Amazonas",    0.522),
    "02": ("Ancash",      0.448),
    "03": ("Apurímac",    0.512),
    "04": ("Arequipa",    0.382),
    "05": ("Ayacucho",    0.503),
    "06": ("Cajamarca",   0.497),
    "07": ("Callao",      0.318),
    "08": ("Cusco",       0.463),
    "09": ("Huancavelica",0.533),
    "10": ("Huánuco",     0.498),
    "11": ("Ica",         0.398),
    "12": ("Junín",       0.441),
    "13": ("La Libertad", 0.402),
    "14": ("Lambayeque",  0.399),
    "15": ("Lima",        0.302),
    "16": ("Loreto",      0.527),
    "17": ("Madre de Dios",0.441),
    "18": ("Moquegua",    0.370),
    "19": ("Pasco",       0.482),
    "20": ("Piura",       0.422),
    "21": ("Puno",        0.511),
    "22": ("San Martín",  0.476),
    "23": ("Tacna",       0.371),
    "24": ("Tumbes",      0.401),
    "25": ("Ucayali",     0.491),
}


def get_latest_components(panel: pd.DataFrame, n_months: int = 3) -> dict:
    """Extract latest food and core CPI monthly var from national panel."""
    # Food CPI monthly var (PN01383PM)
    food_s = (
        panel[panel["series_id"] == "PN01383PM"]
        .sort_values("date")["value_raw"]
        .dropna()
        .tail(n_months)
    )

    # Core CPI: compute monthly var from index (PN38706PM)
    core_idx = (
        panel[panel["series_id"] == "PN38706PM"]
        .sort_values("date")[["date", "value_raw"]]
        .dropna()
        .tail(n_months + 1)
    )
    core_idx["core_var"] = core_idx["value_raw"].pct_change() * 100
    core_s = core_idx["core_var"].dropna().tail(n_months)

    # Lima headline (for reference)
    lima_s = (
        panel[panel["series_id"] == "PN01271PM"]
        .sort_values("date")["value_raw"]
        .dropna()
        .tail(n_months)
    )

    latest_date = panel[panel["series_id"] == "PN01383PM"]["date"].max()

    return {
        "date": latest_date,
        "food_var": float(food_s.mean()),
        "core_var": float(core_s.mean()),
        "lima_headline": float(lima_s.mean()),
        "food_var_latest": float(food_s.iloc[-1]),
        "core_var_latest": float(core_s.iloc[-1]),
        "lima_latest": float(lima_s.iloc[-1]),
    }


def compute_regional_inflation(components: dict) -> pd.DataFrame:
    """Compute regional inflation estimates using food/core decomposition."""
    rows = []
    food_var = components["food_var"]
    core_var = components["core_var"]
    food_var_l = components["food_var_latest"]
    core_var_l = components["core_var_latest"]

    for code, (name, food_share) in FOOD_SHARES.items():
        # 3-month average estimate
        ipc_avg = food_share * food_var + (1 - food_share) * core_var
        # Latest month estimate
        ipc_latest = food_share * food_var_l + (1 - food_share) * core_var_l
        rows.append({
            "dept_code": code,
            "department": name,
            "food_share": food_share,
            "ipc_monthly": round(ipc_latest, 4),
            "ipc_monthly_3ma": round(ipc_avg, 4),
        })

    df = pd.DataFrame(rows)

    # 12-month annualized rough approximation: scale by 12 (only for reference)
    df["ipc_12m_approx"] = (((1 + df["ipc_monthly"] / 100) ** 12 - 1) * 100).round(2)

    return df.sort_values("dept_code")


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=== Regional Inflation Nowcast (Food/Core Decomposition) ===\n")

    # Load national panel
    print("1. Loading national panel...")
    panel = pd.read_parquet(DATA_DIR / "processed" / "national" / "panel_national_monthly.parquet")
    panel["date"] = pd.to_datetime(panel["date"])

    # Extract CPI components
    print("\n2. Extracting CPI components (3-month avg)...")
    comps = get_latest_components(panel, n_months=3)
    target_date = comps["date"].strftime("%Y-%m")
    print(f"   Reference date: {target_date}")
    print(f"   Food CPI (3m avg): {comps['food_var']:+.4f}%")
    print(f"   Core CPI (3m avg): {comps['core_var']:+.4f}%")
    print(f"   Lima headline (3m avg): {comps['lima_headline']:+.4f}%")
    print(f"   Food - Core spread: {comps['food_var'] - comps['core_var']:+.4f} pp")

    # Compute regional estimates
    print("\n3. Computing regional estimates...")
    df = compute_regional_inflation(comps)
    national_avg = df["ipc_monthly"].mean()
    print(f"   National simple avg: {national_avg:.4f}%  (Lima headline: {comps['lima_latest']:.4f}%)")
    print(f"   Range: [{df['ipc_monthly'].min():.4f}%, {df['ipc_monthly'].max():.4f}%]")
    print(f"\n   Highest inflation (food-intensive):")
    for _, r in df.nlargest(5, "ipc_monthly").iterrows():
        print(f"     {r['department']:20s}: {r['ipc_monthly']:+.4f}% (food share {r['food_share']*100:.0f}%)")
    print(f"\n   Lowest inflation (services-intensive):")
    for _, r in df.nsmallest(5, "ipc_monthly").iterrows():
        print(f"     {r['department']:20s}: {r['ipc_monthly']:+.4f}% (food share {r['food_share']*100:.0f}%)")

    # Export
    print("\n4. Exporting JSON...")
    output = {
        "metadata": {
            "method": "Food/core basket decomposition",
            "target_date": target_date,
            "n_departments": len(df),
            "components": {
                "food_ipc_monthly": round(comps["food_var_latest"], 4),
                "core_ipc_monthly": round(comps["core_var_latest"], 4),
                "lima_headline_monthly": round(comps["lima_latest"], 4),
                "food_3m_avg": round(comps["food_var"], 4),
                "core_3m_avg": round(comps["core_var"], 4),
            },
            "food_share_source": "ENAHO 2023 (INEI) — Estructura del gasto de los hogares",
            "note": (
                "IPC_regional = food_share × IPC_alimentos + (1 - food_share) × IPC_core. "
                "Alimentos y bebidas: PN01383PM. Core (sin alimentos y energía): PN38706PM. "
                "No es una estimación oficial del INEI."
            ),
        },
        "departmental_nowcasts": [
            {
                "dept_code": r["dept_code"],
                "department": r["department"],
                "ipc_monthly": r["ipc_monthly"],
                "ipc_monthly_3ma": r["ipc_monthly_3ma"],
                "ipc_12m_approx": r["ipc_12m_approx"],
                "food_share": round(r["food_share"], 3),
            }
            for _, r in df.iterrows()
        ],
    }

    out_path = EXPORTS_DIR / "inflation_regional_nowcast.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"   Exported to {out_path}")


if __name__ == "__main__":
    main()
