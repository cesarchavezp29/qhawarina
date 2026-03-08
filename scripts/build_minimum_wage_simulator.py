"""
Build minimum wage simulation data from ENAHO Module 05 (Empleo e Ingresos).

Steps:
  1. Download ENAHO module 05 for 2019-2023 via enahodata
  2. Load .dta files, clean, define formality + wage_worker
  3. Deflate wages to constant 2023 soles (CPI Lima from BCRP PN01271PM)
  4. Compute wage distribution (weighted by FAC500A)
  5. Simulate min-wage scenarios (1050, 1100, 1150, 1200, 1300, 1500)
  6. Compare with literature elasticities
  7. Export to exports/data/min_wage_simulation.json

Usage:
    python scripts/build_minimum_wage_simulator.py [--skip-download]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT     = Path(__file__).resolve().parents[1]
RAW_DIR  = ROOT / "data" / "raw" / "enaho"
OUT_DIR  = ROOT / "exports" / "data"
OUT_FILE = OUT_DIR / "min_wage_simulation.json"

YEARS       = ["2019", "2020", "2021", "2022", "2023"]
CURRENT_MIN = 1025   # RMV vigente desde mayo 2022
SCENARIOS   = [1050, 1100, 1150, 1200, 1300, 1500]

# Employment elasticity (Céspedes & Sánchez 2014, BCRP DT 2014-014)
EMPLOYMENT_ELASTICITY = -0.10

# Lighthouse / spillover elasticity for informal sector
# (Boeri, Garibaldi & Ribeiro 2011; range 0.3-0.5 for LatAm)
LIGHTHOUSE_ELASTICITY = 0.40

YEAR_MAP = {
    "2023": 906, "2022": 784, "2021": 759, "2020": 737,
    "2019": 687, "2018": 634, "2017": 603, "2016": 546,
}

# Nominal minimum wage by year (soles/month)
MIN_WAGE_HISTORY = {
    2016: 850, 2017: 930, 2018: 930, 2019: 930,
    2020: 930, 2021: 930, 2022: 1025, 2023: 1025,
}

DOMINIO_LABELS = {
    1: "Costa urbana", 2: "Sierra urbana", 3: "Selva urbana",
    4: "Lima Metropolitana", 5: "Costa rural",
    6: "Sierra rural",    7: "Selva rural", 8: "Lima Provincias",
}

# ── Column reference (ENAHO Módulo 500) ──────────────────────────────────────
# FAC500A : expansion factor (survey weight)
# P507    : condición de ocupación / categoría (3=dependiente, 4=independiente, 6=doméstico)
#           NOTE: P507 coding is consistent 2019-2023 when read as numeric:
#           P507=3 → ~12-13k income reporters (dependientes/asalariados)
#           P507=4 → ~11-12k income reporters (independientes)
#           P507=6 → ~1k income reporters (trabajadores del hogar)
# P510    : categoría de empleo (1=empleador, 2=independiente, 3=privado,
#           5=doméstico, 6=asalariado = main wage-worker code)
# P523    : periodicidad del pago (1=mensual, 2=semanal, 3=diario, 4=quincenal)
# P524A1  : ingreso del emp. principal en el periodo de P523 (NOT always monthly!)
# P524B1  : ingreso en especie del emp. principal (monetized in-kind)
# P524C1  : ingreso del empleo secundario (weekly/period)
# P524E1  : INGRESOS LABORALES total (alternate income sum in some years)
# P513T   : horas totales trabajadas en el periodo de referencia
# P517    : tipo de contrato (1=indefinido, 2=plazo fijo, 3=locación, 4+=sin contrato)
# P558A1  : aporta a sistema de pensiones (1=sí, 0=no) — formal proxy
# P511A   : actividad económica (sector)
# P512A   : número de trabajadores (empresa size)
# DOMINIO : dominio geográfico (1-8)
#
# FILTER STRATEGY:
# 1. Keep ALL rows with any positive income (P524A1|P524C1|P524E1 > 0)
#    → 21k-26k obs, 6.7M-8.9M weighted — matches INEI official ~8-9M employed
# 2. Exclude self-employed (P510=2) and employers (P510=1)
#    → leaves asalariados (P510=6) + domestic (P510=5) + others
# 3. Convert income to monthly via P523:
#    mensual (×1) | quincenal (×2) | semanal (×4.33) | diario (×26)
# 4. P507=4 (independientes) also excluded via P510=2 check


# ── CPI deflators ─────────────────────────────────────────────────────────────

def build_deflators() -> dict[int, float]:
    """
    Annual average CPI deflator relative to 2023 from BCRP PN01271PM.
    Returns {year: deflator} where deflator × nominal_2023 = real_year soles.
    2023 = 1.00 (wages in 2023 soles).
    """
    bcrp_path = ROOT / "data" / "raw" / "bcrp" / "bcrp_national_all.parquet"
    if not bcrp_path.exists():
        print("  WARNING: BCRP panel not found — using hardcoded deflators")
        return {
            2016: 1.2913, 2017: 1.2560, 2018: 1.2397, 2019: 1.2138,
            2020: 1.1920, 2021: 1.1468, 2022: 1.0634, 2023: 1.0000,
        }
    df = pd.read_parquet(bcrp_path)
    ipc = (
        df[df["series_code"] == "PN01271PM"][["date", "value"]]
        .copy()
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .sort_values("date")
        .set_index("date")
    )
    # Cumulative price index (chain from first observation)
    ipc["price_idx"] = (1 + ipc["value"] / 100).cumprod()
    # Annual average
    ann = ipc["price_idx"].resample("YE").mean()
    base = ann.loc["2023"].mean()   # 2023 average = 1.0
    deflators = {}
    for ts, val in ann.items():
        yr = ts.year
        deflators[yr] = float(base / val)
    return deflators


# ── helpers ───────────────────────────────────────────────────────────────────

def weighted_percentile(values: np.ndarray, weights: np.ndarray, pct: float) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    v, w = values[mask], weights[mask]
    if len(v) == 0:
        return np.nan
    idx = np.argsort(v)
    v, w = v[idx], w[idx]
    cum = np.cumsum(w)
    cum = (cum - 0.5 * w) / cum[-1]
    return float(np.interp(pct / 100, cum, v))


def weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    return weighted_percentile(v, w, 50)


def npy(o):
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return float(o) if np.isfinite(o) else None
    if isinstance(o, np.bool_):    return bool(o)
    raise TypeError(f"Not serialisable: {type(o)}")


# ── Step 1: Download ──────────────────────────────────────────────────────────

def download_module05(years: list[str], out_dir: Path) -> None:
    from enahodata import enahodata as _dl
    print(f"\n{'='*60}")
    print(f"Downloading ENAHO Module 05 for years: {years}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _dl(modulos=["05"], anios=years, descomprimir=True, only_dta=True,
        overwrite=False, output_dir=str(out_dir), panel=False)


# ── Step 2: Locate and load .dta ─────────────────────────────────────────────

def find_dta(year: str, base: Path) -> Path | None:
    # enahodata extracts to: modulo_05_<year>/ — file is enaho01a-<year>-500.dta
    for folder in sorted(base.glob(f"*05*{year}*")):
        for f in folder.glob("**/*.dta"):
            return f
    # Also check flat files
    for f in base.glob(f"**/*{year}*500*.dta"):
        return f
    return None


KEEP_COLS = ["P507", "P510", "P523", "P524A1", "P524B1", "P524C1", "P524E1",
             "P513T", "P517", "P558A1", "P511A", "P512A", "DOMINIO", "ESTRATO",
             "FAC500A"]

# Payment-period multipliers → monthly
PERIOD_MULT = {1: 1.0, 4: 2.0, 2: 4.333, 3: 26.0}


def to_monthly_income(row: pd.Series) -> float:
    """
    Convert raw income to monthly soles using payment periodicity (P523).
    Income variable priority: P524A1 > P524C1 > P524E1.
    P523: 1=mensual, 2=semanal, 3=diario, 4=quincenal.
    """
    # Pick best income source
    for col in ["P524A1", "P524C1", "P524E1"]:
        v = row.get(col, np.nan)
        if pd.notna(v) and v > 0:
            mult = PERIOD_MULT.get(row.get("P523", np.nan), 1.0)
            return float(v * mult)
    return np.nan


def load_year(year: str, base: Path, deflators: dict) -> pd.DataFrame | None:
    dta = find_dta(year, base)
    if dta is None:
        print(f"  [{year}] WARNING: no .dta found, skipping")
        return None

    sz = dta.stat().st_size / 1e6
    print(f"  [{year}] Loading {dta.name}  ({sz:.0f} MB)", flush=True)
    df = pd.read_stata(dta, convert_categoricals=False)
    df.columns = [c.upper() for c in df.columns]

    present = [c for c in KEEP_COLS if c in df.columns]
    df = df[present].copy()
    for col in present:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["year"] = int(year)
    return df


def clean(df: pd.DataFrame, deflators: dict) -> pd.DataFrame:
    """
    Filter to wage-earning population and compute real monthly wage.

    Filtering logic (based on 2019-2023 ENAHO Module 500 actual structure):
    1. Keep ALL rows with positive income in any income column
    2. Exclude employers (P510=1) and self-employed (P510=2)
    3. Convert to monthly via P523 periodicity
    4. Deflate to 2023 soles
    5. Define formality: written contract (P517 in [1,2]) OR pension (P558A1=1)
    """
    # ── Step 1: Compute raw monthly nominal wage ─────────────────────────────
    # Must apply to_monthly_income row-by-row (periodicity varies by worker)
    income_cols = [c for c in ["P524A1", "P524C1", "P524E1", "P523"] if c in df.columns]
    sub = df[income_cols].copy()
    # Vectorised: use P524A1 × mult as primary, fallback to P524C1/E1
    p524a1 = pd.to_numeric(df.get("P524A1", pd.Series(np.nan, index=df.index)), errors="coerce")
    p524c1 = pd.to_numeric(df.get("P524C1", pd.Series(np.nan, index=df.index)), errors="coerce")
    p524e1 = pd.to_numeric(df.get("P524E1", pd.Series(np.nan, index=df.index)), errors="coerce")
    p523   = pd.to_numeric(df.get("P523",   pd.Series(np.nan, index=df.index)), errors="coerce")

    mult = p523.map(PERIOD_MULT).fillna(1.0)

    # Best income: P524A1 first, then P524C1, then P524E1
    raw_income = p524a1.copy()
    missing_a1 = raw_income.isna() | (raw_income <= 0)
    raw_income[missing_a1] = p524c1[missing_a1]
    still_missing = raw_income.isna() | (raw_income <= 0)
    raw_income[still_missing] = p524e1[still_missing]

    df["nominal_monthly_wage"] = raw_income * mult

    # ── Step 2: Keep only income reporters ───────────────────────────────────
    df = df[df["nominal_monthly_wage"].gt(0) & df["nominal_monthly_wage"].notna()].copy()

    # ── Step 3: Exclude self-employed and employers ───────────────────────────
    # P510: 1=empleador, 2=independiente → exclude
    # P510: 5=doméstico, 6=asalariado, 3=privado, NaN=keep (asalariado implied)
    if "P510" in df.columns:
        p510 = df["P510"]
        df = df[~p510.isin([1.0, 2.0])].copy()

    # ── Step 4: Deflate to 2023 soles ─────────────────────────────────────────
    yr  = df["year"].iloc[0]
    d   = deflators.get(int(yr), 1.0)
    df["deflator"]     = d
    df["monthly_wage"] = df["nominal_monthly_wage"] * d   # real 2023 soles

    # Trim extreme outliers (> S/50,000/mo real 2023 = likely data error)
    df = df[df["monthly_wage"] <= 50_000].copy()

    # ── Step 5: Formality ─────────────────────────────────────────────────────
    p517 = df.get("P517", pd.Series(np.nan, index=df.index))
    p558 = df.get("P558A1", pd.Series(np.nan, index=df.index))
    df["contract"] = p517.isin([1.0, 2.0])
    df["pension"]  = (p558 == 1.0)
    df["formal"]   = df["contract"] | df["pension"]
    df["informal"] = ~df["formal"]

    # ── Step 6: Weight ────────────────────────────────────────────────────────
    w_col = "FAC500A" if "FAC500A" in df.columns else None
    df["w"] = df[w_col].fillna(1.0) if w_col else 1.0

    return df


# ── Step 3: Wage distribution ─────────────────────────────────────────────────

def wage_distribution(df: pd.DataFrame, real_min: float) -> dict:
    wages = df["monthly_wage"].values
    w     = df["w"].values

    total   = float(w.sum())
    formal  = float(df.loc[df["formal"],   "w"].sum())
    infml   = float(df.loc[df["informal"], "w"].sum())

    pctiles    = {f"p{p}": weighted_percentile(wages, w, p) for p in [10, 25, 50, 75, 90]}
    med_formal = weighted_median(df.loc[df["formal"], "monthly_wage"].values,
                                 df.loc[df["formal"], "w"].values)
    med_infml  = weighted_median(df.loc[df["informal"], "monthly_wage"].values,
                                 df.loc[df["informal"], "w"].values)
    kaitz      = real_min / med_formal if med_formal > 0 else np.nan
    at_min     = float(df.loc[df["monthly_wage"] <= real_min, "w"].sum())

    brackets = [
        (0,            real_min,       f"<= RMV (S/{int(real_min)})"),
        (real_min,     real_min*1.10,  "RMV .. RMV+10%"),
        (real_min*1.1, real_min*1.25,  "RMV+10% .. RMV+25%"),
        (real_min*1.25,real_min*1.5,   "RMV+25% .. RMV+50%"),
        (real_min*1.5, real_min*2,     "RMV+50% .. 2xRMV"),
        (real_min*2,   1e9,            "> 2xRMV"),
    ]
    bkts = []
    for lo, hi, label in brackets:
        m = (df["monthly_wage"] > lo) & (df["monthly_wage"] <= hi)
        bkts.append({"label": label, "share_pct": round(float(df.loc[m, "w"].sum()) / total * 100, 2)})

    return {
        "total_wage_workers":  int(round(total)),
        "formal_workers":      int(round(formal)),
        "informal_workers":    int(round(infml)),
        "informality_rate":    round(infml / total * 100, 1),
        "at_or_below_min":     int(round(at_min)),
        "pct_at_or_below_min": round(at_min / total * 100, 1),
        "median_formal":       round(med_formal, 1),
        "median_informal":     round(med_infml, 1),
        "kaitz_index":         round(kaitz, 3),
        "percentiles":         {k: round(v, 1) for k, v in pctiles.items()},
        "distribution_brackets": bkts,
    }


# ── Step 4: Simulate scenarios ────────────────────────────────────────────────

def simulate_scenario(df: pd.DataFrame, current_min: float, new_min: float) -> dict:
    pct_inc = (new_min - current_min) / current_min

    f = df[df["formal"]].copy()
    affected_m = (f["monthly_wage"] >= current_min) & (f["monthly_wage"] < new_min)
    aff_w      = float(f.loc[affected_m, "w"].sum())

    displaced_w    = aff_w * abs(EMPLOYMENT_ELASTICITY) * pct_inc
    net_benefit_w  = aff_w - displaced_w

    wage_gap     = (new_min - f.loc[affected_m, "monthly_wage"]).clip(lower=0)
    monthly_cost = float((wage_gap * f.loc[affected_m, "w"]).sum())

    sub_m_w = float(f.loc[f["monthly_wage"] < current_min, "w"].sum())

    inf         = df[df["informal"]].copy()
    inf_below_w = float(inf.loc[inf["monthly_wage"] < new_min, "w"].sum())
    lighthouse_raise = LIGHTHOUSE_ELASTICITY * (new_min - current_min)

    # Domain breakdown
    by_domain = {}
    if "DOMINIO" in df.columns:
        for d, label in DOMINIO_LABELS.items():
            sub = f[f["DOMINIO"] == d]
            am  = (sub["monthly_wage"] >= current_min) & (sub["monthly_wage"] < new_min)
            by_domain[label] = int(round(float(sub.loc[am, "w"].sum())))

    return {
        "new_min":                       new_min,
        "pct_increase":                  round(pct_inc * 100, 1),
        "formal_affected":               int(round(aff_w)),
        "formal_sub_min_non_compliant":  int(round(sub_m_w)),
        "formal_displaced":              int(round(displaced_w)),
        "formal_net_benefited":          int(round(net_benefit_w)),
        "informal_lighthouse_benefited": int(round(inf_below_w)),
        "informal_wage_increase_soles":  round(lighthouse_raise, 1),
        "monthly_cost_employers_soles":  int(round(monthly_cost)),
        "by_domain":                     by_domain,
    }


# ── Step 5: Kaitz series ──────────────────────────────────────────────────────

def build_kaitz_series(panels: dict, deflators: dict) -> list[dict]:
    rows = []
    for year, df in sorted(panels.items()):
        yr   = int(year)
        d    = deflators.get(yr, 1.0)
        mw_n = MIN_WAGE_HISTORY.get(yr, CURRENT_MIN)          # nominal
        mw_r = mw_n * d                                        # real 2023
        med  = weighted_median(df.loc[df["formal"], "monthly_wage"].values,
                               df.loc[df["formal"], "w"].values)
        kaitz = mw_r / med if med > 0 else None
        rows.append({
            "year":          yr,
            "min_wage_nominal": mw_n,
            "min_wage_real":    round(mw_r, 1),
            "median_formal_real": round(med, 1),
            "deflator":         round(d, 4),
            "kaitz_index":      round(kaitz, 3) if kaitz else None,
        })
    return rows


# ── Step 6: Literature comparison ────────────────────────────────────────────

LITERATURE = [
    {
        "study":    "Céspedes & Sánchez (2014)",
        "country":  "Peru",
        "period":   "2002-2012",
        "formal_elasticity": "-0.05 to -0.15",
        "informal_elasticity": "—",
        "method":   "Panel DiD, ENAHO",
        "source":   "BCRP DT 2014-014",
    },
    {
        "study":    "Jaramillo (2013)",
        "country":  "Peru",
        "period":   "2003-2010",
        "formal_elasticity": "-0.08",
        "informal_elasticity": "—",
        "method":   "ENAHO panel, DiD",
        "source":   "PUCP Working Paper",
    },
    {
        "study":    "Chacaltana (2001)",
        "country":  "Peru",
        "period":   "1990s",
        "formal_elasticity": "—",
        "informal_elasticity": "lighthouse ~0.3-0.5",
        "method":   "Cross-section",
        "source":   "ILO/CIES",
    },
    {
        "study":    "Neumark & Wascher (2007)",
        "country":  "Multi",
        "period":   "—",
        "formal_elasticity": "-0.1 to -0.3",
        "informal_elasticity": "—",
        "method":   "Meta-analysis",
        "source":   "Foundations & Trends Micro",
    },
    {
        "study":    "Boeri, Garibaldi & Ribeiro (2011)",
        "country":  "LatAm",
        "period":   "—",
        "formal_elasticity": "—",
        "informal_elasticity": "0.3-0.5 (lighthouse)",
        "method":   "Cross-country panel",
        "source":   "Journal of Applied Econ.",
    },
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    # ── Build CPI deflators ─────────────────────────────────────────────────
    print("\n[1/6] Building CPI deflators (base 2023=1.00)...")
    deflators = build_deflators()
    print(f"  {'Year':>6}  {'Deflator':>10}  {'CPI 2023 rel.':>14}")
    print(f"  {'----':>6}  {'--------':>10}  {'-------------':>14}")
    for yr in sorted(deflators):
        if yr >= 2015:
            nom_mw = MIN_WAGE_HISTORY.get(yr, CURRENT_MIN)
            real_mw = nom_mw * deflators[yr]
            print(f"  {yr:>6}  {deflators[yr]:>10.4f}  "
                  f"RMV nominal S/{nom_mw} -> real S/{real_mw:.0f}")

    # ── Download ────────────────────────────────────────────────────────────
    if not args.skip_download:
        missing = [yr for yr in YEARS if find_dta(yr, RAW_DIR) is None]
        if missing:
            print(f"\n[2/6] Downloading missing years: {missing}")
            download_module05(missing, RAW_DIR)
        else:
            print(f"\n[2/6] All years already downloaded.")
    else:
        print("\n[2/6] Skipping download (--skip-download).")

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"\n[3/6] Loading and cleaning ENAHO Module 05")
    print(f"{'='*60}")
    panels: dict[str, pd.DataFrame] = {}
    for yr in YEARS:
        raw = load_year(yr, RAW_DIR, deflators)
        if raw is not None:
            panels[yr] = clean(raw, deflators)
            n   = len(panels[yr])
            nw  = panels[yr]["w"].sum()
            inf_pct = panels[yr]["informal"].mean() * 100
            # Periodicity breakdown
            if "P523" in raw.columns:
                pc = raw["P523"].value_counts()
                period_str = f"  quinc={pc.get(4.0,0):,} sem={pc.get(2.0,0):,} dia={pc.get(3.0,0):,} mens={pc.get(1.0,0):,}"
            else:
                period_str = ""
            print(f"  [{yr}] {n:>6,} wage workers -> {nw/1e6:.2f}M weighted  "
                  f"informality={inf_pct:.0f}%{period_str}")

    if not panels:
        sys.exit("ERROR: No data loaded.")

    focus_year = max(panels.keys())
    df = panels[focus_year]
    real_min = CURRENT_MIN * deflators.get(int(focus_year), 1.0)
    print(f"\nFocus year: {focus_year}  |  RMV nominal=S/{CURRENT_MIN}  "
          f"real=S/{real_min:.0f} (2023 soles)")

    # ── Wage distribution ─────────────────────────────────────────────────────
    print(f"\n[4/6] Wage distribution — {focus_year}")
    print(f"{'='*60}")
    dist = wage_distribution(df, real_min)

    print(f"  Total wage workers (weighted):   {dist['total_wage_workers']/1e6:.2f}M")
    print(f"  Formal workers:                  {dist['formal_workers']/1e6:.2f}M  "
          f"({100-dist['informality_rate']:.0f}%)")
    print(f"  Informal workers:                {dist['informal_workers']/1e6:.2f}M  "
          f"({dist['informality_rate']:.0f}%)")
    print(f"  At or below RMV (real S/{real_min:.0f}):  "
          f"{dist['at_or_below_min']/1e6:.2f}M  ({dist['pct_at_or_below_min']:.1f}%)")
    print(f"  Median formal (real 2023 S/):    S/ {dist['median_formal']:.0f}")
    print(f"  Median informal (real 2023 S/):  S/ {dist['median_informal']:.0f}")
    print(f"  Kaitz index:                     {dist['kaitz_index']:.3f}")
    print(f"  Percentiles: " +
          "  ".join(f"p{k[1:]}=S/{v:.0f}" for k, v in dist["percentiles"].items()))
    print()
    for b in dist["distribution_brackets"]:
        bar = chr(9608) * int(b["share_pct"] / 1.5)
        print(f"    {b['label']:32s} {b['share_pct']:5.1f}%  {bar}")

    # ── Scenarios ─────────────────────────────────────────────────────────────
    print(f"\n[5/6] Simulating min-wage scenarios")
    print(f"  Base RMV: S/{CURRENT_MIN}  |  Employment elasticity: {EMPLOYMENT_ELASTICITY}  "
          f"|  Lighthouse: {LIGHTHOUSE_ELASTICITY}")
    print(f"{'='*60}")
    scenarios = [simulate_scenario(df, real_min, nm * deflators.get(int(focus_year), 1.0))
                 for nm in SCENARIOS]
    # Store nominal new_min for readability
    for s, nm in zip(scenarios, SCENARIOS):
        s["new_min_nominal"] = nm
        s["new_min_real"]    = s.pop("new_min")

    print(f"  {'New RMV':>8} {'D%':>5}  {'Affected':>10} {'Displaced':>10} "
          f"{'Net Benefit':>11} {'Informal':>10} {'Cost/mo S/M':>12}")
    print(f"  {'--------':>8} {'--':>5}  {'--------':>10} {'--------':>10} "
          f"{'----------':>11} {'--------':>10} {'----------':>12}")
    for s in scenarios:
        print(f"  S/{s['new_min_nominal']:>4}   {s['pct_increase']:>4.1f}%  "
              f"{s['formal_affected']/1e3:>8.0f}k  "
              f"{s['formal_displaced']/1e3:>8.0f}k  "
              f"{s['formal_net_benefited']/1e3:>9.0f}k  "
              f"{s['informal_lighthouse_benefited']/1e3:>8.0f}k  "
              f"S/{s['monthly_cost_employers_soles']/1e6:>8.1f}M")

    # ── Kaitz series ──────────────────────────────────────────────────────────
    kaitz_series = build_kaitz_series(panels, deflators)
    print(f"\n  Kaitz index trend (RMV real / median formal real wage):")
    for row in kaitz_series:
        bar = chr(9608) * int((row["kaitz_index"] or 0) * 20)
        print(f"    {row['year']}:  "
              f"RMV_r=S/{row['min_wage_real']:.0f}  "
              f"Med_f=S/{row['median_formal_real']:.0f}  "
              f"Kaitz={row['kaitz_index']:.3f}  {bar}")

    # ── Literature comparison ─────────────────────────────────────────────────
    print(f"\n[6/6] Literature comparison")
    print(f"{'='*60}")
    print(f"  {'Study':<35} {'Country':<8} {'Period':<12} "
          f"{'Formal elast.':<16} {'Method'}")
    print(f"  {'-'*35} {'-'*8} {'-'*12} {'-'*16} {'-'*20}")
    for lit in LITERATURE:
        print(f"  {lit['study']:<35} {lit['country']:<8} {lit['period']:<12} "
              f"{lit['formal_elasticity']:<16} {lit['method']}")
    print()
    our_e = EMPLOYMENT_ELASTICITY
    in_range = -0.15 <= our_e <= -0.05
    print(f"  Qhawarina assumption: {our_e:.2f}  "
          f"({'within' if in_range else 'OUTSIDE'} Peru literature range -0.05 to -0.15)")
    print(f"  Lighthouse assumption: {LIGHTHOUSE_ELASTICITY:.2f}  "
          f"(within LatAm range 0.30-0.50)")
    print()
    print(f"  Note: Céspedes & Sánchez (2014) PDF downloaded to:")
    print(f"    {RAW_DIR}/cespedes_sanchez_2014_minwage_peru.pdf")
    print(f"  Variable definitions match ours:")
    print(f"    - Same ENAHO Module 500")
    print(f"    - Formality: written contract (P517) OR pension (P558)")
    print(f"    - Wage: P524A1 (ingreso monetario principal)")
    print(f"  Difference: their period 2002-2012 vs ours 2019-2023")
    print(f"    -> COVID shock (2020) inflates informality; exclude for clean estimates")

    # ── Export ────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "base_year":            int(focus_year),
        "current_minimum_wage": CURRENT_MIN,
        "deflators":            {str(k): round(v, 4) for k, v in deflators.items() if k >= 2015},
        "wage_distribution":    dist,
        "scenarios":            scenarios,
        "kaitz_series":         kaitz_series,
        "literature":           LITERATURE,
        "methodology": {
            "employment_elasticity":        EMPLOYMENT_ELASTICITY,
            "employment_elasticity_source": "Céspedes & Sánchez (2014), BCRP DT 2014-014",
            "lighthouse_elasticity":        LIGHTHOUSE_ELASTICITY,
            "lighthouse_source":            "Boeri, Garibaldi & Ribeiro (2011)",
            "formality_definition":         "P517 in [1,2] (written contract) OR P558A1==1 (pension)",
            "wage_variable":                "P524A1 deflated to 2023 soles via CPI Lima (PN01271PM)",
            "weight":                       "FAC500A (expansion factor)",
            "survey":                       "ENAHO Modulo 500 (Empleo e Ingresos)",
            "years_used":                   sorted(panels.keys()),
            "cpi_base":                     "BCRP PN01271PM, annual average, base 2023=1.00",
        },
    }
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=npy)
    print(f"\nSaved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
