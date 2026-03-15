"""
build_mw_regional.py
====================
Process ENAHO 2023 Module 05 (srienaho code 906) into department-level
minimum wage statistics for the Qhawarina MW Simulator.

Output: exports/data/mw_regional_data.json

Columns used
------------
ubigeo   : 6-char UBIGEO (first 2 digits = dept code 01-25)
fac500a  : expansion / survey weight  (MUST use for all estimates)
ocu500   : 1 = employed (filter to this)
ocupinf  : informality status  1=informal  2=formal  (INEI official definition)
p524a1   : gross monthly income from main job (soles/month, current) ← wage variable
p207     : sex  (1=male 2=female)
p208a    : age

Formality definition: ocupinf == 2  (INEI official formal worker classification)
Wage variable: p524a1  (monthly soles, raw reported; ~42% coverage among employed)
  — NOT i524a1 (that is imputed ANNUAL income, needs /12)
  — NOT d529t  (deflated annual, only covers independents/obreros)

Department codes (INEI, 2-digit string):
  01 Amazonas  02 Ancash   03 Apurimac  04 Arequipa  05 Ayacucho
  06 Cajamarca 07 Callao   08 Cusco     09 Huancavelica 10 Huanuco
  11 Ica       12 Junin    13 La Libertad 14 Lambayeque 15 Lima
  16 Loreto    17 Madre de Dios 18 Moquegua 19 Pasco  20 Piura
  21 Puno      22 San Martin  23 Tacna   24 Tumbes  25 Ucayali
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
SAV     = ROOT / "data/raw/enaho/906/906-Modulo05/Enaho01a-2023-500.sav"
OUT_DIR = ROOT / "exports/data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT     = OUT_DIR / "mw_regional_data.json"

# ── Constants ─────────────────────────────────────────────────────────────────
MW_REF   = 1025          # Apr 2022 EPE reference MW (soles/month)
MW_VALS  = [1050, 1075, 1100, 1130, 1200, 1300, 1500]  # scenarios
BAND_LO  = 0.85          # treat_lo  → lower edge of treatment band
BAND_HI  = 1.40          # ctrl_hi   → upper edge of control band
KDE_BINS = 200           # histogram bins for wage_distribution
WAGE_MIN = 0
WAGE_MAX = 10_000        # soles/month cap for KDE

DEPT_NAMES = {
    "01": "Amazonas",       "02": "Ancash",         "03": "Apurímac",
    "04": "Arequipa",       "05": "Ayacucho",        "06": "Cajamarca",
    "07": "Callao",         "08": "Cusco",           "09": "Huancavelica",
    "10": "Huánuco",        "11": "Ica",             "12": "Junín",
    "13": "La Libertad",    "14": "Lambayeque",      "15": "Lima",
    "16": "Loreto",         "17": "Madre de Dios",   "18": "Moquegua",
    "19": "Pasco",          "20": "Piura",           "21": "Puno",
    "22": "San Martín",     "23": "Tacna",           "24": "Tumbes",
    "25": "Ucayali",
}

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading {SAV} …")
try:
    import pyreadstat
    df, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False)
    print(f"  Loaded via pyreadstat: {df.shape}")
except ImportError:
    import savReaderWriter
    with savReaderWriter.SavReader(str(SAV), returnHeader=True) as r:
        header = next(r)
        rows   = list(r)
    df = pd.DataFrame(rows, columns=header)
    print(f"  Loaded via savReaderWriter: {df.shape}")

# Normalise column names to lower-case
df.columns = [c.lower() for c in df.columns]
print(f"  Columns (first 30): {list(df.columns[:30])}")

# ── Column selection & type coercion ─────────────────────────────────────────
need = ["ubigeo", "fac500a", "ocu500", "ocupinf", "p524a1", "p207", "p208a"]

missing = [c for c in need if c not in df.columns]
if missing:
    print(f"  WARNING — missing columns: {missing}")
    for c in missing:
        df[c] = np.nan

df = df[need].copy()
for c in ["fac500a", "ocu500", "ocupinf", "p524a1", "p207", "p208a"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ubigeo: coerce to zero-padded 6-char string, extract dept (first 2)
df["ubigeo"] = df["ubigeo"].astype(str).str.strip().str.zfill(6)
df["dept"]   = df["ubigeo"].str[:2]

print(f"  Departments found: {sorted(df['dept'].unique())}")
print(f"  fac500a range: {df['fac500a'].min():.1f} – {df['fac500a'].max():.1f}")

# ── Filter to employed workers ────────────────────────────────────────────────
employed = df[df["ocu500"] == 1].copy()
print(f"  Employed rows: {len(employed):,}  (weighted sum: {employed['fac500a'].sum():,.0f})")

# ── Monthly wage ──────────────────────────────────────────────────────────────
# p524a1 = gross monthly income from main job (soles, current prices)
# This is the standard variable for MW analysis in Peru labor economics.
employed["wage_monthly"] = employed["p524a1"].copy()

# Keep only positive wages (workers who report monetary income)
valid_wage = employed["wage_monthly"] > 0
print(f"  Workers with valid p524a1 wage: {valid_wage.sum():,} / {len(employed):,}")
employed = employed[valid_wage].copy()

# ── Formality (INEI official definition) ─────────────────────────────────────
# ocupinf: 1 = informal worker, 2 = formal worker
employed["is_formal"] = (employed["ocupinf"] == 2).astype(int)

n_total  = employed["fac500a"].sum()
n_formal = employed.loc[employed["is_formal"] == 1, "fac500a"].sum()
print(f"  National formality rate (weighted): {n_formal/n_total:.1%}")

# Diagnostic: p524a1 percentiles for formal vs informal
f_wages = employed.loc[employed["is_formal"]==1, "wage_monthly"]
i_wages = employed.loc[employed["is_formal"]==0, "wage_monthly"]
print(f"  Formal wage p25/p50/p75:   {f_wages.quantile(.25):.0f} / {f_wages.quantile(.50):.0f} / {f_wages.quantile(.75):.0f}")
print(f"  Informal wage p25/p50/p75: {i_wages.quantile(.25):.0f} / {i_wages.quantile(.50):.0f} / {i_wages.quantile(.75):.0f}")

# ── Weighted quantile helper ──────────────────────────────────────────────────
def weighted_quantile(values, weights, quantiles):
    """Compute weighted quantiles. quantiles in [0,1]."""
    values  = np.asarray(values,  dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask    = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[mask], weights[mask]
    if len(values) == 0:
        return [np.nan] * len(quantiles)
    sorter   = np.argsort(values)
    values   = values[sorter]
    weights  = weights[sorter]
    cum_w    = np.cumsum(weights)
    total_w  = cum_w[-1]
    results  = []
    for q in quantiles:
        target = q * total_w
        idx    = np.searchsorted(cum_w, target)
        idx    = min(idx, len(values) - 1)
        results.append(float(values[idx]))
    return results

# ── KDE builder ───────────────────────────────────────────────────────────────
def build_kde(wages, weights, x_grid):
    """Return list of [x, density] bins using weighted KDE."""
    wages   = np.asarray(wages,   dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask    = np.isfinite(wages) & np.isfinite(weights) & (wages > 0) & (weights > 0)
    wages, weights = wages[mask], weights[mask]
    if len(wages) < 5:
        return [[float(x), 0.0] for x in x_grid]
    # Subsample for speed (max 5000 points, proportional to weight)
    if len(wages) > 5000:
        prob = weights / weights.sum()
        idx  = np.random.choice(len(wages), size=5000, replace=False, p=prob)
        wages   = wages[idx]
        weights = weights[idx]
    try:
        kde = gaussian_kde(wages, weights=weights, bw_method="scott")
        dens = kde(x_grid)
    except Exception:
        dens = np.zeros_like(x_grid)
    return [[float(x), float(d)] for x, d in zip(x_grid, dens)]

x_grid = np.linspace(WAGE_MIN, WAGE_MAX, KDE_BINS)

# ── Per-department computation ────────────────────────────────────────────────
print("\nComputing department-level statistics …")

results = {}

depts_to_process = sorted(DEPT_NAMES.keys())

for dept in depts_to_process:
    name = DEPT_NAMES[dept]
    sub  = employed[employed["dept"] == dept].copy()

    if len(sub) == 0:
        print(f"  {dept} {name}: NO DATA")
        continue

    w    = sub["fac500a"].values
    wage = sub["wage_monthly"].values
    frm  = sub["is_formal"].values

    formal_mask   = frm == 1
    informal_mask = frm == 0

    sub_formal   = sub[formal_mask]
    sub_informal = sub[informal_mask]

    w_f  = sub_formal["fac500a"].values
    wg_f = sub_formal["wage_monthly"].values
    w_i  = sub_informal["fac500a"].values

    # Counts
    n_total_w   = float(w.sum())
    n_formal_w  = float(w_f.sum())   if len(w_f)  > 0 else 0.0
    n_informal_w= float(w_i.sum())   if len(w_i)  > 0 else 0.0
    pct_formal  = n_formal_w / n_total_w if n_total_w > 0 else 0.0

    # Medians
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    p_all     = weighted_quantile(wage, w, qs)
    p_formal  = weighted_quantile(wg_f, w_f, qs) if len(w_f) > 10 else [np.nan]*5
    p_informal= weighted_quantile(sub_informal["wage_monthly"].values, w_i, qs) if len(w_i) > 10 else [np.nan]*5

    med_formal = p_formal[2]   # p50
    med_all    = p_all[2]

    # Kaitz index
    kaitz_formal = (MW_REF / med_formal)  if (med_formal and med_formal > 0) else None
    kaitz_all    = (MW_REF / med_all)     if (med_all    and med_all    > 0) else None

    # Workers in band for each MW scenario
    in_band = {}
    below_mw_ref = {}
    for mw in MW_VALS:
        lo = BAND_LO * mw
        hi = BAND_HI * mw
        band_mask_f = formal_mask & (wage >= lo) & (wage <= hi)
        in_band[str(mw)] = float(sub.loc[band_mask_f, "fac500a"].sum())
        below_mask_f     = formal_mask & (wage < mw)
        below_mw_ref[str(mw)] = float(sub.loc[below_mask_f, "fac500a"].sum())

    # KDE for formal wage distribution
    kde_formal = build_kde(wg_f, w_f, x_grid) if len(w_f) > 10 else []

    dept_result = {
        "dept_code":          dept,
        "dept_name":          name,
        "n_workers":          round(n_total_w),
        "n_formal":           round(n_formal_w),
        "n_informal":         round(n_informal_w),
        "pct_formal":         round(pct_formal, 4),
        "median_formal_wage": round(med_formal, 2) if med_formal and np.isfinite(med_formal) else None,
        "median_all_wage":    round(med_all,    2) if med_all    and np.isfinite(med_all)    else None,
        "kaitz_formal":       round(kaitz_formal, 4) if kaitz_formal and np.isfinite(kaitz_formal) else None,
        "kaitz_all":          round(kaitz_all,    4) if kaitz_all    and np.isfinite(kaitz_all)    else None,
        "wage_percentiles": {
            "formal": {
                "p10": round(p_formal[0], 2) if np.isfinite(p_formal[0]) else None,
                "p25": round(p_formal[1], 2) if np.isfinite(p_formal[1]) else None,
                "p50": round(p_formal[2], 2) if np.isfinite(p_formal[2]) else None,
                "p75": round(p_formal[3], 2) if np.isfinite(p_formal[3]) else None,
                "p90": round(p_formal[4], 2) if np.isfinite(p_formal[4]) else None,
            },
            "all": {
                "p10": round(p_all[0], 2) if np.isfinite(p_all[0]) else None,
                "p25": round(p_all[1], 2) if np.isfinite(p_all[1]) else None,
                "p50": round(p_all[2], 2) if np.isfinite(p_all[2]) else None,
                "p75": round(p_all[3], 2) if np.isfinite(p_all[3]) else None,
                "p90": round(p_all[4], 2) if np.isfinite(p_all[4]) else None,
            },
        },
        "n_formal_in_band":   in_band,
        "n_formal_below_mw":  below_mw_ref,
        "wage_kde_formal":    kde_formal,
    }
    results[dept] = dept_result
    print(f"  {dept} {name:18s}: n={n_total_w:>9,.0f}  formal={pct_formal:.1%}  "
          f"med_formal={med_formal:>7.0f}  kaitz={kaitz_formal:.3f}" if kaitz_formal else
          f"  {dept} {name:18s}: n={n_total_w:>9,.0f}  formal={pct_formal:.1%}  med_formal=N/A")

# ── National aggregate ────────────────────────────────────────────────────────
print("\nComputing national aggregate …")
w    = employed["fac500a"].values
wage = employed["wage_monthly"].values
frm  = employed["is_formal"].values

formal_mask   = frm == 1
sub_formal    = employed[formal_mask]
w_f           = sub_formal["fac500a"].values
wg_f          = sub_formal["wage_monthly"].values

qs            = [0.10, 0.25, 0.50, 0.75, 0.90]
p_all         = weighted_quantile(wage, w, qs)
p_formal      = weighted_quantile(wg_f, w_f, qs)

n_total_w     = float(w.sum())
n_formal_w    = float(w_f.sum())
pct_formal    = n_formal_w / n_total_w
med_formal    = p_formal[2]
med_all       = p_all[2]
kaitz_formal  = MW_REF / med_formal if med_formal > 0 else None
kaitz_all     = MW_REF / med_all    if med_all    > 0 else None

in_band = {}
below_mw_ref = {}
for mw in MW_VALS:
    lo = BAND_LO * mw
    hi = BAND_HI * mw
    band_mask = formal_mask & (wage >= lo) & (wage <= hi)
    in_band[str(mw)]     = float(employed.loc[band_mask,   "fac500a"].sum())
    below_mask           = formal_mask & (wage < mw)
    below_mw_ref[str(mw)] = float(employed.loc[below_mask, "fac500a"].sum())

kde_national_formal = build_kde(wg_f, w_f, x_grid)

results["00"] = {
    "dept_code":          "00",
    "dept_name":          "Nacional",
    "n_workers":          round(n_total_w),
    "n_formal":           round(n_formal_w),
    "n_informal":         round(n_total_w - n_formal_w),
    "pct_formal":         round(pct_formal, 4),
    "median_formal_wage": round(med_formal, 2),
    "median_all_wage":    round(med_all,    2),
    "kaitz_formal":       round(kaitz_formal, 4) if kaitz_formal else None,
    "kaitz_all":          round(kaitz_all,    4) if kaitz_all    else None,
    "wage_percentiles": {
        "formal": {f"p{int(q*100)}": round(v, 2) for q, v in zip([0.1,0.25,0.5,0.75,0.9], p_formal)},
        "all":    {f"p{int(q*100)}": round(v, 2) for q, v in zip([0.1,0.25,0.5,0.75,0.9], p_all)},
    },
    "n_formal_in_band":   in_band,
    "n_formal_below_mw":  below_mw_ref,
    "wage_kde_formal":    kde_national_formal,
}

print(f"  Nacional: n={n_total_w:,.0f}  formal={pct_formal:.1%}  "
      f"med_formal={med_formal:.0f}  kaitz={kaitz_formal:.3f}")

# ── Validation summary ────────────────────────────────────────────────────────
print("\n=== VALIDATION SUMMARY ===")
print(f"{'Dept':4} {'Name':20} {'Formal%':>8} {'Median F':>10} {'Kaitz':>7} {'InBand1130':>12}")
print("-" * 68)
for code in sorted(results.keys()):
    r = results[code]
    kaitz_str   = f"{r['kaitz_formal']:.3f}" if r.get('kaitz_formal') else "  N/A"
    in_band_str = f"{r['n_formal_in_band'].get('1130', 0):>12,.0f}"
    med_str     = f"{r['median_formal_wage']:>10,.0f}" if r.get('median_formal_wage') else "       N/A"
    print(f"{code:4} {r['dept_name']:20} {r['pct_formal']:>8.1%} {med_str} {kaitz_str:>7} {in_band_str}")

# ── Metadata ──────────────────────────────────────────────────────────────────
output = {
    "metadata": {
        "source":       "ENAHO 2023 Módulo 05 Empleo e Ingresos (srienaho code 906)",
        "reference_mw": MW_REF,
        "mw_scenarios": MW_VALS,
        "band_lo":      BAND_LO,
        "band_hi":      BAND_HI,
        "formality":    "ocupinf==2 (INEI official formal worker classification)",
        "wage_var":     "p524a1 (gross monthly income from main job, soles, current prices)",
        "filter":       "ocu500==1 (employed), p524a1>0",
        "generated":    "2026-03-15",
        "x_grid_min":   WAGE_MIN,
        "x_grid_max":   WAGE_MAX,
        "kde_bins":     KDE_BINS,
    },
    "departments": results,
}

# ── Write JSON ────────────────────────────────────────────────────────────────
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

size_kb = OUT.stat().st_size / 1024
print(f"\nWrote {OUT}  ({size_kb:,.1f} KB)")
print("Done.")
