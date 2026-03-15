"""
build_mw_regional_did_panel.py
================================
Build estimation dataset for the regional DiD following Dube, Lester & Reich (2010):

  Y_dt = α_d + γ_t + β(Post_t × Kaitz_d_pre) + X'θ + ε_dt

Source: ENAHO Panel 2020-2024 (srienaho code 978, Module 1477)
        One SAV file covers all years in WIDE format.

Steps:
  1. Load wide panel → melt to long format
  2. Compute dept-level Kaitz_pre from 2021 (MW=930 in effect all year)
  3. Merge treatment intensity by department
  4. Create outcome variables
  5. Save to parquet + print diagnostics

Wage: i524a1_{YY} / 12  (imputed annual income → monthly, standard INEI convention)
Formality: ocupinf_{YY} == 2  (INEI official)
Weight: fac500a_{YY}

Panel structure: WIDE (one row per person, year suffix _20 / _21 / _22 / _23 / _24)
Person ID: numper

MW schedule (annual, in effect):
  2020: S/930  (nominal, COVID freeze year)
  2021: S/930
  2022: S/930 → S/1,025 (raised May 1 2022, partial exposure)
  2023: S/1,025 (full year in effect)
  2024: S/1,025
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
SAV     = ROOT / "data/raw/enaho/978_panel/Modulo1477_Empleo_Ingresos/978-Modulo1477/enaho01a-2020-2024-500-panel.sav"
OUT_DIR = ROOT / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_OUT = OUT_DIR / "mw_regional_did_panel.parquet"
KAITZ_OUT = ROOT / "exports/data/mw_pre_policy_kaitz.json"

# ── Constants ─────────────────────────────────────────────────────────────────
MW_BY_YEAR = {2020: 930, 2021: 930, 2022: 977.5, 2023: 1025, 2024: 1025}
# 2022 midpoint: MW raised May 1 → avg ~930 for 4 months + 1025 for 8 months ≈ 993
# Use 977.5 = average(930, 1025) for partial-year exposure

YEARS   = [2021, 2022, 2023]   # core window (2021=pre, 2022=transition, 2023=post)
SUFFIXES = {2021: "21", 2022: "22", 2023: "23"}

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

# ── Load wide panel (targeted columns only) ────────────────────────────────────
print("Loading panel Module 1477 (targeted columns) …")

stems    = ["ubigeo", "ocu500", "ocupinf", "i524a1", "fac500a", "p207", "p208a", "p507"]
all_sfx  = ["20", "21", "22", "23", "24"]
usecols  = ["numper", "conglome", "vivienda"] + [f"{s}_{y}" for s in stems for y in all_sfx]

import pyreadstat
df_wide, _ = pyreadstat.read_sav(str(SAV), apply_value_formats=False, usecols=usecols)
df_wide.columns = [c.lower() for c in df_wide.columns]
print(f"  Wide shape: {df_wide.shape}")
print(f"  Unique persons (numper): {df_wide['numper'].nunique():,}")

# ── Melt to long format ────────────────────────────────────────────────────────
print("\nMelting to long format …")

frames = []
for year, sfx in {2020: "20", 2021: "21", 2022: "22", 2023: "23", 2024: "24"}.items():
    cols_needed = {
        "ubigeo":   f"ubigeo_{sfx}",
        "ocu500":   f"ocu500_{sfx}",
        "ocupinf":  f"ocupinf_{sfx}" if f"ocupinf_{sfx}" in df_wide.columns else None,
        "i524a1":   f"i524a1_{sfx}",
        "fac500a":  f"fac500a_{sfx}",
        "p207":     f"p207_{sfx}",
        "p208a":    f"p208a_{sfx}",
        "p507":     f"p507_{sfx}"    if f"p507_{sfx}"    in df_wide.columns else None,
    }

    sub = df_wide[["numper"]].copy()
    sub["year"] = year

    for out_col, src_col in cols_needed.items():
        if src_col and src_col in df_wide.columns:
            sub[out_col] = pd.to_numeric(df_wide[src_col], errors="coerce")
        else:
            sub[out_col] = np.nan

    # ubigeo as string
    sub["ubigeo"] = df_wide[f"ubigeo_{sfx}"].astype(str).str.strip().str.zfill(6)
    sub["dept"]   = sub["ubigeo"].str[:2]

    frames.append(sub)

df = pd.concat(frames, ignore_index=True)
print(f"  Long shape: {df.shape}")

# ── Filter to core estimation window (2021-2023) ───────────────────────────────
df_core = df[df["year"].isin(YEARS)].copy()
print(f"  Core window (2021-2023): {len(df_core):,} person-year obs")

# ── Monthly wage ───────────────────────────────────────────────────────────────
df_core["wage_monthly"] = df_core["i524a1"] / 12.0

# ── Working-age filter (14-65) ─────────────────────────────────────────────────
df_core = df_core[(df_core["p208a"] >= 14) & (df_core["p208a"] <= 65)].copy()
print(f"  Working-age (14-65): {len(df_core):,}")

# ── Outcome variables ──────────────────────────────────────────────────────────
df_core["formal"]          = (df_core["ocupinf"] == 2).astype(float)
df_core["informal"]        = (df_core["ocupinf"] == 1).astype(float)
df_core["employed"]        = (df_core["ocu500"]  == 1).astype(float)
df_core["formal_salaried"] = ((df_core["ocupinf"] == 2) & df_core["p507"].isin([1.0, 2.0])).astype(float)

# log_wage — only for employed formal with positive wage
wage_valid = (df_core["ocu500"] == 1) & (df_core["ocupinf"] == 2) & (df_core["wage_monthly"] > 0)
df_core["log_wage"] = np.nan
df_core.loc[wage_valid, "log_wage"] = np.log(df_core.loc[wage_valid, "wage_monthly"])

# below_mw — formal employed with wage below MW of that year
mw_year_map = df_core["year"].map(MW_BY_YEAR)
df_core["below_mw"] = np.nan
df_core.loc[wage_valid, "below_mw"] = (
    df_core.loc[wage_valid, "wage_monthly"] < mw_year_map[wage_valid]
).astype(float)

# ── Weighted quantile helper ───────────────────────────────────────────────────
def weighted_median(values, weights):
    values  = np.asarray(values,  dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask    = np.isfinite(values) & np.isfinite(weights) & (weights > 0) & (values > 0)
    if mask.sum() < 5:
        return np.nan
    v, w = values[mask], weights[mask]
    sorter = np.argsort(v)
    v, w   = v[sorter], w[sorter]
    cumw   = np.cumsum(w)
    idx    = np.searchsorted(cumw, 0.5 * cumw[-1])
    return float(v[min(idx, len(v)-1)])

# ── Compute Kaitz_pre from 2021 ────────────────────────────────────────────────
print("\nComputing pre-policy (2021) Kaitz by department …")
MW_PRE = 930  # MW in 2021

df_2021 = df_core[
    (df_core["year"] == 2021) &
    (df_core["ocu500"] == 1) &
    (df_core["ocupinf"] == 2) &
    (df_core["wage_monthly"] > 0)
].copy()

kaitz_pre = {}
dept_stats = []

for dept in sorted(DEPT_NAMES.keys()):
    sub  = df_2021[df_2021["dept"] == dept]
    if len(sub) < 10:
        print(f"  {dept} {DEPT_NAMES[dept]:18s}: TOO FEW OBS ({len(sub)})")
        continue

    med = weighted_median(sub["wage_monthly"].values, sub["fac500a"].values)
    if np.isnan(med) or med <= 0:
        continue

    # Kaitz = MW_pre / median_formal_wage_pre
    kaitz = MW_PRE / med

    # Bindingness: share with wage in [0.9×930, 1.1×1025] in 2021
    w_all     = pd.to_numeric(sub["fac500a"], errors="coerce").fillna(0).values
    wage_all  = sub["wage_monthly"].values
    at_risk   = ((wage_all >= 0.9 * MW_PRE) & (wage_all <= 1.1 * 1025)).astype(float)
    share_at_risk = float(np.average(at_risk, weights=w_all))

    # Share below new MW (1025) in 2021
    below_new = (wage_all < 1025).astype(float)
    share_below_new = float(np.average(below_new, weights=w_all))

    n_weighted = float(sub["fac500a"].sum())

    kaitz_pre[dept] = {
        "dept_code":         dept,
        "dept_name":         DEPT_NAMES[dept],
        "median_formal_2021": round(med, 2),
        "kaitz_pre":          round(kaitz, 4),
        "share_at_risk":      round(share_at_risk, 4),
        "share_below_new_mw": round(share_below_new, 4),
        "n_formal_2021":      round(n_weighted),
    }
    dept_stats.append(kaitz_pre[dept])
    print(f"  {dept} {DEPT_NAMES[dept]:18s}: med={med:>7.0f}  kaitz={kaitz:.3f}  "
          f"at_risk={share_at_risk:.1%}  below_new={share_below_new:.1%}")

# Save Kaitz JSON
with open(KAITZ_OUT, "w", encoding="utf-8") as f:
    json.dump({
        "metadata": {
            "mw_pre":  MW_PRE,
            "mw_post": 1025,
            "year_pre": 2021,
            "source":  "ENAHO Panel 2020-2024 (srienaho-978, Mod-1477)",
            "generated": "2026-03-15",
        },
        "departments": kaitz_pre,
    }, f, ensure_ascii=False, indent=2)
print(f"\nWrote {KAITZ_OUT}")

# ── Merge Kaitz_pre into panel ─────────────────────────────────────────────────
print("\nMerging Kaitz_pre into panel …")
kaitz_df = pd.DataFrame(dept_stats)[["dept_code","kaitz_pre","share_at_risk","share_below_new_mw","median_formal_2021"]]
kaitz_df.rename(columns={"dept_code": "dept"}, inplace=True)

df_core = df_core.merge(kaitz_df, on="dept", how="left")
missing_kaitz = df_core["kaitz_pre"].isna().sum()
print(f"  Observations with Kaitz_pre: {(~df_core['kaitz_pre'].isna()).sum():,}  "
      f"(missing: {missing_kaitz:,})")

# ── Policy timing indicators ───────────────────────────────────────────────────
df_core["post"]      = (df_core["year"] == 2023).astype(int)
df_core["trans"]     = (df_core["year"] == 2022).astype(int)  # transition year
df_core["post_2022"] = (df_core["year"] == 2022).astype(int)
df_core["post_2023"] = (df_core["year"] == 2023).astype(int)
df_core["mw"]        = df_core["year"].map(MW_BY_YEAR)

# ── Save panel ─────────────────────────────────────────────────────────────────
keep_cols = [
    "numper", "year", "dept", "ubigeo",
    "ocu500", "ocupinf", "i524a1", "wage_monthly", "fac500a",
    "p207", "p208a", "p507",
    "employed", "formal", "informal", "formal_salaried",
    "log_wage", "below_mw",
    "kaitz_pre", "share_at_risk", "share_below_new_mw", "median_formal_2021",
    "post", "trans", "post_2022", "post_2023", "mw",
]
df_panel = df_core[keep_cols].copy()
df_panel.to_parquet(PANEL_OUT, index=False)
print(f"\nWrote {PANEL_OUT}  ({PANEL_OUT.stat().st_size/1e6:.1f} MB)")

# ── DIAGNOSTICS ───────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)

# N by year
print("\n--- N by year (all working-age) ---")
print(df_panel.groupby("year").agg(
    N=("numper","count"),
    N_employed=("employed","sum"),
    N_formal=("formal","sum"),
    wgt_sum=("fac500a","sum"),
).round(0).to_string())

# Mean outcomes by year (weighted)
print("\n--- Weighted mean outcomes by year ---")
for yr in YEARS:
    sub = df_panel[df_panel["year"]==yr].copy()
    w   = sub["fac500a"].fillna(0).values
    if w.sum() == 0:
        continue
    emp_rate    = np.average(sub["employed"].fillna(0).values,         weights=w)
    formal_rate = np.average(sub["formal"].fillna(0).values,           weights=w)
    sub_formal  = sub[(sub["ocu500"]==1) & (sub["ocupinf"]==2) & (sub["wage_monthly"]>0)]
    wf          = sub_formal["fac500a"].fillna(0).values
    med_w       = weighted_median(sub_formal["wage_monthly"].values, wf) if len(sub_formal) > 10 else np.nan
    below_mw_r  = float(np.average(sub_formal["below_mw"].fillna(0).values, weights=wf)) if len(sub_formal) > 10 else np.nan
    print(f"  {yr}: employed={emp_rate:.3f}  formal={formal_rate:.3f}  "
          f"med_formal_wage={med_w:.0f}  below_mw={below_mw_r:.3f}")

# Kaitz_pre by department (sorted)
print("\n--- Kaitz_pre by department (sorted low to high) ---")
print(f"{'Dept':4} {'Name':20} {'Med2021':>8} {'Kaitz':>7} {'AtRisk':>8} {'BelowNew':>10}")
print("-" * 65)
for r in sorted(dept_stats, key=lambda x: x["kaitz_pre"]):
    print(f"{r['dept_code']:4} {r['dept_name']:20} {r['median_formal_2021']:>8.0f} "
          f"{r['kaitz_pre']:>7.3f} {r['share_at_risk']:>8.1%} {r['share_below_new_mw']:>10.1%}")

print("\n" + "="*70)
print("DONE. Dataset ready for estimation.")
print(f"  Panel: {PANEL_OUT}")
print(f"  Kaitz: {KAITZ_OUT}")
