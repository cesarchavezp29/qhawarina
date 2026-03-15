"""
build_mw_v4_formality_panel.py
================================
Constructs the V4 formality definition and rebuilds the DiD estimation panel.

V4 algorithm (validated vs INEI ocupinf_23: 90.5% accuracy, +0.4pp gap):

  ASALARIADOS (p507 in [3, 4, 6]):
    formal = 1 if p419a1==1 OR p510 in [1,2,3,5] OR p512a > 1

  EMPLEADORES (p507 == 1):
    formal = 1 if p511a==1 OR p512a > 1 OR p517==1

  INDEPENDIENTES (p507 == 2):
    formal = 1 if p511a==1 OR p517==1

  TFNR (p507 == 5) and OTRO (p507 == 7):
    formal = 0 (always informal)

Sources:
  Employment: D:/Nexus/nexus/data/raw/enaho/978_panel/enaho01a-2020-2024-500-panel.csv
  Health:     D:/Nexus/nexus/data/raw/enaho/978_panel/enaho01a-2020-2024-400-panel.dta
              (only p419a1_{yy} columns needed)

Output:
  D:/Nexus/nexus/data/processed/mw_regional_did_panel.parquet  (overwritten)
  D:/Nexus/nexus/exports/data/mw_pre_policy_kaitz.json          (overwritten)
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
RAW     = ROOT / "data/raw/enaho/978_panel"
EMP_CSV = RAW / "enaho01a-2020-2024-500-panel.csv"
HTH_DTA = RAW / "enaho01a-2020-2024-400-panel.dta"
# Fallback: use SAV if CSV not yet copied
EMP_SAV = RAW / "Modulo1477_Empleo_Ingresos/978-Modulo1477/enaho01a-2020-2024-500-panel.sav"

OUT_DIR    = ROOT / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_OUT  = OUT_DIR / "mw_regional_did_panel.parquet"
KAITZ_OUT  = ROOT / "exports/data/mw_pre_policy_kaitz.json"

# ── Constants ─────────────────────────────────────────────────────────────────
MW_BY_YEAR = {2020: 930, 2021: 930, 2022: 977.5, 2023: 1025, 2024: 1025}
YEARS_CORE = [2021, 2022, 2023]
ALL_SFX    = ["20", "21", "22", "23", "24"]

DEPT_NAMES = {
    "01": "Amazonas",       "02": "Ancash",         "03": "Apurimac",
    "04": "Arequipa",       "05": "Ayacucho",        "06": "Cajamarca",
    "07": "Callao",         "08": "Cusco",           "09": "Huancavelica",
    "10": "Huanuco",        "11": "Ica",             "12": "Junin",
    "13": "La Libertad",    "14": "Lambayeque",      "15": "Lima",
    "16": "Loreto",         "17": "Madre de Dios",   "18": "Moquegua",
    "19": "Pasco",          "20": "Piura",           "21": "Puno",
    "22": "San Martin",     "23": "Tacna",           "24": "Tumbes",
    "25": "Ucayali",
}

# ── Step 1: Load employment data ───────────────────────────────────────────────
EMP_STEMS = [
    "ubigeo", "ocu500", "ocupinf", "i524a1", "fac500a",
    "p207", "p208a",
    "p507",                    # occupation category → determines V4 branch
    "p510",                    # contract type (asalariados)
    "p511a",                   # SUNAT registration (empleadores / independientes)
    "p512a",                   # firm size
    "p517",                    # business registration
]

emp_cols = ["numper"]
for stem in EMP_STEMS:
    for sfx in ALL_SFX:
        cand = f"{stem}_{sfx}"
        emp_cols.append(cand)

print("Step 1: Loading employment data …")

if EMP_CSV.exists():
    print(f"  Using CSV: {EMP_CSV}")
    # Read only needed columns — pandas will scan all cols in CSV but usecols is still faster
    # Get actual header first to resolve which cols exist
    # Detect encoding from first bytes
    with open(EMP_CSV, "rb") as fb:
        raw = fb.read(4096)
    enc = "latin-1" if b"\xd1" in raw or b"\xc3" in raw else "utf-8"
    print(f"  Detected encoding: {enc}")

    with open(EMP_CSV, encoding=enc, errors="replace") as f:
        hdr = f.readline().strip()

    # Build actual usecols with original case from header
    header_lower = [c.strip().lower() for c in hdr.split(",")]
    use_actual   = []
    for c in emp_cols:
        if c in header_lower:
            idx = header_lower.index(c)
            use_actual.append(hdr.split(",")[idx].strip())

    print(f"  Requesting {len(use_actual)} cols from CSV …", flush=True)
    emp = pd.read_csv(EMP_CSV, usecols=use_actual, low_memory=False,
                      encoding=enc, on_bad_lines="skip")
    emp.columns = [c.strip().lower() for c in emp.columns]
else:
    print(f"  CSV not found, falling back to SAV: {EMP_SAV}")
    import pyreadstat
    # Only request columns that exist in SAV
    df_tmp, meta_tmp = pyreadstat.read_sav(str(EMP_SAV), row_limit=0)
    sav_cols = [c.lower() for c in meta_tmp.column_names]
    use_sav  = [c for c in emp_cols if c in sav_cols]
    emp, _   = pyreadstat.read_sav(str(EMP_SAV), apply_value_formats=False, usecols=use_sav)
    emp.columns = [c.lower() for c in emp.columns]

print(f"  Employment shape: {emp.shape}")
print(f"  numper unique: {emp['numper'].nunique():,}")

# ── Step 2: Load health module (p419a1) ───────────────────────────────────────
print("\nStep 2: Loading health module (p419a1) …")

p419_cols = ["numper"] + [f"p419a1_{sfx}" for sfx in ALL_SFX]

if HTH_DTA.exists():
    print(f"  Loading DTA: {HTH_DTA}  ({HTH_DTA.stat().st_size/1e9:.1f} GB)")
    import pyreadstat
    # Read only p419a1 columns — much faster than full file
    hth_tmp, hth_meta = pyreadstat.read_dta(str(HTH_DTA), row_limit=0)
    dta_cols = [c.lower() for c in hth_meta.column_names]
    # Check which p419a1 columns exist
    p419_exist = [c for c in p419_cols if c in dta_cols]
    missing    = [c for c in p419_cols if c not in dta_cols]
    if missing:
        print(f"  WARNING — missing in DTA: {missing}")

    print(f"  Reading {len(p419_exist)} columns …", flush=True)
    hth, _ = pyreadstat.read_dta(str(HTH_DTA), apply_value_formats=False,
                                  usecols=[c for c in hth_meta.column_names
                                           if c.lower() in p419_exist])
    hth.columns = [c.lower() for c in hth.columns]
    # Add missing p419a1 cols as NaN
    for c in p419_cols:
        if c not in hth.columns:
            hth[c] = np.nan
    print(f"  Health shape: {hth.shape}")
else:
    print(f"  DTA not found at {HTH_DTA}")
    print("  Proceeding WITHOUT p419a1 (will be NaN). Formality estimate will be lower bound.")
    hth = pd.DataFrame({"numper": emp["numper"].unique()})
    for sfx in ALL_SFX:
        hth[f"p419a1_{sfx}"] = np.nan

# ── Step 3: Merge health into employment ──────────────────────────────────────
print("\nStep 3: Merging health module into employment data …")
merged = emp.merge(hth[p419_cols], on="numper", how="left")
print(f"  Merged shape: {merged.shape}")
p419_fill = {f"p419a1_{sfx}": 0 for sfx in ALL_SFX}
merged.fillna(p419_fill, inplace=True)

# ── Step 4: V4 formality construction ─────────────────────────────────────────
print("\nStep 4: Constructing V4 formality per year …")

def to_num(df, col):
    """Return numeric series from col, NaN if missing."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)

for sfx in ALL_SFX:
    p507   = to_num(merged, f"p507_{sfx}")
    p419a1 = to_num(merged, f"p419a1_{sfx}")
    p510   = to_num(merged, f"p510_{sfx}")
    p511a  = to_num(merged, f"p511a_{sfx}")
    p512a  = to_num(merged, f"p512a_{sfx}")
    p517   = to_num(merged, f"p517_{sfx}")

    # Branch masks
    asalariados   = p507.isin([3.0, 4.0, 6.0])
    empleadores   = p507 == 1.0
    independientes= p507 == 2.0
    tfnr_otro     = p507.isin([5.0, 7.0])

    # Formality conditions per branch
    f_asal = (p419a1 == 1) | p510.isin([1.0, 2.0, 3.0, 5.0]) | (p512a > 1)
    f_empl = (p511a == 1) | (p512a > 1) | (p517 == 1)
    f_ind  = (p511a == 1) | (p517 == 1)

    formal_v4 = pd.Series(np.nan, index=merged.index)
    formal_v4[asalariados]    = f_asal[asalariados].astype(float)
    formal_v4[empleadores]    = f_empl[empleadores].astype(float)
    formal_v4[independientes] = f_ind[independientes].astype(float)
    formal_v4[tfnr_otro]      = 0.0

    merged[f"formal_v4_{sfx}"] = formal_v4

    # INEI's original (exists 20-23, NaN for 24)
    ocupinf_col = f"ocupinf_{sfx}"
    if ocupinf_col in merged.columns:
        merged[f"ocupinf_inei_{sfx}"] = (to_num(merged, ocupinf_col) == 2).astype(float)
        merged.loc[to_num(merged, ocupinf_col).isna(), f"ocupinf_inei_{sfx}"] = np.nan
    else:
        merged[f"ocupinf_inei_{sfx}"] = np.nan

# ── Step 5: Validation ────────────────────────────────────────────────────────
print("\nStep 5: Validation — V4 vs INEI ocupinf (years 2020-2023)")
print(f"{'Year':>6} {'N_emp':>8} {'V4_rate':>9} {'INEI_rate':>10} {'Diff_pp':>8} {'Accuracy':>10}")
print("-" * 58)

for sfx, yr in [("20",2020),("21",2021),("22",2022),("23",2023)]:
    ocu = to_num(merged, f"ocu500_{sfx}")
    fac = to_num(merged, f"fac500a_{sfx}")
    v4  = merged[f"formal_v4_{sfx}"]
    ini = merged[f"ocupinf_inei_{sfx}"]

    emp_mask = (ocu == 1) & fac.notna() & v4.notna() & ini.notna()
    if emp_mask.sum() < 10:
        print(f"  {yr}: too few obs")
        continue

    w     = fac[emp_mask].fillna(0)
    v4_r  = float(np.average(v4[emp_mask].fillna(0),  weights=w))
    ini_r = float(np.average(ini[emp_mask].fillna(0), weights=w))
    diff  = v4_r - ini_r

    # Accuracy: per-worker agreement
    agree = (v4[emp_mask] == ini[emp_mask]).sum()
    acc   = agree / emp_mask.sum()

    print(f"  {yr}: {emp_mask.sum():>8,} {v4_r:>9.3f} {ini_r:>10.3f} {diff*100:>+7.2f}pp {acc:>9.1%}")

# 2024 — no INEI benchmark
sfx, yr = "24", 2024
ocu = to_num(merged, f"ocu500_{sfx}")
fac = to_num(merged, f"fac500a_{sfx}")
v4  = merged[f"formal_v4_{sfx}"]
emp_mask = (ocu == 1) & fac.notna() & v4.notna()
if emp_mask.sum() > 0:
    w    = fac[emp_mask].fillna(0)
    v4_r = float(np.average(v4[emp_mask].fillna(0), weights=w))
    print(f"  {yr}: {emp_mask.sum():>8,} {v4_r:>9.3f}   (no INEI benchmark)")

# ── Step 6: Melt to long format ───────────────────────────────────────────────
print("\nStep 6: Melting to long format (years 2021-2023) …")

MW_PRE = 930

frames = []
yr_sfx = {2020:"20", 2021:"21", 2022:"22", 2023:"23", 2024:"24"}

for year in [2021, 2022, 2023]:   # core DiD window
    sfx = yr_sfx[year]

    def col(stem):
        c = f"{stem}_{sfx}"
        return c if c in merged.columns else None

    sub = merged[["numper"]].copy()
    sub["year"] = year

    for out, stem in [
        ("ubigeo","ubigeo"), ("ocu500","ocu500"), ("i524a1","i524a1"),
        ("fac500a","fac500a"), ("p207","p207"), ("p208a","p208a"), ("p507","p507"),
        ("p510","p510"), ("p511a","p511a"), ("p512a","p512a"), ("p517","p517"),
        ("p419a1","p419a1"),
    ]:
        c = col(stem)
        sub[out] = pd.to_numeric(merged[c], errors="coerce") if c else np.nan

    sub["ubigeo"] = merged[f"ubigeo_{sfx}"].astype(str).str.strip().str.zfill(6)
    sub["dept"]   = sub["ubigeo"].str[:2]
    sub["formal_v4"]     = merged[f"formal_v4_{sfx}"].values
    sub["ocupinf_inei"]  = merged[f"ocupinf_inei_{sfx}"].values

    frames.append(sub)

df = pd.concat(frames, ignore_index=True)
print(f"  Long shape: {df.shape}")

# Working-age filter
df = df[(df["p208a"] >= 14) & (df["p208a"] <= 65)].copy()
print(f"  Working-age (14-65): {df.shape[0]:,}")

# Outcomes
df["wage_monthly"] = df["i524a1"] / 12.0
df["employed"]     = (df["ocu500"] == 1).astype(float)
df["informal"]     = (df["formal_v4"] == 0) & (df["ocu500"] == 1)

wage_valid = (df["ocu500"] == 1) & (df["formal_v4"] == 1) & (df["wage_monthly"] > 0)
df["log_wage"] = np.nan
df.loc[wage_valid, "log_wage"] = np.log(df.loc[wage_valid, "wage_monthly"])

mw_map = df["year"].map(MW_BY_YEAR)
df["below_mw"] = np.nan
df.loc[wage_valid, "below_mw"] = (df.loc[wage_valid, "wage_monthly"] < mw_map[wage_valid]).astype(float)

df["mw"] = mw_map

# ── Step 7: Compute Kaitz_pre (2021, V4 formal) ───────────────────────────────
print("\nStep 7: Computing Kaitz_pre from 2021 (V4 formal) …")

def weighted_median(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0) & (v > 0)
    if m.sum() < 5:
        return np.nan
    v, w = v[m], w[m]
    s    = np.argsort(v)
    v, w = v[s], w[s]
    cw   = np.cumsum(w)
    idx  = np.searchsorted(cw, 0.5 * cw[-1])
    return float(v[min(idx, len(v)-1)])

df_2021 = df[
    (df["year"] == 2021) &
    (df["ocu500"] == 1) &
    (df["formal_v4"] == 1) &
    (df["wage_monthly"] > 0)
].copy()

kaitz_pre = {}
dept_stats = []

for dept in sorted(DEPT_NAMES.keys()):
    sub = df_2021[df_2021["dept"] == dept]
    if len(sub) < 5:
        continue
    med = weighted_median(sub["wage_monthly"].values, sub["fac500a"].values)
    if np.isnan(med) or med <= 0:
        continue

    kaitz = MW_PRE / med
    w_all = sub["fac500a"].fillna(0).values
    wg    = sub["wage_monthly"].values

    share_at_risk   = float(np.average(((wg >= 0.9*MW_PRE) & (wg <= 1.1*1025)).astype(float), weights=w_all))
    share_below_new = float(np.average((wg < 1025).astype(float), weights=w_all))

    rec = {
        "dept_code":          dept,
        "dept_name":          DEPT_NAMES[dept],
        "median_formal_2021": round(med, 2),
        "kaitz_pre":          round(kaitz, 4),
        "share_at_risk":      round(share_at_risk, 4),
        "share_below_new_mw": round(share_below_new, 4),
        "n_formal_2021":      round(float(sub["fac500a"].sum())),
    }
    kaitz_pre[dept] = rec
    dept_stats.append(rec)

# Save Kaitz JSON
with open(KAITZ_OUT, "w", encoding="utf-8") as f:
    json.dump({
        "metadata": {
            "mw_pre": MW_PRE, "mw_post": 1025,
            "year_pre": 2021,
            "formality": "V4 algorithm (p419a1, p510, p511a, p512a, p517)",
            "source": "ENAHO Panel 2020-2024 (srienaho-978, Mod-1477 + Mod-400)",
            "generated": "2026-03-15",
        },
        "departments": kaitz_pre,
    }, f, ensure_ascii=False, indent=2)

# ── Merge Kaitz_pre ───────────────────────────────────────────────────────────
kaitz_df = pd.DataFrame(dept_stats)[["dept_code","kaitz_pre","share_at_risk","share_below_new_mw","median_formal_2021"]]
kaitz_df.rename(columns={"dept_code":"dept"}, inplace=True)
df = df.merge(kaitz_df, on="dept", how="left")

# DiD time indicators
df["post"]      = (df["year"] == 2023).astype(int)
df["post_2022"] = (df["year"] == 2022).astype(int)
df["post_2023"] = (df["year"] == 2023).astype(int)

# ── Save ──────────────────────────────────────────────────────────────────────
keep = [
    "numper", "year", "dept", "ubigeo",
    "ocu500", "i524a1", "wage_monthly", "fac500a",
    "p207", "p208a", "p507", "p510", "p511a", "p512a", "p517", "p419a1",
    "employed", "formal_v4", "ocupinf_inei", "informal", "log_wage", "below_mw",
    "kaitz_pre", "share_at_risk", "share_below_new_mw", "median_formal_2021",
    "post", "post_2022", "post_2023", "mw",
]
df[keep].to_parquet(PANEL_OUT, index=False)
print(f"\nWrote {PANEL_OUT}  ({PANEL_OUT.stat().st_size/1e6:.1f} MB)")

# ── DIAGNOSTICS ───────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)

print("\n--- Weighted mean outcomes by year ---")
for yr in [2021, 2022, 2023]:
    sub = df[df["year"] == yr].copy()
    w   = sub["fac500a"].fillna(0).values
    if w.sum() == 0:
        continue
    emp_r  = float(np.average(sub["employed"].fillna(0).values,   weights=w))
    v4_r   = float(np.average(sub["formal_v4"].fillna(0).values,  weights=w))
    ini_r  = float(np.average(sub["ocupinf_inei"].fillna(0).values, weights=w))
    sub_fw = sub[sub["formal_v4"]==1]
    wf     = sub_fw["fac500a"].fillna(0).values
    med_w  = weighted_median(sub_fw["wage_monthly"].values, wf) if len(sub_fw)>10 else np.nan
    below  = float(np.average(sub_fw["below_mw"].fillna(0).values, weights=wf)) if len(sub_fw)>10 else np.nan
    print(f"  {yr}: emp={emp_r:.3f}  formal_v4={v4_r:.3f}  inei={ini_r:.3f}  "
          f"med_wage_formal={med_w:.0f}  below_mw={below:.3f}")

print("\n--- Kaitz_pre by department (V4, 2021) ---")
print(f"{'Dept':4} {'Name':20} {'Med21':>7} {'Kaitz':>7} {'AtRisk':>8} {'BelowNew':>10}")
print("-" * 63)
for r in sorted(dept_stats, key=lambda x: x["kaitz_pre"]):
    print(f"{r['dept_code']:4} {r['dept_name']:20} {r['median_formal_2021']:>7.0f} "
          f"{r['kaitz_pre']:>7.3f} {r['share_at_risk']:>8.1%} {r['share_below_new_mw']:>10.1%}")

print("\n" + "="*70)
print("DONE. Panel ready for estimation.")
print(f"  Panel: {PANEL_OUT}")
print(f"  Kaitz: {KAITZ_OUT}")
