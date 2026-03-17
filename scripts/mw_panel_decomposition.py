"""
Priority 1: ENAHO Panel 978 — Near-MW Decomposition (Event C)
Chunked loading of wide-format panel. Treatment = formal-dep near MW in 2021.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import json
from datetime import datetime

PANEL_PATH = 'D:/Nexus/nexus/data/raw/enaho/978_panel/enaho01a-2020-2024-500-panel.csv'
MW_OLD = 930   # Event C pre-MW (2021)
MW_NEW = 1025  # Event C new MW (2023)

# ── Step 1: Select only needed columns ────────────────────────────────────────
print("Step 1: Detecting columns...")
all_cols = pd.read_csv(PANEL_PATH, nrows=0, encoding='latin-1').columns.tolist()
all_cols_l = [c.lower() for c in all_cols]

targets = {
    'ocu500':     ['ocu500_21', 'ocu500_23'],
    'p507':       ['p507_21', 'p507_23'],         # employment category (dep=3,4,6)
    'cat07p500a1':['cat07p500a1_21','cat07p500a1_23'],  # alt dep var
    'p524a1':     ['p524a1_21', 'p524a1_23'],     # monthly cash wage
    'i524a1':     ['i524a1_21', 'i524a1_23'],     # annual income (fallback)
    'ocupinf':    ['ocupinf_21', 'ocupinf_23'],   # formality
    'p513t':      ['p513t_21', 'p513t_23'],       # weekly hours
    'p208a':      ['p208a_21', 'p208a_23'],       # age
    'p207':       ['p207_21', 'p207_23'],         # sex
    'fac500a':    ['fac500a_21', 'fac500a_23'],   # annual weight
    'facpanel':   ['facpanel2123'],               # panel weight 2021-2023
    'ubigeo':     ['ubigeo_21'],
}

keep = set()
for base_list in targets.values():
    for t in base_list:
        # find exact match in all_cols_l
        matches = [c for c, cl in zip(all_cols, all_cols_l) if cl == t]
        keep.update(matches)

# Also add dept identifier from ubigeo
dept_cols = [c for c, cl in zip(all_cols, all_cols_l) if 'ubigeo_21' in cl]
keep.update(dept_cols)

keep = list(keep)
print(f"Loading {len(keep)} columns: {sorted([c.lower() for c in keep])[:20]}")

# ── Step 2: Load filtered columns ─────────────────────────────────────────────
print("\nStep 2: Loading data...")
df = pd.read_csv(PANEL_PATH, usecols=keep, encoding='latin-1', low_memory=False)
df.columns = [c.lower() for c in df.columns]
print(f"Shape: {df.shape}  Memory: {df.memory_usage(deep=True).sum()/1e6:.0f} MB")

# ── Step 3: Build variables ────────────────────────────────────────────────────
print("\nStep 3: Building sample...")

def to_num(s, col):
    if col in s.columns:
        return pd.to_numeric(s[col], errors='coerce')
    return pd.Series(np.nan, index=s.index)

# Employment (employed = ocu500==1)
emp21 = to_num(df, 'ocu500_21') == 1
emp23 = to_num(df, 'ocu500_23') == 1

# Dependent employment: p507 values 3(empleado),4(obrero),6(trab.hogar)
# OR cat07p500a1 == 2
p507_21 = to_num(df, 'p507_21')
p507_23 = to_num(df, 'p507_23')
dep21 = p507_21.isin([3, 4, 6])
dep23 = p507_23.isin([3, 4, 6])
if dep21.sum() < 1000:  # fallback
    cat_21 = to_num(df, 'cat07p500a1_21')
    cat_23 = to_num(df, 'cat07p500a1_23')
    dep21 = (cat_21 == 2)
    dep23 = (cat_23 == 2)
    print(f"  Using cat07p500a1 for dep (p507 had too few): dep21={dep21.sum():,}")

# Formality (ocupinf==2)
inf21 = to_num(df, 'ocupinf_21') == 2
inf23 = to_num(df, 'ocupinf_23') == 2

# Wages (p524a1 priority, fallback i524a1/12)
w21_p = to_num(df, 'p524a1_21')
w21_i = to_num(df, 'i524a1_21') / 12.0
w21 = w21_p.where(w21_p > 0, w21_i)

w23_p = to_num(df, 'p524a1_23')
w23_i = to_num(df, 'i524a1_23') / 12.0
w23 = w23_p.where(w23_p > 0, w23_i)

# Hours
h21 = to_num(df, 'p513t_21')
h23 = to_num(df, 'p513t_23')

# Weight (panel weight preferred)
wt_panel = to_num(df, 'facpanel2123')
wt_21    = to_num(df, 'fac500a_21')
wt = wt_panel.where(wt_panel > 0, wt_21)

# Matched = actually observed in BOTH 2021 and 2023 waves (facpanel2123 > 0)
# People with facpanel2123 == NaN or 0 were NOT re-interviewed in 2023
# We must NOT count them as "unemployed" — they are panel attrition
matched_23 = (wt_panel > 0) & wt_panel.notna()
# Alternative: ocu500_23 is non-null (observed in 2023 regardless of employment)
observed_23 = to_num(df, 'ocu500_23').notna()
in_panel_23 = matched_23 | observed_23  # either weight or response present

# ── Step 4: Feasibility counts ─────────────────────────────────────────────────
formal_dep_21    = emp21 & dep21 & inf21 & (w21 > 0)
formal_dep_23    = emp23 & dep23 & inf23 & (w23 > 0)
both_years       = formal_dep_21 & in_panel_23  # present in 2021 AND observed in 2023

treat_lo = 0.85 * MW_OLD   # 790.5
ctrl_lo  = 1.2 * MW_NEW    # 1230
ctrl_hi  = 2.5 * MW_NEW    # 2562.5

treatment = formal_dep_21 & (w21 >= treat_lo) & (w21 < MW_NEW)
control   = formal_dep_21 & (w21 >= ctrl_lo)  & (w21 <= ctrl_hi)

print(f"\n=== PANEL FEASIBILITY (Event C, 2021→2023) ===")
print(f"Total panel rows:              {len(df):,}")
print(f"Employed in 2021:              {emp21.sum():,}")
print(f"Formal dep employed 2021:      {formal_dep_21.sum():,}")
print(f"Formal dep employed 2023:      {formal_dep_23.sum():,}")
print(f"With data in both years:       {both_years.sum():,}")
print(f"")
print(f"TREATMENT (near MW, 2021):     {treatment.sum():,}   "
      f"wage=[S/{treat_lo:.0f}, S/{MW_NEW})")
print(f"CONTROL (high wage, 2021):     {control.sum():,}   "
      f"wage=[S/{ctrl_lo:.0f}, S/{ctrl_hi:.0f}]")

n_treat = int(treatment.sum())
n_ctrl  = int(control.sum())
THRESHOLD = 200

if n_treat < THRESHOLD:
    print(f"\nPanel infeasible: only {n_treat} treatment workers (need >= {THRESHOLD}).")
    print("Panel 978 cannot support near-MW decomposition for Event C.")
    results = {
        'feasible': False,
        'n_treatment': n_treat,
        'n_control': n_ctrl,
        'n_formal_dep_2021': int(formal_dep_21.sum()),
        'n_both_years': int(both_years.sum()),
        'threshold': THRESHOLD,
        'note': f'Insufficient treatment workers ({n_treat} < {THRESHOLD}). Panel 25% annual retention means ~17% survive 2021→2023 gap.'
    }
else:
    print(f"\nPanel feasible ({n_treat} >= {THRESHOLD}). Running decomposition...")
    # ── Step 5: Near-MW decomposition ─────────────────────────────────────────
    for group_name, mask_21 in [('Treatment', treatment), ('Control', control)]:
        sub = df[mask_21].copy()
        sub_w21 = w21[mask_21]
        sub_w23 = w23[mask_21]
        sub_h21 = h21[mask_21]
        sub_h23 = h23[mask_21]
        sub_emp23   = emp23[mask_21]
        sub_dep23   = dep23[mask_21]
        sub_inf23   = inf23[mask_21]
        sub_wt      = wt[mask_21]

        n = mask_21.sum()
        # Employment retention
        emp_ret = float(sub_emp23.mean())
        # Formal retention (conditional on employed)
        form_ret = float((sub_inf23 & sub_dep23 & sub_emp23).sum() / n) if n > 0 else np.nan
        # Wage change (stayers: employed and formal in both)
        stayer = sub_emp23 & sub_dep23 & sub_inf23 & (sub_w23 > 0)
        if stayer.sum() > 10:
            dlogw = float(np.log(sub_w23[stayer]).mean() - np.log(sub_w21[stayer]).mean())
        else:
            dlogw = np.nan
        # Hours change
        hrs_ok = stayer & h21[mask_21].notna() & h23[mask_21].notna() & (h21[mask_21] > 0)
        if hrs_ok.sum() > 5:
            dh = float(h23[mask_21][hrs_ok].mean() - h21[mask_21][hrs_ok].mean())
        else:
            dh = np.nan
        print(f"\n  {group_name} (N={n:,}):")
        print(f"    Employment retention:  {emp_ret:.1%}")
        print(f"    Formal retention:      {form_ret:.1%}")
        print(f"    Δ log wage (stayers):  {dlogw:+.3f}") if dlogw else print(f"    Δ log wage (stayers):  —")
        print(f"    Δ hours (stayers):     {dh:+.1f}h/wk") if dh else print(f"    Δ hours (stayers):     —")

    # Transition matrix (treatment)
    treat_sub = df[treatment]
    te23 = emp23[treatment]; td23 = dep23[treatment]; ti23 = inf23[treatment]
    n_t = treatment.sum()
    n_formal23    = int((te23 & td23 & ti23).sum())
    n_informal23  = int((te23 & td23 & ~ti23).sum())
    n_unemp23     = int((~te23 & te23.notna()).sum())
    n_olf23       = int(n_t - n_formal23 - n_informal23 - n_unemp23)

    print(f"\n  Transition matrix (Treatment, N={n_t:,}):")
    print(f"    Formal 2021 → Formal dep 2023:    {n_formal23:4d} ({n_formal23/n_t:.1%})")
    print(f"    Formal 2021 → Informal dep 2023:  {n_informal23:4d} ({n_informal23/n_t:.1%})")
    print(f"    Formal 2021 → Unemployed 2023:    {n_unemp23:4d} ({n_unemp23/n_t:.1%})")
    print(f"    Formal 2021 → OLF/other 2023:     {n_olf23:4d} ({n_olf23/n_t:.1%})")

    results = {'feasible': True, 'n_treatment': n_treat, 'n_control': n_ctrl}

# ── Save ──────────────────────────────────────────────────────────────────────
output = {
    'metadata': {'generated': datetime.now().isoformat(),
                 'description': 'Panel 978 feasibility and near-MW decomposition (Event C)'},
    'results': results,
}
with open('D:/Nexus/nexus/exports/data/mw_panel_decomposition.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print("\nSaved: exports/data/mw_panel_decomposition.json")
