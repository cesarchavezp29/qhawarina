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
        matches = [c for c, cl in zip(all_cols, all_cols_l) if cl == t]
        keep.update(matches)

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

# Raw ocu500 series (keep NaN distinct from 0/1 — used for attrition detection)
ocu500_21_raw = to_num(df, 'ocu500_21')
ocu500_23_raw = to_num(df, 'ocu500_23')

# Employment (employed = ocu500==1)
emp21 = ocu500_21_raw == 1
emp23 = ocu500_23_raw == 1   # NaN → False (not-employed); separated below

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

# Weights
# facpanel2123: INEI panel weight for 2021→2023 matched individuals (> 0 = re-interviewed)
# Fallback: fac500a_21 for 2021-only sample stats
wt_panel = to_num(df, 'facpanel2123')
wt_21    = to_num(df, 'fac500a_21')
# For panel outcomes, use facpanel2123; for 2021-only counts use fac500a_21
# Re-interviewed = facpanel2123 > 0 OR ocu500_23 is not NaN
re_interviewed_global = (wt_panel > 0) & wt_panel.notna()
observed_23_global    = ocu500_23_raw.notna()
in_panel_23 = re_interviewed_global | observed_23_global

# ── Step 4: Feasibility counts ─────────────────────────────────────────────────
formal_dep_21    = emp21 & dep21 & inf21 & (w21 > 0)
formal_dep_23    = emp23 & dep23 & inf23 & (w23 > 0)
both_years       = formal_dep_21 & in_panel_23

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
        'note': f'Insufficient treatment workers ({n_treat} < {THRESHOLD}).',
    }
else:
    print(f"\nPanel feasible ({n_treat} >= {THRESHOLD}). Running decomposition...")

    groups_out = {}

    # ── Step 5: Near-MW decomposition ─────────────────────────────────────────
    for group_name, mask_21 in [('treatment', treatment), ('control', control)]:
        n = int(mask_21.sum())

        sub_ocu23_raw = ocu500_23_raw[mask_21]
        sub_emp23     = emp23[mask_21]          # NaN→False
        sub_dep23     = dep23[mask_21]
        sub_inf23     = inf23[mask_21]
        sub_w21       = w21[mask_21]
        sub_w23       = w23[mask_21]
        sub_h21       = h21[mask_21]
        sub_h23       = h23[mask_21]
        sub_wt_panel  = wt_panel[mask_21]

        # ── BUG 1 FIX: separate attrition from non-employment ─────────────────
        # Re-interviewed = facpanel2123 > 0  OR  ocu500_23 is not NaN
        re_int   = ((sub_wt_panel > 0) & sub_wt_panel.notna()) | sub_ocu23_raw.notna()
        n_reint  = int(re_int.sum())
        n_attr   = n - n_reint
        attr_rate = n_attr / n if n > 0 else np.nan

        # Panel weight for re-interviewed subset
        # Use facpanel2123 where positive; fall back to fac500a_21 for those
        # observed_23 but lacking panel weight (rare edge case)
        sub_wt_reint = sub_wt_panel[re_int]
        sub_wt_reint = sub_wt_reint.where(sub_wt_reint > 0, wt_21[mask_21][re_int])
        wt_ok = sub_wt_reint.notna() & (sub_wt_reint > 0)

        # ── BUG 2 FIX: survey-weighted statistics among re-interviewed ─────────
        sub_emp23_reint = sub_emp23[re_int]
        sub_dep23_reint = sub_dep23[re_int]
        sub_inf23_reint = sub_inf23[re_int]
        sub_w23_reint   = sub_w23[re_int]
        sub_w21_reint   = sub_w21[re_int]
        sub_h21_reint   = sub_h21[re_int]
        sub_h23_reint   = sub_h23[re_int]

        if wt_ok.sum() > 0:
            emp_ret = float(np.average(sub_emp23_reint[wt_ok],
                                       weights=sub_wt_reint[wt_ok]))
            formal_dep_ret = float(
                np.average((sub_emp23_reint & sub_dep23_reint & sub_inf23_reint)[wt_ok],
                           weights=sub_wt_reint[wt_ok]))
        else:
            emp_ret        = float(sub_emp23_reint.mean())
            formal_dep_ret = float((sub_emp23_reint & sub_dep23_reint & sub_inf23_reint).mean())

        # Stayers: formal dep employed in both years with positive wage
        stayer = (sub_emp23_reint & sub_dep23_reint & sub_inf23_reint
                  & (sub_w23_reint > 0))
        stayer_wt = sub_wt_reint[stayer] if wt_ok[stayer].any() else None

        if stayer.sum() > 10:
            lw23 = np.log(sub_w23_reint[stayer])
            lw21 = np.log(sub_w21_reint[stayer].clip(lower=1))
            if stayer_wt is not None and (stayer_wt > 0).sum() > 5:
                wt_s = stayer_wt.where(stayer_wt > 0)
                ok_s = wt_s.notna()
                dlogw = float(np.average(lw23[ok_s], weights=wt_s[ok_s])
                              - np.average(lw21[ok_s], weights=wt_s[ok_s]))
            else:
                dlogw = float(lw23.mean() - lw21.mean())
        else:
            dlogw = np.nan

        hrs_ok = (stayer
                  & sub_h21_reint.notna() & sub_h23_reint.notna()
                  & (sub_h21_reint > 0))
        if hrs_ok.sum() > 5:
            if stayer_wt is not None and (stayer_wt[hrs_ok] > 0).sum() > 5:
                wt_h = stayer_wt[hrs_ok].where(stayer_wt[hrs_ok] > 0)
                ok_h = wt_h.notna()
                dh = float(np.average(sub_h23_reint[hrs_ok][ok_h], weights=wt_h[ok_h])
                           - np.average(sub_h21_reint[hrs_ok][ok_h], weights=wt_h[ok_h]))
            else:
                dh = float(sub_h23_reint[hrs_ok].mean() - sub_h21_reint[hrs_ok].mean())
        else:
            dh = np.nan

        # Transition counts (unweighted, among all in group)
        n_formal23   = int((sub_emp23 & sub_dep23 & sub_inf23).sum())
        n_informal23 = int((sub_emp23 & sub_dep23 & ~sub_inf23).sum())
        n_unemp23    = int((re_int & ~sub_emp23).sum())
        n_attrited   = n_attr

        label = 'Treatment' if group_name == 'treatment' else 'Control'
        print(f"\n  {label} (N={n:,}):")
        print(f"    Re-interviewed:             {n_reint:,} ({n_reint/n:.1%})")
        print(f"    Attrited (not re-int.):     {n_attrited:,} ({attr_rate:.1%})")
        print(f"    Among re-interviewed (weighted):")
        print(f"      Employment retention:     {emp_ret:.1%}")
        print(f"      Formal dep retention:     {formal_dep_ret:.1%}")
        print(f"    Δ log wage (stayers, wtd):  {dlogw:+.3f}" if not np.isnan(dlogw) else
              f"    Δ log wage (stayers):       —")
        print(f"    Δ hours (stayers, wtd):     {dh:+.1f}h/wk" if not np.isnan(dh) else
              f"    Δ hours (stayers):          —")
        print(f"    Transition counts (unweighted):")
        print(f"      → Formal dep 2023:        {n_formal23:4d} ({n_formal23/n:.1%})")
        print(f"      → Informal dep 2023:      {n_informal23:4d} ({n_informal23/n:.1%})")
        print(f"      → Not employed 2023:      {n_unemp23:4d} ({n_unemp23/n:.1%})")
        print(f"      → Not re-interviewed:     {n_attrited:4d} ({attr_rate:.1%})")

        # ── BUG 3 FIX: save all stats ─────────────────────────────────────────
        groups_out[group_name] = {
            'n_2021':                    n,
            'n_reinterviewed':           n_reint,
            'n_attrited':                n_attrited,
            'attrition_rate':            round(attr_rate, 4),
            'emp_ret_reint_wtd':         round(emp_ret, 4),
            'formal_dep_ret_reint_wtd':  round(formal_dep_ret, 4),
            'dlogw_stayers_wtd':         round(dlogw, 4) if not np.isnan(dlogw) else None,
            'dhours_stayers_wtd':        round(dh, 2)   if not np.isnan(dh)    else None,
            'n_stayers':                 int(stayer.sum()),
            'transition_formal23':       n_formal23,
            'transition_informal23':     n_informal23,
            'transition_not_emp23':      n_unemp23,
            'transition_attrited':       n_attrited,
        }

    # DiD
    t = groups_out.get('treatment', {})
    c = groups_out.get('control', {})
    did = {}
    for k in ['emp_ret_reint_wtd', 'formal_dep_ret_reint_wtd',
              'dlogw_stayers_wtd', 'dhours_stayers_wtd']:
        tv, cv = t.get(k), c.get(k)
        did[f'did_{k}'] = (round(tv - cv, 4)
                           if tv is not None and cv is not None else None)

    print(f"\n  DiD (Treatment − Control, among re-interviewed):")
    print(f"    Employment retention:  "
          f"{t.get('emp_ret_reint_wtd',0)*100:.1f}% − "
          f"{c.get('emp_ret_reint_wtd',0)*100:.1f}% = "
          f"{did.get('did_emp_ret_reint_wtd', np.nan)*100:+.1f}pp")
    print(f"    Formal dep retention:  "
          f"{t.get('formal_dep_ret_reint_wtd',0)*100:.1f}% − "
          f"{c.get('formal_dep_ret_reint_wtd',0)*100:.1f}% = "
          f"{did.get('did_formal_dep_ret_reint_wtd', np.nan)*100:+.1f}pp")
    dw = did.get('did_dlogw_stayers_wtd')
    print(f"    Δ log wage DiD:        "
          f"{t.get('dlogw_stayers_wtd', np.nan):+.3f} − "
          f"{c.get('dlogw_stayers_wtd', np.nan):+.3f} = "
          f"{dw:+.3f}" if dw is not None else "    Δ log wage DiD:        —")
    dhr = did.get('did_dhours_stayers_wtd')
    print(f"    Δ hours DiD:           "
          f"{t.get('dhours_stayers_wtd', np.nan):+.1f} − "
          f"{c.get('dhours_stayers_wtd', np.nan):+.1f} = "
          f"{dhr:+.1f}h/wk" if dhr is not None else "    Δ hours DiD:           —")

    results = {
        'feasible':    True,
        'n_treatment': n_treat,
        'n_control':   n_ctrl,
        'treatment':   groups_out.get('treatment'),
        'control':     groups_out.get('control'),
        'did':         did,
        'notes': [
            'emp_ret and formal_dep_ret are among re-interviewed workers only (weighted by facpanel2123)',
            'attrition = facpanel2123 <= 0 AND ocu500_23 is NaN — not re-interviewed, not counted as unemployed',
            'dlogw and dhours are weighted means among formal-dep stayers',
            'transition counts are unweighted',
        ],
    }

# ── Save ──────────────────────────────────────────────────────────────────────
output = {
    'metadata': {'generated': datetime.now().isoformat(),
                 'description': 'Panel 978 feasibility and near-MW decomposition (Event C)'},
    'results': results,
}
with open('D:/Nexus/nexus/exports/data/mw_panel_decomposition.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved: exports/data/mw_panel_decomposition.json")
