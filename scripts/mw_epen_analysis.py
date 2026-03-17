"""
Strategy 5 + 6: EPEN Departamentos and Ciudades MW Analysis.

EPEN_DEP (Departamentos):
  Annual dataset 2022 + 2023. Has ccdd (24 departments).
  Monthly data allows within-year pre/post split for Event C (May 2022).
  Pre: Jan-Apr 2022  |  Post: Jul-Dec 2022 (tight window)
  Also: Jan-Apr 2022 vs Jan-Apr 2023 (12-month comparison)

EPEN_CIU (Ciudades):
  Quarterly data Q1-Q4 2022 + Q1-Q3 2023.
  No ccdd, only c201 (strata/city), c203.
  Used for Cengiz bunching (Q1 2022 pre vs Q3 2022 post).

Event C: S/930->S/1025, effective May 2022.

VARIABLES (printed at runtime for audit):
  ocup300: employment status (1=employed)
  ingtrabw: monthly labor income (soles)
  informal_p: INEI pre-computed informality (1=formal, 2=informal)
  ccdd: department code (1-25, 24 departments)
  fac300_anual: annual expansion factor (EPEN_DEP)
  fac_t300: quarterly expansion factor (EPEN_CIU)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import json
import os
import shutil
import glob
from scipy import stats

DEP_DIR = 'D:/Nexus/nexus/data/raw/epen/dep/csv'
CIU_DIR = 'D:/Nexus/nexus/data/raw/epen/ciu/csv'

MW_C_OLD = 930
MW_C_NEW = 1025
BIN_WIDTH = 25
WAGE_MAX  = 5000


# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────

def load_dep(year):
    pattern = f'{DEP_DIR}/epen_dep_{year}_anual/**/*.csv'
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f'EPEN_DEP {year} not found: {pattern}')
    f = files[0]
    print(f'  Loading EPEN_DEP {year}: {os.path.basename(f)}')
    df = pd.read_csv(f, encoding='latin-1', low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    return df


def load_ciu(label):
    pattern = f'{CIU_DIR}/{label}/**/*.csv'
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f'EPEN_CIU {label} not found')
    f = files[0]
    print(f'  Loading EPEN_CIU {label}: {os.path.basename(f)}')
    df = pd.read_csv(f, encoding='latin-1', low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    return df


# ─────────────────────────────────────────────
# VARIABLE AUDIT — print full construction
# ─────────────────────────────────────────────

def audit_variables(df, label, weight_col, is_dep=True):
    """Print detailed variable audit for verification."""
    print(f'\n  {"="*55}')
    print(f'  VARIABLE AUDIT: {label}')
    print(f'  {"="*55}')
    print(f'  Rows: {len(df):,}')
    print(f'  Columns: {len(df.columns)}')

    # --- Employment ---
    emp_raw = pd.to_numeric(df['ocup300'], errors='coerce')
    emp = (emp_raw == 1).astype(float)
    emp_rate = emp.mean()
    print(f'\n  [EMPLOYMENT] ocup300 == 1')
    print(f'  Distribution: {dict(emp_raw.value_counts().sort_index().head(6))}')
    print(f'  Employed (==1): {emp.sum():,.0f} / {len(df):,} = {emp_rate:.3f}')

    # --- Wages ---
    wage = pd.to_numeric(df['ingtrabw'], errors='coerce')
    wage_pos = wage[(emp == 1) & (wage > 0)]
    print(f'\n  [WAGES] ingtrabw (monthly labor income)')
    print(f'  Non-null: {wage.notna().sum():,}  Positive (employed): {len(wage_pos):,}')
    if len(wage_pos) > 0:
        print(f'  Among employed+positive: median={wage_pos.median():.0f}  '
              f'p10={wage_pos.quantile(0.1):.0f}  '
              f'p25={wage_pos.quantile(0.25):.0f}  '
              f'p75={wage_pos.quantile(0.75):.0f}  '
              f'p90={wage_pos.quantile(0.9):.0f}  '
              f'mean={wage_pos.mean():.0f}')

    # --- Formality ---
    if 'informal_p' in df.columns:
        inf_raw = df['informal_p'].astype(str).str.strip()
        print(f'\n  [FORMALITY] informal_p (INEI pre-computed: 1=formal, 2=informal)')
        print(f'  Distribution: {inf_raw.value_counts().to_dict()}')
        formal = (inf_raw == '1').astype(float)
        informal = (inf_raw == '2').astype(float)
        missing = (inf_raw == '').astype(float)
        print(f'  Formal (==1): {formal.sum():,.0f}  Informal (==2): {informal.sum():,.0f}  '
              f'Missing (blank/not applicable): {missing.sum():,.0f}')
        # Among employed
        emp_mask = (emp == 1)
        print(f'  Among employed: formal={formal[emp_mask].sum():,.0f}  '
              f'informal={informal[emp_mask].sum():,.0f}  '
              f'missing={missing[emp_mask].sum():,.0f}  '
              f'formality_rate={formal[emp_mask].mean():.3f}')
    else:
        print(f'\n  [FORMALITY] informal_p NOT PRESENT in this file')

    # --- Department ---
    if 'ccdd' in df.columns:
        dept = pd.to_numeric(df['ccdd'], errors='coerce')
        print(f'\n  [DEPARTMENT] ccdd')
        print(f'  Unique departments: {dept.nunique()}  '
              f'Range: [{dept.min():.0f}, {dept.max():.0f}]')
        print(f'  Values: {dict(sorted(dept.value_counts().items())[:10])} ...')
    else:
        print(f'\n  [DEPARTMENT] ccdd NOT PRESENT')
        if 'c201' in df.columns:
            print(f'  c201 available: {dict(df["c201"].value_counts().sort_index().head(8))}')

    # --- Weight ---
    if weight_col in df.columns:
        wt = pd.to_numeric(df[weight_col], errors='coerce')
        print(f'\n  [WEIGHT] {weight_col}')
        print(f'  Non-null: {wt.notna().sum():,}  Zero/neg: {(wt <= 0).sum():,}  '
              f'Sum: {wt.sum():,.0f}  '
              f'Median: {wt.median():.1f}')
    else:
        print(f'\n  [WEIGHT] {weight_col} NOT PRESENT')
        fa_cols = [c for c in df.columns if c.startswith('fac') or c.startswith('fa_')]
        print(f'  Available weight cols: {fa_cols}')


# ─────────────────────────────────────────────
# FEATURIZE
# ─────────────────────────────────────────────

def featurize_dep(df, period_label, month_filter=None):
    """Extract features from EPEN_DEP slice."""
    sub = df.copy()
    if month_filter is not None:
        mes = pd.to_numeric(sub['mes'], errors='coerce')
        sub = sub[mes.isin(month_filter)].copy()
        print(f'  {period_label}: filtered to months {month_filter} → {len(sub):,} rows')

    out = pd.DataFrame(index=sub.index)
    out['dept'] = pd.to_numeric(sub['ccdd'], errors='coerce').fillna(0).astype(int)
    out['employed'] = (pd.to_numeric(sub['ocup300'], errors='coerce') == 1).astype(float)
    out['wage'] = pd.to_numeric(sub['ingtrabw'], errors='coerce')
    out['log_wage'] = np.where(
        (out['employed'] == 1) & (out['wage'] > 0),
        np.log(out['wage']), np.nan
    )

    inf_raw = sub['informal_p'].astype(str).str.strip() if 'informal_p' in sub.columns else pd.Series('', index=sub.index)
    # informal_p: 1=formal employee, 2=informal. blank = not in paid employment.
    out['formal'] = np.where(inf_raw == '1', 1.0, np.where(inf_raw == '2', 0.0, np.nan))
    # only meaningful for employed workers
    out.loc[out['employed'] == 0, 'formal'] = np.nan

    out['wt'] = pd.to_numeric(sub['fac300_anual'], errors='coerce').fillna(0)
    out['period'] = period_label
    return out


# ─────────────────────────────────────────────
# KAITZ COMPUTATION
# ─────────────────────────────────────────────

def compute_kaitz_dep(df_pre, mw_old):
    """Dept-level Kaitz = MW_old / median_employed_wage_dept."""
    sub = df_pre[(df_pre['employed'] == 1) & (df_pre['wage'] > 0) & (df_pre['wt'] > 0)].copy()

    def wmedian(g):
        g = g.dropna(subset=['wage']).sort_values('wage')
        cum = g['wt'].cumsum()
        tot = g['wt'].sum()
        if tot == 0 or len(g) == 0:
            return np.nan
        idx = (cum >= tot / 2).idxmax()
        return float(g.loc[idx, 'wage'])

    med = sub.groupby('dept').apply(wmedian, include_groups=False)
    if isinstance(med, pd.DataFrame):
        med = med.iloc[:, 0]
    med = pd.to_numeric(med, errors='coerce')
    kaitz = mw_old / med
    kaitz.name = 'kaitz_pre'

    print(f'\n  KAITZ (MW={mw_old} / dept_median_wage):')
    print(f'  n_depts={kaitz.notna().sum()}  range=[{kaitz.min():.3f}, {kaitz.max():.3f}]  '
          f'mean={kaitz.mean():.3f}  median={kaitz.median():.3f}')
    print(f'  Dept medians:')
    for d, k in sorted(zip(med.index, med.values)):
        kv = mw_old / k if k > 0 and not np.isnan(k) else np.nan
        print(f'    dept={d:02.0f}  median_wage={k:.0f}  Kaitz={kv:.3f}')
    return kaitz


# ─────────────────────────────────────────────
# DiD REGRESSION
# ─────────────────────────────────────────────

def run_dep_did(df_all, outcomes, label):
    """Regional Kaitz DiD: Y_idt = alpha_d + gamma_t + beta*(post*kaitz_pre) + epsilon."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print('ERROR: statsmodels not installed'); return {}

    results = {}
    for outcome in outcomes:
        sub = df_all.dropna(subset=[outcome, 'kaitz_pre', 'wt']).copy()
        sub = sub[sub['wt'] > 0]

        n_obs = len(sub)
        n_depts = sub['dept'].nunique()

        if n_obs < 200 or n_depts < 5:
            print(f'  {outcome}: skip (n={n_obs}, depts={n_depts})')
            results[outcome] = None
            continue

        try:
            formula = f'{outcome} ~ post + post:kaitz_pre + C(dept)'
            mod = smf.wls(formula, data=sub, weights=sub['wt']).fit(
                cov_type='cluster', cov_kwds={'groups': sub['dept']}
            )
            key = 'post:kaitz_pre'
            beta  = float(mod.params.get(key, np.nan))
            se    = float(mod.bse.get(key, np.nan))
            pval  = float(mod.pvalues.get(key, np.nan))
            ci    = mod.conf_int()
            ci_lo = float(ci.loc[key, 0]) if key in ci.index else np.nan
            ci_hi = float(ci.loc[key, 1]) if key in ci.index else np.nan

            st = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.10 else '   '))
            print(f'  {outcome:12s}: beta={beta:+.4f} SE={se:.4f} '
                  f'[{ci_lo:+.4f},{ci_hi:+.4f}] p={pval:.3f}{st} '
                  f'N={n_obs:,} depts={n_depts}')

            results[outcome] = dict(beta=beta, se=se, ci_low=ci_lo, ci_high=ci_hi,
                                    p=pval, n_obs=n_obs, n_depts=n_depts)
        except Exception as e:
            print(f'  {outcome}: FAILED — {e}')
            results[outcome] = None
    return results


# ─────────────────────────────────────────────
# CENGIZ BUNCHING (EPEN_CIU)
# ─────────────────────────────────────────────

def cengiz_ciu(pre_label, post_label, mw_new, event_name):
    """Cengiz bunching on EPEN_CIU quarterly data."""
    print(f'\n{"="*60}')
    print(f'CENGIZ BUNCHING (CIU): {event_name}')
    print(f'  Pre: {pre_label}  |  Post: {post_label}')
    print(f'{"="*60}')

    results = {}
    for label, dir_label in [('pre', pre_label), ('post', post_label)]:
        df = load_ciu(dir_label)
        audit_variables(df, f'EPEN_CIU {dir_label}', 'fac_t300', is_dep=False)

        emp = pd.to_numeric(df['ocup300'], errors='coerce') == 1
        w   = pd.to_numeric(df['ingtrabw'], errors='coerce')
        wt  = pd.to_numeric(df['fac_t300'], errors='coerce').fillna(0)
        mask = emp & (w > 0) & (w < WAGE_MAX) & (wt > 0)

        wv  = w[mask].values
        wtv = wt[mask].values
        print(f'\n  {label}: {mask.sum():,} employed | '
              f'median={np.median(wv):.0f} p10={np.percentile(wv,10):.0f} '
              f'p90={np.percentile(wv,90):.0f} '
              f'(MW_new={mw_new})')

        bin_edges   = np.arange(0, WAGE_MAX + BIN_WIDTH, BIN_WIDTH)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        counts, _   = np.histogram(wv, bins=bin_edges, weights=wtv)
        total       = counts.sum()
        shares      = counts / total if total > 0 else counts * 0
        results[label] = dict(counts=counts, shares=shares,
                              total=float(total), bin_centers=bin_centers)

    delta      = results['post']['shares'] - results['pre']['shares']
    bc         = results['pre']['bin_centers']
    below_new  = bc < mw_new
    at_new     = (bc >= mw_new) & (bc < mw_new + 4 * BIN_WIDTH)

    missing    = -delta[below_new & (delta < 0)].sum()
    excess     = delta[at_new    & (delta > 0)].sum()
    total_pre  = results['pre']['total']
    total_post = results['post']['total']
    emp_chg    = (total_post / total_pre - 1) * 100 if total_pre > 0 else 0.0
    ratio      = excess / missing if missing > 0 else float('inf')

    print(f'\n  Missing mass below S/{mw_new}: {missing*100:.3f}pp')
    print(f'  Excess mass at S/{mw_new}:      {excess*100:.3f}pp')
    print(f'  RATIO:                        {ratio:.3f}')
    print(f'  Employment change (weighted): {emp_chg:+.2f}%')

    return dict(missing_mass_pp=float(missing*100), excess_mass_pp=float(excess*100),
                ratio=float(ratio), employment_change_pct=float(emp_chg),
                total_pre=float(total_pre), total_post=float(total_post),
                bin_data=dict(bin_centers=bc.tolist(), delta=delta.tolist(),
                              shares_pre=results['pre']['shares'].tolist(),
                              shares_post=results['post']['shares'].tolist()))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    all_results = {}

    # ── EPEN_DEP: Event C DiD ────────────────
    print('\n' + '#'*65)
    print('# STRATEGY 5: EPEN_DEP — Regional Kaitz DiD (Event C)')
    print('#'*65)
    print('\nEvent C: S/930->S/1025 (May 2022)')
    print('Pre:  Jan-Apr 2022  (months 1-4, before May increase)')
    print('Post: Jul-Dec 2022  (months 7-12, tight post window)')

    try:
        df_dep2022 = load_dep(2022)

        print('\n--- AUDIT: EPEN_DEP 2022 (full year) ---')
        audit_variables(df_dep2022, 'EPEN_DEP 2022 full', 'fac300_anual')

        print('\n--- Building pre/post slices ---')
        feat_pre  = featurize_dep(df_dep2022, 'pre',  month_filter=list(range(1,5)))
        feat_post = featurize_dep(df_dep2022, 'post', month_filter=list(range(7,13)))

        print('\n--- Kaitz computation (pre-period) ---')
        kaitz = compute_kaitz_dep(feat_pre, MW_C_OLD)

        feat_pre['post']  = 0
        feat_post['post'] = 1
        df_all = pd.concat([feat_pre, feat_post], ignore_index=True)
        kaitz_df = kaitz.reset_index()
        kaitz_df.columns = ['dept', 'kaitz_pre']
        df_all = pd.concat([feat_pre, feat_post], ignore_index=True)
        df_all = df_all.merge(kaitz_df, on='dept', how='left')

        print(f'\nCombined sample: {len(df_all):,} obs  '
              f'(pre={len(feat_pre):,}, post={len(feat_post):,})')
        print(f'Missing kaitz_pre: {df_all["kaitz_pre"].isna().sum():,}')

        print('\n--- DiD RESULTS: Event C (tight window) ---')
        outcomes = ['employed', 'formal', 'log_wage']
        dep_tight = run_dep_did(df_all, outcomes, 'EPEN_DEP tight (Jan-Apr vs Jul-Dec 2022)')
        all_results['dep_event_c_tight'] = dep_tight

        # ── Pre-trend test: Jan-Feb vs Mar-Apr 2022 (no MW change) ──
        print('\n--- PRE-TREND TEST: Jan-Feb vs Mar-Apr 2022 ---')
        print('(No MW change within this window; beta should be ~0)')
        feat_pt_pre  = featurize_dep(df_dep2022, 'pt_pre',  month_filter=[1, 2])
        feat_pt_post = featurize_dep(df_dep2022, 'pt_post', month_filter=[3, 4])
        feat_pt_pre['post']  = 0
        feat_pt_post['post'] = 1
        df_pt = pd.concat([feat_pt_pre, feat_pt_post], ignore_index=True)
        df_pt = df_pt.merge(kaitz_df, on='dept', how='left')
        pt_results = run_dep_did(df_pt, outcomes, 'pre-trend')
        all_results['dep_pretrend'] = pt_results

        # ── 2022 Jan-Apr vs 2023 Jan-Apr (12-month robustness) ──
        print('\n--- ROBUSTNESS: Jan-Apr 2022 vs Jan-Apr 2023 ---')
        df_dep2023 = load_dep(2023)
        print('\n--- AUDIT: EPEN_DEP 2023 (full year) ---')
        audit_variables(df_dep2023, 'EPEN_DEP 2023 full', 'fac300_anual')

        feat_pre_2022  = featurize_dep(df_dep2022, 'pre',  month_filter=list(range(1,5)))
        feat_post_2023 = featurize_dep(df_dep2023, 'post', month_filter=list(range(1,5)))
        kaitz_2022 = compute_kaitz_dep(feat_pre_2022, MW_C_OLD)
        kaitz_df22 = kaitz_2022.reset_index()
        kaitz_df22.columns = ['dept', 'kaitz_pre']
        feat_pre_2022['post']  = 0
        feat_post_2023['post'] = 1
        df_rob = pd.concat([feat_pre_2022, feat_post_2023], ignore_index=True)
        df_rob = df_rob.merge(kaitz_df22, on='dept', how='left')
        print('\n--- DiD RESULTS: Jan-Apr 2022 vs Jan-Apr 2023 ---')
        rob_results = run_dep_did(df_rob, outcomes, 'EPEN_DEP 12-month')
        all_results['dep_event_c_12m'] = rob_results

    except Exception as e:
        print(f'\nERROR in EPEN_DEP analysis: {e}')
        import traceback; traceback.print_exc()

    # ── EPEN_CIU: Cengiz bunching Q1 vs Q3 2022 ─────────────────
    print('\n' + '#'*65)
    print('# STRATEGY 6: EPEN_CIU — Cengiz Bunching (Cities, Q1 vs Q3 2022)')
    print('#'*65)
    print('NOTE: EPEN_CIU has no ccdd (no dept breakdown).')
    print('      Running wage-distribution bunching only.')

    try:
        ciu_tight = cengiz_ciu(
            'epen_ciu_2022_q1', 'epen_ciu_2022_q3',
            mw_new=MW_C_NEW,
            event_name='S/930->S/1025 (May 2022), Q1 vs Q3 2022'
        )
        all_results['ciu_bunching_tight'] = ciu_tight

        # Q1 2022 vs Q1 2023 (longer window)
        ciu_long = cengiz_ciu(
            'epen_ciu_2022_q1', 'epen_ciu_2023_q1',
            mw_new=MW_C_NEW,
            event_name='S/930->S/1025 (May 2022), Q1 2022 vs Q1 2023'
        )
        all_results['ciu_bunching_12m'] = ciu_long

    except Exception as e:
        print(f'\nERROR in EPEN_CIU analysis: {e}')
        import traceback; traceback.print_exc()

    # ── SUMMARY ─────────────────────────────────────────────────
    print('\n' + '='*65)
    print('EPEN SUMMARY')
    print('='*65)

    print('\nEPEN_DEP Regional Kaitz DiD (Event C: S/930->S/1025, May 2022):')
    print(f'  {"Method":<35} {"employed":>10} {"formal":>10} {"log_wage":>10}')
    print('  ' + '-'*68)
    for key, label in [
        ('dep_event_c_tight', 'Tight (Jan-Apr vs Jul-Dec 2022)'),
        ('dep_event_c_12m',   '12-month (2022 Q1 vs 2023 Q1)'),
    ]:
        r = all_results.get(key, {})
        betas = []
        for outcome in ['employed', 'formal', 'log_wage']:
            rv = r.get(outcome)
            if rv:
                b, p = rv['beta'], rv['p']
                st = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))
                betas.append(f'{b:+.4f}{st}')
            else:
                betas.append('  N/A')
        print(f'  {label:<35} {betas[0]:>12} {betas[1]:>10} {betas[2]:>10}')

    print(f'\n  {"Pre-trend test (Jan-Feb vs Mar-Apr)":<35}', end='')
    pt = all_results.get('dep_pretrend', {})
    for outcome in ['employed', 'formal', 'log_wage']:
        rv = pt.get(outcome)
        if rv:
            b, p = rv['beta'], rv['p']
            verdict = 'PASS' if abs(b) < 2*rv['se'] else 'FAIL'
            print(f' {b:+.4f}[{verdict}]', end='')
    print()

    print('\nEPEN_CIU Cengiz Bunching (Event C):')
    print(f'  {"Window":<35} {"Missing":>10} {"Excess":>10} {"Ratio":>8} {"EmpChg":>8}')
    print('  ' + '-'*75)
    for key, label in [
        ('ciu_bunching_tight', 'Q1 vs Q3 2022 (tight)'),
        ('ciu_bunching_12m',   'Q1 2022 vs Q1 2023 (12m)'),
    ]:
        r = all_results.get(key, {})
        if r:
            print(f'  {label:<35} '
                  f'{r["missing_mass_pp"]:>9.3f}pp '
                  f'{r["excess_mass_pp"]:>9.3f}pp '
                  f'{r["ratio"]:>8.3f} '
                  f'{r["employment_change_pct"]:>+7.2f}%')

    # ── SAVE ─────────────────────────────────
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    output = {
        'strategy_5': 'EPEN_DEP regional Kaitz DiD — Event C (S/930->S/1025, May 2022)',
        'strategy_6': 'EPEN_CIU Cengiz bunching — Event C, Q1 vs Q3 2022',
        'variables': {
            'employment': 'ocup300 == 1',
            'wages':      'ingtrabw (monthly labor income, soles)',
            'formality':  'informal_p: 1=formal, 2=informal (INEI pre-computed)',
            'department': 'ccdd (1-24 for EPEN_DEP; not available in EPEN_CIU)',
            'weight_dep': 'fac300_anual (EPEN_DEP annual)',
            'weight_ciu': 'fac_t300 (EPEN_CIU quarterly)',
            'kaitz':      f'MW_old ({MW_C_OLD}) / weighted_dept_median_wage (pre-period)',
        },
        'event_c': {'mw_old': MW_C_OLD, 'mw_new': MW_C_NEW, 'effective': 'May 2022'},
        'results': clean(all_results),
    }

    out_path = 'D:/Nexus/nexus/exports/data/mw_epen_results.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\nSaved to {out_path}')

    qhaw_path = 'D:/qhawarina/public/assets/data/mw_epen_results.json'
    os.makedirs(os.path.dirname(qhaw_path), exist_ok=True)
    shutil.copy(out_path, qhaw_path)
    print(f'Copied to {qhaw_path}')
