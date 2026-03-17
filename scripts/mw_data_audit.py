"""
mw_data_audit.py
================
Full data audit for paper Section 3 (Data) and Table 1 (Summary Statistics).
Audits: ENAHO cross-section, EPE Lima, EPEN Departamentos.
Saves all tables to exports/data/mw_data_audit.json.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import json, glob, os
from pathlib import Path
from datetime import datetime

# ==============================================================================
# PATHS & CONSTANTS
# ==============================================================================
BASE       = Path('D:/Nexus/nexus/data/raw/enaho/cross_section')
EPE_BASE   = Path('D:/Nexus/nexus/data/raw/epe/csv')
EPEN_BASE  = Path('D:/Nexus/nexus/data/raw/epen/dep/csv')
OUT_JSON   = Path('D:/Nexus/nexus/exports/data/mw_data_audit.json')

MW_BY_YEAR = {
    2015: 750, 2016: 850, 2017: 850, 2018: 930,
    2019: 930, 2021: 930, 2022: 1025, 2023: 1025,
}

ENAHO_YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]

# Sector mapping (CIIU rev4, 2-digit)
def ciiu2_sector(code):
    try:
        c = int(float(code))
    except (ValueError, TypeError):
        return 'other'
    if 1 <= c <= 9:   return 'agri_mining'
    if 10 <= c <= 33: return 'manufacturing'
    if 41 <= c <= 43: return 'construction'
    if 45 <= c <= 47: return 'commerce'
    if 49 <= c <= 56: return 'transport'
    if 58 <= c <= 83: return 'finance_prof'
    if c == 84:       return 'public_admin'
    if 85 <= c <= 88: return 'edu_health'
    return 'other_services'

# ==============================================================================
# HELPERS
# ==============================================================================
def weighted_percentile(vals, wts, q):
    """Weighted percentile."""
    mask = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    v, w = vals[mask], wts[mask]
    if len(v) == 0:
        return np.nan
    idx  = np.argsort(v)
    v, w = v[idx], w[idx]
    cum  = np.cumsum(w) - w / 2
    cum /= cum[-1]
    return float(np.interp(q / 100.0, cum, v))

def gini(vals, wts):
    mask = np.isfinite(vals) & np.isfinite(wts) & (wts > 0) & (vals >= 0)
    v, w = vals[mask], wts[mask]
    if len(v) < 10:
        return np.nan
    idx = np.argsort(v)
    v, w = v[idx], w[idx]
    cw = np.cumsum(w)
    total_w = cw[-1]
    total_v = np.sum(v * w)
    if total_v == 0:
        return np.nan
    B = np.sum(v * (total_w - cw + w / 2) * w) / (total_w * total_v)
    return float(1 - 2 * B)

def fmt_n(x):
    return f'{int(x):,}' if pd.notna(x) else 'N/A'

def fmt_f(x, d=1):
    return f'{x:.{d}f}' if pd.notna(x) else 'N/A'

def fmt_pct(x, d=1):
    return f'{x*100:.{d}f}%' if pd.notna(x) else 'N/A'

# ==============================================================================
# LOAD ENAHO MODULE 500
# ==============================================================================
def load_enaho(year):
    patterns = [
        BASE / f'modulo_05_{year}' / f'enaho01a-{year}-500.dta',
        BASE / f'modulo_05_{year}' / f'enaho01a-{year}_500.dta',
    ]
    for p in patterns:
        if p.exists():
            df = pd.read_stata(str(p), convert_categoricals=False)
            df.columns = [c.lower() for c in df.columns]
            return df
    raise FileNotFoundError(f'ENAHO Module 500 not found for {year}')

# ==============================================================================
# AUDIT 1A: SAMPLE COUNTS
# ==============================================================================
def audit_sample_counts(df, year):
    emp  = pd.to_numeric(df.get('ocu500', pd.Series(1, index=df.index)), errors='coerce') == 1
    p507 = pd.to_numeric(df.get('p507', pd.Series(np.nan, index=df.index)), errors='coerce')
    dep  = p507.isin([3, 4])
    self_emp = p507 == 2
    employer = p507 == 1
    dom      = p507 == 6
    unpaid   = p507 == 5

    ocinf = pd.to_numeric(df.get('ocupinf', pd.Series(np.nan, index=df.index)), errors='coerce')
    formal_dep   = emp & dep & (ocinf == 2)
    informal_dep = emp & dep & (ocinf == 1)

    wage = pd.to_numeric(df.get('p524a1', pd.Series(np.nan, index=df.index)), errors='coerce')
    wt   = pd.to_numeric(df.get('fac500a', pd.Series(np.nan, index=df.index)), errors='coerce')

    return {
        'year': year,
        'total_rows': len(df),
        'employed': int(emp.sum()),
        'dependent': int((emp & dep).sum()),
        'formal_dep': int(formal_dep.sum()),
        'informal_dep': int(informal_dep.sum()),
        'self_employed': int((emp & self_emp).sum()),
        'employer': int((emp & employer).sum()),
        'domestic': int((emp & dom).sum()),
        'unpaid': int((emp & unpaid).sum()),
        'wage_pos': int((emp & dep & (wage > 0)).sum()),
        'wt_pos': int((wt > 0).sum()),
        'formal_dep_wt_sum': float((wt[formal_dep].sum())) if formal_dep.sum() > 0 else 0,
        'emp_wt_sum': float(wt[emp].sum()) if emp.sum() > 0 else 0,
    }

# ==============================================================================
# AUDIT 1B: WAGE DISTRIBUTION
# ==============================================================================
def audit_wage_dist(df, year, mw):
    emp  = pd.to_numeric(df.get('ocu500', pd.Series(1, index=df.index)), errors='coerce') == 1
    p507 = pd.to_numeric(df.get('p507', pd.Series(np.nan, index=df.index)), errors='coerce')
    dep  = p507.isin([3, 4])
    ocinf = pd.to_numeric(df.get('ocupinf', pd.Series(np.nan, index=df.index)), errors='coerce')
    mask  = emp & dep & (ocinf == 2)

    wage = pd.to_numeric(df.get('p524a1', pd.Series(np.nan, index=df.index)), errors='coerce')
    wt   = pd.to_numeric(df.get('fac500a', pd.Series(np.nan, index=df.index)), errors='coerce').fillna(0)

    m = mask & (wage > 0) & (wt > 0)
    w = wage[m].values.astype(float)
    wts = wt[m].values.astype(float)
    n = m.sum()

    if n < 10:
        return {'year': year, 'mw': mw, 'n': n, 'error': 'insufficient'}

    mean_w  = float(np.average(w, weights=wts))
    median_ = weighted_percentile(w, wts, 50)
    p10     = weighted_percentile(w, wts, 10)
    p25     = weighted_percentile(w, wts, 25)
    p75     = weighted_percentile(w, wts, 75)
    p90     = weighted_percentile(w, wts, 90)
    p99     = weighted_percentile(w, wts, 99)

    at_mw   = float(np.sum(wts[np.abs(w - mw) <= 25]) / np.sum(wts))
    below   = float(np.sum(wts[w < mw]) / np.sum(wts))
    below85 = float(np.sum(wts[w < 0.85 * mw]) / np.sum(wts))
    g       = gini(w, wts)

    return {
        'year': year, 'mw': mw, 'n': int(n),
        'mean': round(mean_w, 1), 'median': round(median_, 1),
        'p10': round(p10, 1), 'p25': round(p25, 1),
        'p75': round(p75, 1), 'p90': round(p90, 1), 'p99': round(p99, 1),
        'at_mw_pct': round(at_mw * 100, 2),
        'below_mw_pct': round(below * 100, 2),
        'below_85mw_pct': round(below85 * 100, 2),
        'gini': round(g, 4) if pd.notna(g) else None,
        'kaitz_median': round(mw / median_, 3) if median_ > 0 else None,
    }

# ==============================================================================
# AUDIT 1C: FORMALITY RATES
# ==============================================================================
def audit_formality(df, year):
    emp   = pd.to_numeric(df.get('ocu500', pd.Series(1, index=df.index)), errors='coerce') == 1
    ocinf = pd.to_numeric(df.get('ocupinf', pd.Series(np.nan, index=df.index)), errors='coerce')
    wt    = pd.to_numeric(df.get('fac500a', pd.Series(np.nan, index=df.index)), errors='coerce').fillna(0)
    formal = (ocinf == 2)

    p507  = pd.to_numeric(df.get('p507', pd.Series(np.nan, index=df.index)), errors='coerce')
    dep   = p507.isin([3, 4])

    # Overall formality among employed
    emp_wt   = wt[emp].sum()
    form_wt  = wt[emp & formal].sum()
    rate_all = form_wt / emp_wt if emp_wt > 0 else np.nan

    # By sector (CIIU)
    svar = 'p506r4' if 'p506r4' in df.columns else ('p506' if 'p506' in df.columns else None)
    sector_rates = {}
    if svar:
        raw = pd.to_numeric(df[svar], errors='coerce')
        ciiu2 = (raw / 100).apply(np.floor)
        for sec in ['agri_mining','manufacturing','commerce','transport','finance_prof','public_admin','edu_health']:
            smask = emp & dep & (ciiu2.apply(ciiu2_sector) == sec)
            sw = wt[smask].sum()
            fw = wt[smask & formal].sum()
            sector_rates[sec] = round(fw / sw, 3) if sw > 0 else None

    # By firm size (p512b)
    size_rates = {}
    if 'p512b' in df.columns:
        p512b = pd.to_numeric(df['p512b'], errors='coerce')
        for label, cond in [('micro', p512b <= 10), ('small', (p512b > 10) & (p512b <= 50)), ('med_plus', p512b > 50)]:
            smask = emp & dep & cond
            sw = wt[smask].sum()
            fw = wt[smask & formal].sum()
            size_rates[label] = round(fw / sw, 3) if sw > 0 else None

    return {
        'year': year,
        'formality_rate': round(float(rate_all), 3) if pd.notna(rate_all) else None,
        'sector_formality': sector_rates,
        'size_formality': size_rates,
    }

# ==============================================================================
# AUDIT 1D: EMPLOYMENT COMPOSITION
# ==============================================================================
def audit_emp_composition(df, year):
    emp  = pd.to_numeric(df.get('ocu500', pd.Series(1, index=df.index)), errors='coerce') == 1
    p507 = pd.to_numeric(df.get('p507', pd.Series(np.nan, index=df.index)), errors='coerce')
    wt   = pd.to_numeric(df.get('fac500a', pd.Series(np.nan, index=df.index)), errors='coerce').fillna(0)

    emp_wt = wt[emp].sum()
    result = {'year': year}
    for val, label in [(1,'employer'),(2,'self_emp'),(3,'empleado'),(4,'obrero'),
                       (5,'unpaid'),(6,'domestic')]:
        m = emp & (p507 == val)
        result[label] = round(wt[m].sum() / emp_wt, 4) if emp_wt > 0 else None

    result['dep_total'] = round((result.get('empleado', 0) or 0) + (result.get('obrero', 0) or 0), 4)
    return result

# ==============================================================================
# AUDIT 1E: DEPARTMENT KAITZ
# ==============================================================================
def audit_dept_kaitz(df, year, mw):
    emp   = pd.to_numeric(df.get('ocu500', pd.Series(1, index=df.index)), errors='coerce') == 1
    p507  = pd.to_numeric(df.get('p507', pd.Series(np.nan, index=df.index)), errors='coerce')
    ocinf = pd.to_numeric(df.get('ocupinf', pd.Series(np.nan, index=df.index)), errors='coerce')
    wage  = pd.to_numeric(df.get('p524a1', pd.Series(np.nan, index=df.index)), errors='coerce')
    wt    = pd.to_numeric(df.get('fac500a', pd.Series(np.nan, index=df.index)), errors='coerce').fillna(0)

    formal_dep = emp & p507.isin([3, 4]) & (ocinf == 2) & (wage > 0) & (wt > 0)

    # Department code: first 2 digits of ubigeo or dedicated dept variable
    dept = None
    for dvar in ['ubigeo', 'dominio', 'ccdd']:
        if dvar in df.columns:
            raw = df[dvar].astype(str).str[:2]
            dept = raw
            break

    if dept is None:
        return {'year': year, 'mw': mw, 'error': 'no dept variable'}

    results = []
    for d_code in dept[formal_dep].unique():
        dmask = formal_dep & (dept == d_code)
        n = dmask.sum()
        if n < 30:
            continue
        w = wage[dmask].values.astype(float)
        wts = wt[dmask].values.astype(float)
        med = weighted_percentile(w, wts, 50)
        kaitz = mw / med if med > 0 else np.nan
        results.append({
            'dept': d_code,
            'n': int(n),
            'median_wage': round(med, 1),
            'kaitz': round(float(kaitz), 3) if pd.notna(kaitz) else None,
        })

    results.sort(key=lambda x: x.get('kaitz') or 0, reverse=True)
    return {'year': year, 'mw': mw, 'dept_kaitz': results}

# ==============================================================================
# AUDIT 1F: HOURS DISTRIBUTION
# ==============================================================================
def audit_hours(df, year):
    emp   = pd.to_numeric(df.get('ocu500', pd.Series(1, index=df.index)), errors='coerce') == 1
    p507  = pd.to_numeric(df.get('p507', pd.Series(np.nan, index=df.index)), errors='coerce')
    ocinf = pd.to_numeric(df.get('ocupinf', pd.Series(np.nan, index=df.index)), errors='coerce')
    wt    = pd.to_numeric(df.get('fac500a', pd.Series(np.nan, index=df.index)), errors='coerce').fillna(0)

    formal_dep = emp & p507.isin([3, 4]) & (ocinf == 2)

    # Hours variable: d540 or p540 (usual weekly hours)
    hrs = None
    for hvar in ['d540', 'p540', 'p541a']:
        if hvar in df.columns:
            hrs_raw = pd.to_numeric(df[hvar], errors='coerce')
            if hrs_raw[formal_dep].notna().sum() > 100:
                hrs = hrs_raw
                break

    if hrs is None:
        return {'year': year, 'error': 'no hours variable'}

    m = formal_dep & hrs.between(1, 120) & (wt > 0)
    h = hrs[m].values.astype(float)
    wts = wt[m].values.astype(float)

    return {
        'year': year,
        'n': int(m.sum()),
        'hours_var': hvar,
        'mean_hrs': round(float(np.average(h, weights=wts)), 1),
        'median_hrs': round(weighted_percentile(h, wts, 50), 1),
        'p10_hrs': round(weighted_percentile(h, wts, 10), 1),
        'p90_hrs': round(weighted_percentile(h, wts, 90), 1),
        'part_time_pct': round(float(np.sum(wts[h < 35]) / np.sum(wts)) * 100, 2),
        'overtime_pct': round(float(np.sum(wts[h > 48]) / np.sum(wts)) * 100, 2),
    }

# ==============================================================================
# AUDIT 2: VARIABLE CONSISTENCY
# ==============================================================================
def audit_variable_consistency(df_2015, df_2023):
    results = {}
    for varname in ['ocupinf', 'p507', 'p510a1', 'ocu500']:
        row = {}
        for year, df in [(2015, df_2015), (2023, df_2023)]:
            if varname in df.columns:
                vc = pd.to_numeric(df[varname], errors='coerce').value_counts().sort_index()
                row[str(year)] = {str(int(k)): int(v) for k, v in vc.items() if pd.notna(k)}
            else:
                row[str(year)] = 'NOT FOUND'
        results[varname] = row

    # p524a1 (wage)
    for year, df in [(2015, df_2015), (2023, df_2023)]:
        if 'p524a1' in df.columns:
            emp   = pd.to_numeric(df.get('ocu500', pd.Series(1, index=df.index)), errors='coerce') == 1
            p507  = pd.to_numeric(df['p507'], errors='coerce')
            ocinf = pd.to_numeric(df.get('ocupinf', pd.Series(np.nan, index=df.index)), errors='coerce')
            m = emp & p507.isin([3,4]) & (ocinf == 2)
            w = pd.to_numeric(df['p524a1'], errors='coerce')[m]
            results.setdefault('p524a1_stats', {})[str(year)] = {
                'count': int(w.notna().sum()),
                'mean': round(float(w.mean()), 1) if w.notna().sum() > 0 else None,
                'median': round(float(w.median()), 1) if w.notna().sum() > 0 else None,
                'min': round(float(w.min()), 1) if w.notna().sum() > 0 else None,
                'max': round(float(w.max()), 1) if w.notna().sum() > 0 else None,
                'pct_gt_0': round(float((w > 0).sum() / w.notna().sum()), 3) if w.notna().sum() > 0 else None,
            }

    # fac500a (weights)
    for year, df in [(2015, df_2015), (2023, df_2023)]:
        if 'fac500a' in df.columns:
            wt = pd.to_numeric(df['fac500a'], errors='coerce')
            emp = pd.to_numeric(df.get('ocu500', pd.Series(1, index=df.index)), errors='coerce') == 1
            results.setdefault('fac500a_stats', {})[str(year)] = {
                'sum_all': round(float(wt.sum()), 0),
                'sum_employed': round(float(wt[emp].sum()), 0),
                'mean': round(float(wt[wt > 0].mean()), 1) if (wt > 0).sum() > 0 else None,
                'min': round(float(wt[wt > 0].min()), 1) if (wt > 0).sum() > 0 else None,
                'max': round(float(wt.max()), 0),
                'n_zero': int((wt == 0).sum()),
            }

    # ubigeo dept count
    for year, df in [(2015, df_2015), (2023, df_2023)]:
        for dvar in ['ubigeo', 'ccdd']:
            if dvar in df.columns:
                n_dept = int(df[dvar].astype(str).str[:2].nunique())
                results.setdefault('dept_codes', {})[str(year)] = {'var': dvar, 'n_unique': n_dept}
                break

    return results

# ==============================================================================
# AUDIT 3: EPE LIMA
# ==============================================================================
def audit_epe():
    results = []
    for folder in sorted(EPE_BASE.iterdir()):
        if not folder.is_dir():
            continue
        csvs = list(folder.glob('*.csv'))
        if not csvs:
            continue
        csv_path = csvs[0]
        quarter_name = folder.name

        try:
            df = pd.read_csv(str(csv_path), encoding='latin1', low_memory=False)
            df.columns = [c.lower().strip() for c in df.columns]
            n_rows = len(df)

            # Key variables
            vars_found = {v: v in df.columns for v in
                          ['ocu_def','p207','ingprin','fac','p222','fac_inf']}

            # Employment rate
            emp_rate = None
            for ev in ['ocu_def','ocu500','empleo']:
                if ev in df.columns:
                    emp_raw = pd.to_numeric(df[ev], errors='coerce')
                    emp_rate = round(float((emp_raw == 1).mean()), 3)
                    break

            # Wage distribution (ingprin = principal income)
            wage_stats = {}
            for wv in ['ingprin', 'ing_prin', 'salario']:
                if wv in df.columns:
                    w = pd.to_numeric(df[wv], errors='coerce')
                    w_pos = w[w > 0]
                    if len(w_pos) > 10:
                        wage_stats = {
                            'var': wv,
                            'n_pos': int(len(w_pos)),
                            'mean': round(float(w_pos.mean()), 1),
                            'median': round(float(w_pos.median()), 1),
                            'p10': round(float(w_pos.quantile(0.1)), 1),
                            'p90': round(float(w_pos.quantile(0.9)), 1),
                        }
                    break

            results.append({
                'quarter': quarter_name,
                'n_rows': n_rows,
                'vars_found': vars_found,
                'emp_rate': emp_rate,
                'wage_stats': wage_stats,
                'columns_sample': list(df.columns[:20]),
            })

        except Exception as e:
            results.append({'quarter': quarter_name, 'error': str(e)})

    return results

# ==============================================================================
# AUDIT 4: EPEN DEPARTAMENTOS
# ==============================================================================
def audit_epen():
    results = []
    for folder in sorted(EPEN_BASE.iterdir()):
        if not folder.is_dir():
            continue
        # Look for CSV inside
        csv_paths = list(folder.rglob('*.csv')) + list(folder.rglob('*.CSV'))
        if not csv_paths:
            # Try .dta
            dta_paths = list(folder.rglob('*.dta'))
            if not dta_paths:
                results.append({'folder': folder.name, 'error': 'no data files'})
                continue
            try:
                df = pd.read_stata(str(dta_paths[0]), convert_categoricals=False)
            except Exception as e:
                results.append({'folder': folder.name, 'error': str(e)})
                continue
        else:
            try:
                df = pd.read_csv(str(csv_paths[0]), encoding='latin1', low_memory=False)
            except Exception as e:
                results.append({'folder': folder.name, 'error': str(e)})
                continue

        df.columns = [c.lower().strip() for c in df.columns]
        n_rows = len(df)
        vars_found = {v: v in df.columns for v in
                      ['ocup300','informal_p','ingtrabw','fac300_anual','ccdd']}

        # Employment rate
        emp_rate = None
        for ev in ['ocup300', 'ocu300', 'empleo']:
            if ev in df.columns:
                emp_raw = pd.to_numeric(df[ev], errors='coerce')
                emp_rate = round(float((emp_raw == 1).mean()), 3)
                break

        # Formality
        form_rate = None
        for fv in ['informal_p', 'formal_p', 'ocupinf']:
            if fv in df.columns:
                fraw = pd.to_numeric(df[fv], errors='coerce')
                if fv == 'informal_p':
                    form_rate = round(float((fraw == 0).mean()), 3)
                else:
                    form_rate = round(float((fraw == 2).mean()), 3)
                break

        # Wage
        wage_stats = {}
        for wv in ['ingtrabw', 'ingprin', 'salario']:
            if wv in df.columns:
                w = pd.to_numeric(df[wv], errors='coerce')
                w_pos = w[w > 0]
                if len(w_pos) > 10:
                    wage_stats = {
                        'var': wv, 'n_pos': int(len(w_pos)),
                        'mean': round(float(w_pos.mean()), 1),
                        'median': round(float(w_pos.median()), 1),
                    }
                break

        # Weight sum
        wt_sum = None
        for wtv in ['fac300_anual', 'fac300', 'factor']:
            if wtv in df.columns:
                wt_sum = round(float(pd.to_numeric(df[wtv], errors='coerce').sum()), 0)
                break

        # Dept count
        n_dept = None
        for dv in ['ccdd', 'ubigeo', 'departamento']:
            if dv in df.columns:
                n_dept = int(df[dv].astype(str).str[:2].nunique())
                break

        results.append({
            'folder': folder.name,
            'n_rows': n_rows,
            'vars_found': vars_found,
            'emp_rate': emp_rate,
            'formality_rate': form_rate,
            'wage_stats': wage_stats,
            'wt_sum': wt_sum,
            'n_departments': n_dept,
            'columns_sample': list(df.columns[:25]),
        })

    return results

# ==============================================================================
# PRINT HELPERS
# ==============================================================================
def print_table(rows, cols, col_widths=None):
    if not rows:
        print('  (no data)')
        return
    if col_widths is None:
        col_widths = [max(len(str(r.get(c, ''))) for r in rows + [{'dummy': ''}]) + 2
                      for c in cols]
        col_widths = [max(len(c) + 2, w) for c, w in zip(cols, col_widths)]

    header = ''.join(str(c).ljust(w) for c, w in zip(cols, col_widths))
    print(header)
    print('-' * sum(col_widths))
    for row in rows:
        line = ''.join(str(row.get(c, '')).ljust(w) for c, w in zip(cols, col_widths))
        print(line)

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    audit = {
        'generated_at': datetime.now().isoformat(),
        'sample_counts': [],
        'wage_dist': [],
        'formality': [],
        'emp_composition': [],
        'hours': [],
        'dept_kaitz': [],
        'variable_consistency': {},
        'epe': [],
        'epen': [],
        'time_series': {},
    }

    print('=' * 80)
    print('FULL DATA AUDIT — ENAHO + EPE + EPEN')
    print('=' * 80)

    # ── ENAHO loop ──
    dfs = {}
    for year in ENAHO_YEARS:
        print(f'\n{"─"*60}\nYear {year}')
        mw = MW_BY_YEAR[year]
        try:
            df = load_enaho(year)
            dfs[year] = df
        except FileNotFoundError as e:
            print(f'  ERROR: {e}')
            continue

        sc = audit_sample_counts(df, year)
        audit['sample_counts'].append(sc)
        print(f'  Rows={sc["total_rows"]:,}  Emp={sc["employed"]:,}  Dep={sc["dependent"]:,}  '
              f'FormalDep={sc["formal_dep"]:,}  InfDep={sc["informal_dep"]:,}  '
              f'SelfEmp={sc["self_employed"]:,}')

        wd = audit_wage_dist(df, year, mw)
        audit['wage_dist'].append(wd)
        print(f'  Wages(FD): N={wd.get("n","?")}  '
              f'Median={wd.get("median","?")}  P10={wd.get("p10","?")}  P90={wd.get("p90","?")}  '
              f'AtMW={wd.get("at_mw_pct","?")}%  Below={wd.get("below_mw_pct","?")}%  '
              f'Kaitz={wd.get("kaitz_median","?")}')

        fm = audit_formality(df, year)
        audit['formality'].append(fm)
        print(f'  Formality rate: {fmt_pct(fm.get("formality_rate"))}')

        ec = audit_emp_composition(df, year)
        audit['emp_composition'].append(ec)
        print(f'  Emp composition: dep={fmt_pct(ec.get("dep_total"))}  '
              f'selfEmp={fmt_pct(ec.get("self_emp"))}  employer={fmt_pct(ec.get("employer"))}')

        hrs = audit_hours(df, year)
        audit['hours'].append(hrs)
        if 'error' not in hrs:
            print(f'  Hours(FD): mean={hrs.get("mean_hrs")}  med={hrs.get("median_hrs")}  '
                  f'partTime={hrs.get("part_time_pct")}%  OT={hrs.get("overtime_pct")}%')

        dkz = audit_dept_kaitz(df, year, mw)
        audit['dept_kaitz'].append(dkz)
        if 'dept_kaitz' in dkz and dkz['dept_kaitz']:
            top3 = dkz['dept_kaitz'][:3]
            bot3 = dkz['dept_kaitz'][-3:]
            print(f'  Dept Kaitz  TOP3: ' +
                  ', '.join(f'{d["dept"]}={d["kaitz"]:.2f}({d["median_wage"]:.0f})' for d in top3))
            print(f'  Dept Kaitz  BOT3: ' +
                  ', '.join(f'{d["dept"]}={d["kaitz"]:.2f}({d["median_wage"]:.0f})' for d in bot3))

    # ── Variable consistency (2015 vs 2023) ──
    print(f'\n{"─"*60}\nAUDIT 2: VARIABLE CONSISTENCY 2015 vs 2023')
    if 2015 in dfs and 2023 in dfs:
        vc = audit_variable_consistency(dfs[2015], dfs[2023])
        audit['variable_consistency'] = vc
        for varname in ['ocupinf', 'p507', 'ocu500', 'p510a1']:
            v = vc.get(varname, {})
            print(f'\n  {varname}:')
            for yr in ['2015', '2023']:
                vv = v.get(yr, 'NOT IN AUDIT')
                print(f'    {yr}: {vv}')
        for stat_key in ['p524a1_stats', 'fac500a_stats']:
            v = vc.get(stat_key, {})
            print(f'\n  {stat_key}:')
            for yr in ['2015', '2023']:
                print(f'    {yr}: {v.get(yr, "N/A")}')
        v = vc.get('dept_codes', {})
        print(f'\n  dept_codes: 2015={v.get("2015")}  2023={v.get("2023")}')
    else:
        print('  Cannot compare: 2015 or 2023 not loaded')
        audit['variable_consistency'] = {'error': 'missing years'}

    # ── Summary tables ──
    print(f'\n{"="*80}')
    print('TABLE A: SAMPLE COUNTS BY YEAR')
    print(f'{"="*80}')
    if audit['sample_counts']:
        cols = ['year','total_rows','employed','dependent','formal_dep','informal_dep','self_employed','wage_pos']
        cw   = [6, 10, 10, 10, 12, 10, 12, 9]
        hdr  = ''.join(c.ljust(w) for c, w in zip(cols, cw))
        print(hdr)
        print('-' * sum(cw))
        for r in audit['sample_counts']:
            line = ''.join(str(r.get(c, '')).ljust(w) for c, w in zip(cols, cw))
            print(line)

    print(f'\n{"="*80}')
    print('TABLE B: WAGE DISTRIBUTION (formal-dep, p524a1) BY YEAR')
    print(f'{"="*80}')
    if audit['wage_dist']:
        cols = ['year','mw','n','mean','median','p10','p25','p75','p90','at_mw_pct','below_mw_pct','kaitz_median','gini']
        cw   = [6,6,7,8,8,7,7,7,7,9,10,13,7]
        hdr  = ''.join(c.ljust(w) for c, w in zip(cols, cw))
        print(hdr)
        print('-' * sum(cw))
        for r in audit['wage_dist']:
            if 'error' in r:
                print(f'{r["year"]} ERROR: {r["error"]}')
                continue
            line = ''.join(str(r.get(c, '')).ljust(w) for c, w in zip(cols, cw))
            print(line)

    print(f'\n{"="*80}')
    print('TABLE C: FORMALITY RATES BY YEAR')
    print(f'{"="*80}')
    if audit['formality']:
        cols = ['year','formality_rate']
        for f in audit['formality']:
            sec = f.get('sector_formality', {})
            sz  = f.get('size_formality', {})
            print(f'  {f["year"]}: overall={fmt_pct(f.get("formality_rate"))}  '
                  f'agri={fmt_pct(sec.get("agri_mining"))}  '
                  f'manuf={fmt_pct(sec.get("manufacturing"))}  '
                  f'commerce={fmt_pct(sec.get("commerce"))}  '
                  f'public={fmt_pct(sec.get("public_admin"))}  '
                  f'edu={fmt_pct(sec.get("edu_health"))}  ||  '
                  f'micro={fmt_pct(sz.get("micro"))}  '
                  f'small={fmt_pct(sz.get("small"))}  '
                  f'med+={fmt_pct(sz.get("med_plus"))}')

    print(f'\n{"="*80}')
    print('TABLE D: EMPLOYMENT COMPOSITION BY YEAR')
    print(f'{"="*80}')
    if audit['emp_composition']:
        print(f'  {"Year":<6} {"DepTotal":>9} {"SelfEmp":>9} {"Employer":>9} {"Unpaid":>9} {"Domestic":>9}')
        print('  ' + '-' * 50)
        for r in audit['emp_composition']:
            print(f'  {r["year"]:<6} {fmt_pct(r.get("dep_total")):>9} '
                  f'{fmt_pct(r.get("self_emp")):>9} {fmt_pct(r.get("employer")):>9} '
                  f'{fmt_pct(r.get("unpaid")):>9} {fmt_pct(r.get("domestic")):>9}')

    print(f'\n{"="*80}')
    print('TABLE E: HOURS (formal-dep) BY YEAR')
    print(f'{"="*80}')
    for r in audit['hours']:
        if 'error' in r:
            print(f'  {r["year"]}: {r["error"]}')
        else:
            print(f'  {r["year"]}: mean={r.get("mean_hrs")}h  '
                  f'med={r.get("median_hrs")}h  '
                  f'p10={r.get("p10_hrs")}h  p90={r.get("p90_hrs")}h  '
                  f'partTime={r.get("part_time_pct")}%  OT={r.get("overtime_pct")}%')

    # ── Time series summary ──
    print(f'\n{"="*80}')
    print('TABLE F: KEY TIME SERIES (formal-dep, national)')
    print(f'{"="*80}')
    ts_rows = []
    for wd, fm, sc in zip(audit['wage_dist'], audit['formality'], audit['sample_counts']):
        yr = wd['year']
        ts_rows.append({
            'year': yr,
            'mw': wd.get('mw'),
            'median_fw': wd.get('median'),
            'kaitz': wd.get('kaitz_median'),
            'form_rate': fm.get('formality_rate'),
            'n_formal_dep': sc.get('formal_dep'),
            'at_mw_pct': wd.get('at_mw_pct'),
            'below_mw_pct': wd.get('below_mw_pct'),
        })
    audit['time_series'] = ts_rows

    print(f'  {"Year":<6} {"MW":>6} {"MedianFW":>10} {"Kaitz":>7} {"Formal%":>8} {"N_FD":>8} {"AtMW%":>8} {"Below%":>8}')
    print('  ' + '-' * 65)
    for r in ts_rows:
        print(f'  {r["year"]:<6} {str(r.get("mw","")):>6} '
              f'{str(r.get("median_fw","")):>10} '
              f'{str(r.get("kaitz","")):>7} '
              f'{fmt_pct(r.get("form_rate")):>8} '
              f'{fmt_n(r.get("n_formal_dep","?")):>8} '
              f'{str(r.get("at_mw_pct","")):>8} '
              f'{str(r.get("below_mw_pct","")):>8}')

    # ── EPE ──
    print(f'\n{"="*80}')
    print('AUDIT 3: EPE LIMA')
    print(f'{"="*80}')
    epe_results = audit_epe()
    audit['epe'] = epe_results
    for r in epe_results:
        if 'error' in r:
            print(f'  {r["quarter"]}: ERROR {r["error"]}')
        else:
            ws = r.get('wage_stats', {})
            print(f'  {r["quarter"]}: N={r["n_rows"]:,}  '
                  f'emp_rate={r.get("emp_rate")}  '
                  f'wage_med={ws.get("median","?")}  '
                  f'vars={[k for k,v in r.get("vars_found",{}).items() if v]}')

    # ── EPEN ──
    print(f'\n{"="*80}')
    print('AUDIT 4: EPEN DEPARTAMENTOS')
    print(f'{"="*80}')
    epen_results = audit_epen()
    audit['epen'] = epen_results
    for r in epen_results:
        if 'error' in r:
            print(f'  {r["folder"]}: ERROR {r["error"]}')
        else:
            ws = r.get('wage_stats', {})
            print(f'  {r["folder"]}: N={r["n_rows"]:,}  '
                  f'emp={r.get("emp_rate")}  '
                  f'formal={r.get("formality_rate")}  '
                  f'wage_med={ws.get("median","?")}  '
                  f'n_dept={r.get("n_departments")}  '
                  f'wt_sum={r.get("wt_sum")}  '
                  f'vars={[k for k,v in r.get("vars_found",{}).items() if v]}')

    # ── Dept Kaitz ranking stability ──
    print(f'\n{"="*80}')
    print('DEPT KAITZ RANKING STABILITY (highest Kaitz = MW most binding)')
    print(f'{"="*80}')
    for dkz in audit['dept_kaitz']:
        yr = dkz['year']
        depts = dkz.get('dept_kaitz', [])
        if depts:
            top5 = depts[:5]
            print(f'  {yr} TOP5: ' + '  '.join(f'{d["dept"]}:{d["kaitz"]:.3f}' for d in top5))

    # ── Save ──
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(audit, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n\nSaved: {OUT_JSON}')
    print(f'File size: {OUT_JSON.stat().st_size / 1024:.1f} KB')

if __name__ == '__main__':
    main()
