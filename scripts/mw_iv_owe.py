"""
Priority 2 + 3: IV/OWE Estimation + Event Study Pre-Trends
Dube & Lindner (2024) structural approach using ENAHO CS.
Instrument: Kaitz_d_pre = MW / median_formal_wage in dept d (pre-period).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

try:
    import pyreadstat
    HAS_PYREADSTAT = True
except ImportError:
    HAS_PYREADSTAT = False

CS_BASE = 'D:/Nexus/nexus/data/raw/enaho/cross_section'
OUT_JSON = 'D:/Nexus/nexus/exports/data/mw_iv_owe.json'

EVENTS = {
    'A': {'mw_old': 750,  'mw_new': 850,  'pre': 2015, 'post': 2017, 'name': 'S/750->850 (May 2016)'},
    'B': {'mw_old': 850,  'mw_new': 930,  'pre': 2017, 'post': 2019, 'name': 'S/850->930 (Apr 2018)'},
    'C': {'mw_old': 930,  'mw_new': 1025, 'pre': 2021, 'post': 2023, 'name': 'S/930->1025 (May 2022)'},
}
EVENT_STUDY_YEARS = {
    'A': [2015, 2016, 2017, 2018, 2019],   # omit 2015 → base; or 2016 → base
    'B': [2015, 2016, 2017, 2018, 2019],   # omit 2017
    'C': [2019, 2021, 2022, 2023],          # omit 2021
}


def load_enaho(year, all_working_age=False):
    """Load ENAHO Module 500 for a given year. Returns DataFrame with standardized cols."""
    import os
    dta_path = f'{CS_BASE}/modulo_05_{year}/enaho01a-{year}-500.dta'
    dta_path2 = f'{CS_BASE}/modulo_05_{year}/enaho01a-{year}_500.dta'

    path = dta_path if os.path.exists(dta_path) else dta_path2
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return None

    if not HAS_PYREADSTAT:
        print("  ERROR: pyreadstat not available")
        return None

    df, meta = pyreadstat.read_dta(path)
    df.columns = [c.lower() for c in df.columns]

    # Standard variable extraction
    # Employment
    emp = pd.to_numeric(df.get('ocu500', df.get('p501', pd.Series(np.nan, index=df.index))), errors='coerce') == 1

    # Dependent employment
    dep_var = 'cat07p500a1' if 'cat07p500a1' in df.columns else 'p507'
    dep_val = pd.to_numeric(df.get(dep_var, pd.Series(np.nan, index=df.index)), errors='coerce')
    dep = dep_val == 2 if dep_var == 'cat07p500a1' else dep_val.isin([3, 4, 6])

    # Formality
    formal = pd.to_numeric(df.get('ocupinf', pd.Series(np.nan, index=df.index)), errors='coerce') == 2

    # Wage
    w_p = pd.to_numeric(df.get('p524a1', pd.Series(np.nan, index=df.index)), errors='coerce')
    w_i = pd.to_numeric(df.get('i524a1', pd.Series(np.nan, index=df.index)), errors='coerce') / 12.0
    wage = w_p.where(w_p > 0, w_i)

    # Hours
    hrs = pd.to_numeric(df.get('p513t', pd.Series(np.nan, index=df.index)), errors='coerce')

    # Demographics
    age  = pd.to_numeric(df.get('p208a', pd.Series(np.nan, index=df.index)), errors='coerce')
    male = (pd.to_numeric(df.get('p207', pd.Series(np.nan, index=df.index)), errors='coerce') == 1).astype(int)

    # Education (p301a or p301)
    edu_var = 'p301a' if 'p301a' in df.columns else 'p301'
    edu = pd.to_numeric(df.get(edu_var, pd.Series(np.nan, index=df.index)), errors='coerce')

    # Weight
    wt = pd.to_numeric(df.get('factor07i500a', df.get('fac500a', pd.Series(np.nan, index=df.index))), errors='coerce')

    # Department (first 2 chars of ubigeo)
    ubigeo = df.get('ubigeo', pd.Series('', index=df.index)).astype(str).str.zfill(6)
    dept = ubigeo.str[:2].astype(str)

    out = pd.DataFrame({
        'year':    year,
        'emp':     emp.astype(int),
        'dep':     dep.astype(int),
        'formal':  formal.astype(int),
        'wage':    wage,
        'hrs':     hrs,
        'age':     age,
        'age_sq':  age ** 2,
        'male':    male,
        'edu':     edu,
        'wt':      wt,
        'dept':    dept,
    })

    working_age = (age >= 14) & (age <= 65)
    if all_working_age:
        return out[working_age].reset_index(drop=True)
    else:
        # Formal dep employed with positive wage
        mask = emp & dep & formal & (wage > 0) & (wage < 15000) & wt.notna()
        return out[working_age & mask].reset_index(drop=True)


def wls_cluster(df, formula_str, weight_col, cluster_col):
    """OLS/WLS with cluster-robust SE using statsmodels."""
    try:
        import statsmodels.formula.api as smf
        mod = smf.wls(formula_str, data=df, weights=df[weight_col])
        res = mod.fit(cov_type='cluster', cov_kwds={'groups': df[cluster_col]})
        return res
    except Exception as e:
        print(f"    WLS error: {e}")
        return None


print("=" * 70)
print("IV / OWE ESTIMATION  +  EVENT STUDY")
print("=" * 70)

all_results = {}

# ─── Load all cross-sections ───────────────────────────────────────────────────
print("\nLoading ENAHO cross-sections...")
enaho_cache = {}
enaho_all_cache = {}
for year in [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]:
    print(f"  Loading {year}...", end=' ')
    df = load_enaho(year, all_working_age=False)
    df_all = load_enaho(year, all_working_age=True)
    if df is not None:
        enaho_cache[year] = df
        print(f"formal_dep N={len(df):,}", end='')
    if df_all is not None:
        enaho_all_cache[year] = df_all
        print(f"  all_working_age N={len(df_all):,}")
    else:
        print()


# ─── Compute department Kaitz (pre-period) for each event ─────────────────────
def dept_kaitz(df_pre, mw_old):
    """Kaitz = MW_old / median_formal_wage by department."""
    dept_med = (df_pre.groupby('dept')
                .apply(lambda g: np.average(g['wage'], weights=g['wt']) if len(g) > 10 else np.nan)
                .rename('med_wage'))
    kaitz = (mw_old / dept_med).rename('kaitz')
    return kaitz.dropna()


# ─── IV/OWE ───────────────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("TABLE 5: OWN-WAGE ELASTICITY (Dube & Lindner 2024)")
print("=" * 70)
print(f"\n{'Event':<8} {'pi_wage':>10} {'SE':>8} {'F-stat':>8} {'beta_emp':>10} {'SE':>8} {'OWE':>8} {'SE':>8}")
print("-" * 75)

owe_results = {}
for ev, cfg in EVENTS.items():
    pre, post, mw_old, mw_new = cfg['pre'], cfg['post'], cfg['mw_old'], cfg['mw_new']

    if pre not in enaho_cache or post not in enaho_cache:
        print(f"  {ev}: missing data  skip")
        continue

    df_pre  = enaho_cache[pre].copy()
    df_post = enaho_cache[post].copy()
    df_pre['post'] = 0
    df_post['post'] = 1
    df_wages = pd.concat([df_pre, df_post], ignore_index=True)

    # Kaitz from pre-period median formal wage by dept
    kaitz_map = dept_kaitz(df_pre, mw_old)
    df_wages['kaitz'] = df_wages['dept'].map(kaitz_map)
    df_wages = df_wages[df_wages['kaitz'].notna() & df_wages['kaitz'].between(0.1, 5)]
    df_wages['post_kaitz'] = df_wages['post'] * df_wages['kaitz']
    df_wages['log_wage'] = np.log(df_wages['wage'].clip(lower=1))

    # First stage: log wage ~ post*kaitz + dept FE + post + controls
    formula_fs = 'log_wage ~ post_kaitz + C(dept) + post + age + age_sq + male'
    res_fs = wls_cluster(df_wages, formula_fs, 'wt', 'dept')

    # Reduced form: employment ~ post*kaitz + dept FE + post + controls
    df_pre_all  = enaho_all_cache.get(pre, df_pre).copy()
    df_post_all = enaho_all_cache.get(post, df_post).copy()
    df_pre_all['post'] = 0; df_post_all['post'] = 1
    df_all = pd.concat([df_pre_all, df_post_all], ignore_index=True)
    df_all['kaitz'] = df_all['dept'].map(kaitz_map)
    df_all = df_all[df_all['kaitz'].notna() & df_all['kaitz'].between(0.1, 5)]
    df_all['post_kaitz'] = df_all['post'] * df_all['kaitz']

    formula_rf = 'emp ~ post_kaitz + C(dept) + post + age + age_sq + male'
    res_rf = wls_cluster(df_all, formula_rf, 'wt', 'dept')

    if res_fs is None or res_rf is None:
        print(f"  {ev}: regression failed")
        continue

    pi      = res_fs.params.get('post_kaitz', np.nan)
    pi_se   = res_fs.bse.get('post_kaitz', np.nan)
    pi_p    = res_fs.pvalues.get('post_kaitz', np.nan)
    beta    = res_rf.params.get('post_kaitz', np.nan)
    beta_se = res_rf.bse.get('post_kaitz', np.nan)
    beta_p  = res_rf.pvalues.get('post_kaitz', np.nan)

    f_stat = (pi / pi_se) ** 2 if pi_se > 0 else np.nan
    owe = beta / pi if abs(pi) > 0.001 else np.nan
    owe_se = (abs(owe) * np.sqrt((beta_se/beta)**2 + (pi_se/pi)**2)
              if (abs(pi) > 0.001 and abs(beta) > 0.001 and not np.isnan(owe)) else np.nan)

    def fs(v, d=4): return f'{v:.{d}f}' if not np.isnan(v) else '—'
    def si(p):
        if np.isnan(p): return ''
        return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'

    print(f"  {ev:<6} {fs(pi):>10} {fs(pi_se):>8} {fs(f_stat,1):>8} "
          f"{fs(beta):>10} {fs(beta_se):>8} {fs(owe,3):>8} {fs(owe_se,3):>8}  "
          f"[pi:{si(pi_p)} emp:{si(beta_p)}]")

    interp = ('destroys employment' if not np.isnan(owe) and owe < -0.05
              else 'negligible employment effect')
    if not np.isnan(owe):
        print(f"         => OWE={owe:.3f}: 10% MW-wage increase => {owe*10:.1f}% employment change ({interp})")
        print(f"         => Dube 2019 US benchmark: OWE~-0.04")

    owe_results[ev] = {
        'pi_wage': round(float(pi), 5) if not np.isnan(pi) else None,
        'pi_se': round(float(pi_se), 5) if not np.isnan(pi_se) else None,
        'pi_p': round(float(pi_p), 4) if not np.isnan(pi_p) else None,
        'f_stat': round(float(f_stat), 2) if not np.isnan(f_stat) else None,
        'beta_emp': round(float(beta), 5) if not np.isnan(beta) else None,
        'beta_se': round(float(beta_se), 5) if not np.isnan(beta_se) else None,
        'beta_p': round(float(beta_p), 4) if not np.isnan(beta_p) else None,
        'owe': round(float(owe), 4) if not np.isnan(owe) else None,
        'owe_se': round(float(owe_se), 4) if not np.isnan(owe_se) else None,
        'n_wage_obs': len(df_wages),
        'n_emp_obs': len(df_all),
        'n_depts': df_wages['dept'].nunique(),
    }

# Pooled OWE (inverse-variance weighted) — all events
valid_owe = [(ev, r) for ev, r in owe_results.items() if r.get('owe') and r.get('owe_se')]
if len(valid_owe) >= 2:
    owe_vals = np.array([r['owe'] for _, r in valid_owe])
    owe_ses  = np.array([r['owe_se'] for _, r in valid_owe])
    ivw = (owe_vals / owe_ses**2).sum() / (1 / owe_ses**2).sum()
    ivw_se = 1 / np.sqrt((1 / owe_ses**2).sum())
    ivw_p = 2 * (1 - __import__('scipy.stats', fromlist=['norm']).norm.cdf(abs(ivw / ivw_se)))
    print(f"\n  Pooled IVW OWE (A+B+C) = {ivw:.3f}  (SE={ivw_se:.3f})")
    print(f"  US Dube (2019) benchmark: OWE ~ -0.04")
    owe_results['pooled'] = {'owe': round(float(ivw), 4), 'owe_se': round(float(ivw_se), 4),
                             'owe_p': round(float(ivw_p), 4)}

# Pooled OWE A+B only (Event C excluded: weak instrument F=1.5)
valid_ab = [(ev, r) for ev, r in owe_results.items()
            if ev in ('A', 'B') and r.get('owe') and r.get('owe_se')]
if len(valid_ab) >= 2:
    owe_ab = np.array([r['owe'] for _, r in valid_ab])
    ses_ab = np.array([r['owe_se'] for _, r in valid_ab])
    ivw_ab = (owe_ab / ses_ab**2).sum() / (1 / ses_ab**2).sum()
    ivw_ab_se = 1 / np.sqrt((1 / ses_ab**2).sum())
    ivw_ab_p = 2 * (1 - __import__('scipy.stats', fromlist=['norm']).norm.cdf(abs(ivw_ab / ivw_ab_se)))
    print(f"  Pooled IVW OWE (A+B only) = {ivw_ab:.3f}  (SE={ivw_ab_se:.3f}, p={ivw_ab_p:.3f})")
    print(f"  [Event C excluded: weak first stage F=1.5]")
    owe_results['pooled_ab'] = {
        'owe': round(float(ivw_ab), 4),
        'owe_se': round(float(ivw_ab_se), 4),
        'owe_p': round(float(ivw_ab_p), 4),
        'note': 'Events A+B only; Event C excluded (F=1.5, weak instrument)',
    }

# ─── EVENT STUDY (Pre-trends) ─────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("FIGURE 1: EVENT STUDY PRE-TRENDS")
print("=" * 70)

es_results = {}
for ev, cfg in EVENTS.items():
    print(f"\n  Event {ev}: {cfg['name']}")
    pre_yr = cfg['pre']
    mw_old = cfg['mw_old']
    years_ev = EVENT_STUDY_YEARS[ev]

    # Load all years for this event window
    dfs = []
    for y in years_ev:
        if y in enaho_all_cache:
            d = enaho_all_cache[y].copy()
            d['year'] = y
            dfs.append(d)
    if len(dfs) < 3:
        print(f"    Insufficient years, skipping")
        continue

    df_es = pd.concat(dfs, ignore_index=True)
    kaitz_ev = dept_kaitz(enaho_cache.get(pre_yr, dfs[0]), mw_old)
    df_es['kaitz'] = df_es['dept'].map(kaitz_ev)
    df_es = df_es[df_es['kaitz'].notna() & df_es['kaitz'].between(0.1, 5)]

    # Omitted year (last pre-period)
    base_year = pre_yr
    non_base = [y for y in years_ev if y != base_year]
    for y in non_base:
        df_es[f'k_x_{y}'] = df_es['kaitz'] * (df_es['year'] == y).astype(int)

    ev_coefs = {}
    for outcome, col in [('employment', 'emp'), ('log_wage', None)]:
        if col == 'emp':
            formula = ('emp ~ ' + ' + '.join([f'k_x_{y}' for y in non_base])
                       + ' + C(dept) + C(year) + age + age_sq + male')
            df_use = df_es[df_es['wt'].notna()]
        else:
            df_es['log_wage'] = np.log(df_es['wage'].clip(lower=1))
            formula = ('log_wage ~ ' + ' + '.join([f'k_x_{y}' for y in non_base])
                       + ' + C(dept) + C(year) + age + age_sq + male')
            df_use = df_es[df_es['wage'] > 0 & df_es['wt'].notna()]

        res = wls_cluster(df_use, formula, 'wt', 'dept')
        if res is None:
            continue

        print(f"    {outcome}:  {'Year':<6} {'beta':>9} {'SE':>8} {'p':>7}  Note")
        betas = {}
        for y in years_ev:
            if y == base_year:
                b, se, p = 0.0, 0.0, 1.0
                note = '<-- BASE'
            else:
                key = f'k_x_{y}'
                b  = res.params.get(key, np.nan)
                se = res.bse.get(key, np.nan)
                p  = res.pvalues.get(key, np.nan)
                is_pre = y < base_year
                note = '<-- must be ~0' if is_pre else ''
                flag = ('***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns') if not np.isnan(p) else ''
                note = f'{flag} {note}'.strip()
            fs2 = lambda v: f'{v:.4f}' if not np.isnan(v) else '—'
            print(f"      {y:<6} {fs2(b):>9} {fs2(se):>8} {fs2(p):>7}  {note}")
            betas[str(y)] = {'beta': float(b) if not np.isnan(b) else None,
                             'se':   float(se) if not np.isnan(se) else None,
                             'p':    float(p)  if not np.isnan(p)  else None}

        # Pre-trend F-test (joint test of pre-period betas = 0)
        pre_keys = [f'k_x_{y}' for y in non_base if y < base_year and f'k_x_{y}' in res.params.index]
        if len(pre_keys) >= 1:
            try:
                f_test = res.f_test(' = '.join([f'{k}' for k in pre_keys]) + ' = 0')
                f_p = float(f_test.pvalue)
                print(f"      Pre-trend F-test (joint zero): p = {f_p:.4f}  "
                      f"{'FAIL (trends exist)' if f_p < 0.05 else 'PASS (parallel trends)'}")
                betas['pretrend_f_pvalue'] = f_p
            except Exception as e:
                print(f"      F-test error: {e}")

        ev_coefs[outcome] = betas

    es_results[ev] = ev_coefs

# ─── Save ──────────────────────────────────────────────────────────────────────
output = {
    'metadata': {'generated': datetime.now().isoformat(),
                 'description': 'IV/OWE (Dube-Lindner) + Event Study pre-trends'},
    'owe': owe_results,
    'event_study': es_results,
}
with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {OUT_JSON}")
