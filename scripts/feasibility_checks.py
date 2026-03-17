"""
Three feasibility checks for stronger IV identification.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import pyreadstat
    HAS_PYREADSTAT = True
except ImportError:
    HAS_PYREADSTAT = False

CS_BASE = 'D:/Nexus/nexus/data/raw/enaho/cross_section'


def load_formal_dep(year, month_filter=None, level='dept'):
    import os
    dta = f'{CS_BASE}/modulo_05_{year}/enaho01a-{year}-500.dta'
    dta2 = f'{CS_BASE}/modulo_05_{year}/enaho01a-{year}_500.dta'
    path = dta if os.path.exists(dta) else dta2
    if not os.path.exists(path):
        return None
    df = pd.read_stata(path, convert_categoricals=False)
    df.columns = [c.lower() for c in df.columns]

    ocu    = pd.to_numeric(df.get('ocu500',  pd.Series(1,   index=df.index)), errors='coerce') == 1
    dep    = pd.to_numeric(df.get('p507',    pd.Series(np.nan, index=df.index)), errors='coerce').isin([3, 4, 6])
    formal = pd.to_numeric(df.get('ocupinf', pd.Series(np.nan, index=df.index)), errors='coerce') == 2
    wage   = pd.to_numeric(df.get('p524a1',  pd.Series(np.nan, index=df.index)), errors='coerce')
    wt     = pd.to_numeric(df.get('fac500a', pd.Series(np.nan, index=df.index)), errors='coerce')
    mes    = pd.to_numeric(df.get('mes',     pd.Series(np.nan, index=df.index)), errors='coerce')
    ubigeo = df.get('ubigeo', pd.Series('', index=df.index)).astype(str).str.zfill(6)
    cluster = ubigeo.str[:4] if level == 'prov' else ubigeo.str[:2]

    mask = ocu & dep & formal & (wage > 0) & (wage < 6000) & wt.notna()
    if month_filter is not None:
        mask = mask & mes.isin(month_filter)

    out = pd.DataFrame({
        'lw': np.log(wage),
        'wt': wt,
        'cluster': cluster,
        'mes': mes,
    })[mask]
    return out.reset_index(drop=True)


def wmedian(wages, weights):
    idx = np.argsort(wages)
    wages = wages[idx]; weights = weights[idx]
    cumw = np.cumsum(weights)
    return float(wages[np.searchsorted(cumw, cumw[-1] / 2)])


def get_kaitz(year, mw, level='dept', month_filter=None):
    import os
    dta  = f'{CS_BASE}/modulo_05_{year}/enaho01a-{year}-500.dta'
    dta2 = f'{CS_BASE}/modulo_05_{year}/enaho01a-{year}_500.dta'
    path = dta if os.path.exists(dta) else dta2
    df = pd.read_stata(path, convert_categoricals=False)
    df.columns = [c.lower() for c in df.columns]

    ocu    = pd.to_numeric(df.get('ocu500',  pd.Series(1,   index=df.index)), errors='coerce') == 1
    dep    = pd.to_numeric(df.get('p507',    pd.Series(np.nan, index=df.index)), errors='coerce').isin([3, 4, 6])
    formal = pd.to_numeric(df.get('ocupinf', pd.Series(np.nan, index=df.index)), errors='coerce') == 2
    wage   = pd.to_numeric(df.get('p524a1',  pd.Series(np.nan, index=df.index)), errors='coerce')
    wt_s   = pd.to_numeric(df.get('fac500a', pd.Series(np.nan, index=df.index)), errors='coerce')
    mes    = pd.to_numeric(df.get('mes',     pd.Series(np.nan, index=df.index)), errors='coerce')
    ubigeo = df.get('ubigeo', pd.Series('', index=df.index)).astype(str).str.zfill(6)
    cluster = ubigeo.str[:4] if level == 'prov' else ubigeo.str[:2]

    mask = ocu & dep & formal & (wage > 0) & (wage < 6000) & wt_s.notna()
    if month_filter:
        mask = mask & mes.isin(month_filter)

    kaitz = {}
    for g, grp in df.loc[mask].groupby(cluster[mask]):
        w   = pd.to_numeric(grp['p524a1'], errors='coerce').values
        wts = pd.to_numeric(grp.get('fac500a', pd.Series(1, index=grp.index)), errors='coerce').values
        ok  = np.isfinite(w) & np.isfinite(wts) & (wts > 0)
        if ok.sum() >= 10:
            kaitz[g] = mw / wmedian(w[ok], wts[ok])
    return pd.Series(kaitz, name='kaitz')


def cluster_se_fstat(y, x, cluster, w=None):
    """WLS within-estimator (cluster FE + post FE removed) with cluster-robust SE."""
    n = len(y)
    if w is None:
        w = np.ones(n)
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(w) & (w > 0)
    y = y[ok]; x = x[ok]; cluster = cluster[ok]; w = w[ok]
    xd = x.reshape(-1, 1)
    wd = w
    beta = np.linalg.lstsq(xd * wd[:, None], y * wd, rcond=None)[0][0]
    resid = y - xd.flatten() * beta
    ucl = np.unique(cluster)
    G = len(ucl)
    V = 0.0
    for cl in ucl:
        mask2 = cluster == cl
        s = (xd[mask2].flatten() * resid[mask2] * wd[mask2]).sum()
        V += s ** 2
    xwx = (xd.flatten() ** 2 * wd).sum()
    V_beta = V / xwx ** 2
    se = np.sqrt(V_beta)
    F = (beta / se) ** 2
    return beta, se, F, G


def run_iv_test(name, df_pre, df_post, kaitz_map):
    df_pre  = df_pre.copy();  df_pre['post']  = 0
    df_post = df_post.copy(); df_post['post'] = 1
    df_ev = pd.concat([df_pre, df_post], ignore_index=True)
    df_ev['kaitz'] = df_ev['cluster'].map(kaitz_map)
    df_ev['ki']    = df_ev['kaitz'] * df_ev['post']
    df_ev = df_ev.dropna(subset=['lw', 'kaitz', 'wt'])
    # Remove cluster FE and post FE by double-demeaning
    df_ev['lw2'] = (df_ev['lw']
                    - df_ev.groupby('cluster')['lw'].transform('mean')
                    - df_ev.groupby('post')['lw'].transform('mean')
                    + df_ev['lw'].mean())
    df_ev['ki2'] = (df_ev['ki']
                    - df_ev.groupby('cluster')['ki'].transform('mean')
                    - df_ev.groupby('post')['ki'].transform('mean')
                    + df_ev['ki'].mean())
    beta, se, F, G = cluster_se_fstat(
        df_ev['lw2'].values, df_ev['ki2'].values,
        df_ev['cluster'].values, df_ev['wt'].values)
    tag = 'PASS F>10' if F > 10 else ('MARGINAL 4-10' if F > 4 else 'FAIL F<4')
    print(f"  {name}:")
    print(f"    N pre={len(df_pre):,}  N post={len(df_post):,}  clusters={G}")
    print(f"    pi={beta:.4f}  SE={se:.4f}  F={F:.2f}  [{tag}]")
    return F


results = {}

# ─── TEST 1: Province-level annual, Event B (2017 pre → 2019 post) ───────────
print('=' * 65)
print('TEST 1: Province-level annual Kaitz IV (Event B: 2017→2019)')
print('=' * 65)
kaitz_prov_2017 = get_kaitz(2017, 850, level='prov')
print(f"  Provinces with Kaitz: {len(kaitz_prov_2017)}")
print(f"  Range: [{kaitz_prov_2017.min():.3f}, {kaitz_prov_2017.max():.3f}]  SD={kaitz_prov_2017.std():.3f}")

d17 = load_formal_dep(2017, level='prov')
d19 = load_formal_dep(2019, level='prov')
F1 = run_iv_test('Province annual Event B (2017→2019)', d17, d19, kaitz_prov_2017)
results['province_annual_B'] = F1

# Also try Event A province-level
print()
kaitz_prov_2015 = get_kaitz(2015, 750, level='prov')
print(f"  (Event A) Provinces with Kaitz: {len(kaitz_prov_2015)}  SD={kaitz_prov_2015.std():.3f}")
d15 = load_formal_dep(2015, level='prov')
d17b = load_formal_dep(2017, level='prov')
F1a = run_iv_test('Province annual Event A (2015→2017)', d15, d17b, kaitz_prov_2015)
results['province_annual_A'] = F1a

# ─── TEST 2: Within-year month split, dept-level, Event B ────────────────────
print()
print('=' * 65)
print('TEST 2: Within-year split, dept-level (Event B 2018, mes 1-3 vs 7-9)')
print('=' * 65)
kaitz_d_q1_18 = get_kaitz(2018, 850, level='dept', month_filter=[1, 2, 3])
print(f"  Depts with Q1 Kaitz: {len(kaitz_d_q1_18)}  SD={kaitz_d_q1_18.std():.3f}")

d18_pre  = load_formal_dep(2018, month_filter=[1, 2, 3], level='dept')
d18_post = load_formal_dep(2018, month_filter=[7, 8, 9], level='dept')
F2 = run_iv_test('Within-year dept (2018 Q1→Q3)', d18_pre, d18_post, kaitz_d_q1_18)
results['within_year_dept_B'] = F2

# ─── TEST 3: Within-year month split, province-level, Event B ────────────────
print()
print('=' * 65)
print('TEST 3: Within-year split, province-level (Event B 2018, mes 1-3 vs 7-9)')
print('=' * 65)
kaitz_p_q1_18 = get_kaitz(2018, 850, level='prov', month_filter=[1, 2, 3])
print(f"  Provinces with Q1 Kaitz: {len(kaitz_p_q1_18)}  SD={kaitz_p_q1_18.std():.3f}")

d18_pre_p  = load_formal_dep(2018, month_filter=[1, 2, 3], level='prov')
d18_post_p = load_formal_dep(2018, month_filter=[7, 8, 9], level='prov')
F3 = run_iv_test('Within-year province (2018 Q1→Q3)', d18_pre_p, d18_post_p, kaitz_p_q1_18)
results['within_year_prov_B'] = F3

# ─── TEST 4: Province × within-year for Event A (May 2016) ───────────────────
print()
print('=' * 65)
print('TEST 4: Within-year province, Event A (2016, mes 1-4 vs 6-12)')
print('  MW increased May 2016 → pre=Jan-Apr, post=Jun-Dec')
print('=' * 65)
kaitz_p_a = get_kaitz(2016, 750, level='prov', month_filter=[1, 2, 3, 4])
print(f"  Provinces with Kaitz: {len(kaitz_p_a)}  SD={kaitz_p_a.std():.3f}")

d16_pre  = load_formal_dep(2016, month_filter=[1, 2, 3, 4],  level='prov')
d16_post = load_formal_dep(2016, month_filter=[6, 7, 8, 9, 10, 11, 12], level='prov')
F4 = run_iv_test('Within-year province Event A (2016 Q1→Q3+)', d16_pre, d16_post, kaitz_p_a)
results['within_year_prov_A'] = F4

# ─── SUMMARY ─────────────────────────────────────────────────────────────────
print()
print('=' * 65)
print('FEASIBILITY SUMMARY')
print('=' * 65)
for name, F_val in results.items():
    tag = 'PASS F>10' if F_val > 10 else ('MARGINAL 4-10' if F_val > 4 else 'FAIL F<4')
    print(f"  {name}: F={F_val:.2f}  [{tag}]")
print()
print('Benchmark: Stock-Yogo (2005) critical value F>10 for 5% maximal IV size distortion.')
print('Angrist-Pischke (2009): F>10 conventional threshold for strong instrument.')
