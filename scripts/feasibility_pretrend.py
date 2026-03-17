"""
Pre-trend and placebo checks for the two passing approaches.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

CS_BASE = 'D:/Nexus/nexus/data/raw/enaho/cross_section'


def load_formal_dep(year, month_filter=None, level='dept'):
    import os
    dta  = f'{CS_BASE}/modulo_05_{year}/enaho01a-{year}-500.dta'
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
    out = pd.DataFrame({'lw': np.log(wage), 'wt': wt, 'cluster': cluster})[mask]
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


def run_iv_test(name, df_pre, df_post, kaitz_map, min_cluster_n=10):
    df_pre  = df_pre.copy();  df_pre['post']  = 0
    df_post = df_post.copy(); df_post['post'] = 1
    df_ev = pd.concat([df_pre, df_post], ignore_index=True)
    df_ev['kaitz'] = df_ev['cluster'].map(kaitz_map)
    df_ev['ki']    = df_ev['kaitz'] * df_ev['post']
    df_ev = df_ev.dropna(subset=['lw', 'kaitz', 'wt'])
    # Drop clusters with <min_cluster_n obs
    counts = df_ev.groupby('cluster')['lw'].count()
    keep = counts[counts >= min_cluster_n].index
    df_ev = df_ev[df_ev['cluster'].isin(keep)]
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
    sign_ok = 'POSITIVE (expected)' if beta > 0 else 'NEGATIVE (unexpected!)'
    print(f"  {name}:")
    print(f"    N={len(df_ev):,}  G={G}")
    print(f"    pi={beta:.4f}  SE={se:.4f}  F={F:.2f}  [{tag}]  sign={sign_ok}")
    return beta, se, F, G


print("=" * 65)
print("PRE-TREND CHECKS")
print("=" * 65)

# ─── 1. Province annual Event B: pre-period tests ────────────────────────────
print()
print("--- Province annual Event B: pre-trends ---")
print("Using Kaitz_province_2017 as instrument (same as main regression)")
kaitz_p17 = get_kaitz(2017, 850, level='prov')

d15 = load_formal_dep(2015, level='prov')
d16 = load_formal_dep(2016, level='prov')
d17 = load_formal_dep(2017, level='prov')

print()
print("Pre-period 1: 2015->2016 (no MW change, MW=750)")
print("  H0: pi~=0 (parallel trends). If PASS, pre-trend exists.")
run_iv_test("2015->2016 pre-trend", d15, d16, kaitz_p17)

print()
print("Pre-period 2: 2016->2017 (MW changed May 2016 - contaminated)")
print("  [Note: 2016 has MW change in May. This is informative but noisy.]")
run_iv_test("2016->2017 (contaminated year)", d16, d17, kaitz_p17)

# Also test with kaitz from 2015 (cleaner pre-period instrument)
print()
print("Alternative: using Kaitz_province_2015 (pre-event), test 2015->2016")
kaitz_p15 = get_kaitz(2015, 750, level='prov')
run_iv_test("2015->2016 with Kaitz_2015", d15, d16, kaitz_p15)

# ─── 2. Within-year province Event B: placebo tests ─────────────────────────
print()
print("--- Within-year province Event B: Q1->Q3 placebos ---")
print("Main: 2018 Q1->Q3 (MW increased April 2018). Should find F>10, pi>0.")
print("Placebo 1: 2017 Q1->Q3 (MW=850 all year, NO change).")
print("  H0: pi~=0. If PASS (F>4), there is seasonal bias.")

kaitz_p17_q1 = get_kaitz(2017, 850, level='prov', month_filter=[1, 2, 3])
d17_q1 = load_formal_dep(2017, month_filter=[1, 2, 3], level='prov')
d17_q3 = load_formal_dep(2017, month_filter=[7, 8, 9], level='prov')
print()
run_iv_test("2017 Q1->Q3 PLACEBO (no MW change)", d17_q1, d17_q3, kaitz_p17_q1)

print()
print("Placebo 2: 2019 Q1->Q3 (MW=930 all year after Event B, no further change).")
kaitz_p19_q1 = get_kaitz(2019, 930, level='prov', month_filter=[1, 2, 3])
d19_q1 = load_formal_dep(2019, month_filter=[1, 2, 3], level='prov')
d19_q3 = load_formal_dep(2019, month_filter=[7, 8, 9], level='prov')
run_iv_test("2019 Q1->Q3 PLACEBO (no MW change)", d19_q1, d19_q3, kaitz_p19_q1)

print()
print("=" * 65)
print("INTERPRETATION GUIDE")
print("=" * 65)
print("""
Province annual Event B:
  - If 2015->2016 pre-trend F>4: province Kaitz predicts pre-existing trends
    => instrument is NOT valid (same failure as department level)
  - If 2015->2016 F<4: province-level gets us past the pre-trend problem

Within-year province Event B:
  - If 2017 Q1->Q3 placebo F<4 AND pi~0: no seasonal confound
    => within-year approach is valid
  - If 2017 Q1->Q3 placebo F>4: seasonal pattern correlated with Kaitz
    => within-year approach is BIASED (seasonal confound)

Decision rule: implement fully only if (1) F>10 in main AND (2) placebo F<4.
""")
