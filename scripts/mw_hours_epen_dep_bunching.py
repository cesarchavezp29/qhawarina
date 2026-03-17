"""
Priority 4 + 5: Hours Analysis (intensive margin) + EPEN DEP Cengiz Bunching
Uses ENAHO CS (p513t) and EPEN DEP annual files.
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

CS_BASE   = 'D:/Nexus/nexus/data/raw/enaho/cross_section'
EPEN_BASE = 'D:/Nexus/nexus/data/raw/epen/dep/csv'

EVENTS = [
    ('A', 2015, 2017, 750,  850),
    ('B', 2017, 2019, 850,  930),
    ('C', 2021, 2023, 930, 1025),
]

# ─── HOURS ANALYSIS ────────────────────────────────────────────────────────────

def load_enaho_formal_dep(year):
    import os
    dta = f'{CS_BASE}/modulo_05_{year}/enaho01a-{year}-500.dta'
    dta2 = f'{CS_BASE}/modulo_05_{year}/enaho01a-{year}_500.dta'
    path = dta if os.path.exists(dta) else dta2
    if not os.path.exists(path): return None
    if not HAS_PYREADSTAT: return None
    df, _ = pyreadstat.read_dta(path)
    df.columns = [c.lower() for c in df.columns]

    emp    = pd.to_numeric(df.get('ocu500', df.get('p501', pd.Series(1, index=df.index))), errors='coerce') == 1
    dep_v  = 'cat07p500a1' if 'cat07p500a1' in df.columns else 'p507'
    dep_n  = pd.to_numeric(df.get(dep_v, pd.Series(np.nan, index=df.index)), errors='coerce')
    dep    = (dep_n == 2) if dep_v == 'cat07p500a1' else dep_n.isin([3, 4, 6])
    formal = pd.to_numeric(df.get('ocupinf', pd.Series(np.nan, index=df.index)), errors='coerce') == 2
    w_p    = pd.to_numeric(df.get('p524a1', pd.Series(np.nan, index=df.index)), errors='coerce')
    w_i    = pd.to_numeric(df.get('i524a1', pd.Series(np.nan, index=df.index)), errors='coerce') / 12.0
    wage   = w_p.where(w_p > 0, w_i)
    hrs    = pd.to_numeric(df.get('p513t', pd.Series(np.nan, index=df.index)), errors='coerce')
    wt     = pd.to_numeric(df.get('factor07i500a', df.get('fac500a', pd.Series(np.nan, index=df.index))), errors='coerce')
    ubigeo = df.get('ubigeo', pd.Series('', index=df.index)).astype(str).str.zfill(6)
    dept   = ubigeo.str[:2]

    mask = emp & dep & formal & (wage > 0) & (wage < 15000) & wt.notna()
    out = pd.DataFrame({'wage': wage, 'hrs': hrs, 'wt': wt, 'dept': dept})[mask]
    return out.reset_index(drop=True)


def wavg(vals, wts):
    ok = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    if ok.sum() == 0: return np.nan
    return np.average(vals[ok], weights=wts[ok])


print("=" * 70)
print("TABLE 6: HOURS DiD (Intensive Margin) — ENAHO CS, p513t")
print("=" * 70)

hours_results = {}
for ev, pre_yr, post_yr, mw_old, mw_new in EVENTS:
    treat_lo = 0.85 * mw_old
    ctrl_lo  = 1.5  * mw_new
    ctrl_hi  = 3.0  * mw_new

    df_pre  = load_enaho_formal_dep(pre_yr)
    df_post = load_enaho_formal_dep(post_yr)

    if df_pre is None or df_post is None:
        print(f"  Event {ev}: data not available")
        hours_results[ev] = None
        continue

    # Pre-period hours by zone
    treat_pre = df_pre[(df_pre['wage'] >= treat_lo) & (df_pre['wage'] < mw_new)]
    ctrl_pre  = df_pre[(df_pre['wage'] >= ctrl_lo)  & (df_pre['wage'] <= ctrl_hi)]

    # Post-period: treatment = workers who would have been in affected zone,
    # proxied by [mw_new, 1.3*mw_new] (bunched-up zone)
    treat_post = df_post[(df_post['wage'] >= mw_new) & (df_post['wage'] <= 1.3 * mw_new)]
    ctrl_post  = df_post[(df_post['wage'] >= ctrl_lo) & (df_post['wage'] <= ctrl_hi)]

    # Weighted mean hours
    h_tp  = wavg(treat_pre['hrs'].values,  treat_pre['wt'].values)
    h_tpo = wavg(treat_post['hrs'].values, treat_post['wt'].values)
    h_cp  = wavg(ctrl_pre['hrs'].values,   ctrl_pre['wt'].values)
    h_cpo = wavg(ctrl_post['hrs'].values,  ctrl_post['wt'].values)

    d_treat = h_tpo - h_tp
    d_ctrl  = h_cpo - h_cp
    did     = d_treat - d_ctrl

    def fs(v, d=1): return f'{v:.{d}f}' if not np.isnan(v) else '—'

    n_tp = len(treat_pre); n_tpo = len(treat_post)
    n_cp = len(ctrl_pre);  n_cpo = len(ctrl_post)

    print(f"\n  Event {ev} ({mw_old}->{mw_new}):  pre={pre_yr} post={post_yr}")
    print(f"    Treatment [S/{treat_lo:.0f}, S/{mw_new}):  pre={fs(h_tp)}h (N={n_tp:,}) "
          f"-> post={fs(h_tpo)}h (N={n_tpo:,})  delta={fs(d_treat)}h")
    print(f"    Control   [S/{ctrl_lo:.0f}, S/{ctrl_hi:.0f}]: pre={fs(h_cp)}h (N={n_cp:,}) "
          f"-> post={fs(h_cpo)}h (N={n_cpo:,})  delta={fs(d_ctrl)}h")
    did_str = fs(did)
    print(f"    DiD = {did_str}h/week  "
          f"({'Hours reduction detected' if not np.isnan(did) and did < -1 else 'No meaningful hours adjustment'})")

    hours_results[ev] = {
        'treat_pre_hrs': float(h_tp) if not np.isnan(h_tp) else None,
        'treat_post_hrs': float(h_tpo) if not np.isnan(h_tpo) else None,
        'ctrl_pre_hrs': float(h_cp) if not np.isnan(h_cp) else None,
        'ctrl_post_hrs': float(h_cpo) if not np.isnan(h_cpo) else None,
        'delta_treat': float(d_treat) if not np.isnan(d_treat) else None,
        'delta_ctrl': float(d_ctrl) if not np.isnan(d_ctrl) else None,
        'did': float(did) if not np.isnan(did) else None,
        'n_treat_pre': n_tp, 'n_treat_post': n_tpo,
        'n_ctrl_pre': n_cp,  'n_ctrl_post': n_cpo,
    }


# ─── EPEN DEP BUNCHING ────────────────────────────────────────────────────────
import os

def load_epen_dep(year, spec='anual'):
    patterns = [
        f'{EPEN_BASE}/epen_dep_{year}_{spec}',
        f'{EPEN_BASE}/epen_dep_{year}_anual',
    ]
    for base in patterns:
        if not os.path.exists(base): continue
        csvs = [f for f in os.listdir(base) if f.lower().endswith('.csv')]
        if not csvs: continue
        path = os.path.join(base, csvs[0])
        df = pd.read_csv(path, encoding='latin1', low_memory=False)
        df.columns = [c.lower() for c in df.columns]
        return df
    # try sub-directory
    for root, dirs, files in os.walk(f'{EPEN_BASE}'):
        if str(year) in root and 'anual' in root.lower():
            for f in files:
                if f.lower().endswith('.csv'):
                    df = pd.read_csv(os.path.join(root, f), encoding='latin1', low_memory=False)
                    df.columns = [c.lower() for c in df.columns]
                    return df
    return None


def cengiz_bunching_epen(df, mw_old, mw_new, bin_w=25):
    """Cengiz revised bunching on EPEN data."""
    employed = (pd.to_numeric(df.get('ocup300', df.get('ocu300', pd.Series(np.nan, index=df.index))),
                               errors='coerce') == 1)
    # Formality: informal_p (string), '2' = formal
    inf_p = df.get('informal_p', df.get('informal_p', pd.Series('', index=df.index)))
    formal = inf_p.astype(str).str.strip() == '2'
    wage   = pd.to_numeric(df.get('ingtrabw', pd.Series(np.nan, index=df.index)), errors='coerce')
    wt     = pd.to_numeric(df.get('fac300_anual', df.get('fac300_annual', pd.Series(np.nan, index=df.index))), errors='coerce')

    mask = employed & formal & (wage > 0) & (wage < 8000) & wt.notna()
    w = wage[mask].values
    wts = wt[mask].values

    bins = np.arange(200, 4001, bin_w)
    bc   = bins[:-1] + bin_w / 2
    counts, _ = np.histogram(w, bins=bins, weights=wts)
    total = counts.sum()
    shares = counts / total if total > 0 else counts

    # Counterfactual from clean zone
    clean = bc > 2 * mw_new
    if clean.sum() < 5:
        print(f"    WARNING: only {clean.sum()} clean bins")
        return None
    try:
        cf = np.polyval(np.polyfit(bc[clean], shares[clean], 4), bc)
        cf = np.maximum(cf, 0)
    except Exception as e:
        print(f"    Poly fit error: {e}")
        return None

    delta = shares - cf
    aff_lo = 0.85 * mw_old
    miss_zone  = (bc >= aff_lo) & (bc < mw_new)
    exc_zone   = (bc >= mw_new) & (bc < mw_new + 10 * bin_w)
    missing = float(max(-delta[miss_zone].sum(), 0))
    excess  = float(delta[exc_zone & (delta > 0)].sum())
    ratio   = excess / missing if missing > 0.003 else None

    mw_idx = int(np.abs(bc - mw_new).argmin())
    mw_share = float(shares[mw_idx])
    mw_cf    = float(cf[mw_idx])

    return {
        'n_obs': int(mask.sum()),
        'weighted_pop': round(float(wts.sum()), 0),
        'median_wage': float(np.median(w)),
        'kaitz': round(mw_new / float(np.median(w)), 3),
        'missing_mass_pp': round(missing * 100, 3),
        'excess_mass_pp': round(excess * 100, 3),
        'bunching_ratio': round(ratio, 3) if ratio else None,
        'ratio_unreliable': missing < 0.003,
        'mw_bin_share': round(mw_share, 5),
        'mw_cf_share': round(mw_cf, 5),
        'mw_excess_factor': round(mw_share / mw_cf, 3) if mw_cf > 0 else None,
    }


print("\n\n" + "=" * 70)
print("TABLE A2: EPEN DEP BUNCHING (Event C — formal workers)")
print("=" * 70)

epen_dep_results = {}
for year_label, year, spec, mw_old, mw_new in [
    ('pre_2022', 2022, 'anual', 930, 1025),
    ('post_2023', 2023, 'anual', 930, 1025),
]:
    df_ep = load_epen_dep(year, spec)
    if df_ep is None:
        print(f"  {year_label} ({year}): data not available")
        epen_dep_results[year_label] = None
        continue

    print(f"\n  {year_label} ({year}):")
    # Check columns
    has_inf = 'informal_p' in df_ep.columns
    has_wage = 'ingtrabw' in df_ep.columns
    print(f"    Columns: informal_p={'YES' if has_inf else 'NO'}  ingtrabw={'YES' if has_wage else 'NO'}")
    if has_inf:
        inf_dist = df_ep['informal_p'].astype(str).str.strip().value_counts().head(5).to_dict()
        print(f"    informal_p dist: {inf_dist}")

    res = cengiz_bunching_epen(df_ep, mw_old, mw_new)
    if res:
        print(f"    N={res['n_obs']:,}  median_wage=S/{res['median_wage']:.0f}  Kaitz={res['kaitz']}")
        print(f"    Missing={res['missing_mass_pp']:.3f}pp  Excess={res['excess_mass_pp']:.3f}pp  "
              f"Ratio={res['bunching_ratio']}  ExcessFactor={res['mw_excess_factor']}x")
    else:
        print(f"    Bunching computation failed")
    epen_dep_results[year_label] = res

# Event C Cengiz ratio: compare pre and post
if epen_dep_results.get('pre_2022') and epen_dep_results.get('post_2023'):
    pre_r = epen_dep_results['pre_2022']
    pos_r = epen_dep_results['post_2023']
    print(f"\n  EPEN DEP Event C Comparison:")
    print(f"    Pre 2022:  median=S/{pre_r['median_wage']:.0f}  at_mw={pre_r['mw_bin_share']:.4f}")
    print(f"    Post 2023: median=S/{pos_r['median_wage']:.0f}  at_mw={pos_r['mw_bin_share']:.4f}")
    print(f"    Increase in MW concentration: {(pos_r['mw_bin_share']/pre_r['mw_bin_share']-1)*100:.1f}%")

# ─── Save ──────────────────────────────────────────────────────────────────────
output = {
    'metadata': {'generated': datetime.now().isoformat()},
    'hours_intensive_margin': hours_results,
    'epen_dep_bunching': epen_dep_results,
}
with open('D:/Nexus/nexus/exports/data/mw_hours_epen_dep.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved: exports/data/mw_hours_epen_dep.json")
