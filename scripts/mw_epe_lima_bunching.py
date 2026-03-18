"""
EPE Lima quarterly bunching — robustness check for ENAHO annual results.
Cengiz et al. (2019) estimator on 6-month pre/post windows.
Three events: A (May 2016), B (April 2018), C (May 2022).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

EPE_BASE = 'D:/Nexus/nexus/data/raw/epe/csv'
BIN_W   = 25
N_BOOT  = 1000
np.random.seed(42)

EVENTS = {
    'A': {'mw_old': 750,  'mw_new': 850,
          'pre_file': f'{EPE_BASE}/epe_2016_ene_feb_mar/Trim Ene-Feb-Mar16.csv',
          'post_file': f'{EPE_BASE}/epe_2016_jun_jul_ago/Trim Jun-Jul-Ago16.csv',
          'pre_wt': 'fa_efm16', 'post_wt': 'fa_jja16', 'label': 'Q1→Q3 2016'},
    'B': {'mw_old': 850,  'mw_new': 930,
          'pre_file': f'{EPE_BASE}/epe_2018_ene_feb_mar/Trim Ene-Feb-Mar18.csv',
          'post_file': f'{EPE_BASE}/epe_2018_jun_jul_ago/Trim Jun-Jul-Ago18.csv',
          'pre_wt': 'fa_efm18', 'post_wt': 'fa_jja18', 'label': 'Q1→Q3 2018'},
    'C': {'mw_old': 930,  'mw_new': 1025,
          'pre_file': f'{EPE_BASE}/epe_2022_ene_feb_mar/Trim Ene-Feb-Mar22.csv',
          'post_file': f'{EPE_BASE}/epe_2022_jun_jul_ago/Trim Jun-Jul-Ago22.csv',
          'pre_wt': 'fa_efm22', 'post_wt': 'fa_jja22', 'label': 'Q1→Q3 2022'},
}


def load_epe(path, wt_col):
    df = pd.read_csv(path, low_memory=False, encoding='latin-1')
    df.columns = [c.lower() for c in df.columns]

    emp    = df['ocu200'].astype(str).str.strip() == '1'
    p206   = df['p206'].astype(str).str.strip()
    dep    = p206.isin(['3', '4', '6'])
    p222   = df['p222'].astype(str).str.strip()
    formal = p222 == '1'                            # EsSalud = registered formal worker
    wage   = pd.to_numeric(df['ingprin'], errors='coerce')
    wt     = pd.to_numeric(df.get(wt_col, pd.Series(np.nan, index=df.index)), errors='coerce')

    mask = emp & dep & formal & (wage > 0) & (wage < 6000) & wt.notna() & (wt > 0)
    return pd.DataFrame({'wage': wage[mask].values, 'wt': wt[mask].values})


def cengiz_epe(df_pre, df_post, mw_old, mw_new, bin_w=BIN_W):
    """Canonical Cengiz spec: arange(0,6025,25), neg-only missing, inv-abs-wt background."""
    bins = np.arange(0, 6000 + bin_w, bin_w)
    bc   = bins[:-1] + bin_w / 2.0

    def hist(df):
        counts, _ = np.histogram(df['wage'], bins=bins, weights=df['wt'])
        total = counts.sum()
        return counts / total if total > 0 else counts

    delta = hist(df_post) - hist(df_pre)

    clean = bc > 2 * mw_new
    if clean.sum() < 5:
        return np.nan, np.nan, np.nan
    bg = np.average(delta[clean], weights=1.0 / (np.abs(delta[clean]) + 1e-8))
    delta_adj = delta - bg

    aff_lo    = 0.85 * mw_old
    miss_zone = (bc >= aff_lo) & (bc < mw_new)
    exc_zone  = (bc >= mw_new) & (bc < mw_new + 10 * bin_w)

    missing = float(-delta_adj[miss_zone & (delta_adj < 0)].sum())
    excess  = float( delta_adj[exc_zone  & (delta_adj > 0)].sum())

    if missing < 0.0005:
        return np.nan, missing, excess
    return excess / missing, missing, excess


print("=" * 70)
print("EPE LIMA BUNCHING — 6-MONTH WINDOWS (Cengiz et al. 2019 canonical spec)")
print(f"N_BOOT={N_BOOT}  BIN_W={BIN_W}  Formality=p222==1 (EsSalud)")
print("=" * 70)

results = {}

for ev, cfg in EVENTS.items():
    mw_old, mw_new = cfg['mw_old'], cfg['mw_new']
    df_pre  = load_epe(cfg['pre_file'],  cfg['pre_wt'])
    df_post = load_epe(cfg['post_file'], cfg['post_wt'])

    print(f"\nEvent {ev} ({cfg['label']}, MW: S/{mw_old}→S/{mw_new})")
    print(f"  N pre={len(df_pre):,}  N post={len(df_post):,}")
    print(f"  Near-MW pre  [{0.85*mw_old:.0f},{mw_new}): "
          f"N={((df_pre['wage']>=0.85*mw_old)&(df_pre['wage']<mw_new)).sum()}")
    print(f"  Near-MW post [{0.85*mw_old:.0f},{mw_new}): "
          f"N={((df_post['wage']>=0.85*mw_old)&(df_post['wage']<mw_new)).sum()}")

    ratio, miss, exc = cengiz_epe(df_pre, df_post, mw_old, mw_new)
    print(f"  Point estimate: ratio={ratio:.3f}  missing={miss:.5f}  excess={exc:.5f}"
          if not np.isnan(ratio) else f"  Point estimate: UNDEFINED (missing={miss:.6f} too small)")

    # Bootstrap CI
    boot_ratios = []
    for _ in range(N_BOOT):
        idx_pre  = np.random.choice(len(df_pre),  len(df_pre),  replace=True)
        idx_post = np.random.choice(len(df_post), len(df_post), replace=True)
        r, _, __ = cengiz_epe(df_pre.iloc[idx_pre], df_post.iloc[idx_post], mw_old, mw_new)
        if not np.isnan(r) and 0 < r < 10:
            boot_ratios.append(r)

    print(f"  Bootstrap: {len(boot_ratios)}/{N_BOOT} valid resamples")
    if len(boot_ratios) >= 50:
        ra = np.array(boot_ratios)
        lo, hi = np.percentile(ra, 2.5), np.percentile(ra, 97.5)
        print(f"  Boot mean={ra.mean():.3f}  SD={ra.std():.3f}  95%CI=[{lo:.3f},{hi:.3f}]")
        rej1  = hi < 1.0
        rej05 = lo > 0.5
        print(f"  Reject R=1? {'YES' if rej1 else 'NO'}  Reject R=0.5? {'YES' if rej05 else 'NO'}")
    else:
        lo, hi, ra = None, None, np.array(boot_ratios) if boot_ratios else np.array([np.nan])
        print(f"  Insufficient valid resamples for CI")

    results[f'Event_{ev}'] = {
        'mw_old': mw_old, 'mw_new': mw_new, 'window': cfg['label'],
        'n_pre': len(df_pre), 'n_post': len(df_post),
        'n_near_mw_pre': int(((df_pre['wage']>=0.85*mw_old)&(df_pre['wage']<mw_new)).sum()),
        'ratio': round(float(ratio), 4) if not np.isnan(ratio) else None,
        'missing': round(float(miss), 6) if not np.isnan(miss) else None,
        'excess':  round(float(exc),  6) if not np.isnan(exc)  else None,
        'boot_mean': round(float(ra.mean()), 4) if len(boot_ratios) >= 50 else None,
        'boot_sd':   round(float(ra.std()),  4) if len(boot_ratios) >= 50 else None,
        'ci_lo': round(float(lo), 4) if lo is not None else None,
        'ci_hi': round(float(hi), 4) if hi is not None else None,
        'n_boot_valid': len(boot_ratios),
    }

# Comparison with ENAHO
print("\n\n" + "=" * 70)
print("COMPARISON: EPE Lima (6-month) vs ENAHO National (2-year)")
print("=" * 70)
print(f"\n{'Event':<8} {'ENAHO ratio':>12} {'EPE ratio':>10} {'EPE 95%CI':>22} {'Match?':>8}")
print("-" * 65)
enaho_ratios = {'A': 0.6963, 'B': 0.8286, 'C': 0.8295}
for ev in 'ABC':
    r = results[f'Event_{ev}']
    epe_r = r.get('ratio')
    lo_r, hi_r = r.get('ci_lo'), r.get('ci_hi')
    ci_str = f"[{lo_r:.3f},{hi_r:.3f}]" if lo_r is not None else "—"
    enaho = enaho_ratios[ev]
    in_ci = (lo_r <= enaho <= hi_r) if (lo_r is not None and hi_r is not None) else None
    match = "YES" if in_ci else ("NO" if in_ci is not None else "—")
    epe_str = f"{epe_r:.3f}" if epe_r is not None else "—"
    print(f"  {ev:<6} {enaho:>12.3f} {epe_str:>10} {ci_str:>22} {match:>8}")

with open('D:/Nexus/nexus/exports/data/mw_epe_lima_bunching.json', 'w', encoding='utf-8') as f:
    json.dump({'metadata': {'generated': datetime.now().isoformat(),
                            'description': 'EPE Lima 6-month bunching (Cengiz canonical spec)',
                            'formality': 'p222==1 (EsSalud)',
                            'bin_w': BIN_W, 'n_boot': N_BOOT},
               'results': results}, f, indent=2, ensure_ascii=False)
print(f"\nSaved: exports/data/mw_epe_lima_bunching.json")
