"""
Strategy 3: EPE Lima MW Analysis
Two complementary approaches:
  1. Cengiz bunching on Lima wage distribution (pre vs post each event)
  2. Panel DiD — match workers across quarters, treated=near MW, control=above

Data: ENAHO Permanente de Empleo (EPE) from datosabiertos.gob.pe
  CSV files, monthly, Lima Metro Area only.

Correct MW events:
  A: S/750->S/850 (May 2016)   pre=Ene-Feb-Mar16, post=Jun-Jul-Ago16
  B: S/850->S/930 (Apr 2018)   pre=Ene-Feb-Mar18, post=Jun-Jul-Ago18
  C: S/930->S/1025 (May 2022)  pre=Ene-Feb-Mar22, post=Jun-Jul-Ago22

Variables:
  ocu200: 1=employed
  ingprin: monthly principal income (soles)
  p222: health insurance — 1=EsSalud, 2=Seguro Privado, 3=Ambos=FORMAL
                          4=Otro, 5=No afiliado=INFORMAL
  fa_*: expansion factor (column name varies by quarter)
  Panel ID: conglome + vivienda + hogar + codperso
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

CSV_BASE = 'D:/Nexus/nexus/data/raw/epe/csv'
BIN_WIDTH = 25
WAGE_MAX = 5000
FORMAL_CODES = {1.0, 2.0, 3.0}   # EsSalud, Seguro Privado, Ambos

EVENTS = {
    'A': {
        'name': 'S/750->S/850 (May 2016)',
        'mw_old': 750, 'mw_new': 850,
        'pre_dir': 'epe_2016_ene_feb_mar',
        'post_dir': 'epe_2016_jun_jul_ago',
    },
    'B': {
        'name': 'S/850->S/930 (Apr 2018)',
        'mw_old': 850, 'mw_new': 930,
        'pre_dir': 'epe_2018_ene_feb_mar',
        'post_dir': 'epe_2018_jun_jul_ago',
    },
    'C': {
        'name': 'S/930->S/1025 (May 2022)',
        'mw_old': 930, 'mw_new': 1025,
        'pre_dir': 'epe_2022_ene_feb_mar',
        'post_dir': 'epe_2022_jun_jul_ago',
    },
}

PRETREND = {
    'A_pretrend': {
        'name': 'Pre-trend A: Jun-Jul-Aug15 -> Ene-Feb-Mar16 (no MW change)',
        'mw_old': 750, 'mw_new': 850,
        'pre_dir': 'epe_2015_jun_jul_ago',
        'post_dir': 'epe_2016_ene_feb_mar',
    },
    'B_pretrend': {
        'name': 'Pre-trend B: Jun-Jul-Aug17 -> Ene-Feb-Mar18 (no MW change)',
        'mw_old': 850, 'mw_new': 930,
        'pre_dir': 'epe_2017_jun_jul_ago',
        'post_dir': 'epe_2018_ene_feb_mar',
    },
}


def load_epe(dir_name):
    """Load EPE CSV from the datosabiertos csv directory."""
    path = os.path.join(CSV_BASE, dir_name)
    csvs = glob.glob(os.path.join(path, '*.csv'))
    if not csvs:
        raise FileNotFoundError(f"No CSV in {path}")
    f = csvs[0]
    print(f"  Loading: {os.path.basename(f)}")
    df = pd.read_csv(f, encoding='latin-1')
    df.columns = [c.lower() for c in df.columns]
    return df


def get_weight(df):
    """Get expansion factor (column name varies by quarter)."""
    fa_cols = [c for c in df.columns if c.startswith('fa_')]
    if fa_cols:
        wt = pd.to_numeric(df[fa_cols[0]], errors='coerce').fillna(0)
        print(f"  Weight: {fa_cols[0]}")
        return wt
    print("  WARNING: No weight column, using uniform")
    return pd.Series(1.0, index=df.index)


def make_pid(df):
    return (df['conglome'].astype(str).str.strip() + '_' +
            df['vivienda'].astype(str).str.strip() + '_' +
            df['hogar'].astype(str).str.strip() + '_' +
            df['codperso'].astype(str).str.strip())


def featurize_epe(df):
    """Extract person-level features from EPE CSV."""
    out = pd.DataFrame(index=df.index)
    out['pid'] = make_pid(df)
    out['employed'] = (pd.to_numeric(df['ocu200'], errors='coerce') == 1).astype(float)
    out['wage'] = pd.to_numeric(df['ingprin'], errors='coerce')
    out['formal'] = pd.to_numeric(df['p222'], errors='coerce').isin(FORMAL_CODES).astype(float)
    # formal only meaningful for employed workers
    out.loc[out['employed'] == 0, 'formal'] = np.nan
    out['wt'] = get_weight(df)
    return out


# ===================================================
# APPROACH 1: CENGIZ BUNCHING (Lima wage distribution)
# ===================================================

def cengiz_lima(event_id, ecfg):
    """Run Cengiz bunching on EPE Lima wage distributions."""
    print(f"\n{'='*60}")
    print(f"CENGIZ BUNCHING (Lima) Event {event_id}: {ecfg['name']}")
    print(f"{'='*60}")

    mw_new = ecfg['mw_new']

    df_pre_raw = load_epe(ecfg['pre_dir'])
    df_post_raw = load_epe(ecfg['post_dir'])

    results = {}
    for label, df_raw in [('pre', df_pre_raw), ('post', df_post_raw)]:
        feat = featurize_epe(df_raw)
        # Use employed workers with positive wage
        mask = (feat['employed'] == 1) & (feat['wage'] > 0) & (feat['wage'] < WAGE_MAX) & (feat['wt'] > 0)
        w = feat.loc[mask, 'wage'].values
        wt = feat.loc[mask, 'wt'].values

        print(f"  {label}: {mask.sum():,} employed workers | "
              f"median={np.median(w):.0f} | "
              f"mean={np.average(w, weights=wt):.0f} | "
              f"p10={np.percentile(w,10):.0f} | p90={np.percentile(w,90):.0f}")

        bin_edges = np.arange(0, WAGE_MAX + BIN_WIDTH, BIN_WIDTH)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        counts, _ = np.histogram(w, bins=bin_edges, weights=wt)
        total = counts.sum()
        shares = counts / total if total > 0 else counts * 0

        results[label] = {
            'counts': counts, 'shares': shares,
            'total': float(total), 'bin_centers': bin_centers,
        }

    delta = results['post']['shares'] - results['pre']['shares']
    bin_centers = results['pre']['bin_centers']

    below_new = bin_centers < mw_new
    at_new = (bin_centers >= mw_new) & (bin_centers < mw_new + 4 * BIN_WIDTH)

    missing = -delta[below_new & (delta < 0)].sum()
    excess = delta[at_new & (delta > 0)].sum()

    total_pre = results['pre']['total']
    total_post = results['post']['total']
    emp_change = (total_post / total_pre - 1) * 100 if total_pre > 0 else 0.0
    ratio = excess / missing if missing > 0 else float('inf')

    print(f"\n  Missing mass below S/{mw_new}: {missing*100:.3f}pp")
    print(f"  Excess mass at S/{mw_new}: {excess*100:.3f}pp")
    print(f"  RATIO: {ratio:.3f}")
    print(f"  Lima employed change: {emp_change:+.2f}%")

    return {
        'missing_mass_pp': float(missing * 100),
        'excess_mass_pp': float(excess * 100),
        'ratio': float(ratio),
        'employment_change_pct': float(emp_change),
        'total_pre': float(total_pre),
        'total_post': float(total_post),
        'bin_data': {
            'bin_centers': bin_centers.tolist(),
            'delta': delta.tolist(),
            'shares_pre': results['pre']['shares'].tolist(),
            'shares_post': results['post']['shares'].tolist(),
        }
    }


# ===================================================
# APPROACH 2: PANEL DiD (matched workers)
# ===================================================

def panel_did_lima(event_id, ecfg, is_pretrend=False):
    """
    Panel DiD: match workers across pre and post quarters.
    Treatment: wage_pre in [0.85*MW_old, MW_new)  (near-MW workers)
    Control:   wage_pre in [1.05*MW_new, 2.0*MW_old)  (safely above MW)
    Outcomes: employment retention, formality change
    """
    print(f"\n{'='*60}")
    print(f"PANEL DiD (Lima) {'PRE-TREND: ' if is_pretrend else ''}Event {event_id}: {ecfg['name']}")
    print(f"{'='*60}")

    mw_old = ecfg['mw_old']
    mw_new = ecfg['mw_new']

    # Treatment and control bands
    treat_lo = 0.85 * mw_old
    treat_hi = mw_new
    ctrl_lo = 1.05 * mw_new
    ctrl_hi = 2.0 * mw_old

    print(f"  Treatment band: [{treat_lo:.0f}, {treat_hi:.0f})")
    print(f"  Control band:   [{ctrl_lo:.0f}, {ctrl_hi:.0f})")

    df_pre_raw = load_epe(ecfg['pre_dir'])
    df_post_raw = load_epe(ecfg['post_dir'])

    pre = featurize_epe(df_pre_raw)
    post = featurize_epe(df_post_raw)

    # Build employed pre-sample
    emp_pre = pre[(pre['employed'] == 1) & (pre['wage'] > 0) & (pre['wt'] > 0)].copy()
    emp_pre = emp_pre.rename(columns={
        'employed': 'employed_pre', 'wage': 'wage_pre',
        'formal': 'formal_pre', 'wt': 'wt_pre'
    })

    # Merge with post on pid
    post_sub = post[['pid', 'employed', 'formal']].copy()
    post_sub = post_sub.rename(columns={
        'employed': 'employed_post', 'formal': 'formal_post'
    })

    panel = emp_pre.merge(post_sub, on='pid', how='inner')
    n_matched = len(panel)
    n_pre = len(emp_pre)
    match_rate = n_matched / n_pre * 100 if n_pre > 0 else 0
    print(f"  Panel match: {n_matched:,} of {n_pre:,} pre-period workers ({match_rate:.1f}%)")

    if n_matched < 50:
        print("  SKIP: too few matched observations")
        return None

    # Assign treatment/control
    panel['treat'] = (
        (panel['wage_pre'] >= treat_lo) & (panel['wage_pre'] < treat_hi)
    ).astype(int)
    panel['control'] = (
        (panel['wage_pre'] >= ctrl_lo) & (panel['wage_pre'] <= ctrl_hi)
    ).astype(int)

    sample = panel[(panel['treat'] == 1) | (panel['control'] == 1)].copy()
    n_treat = (sample['treat'] == 1).sum()
    n_ctrl = (sample['control'] == 1).sum()
    print(f"  Treated: {n_treat:,}, Control: {n_ctrl:,}")

    if n_treat < 20 or n_ctrl < 20:
        print("  SKIP: too few treat/control obs")
        return None

    results = {}

    # Employment retention DiD
    t = sample[sample['treat'] == 1]
    c = sample[sample['control'] == 1]

    emp_pre_t = t['employed_post'].notna().astype(float)  # matched = kept job
    emp_pre_c = c['employed_post'].notna().astype(float)

    # For employed_post: we matched on pid, so all matched workers were employed in pre
    # employed_post = 1 if still employed in post quarter
    emp_post_t = t['employed_post'].fillna(0)
    emp_post_c = c['employed_post'].fillna(0)

    # DiD for employment retention
    d_emp = (emp_post_t.mean() - 1.0) - (emp_post_c.mean() - 1.0)
    # Equivalently: employment change treat vs employment change control
    # All were employed in pre (ocu200==1), compare post employment rate
    d_emp2 = emp_post_t.mean() - emp_post_c.mean()

    print(f"\n  Employment retention:")
    print(f"    Treat post-emp rate: {emp_post_t.mean():.3f}")
    print(f"    Control post-emp rate: {emp_post_c.mean():.3f}")
    print(f"    DiD (treat - control): {d_emp2:+.4f}")

    # Formal DiD (among employed in both pre and post)
    stayers = sample[
        (sample['employed_post'] == 1) &
        sample['formal_pre'].notna() &
        sample['formal_post'].notna()
    ].copy()

    n_stayers_t = (stayers['treat'] == 1).sum()
    n_stayers_c = (stayers['control'] == 1).sum()
    print(f"\n  Formality (stayers: treat={n_stayers_t}, ctrl={n_stayers_c}):")

    formal_results = None
    if n_stayers_t >= 10 and n_stayers_c >= 10:
        st = stayers[stayers['treat'] == 1]
        sc = stayers[stayers['control'] == 1]

        d_formal = (st['formal_post'].mean() - st['formal_pre'].mean()) - \
                   (sc['formal_post'].mean() - sc['formal_pre'].mean())

        print(f"    Treat: {st['formal_pre'].mean():.3f} -> {st['formal_post'].mean():.3f} (Δ={st['formal_post'].mean()-st['formal_pre'].mean():+.3f})")
        print(f"    Ctrl:  {sc['formal_pre'].mean():.3f} -> {sc['formal_post'].mean():.3f} (Δ={sc['formal_post'].mean()-sc['formal_pre'].mean():+.3f})")
        print(f"    DiD (formality): {d_formal:+.4f}")

        # OLS with formal_pre control
        try:
            import statsmodels.api as sm
            X = sm.add_constant(stayers[['treat', 'formal_pre']])
            y = stayers['formal_post']
            mod = sm.OLS(y, X).fit(cov_type='HC3')
            beta = float(mod.params.get('treat', np.nan))
            se = float(mod.bse.get('treat', np.nan))
            p = float(mod.pvalues.get('treat', np.nan))
            stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))
            print(f"    OLS (formal_post ~ treat + formal_pre): beta={beta:+.4f} SE={se:.4f} p={p:.3f}{stars}")
            formal_results = {'beta': beta, 'se': se, 'p': p,
                              'n_treat': int(n_stayers_t), 'n_ctrl': int(n_stayers_c),
                              'did_raw': float(d_formal)}
        except Exception as e:
            print(f"    OLS failed: {e}")
            formal_results = {'did_raw': float(d_formal),
                              'n_treat': int(n_stayers_t), 'n_ctrl': int(n_stayers_c)}
    else:
        print(f"    Insufficient stayers for formality DiD")

    # Log wage DiD (among stayers with positive wage in both periods)
    # Note: post wages not available in cross-sectional match, so we skip wage DiD

    return {
        'n_matched': int(n_matched),
        'match_rate': float(match_rate),
        'n_treat': int(n_treat),
        'n_ctrl': int(n_ctrl),
        'employment': {
            'emp_rate_treat_post': float(emp_post_t.mean()),
            'emp_rate_ctrl_post': float(emp_post_c.mean()),
            'did': float(d_emp2),
            'n_treat': int(n_treat), 'n_ctrl': int(n_ctrl),
        },
        'formality': formal_results,
    }


# ===================================================
# MAIN
# ===================================================
if __name__ == '__main__':
    all_bunching = {}
    all_panel = {}
    all_pretrend = {}

    # Run all events
    for eid, ecfg in EVENTS.items():
        print(f"\n{'#'*65}")
        print(f"# EVENT {eid}: {ecfg['name']}")
        print(f"{'#'*65}")

        try:
            bres = cengiz_lima(eid, ecfg)
            all_bunching[eid] = bres
        except Exception as e:
            print(f"  Bunching ERROR: {e}")
            import traceback; traceback.print_exc()

        try:
            pres = panel_did_lima(eid, ecfg)
            all_panel[eid] = pres
        except Exception as e:
            print(f"  Panel DiD ERROR: {e}")
            import traceback; traceback.print_exc()

    # Pre-trend tests
    print(f"\n{'#'*65}")
    print("# PRE-TREND TESTS")
    print(f"{'#'*65}")
    for pt_id, pt_cfg in PRETREND.items():
        try:
            pt_res = panel_did_lima(pt_id, pt_cfg, is_pretrend=True)
            all_pretrend[pt_id] = pt_res
        except Exception as e:
            print(f"  Pre-trend {pt_id} ERROR: {e}")

    # Summary
    print(f"\n{'='*65}")
    print("EPE LIMA — SUMMARY")
    print(f"{'='*65}")

    print("\nCENGIZ BUNCHING (Lima wage distribution):")
    print(f"{'Event':<35} {'Missing':>10} {'Excess':>10} {'Ratio':>8} {'EmpChg':>8}")
    print("-" * 75)
    for eid, r in all_bunching.items():
        print(f"{EVENTS[eid]['name']:<35} "
              f"{r['missing_mass_pp']:>9.3f}pp "
              f"{r['excess_mass_pp']:>9.3f}pp "
              f"{r['ratio']:>8.3f} "
              f"{r['employment_change_pct']:>+7.2f}%")

    print("\nPANEL DiD (treated=near MW, control=above MW):")
    print(f"{'Event':<35} {'Emp DiD':>10} {'Formal DiD':>12} {'N matched':>10}")
    print("-" * 70)
    for eid, r in all_panel.items():
        if r is None:
            print(f"{EVENTS[eid]['name']:<35} INSUFFICIENT DATA")
            continue
        emp_did = r['employment']['did']
        frm_did = r['formality']['did_raw'] if r.get('formality') else np.nan
        print(f"{EVENTS[eid]['name']:<35} "
              f"{emp_did:>+9.4f}  "
              f"{frm_did:>+11.4f}  "
              f"{r['n_matched']:>9,}")

    print("\nPRE-TREND TESTS (employment DiD should be ~0):")
    for pt_id, r in all_pretrend.items():
        if r is None:
            print(f"  {pt_id}: INSUFFICIENT DATA")
            continue
        emp_did = r['employment']['did']
        verdict = "PASS" if abs(emp_did) < 0.10 else "FAIL"
        print(f"  {pt_id}: emp DiD = {emp_did:+.4f}  [{verdict}]")

    # Save
    output = {
        'methodology_1': 'Cengiz bunching on Lima wage distribution (EPE)',
        'methodology_2': 'Panel DiD — matched workers pre/post, treat=near MW, ctrl=above MW',
        'data_source': 'EPE Lima (datosabiertos.gob.pe)',
        'formality_definition': 'p222 in {1,2,3} = EsSalud/Seguro Privado/Ambos',
        'events': {eid: EVENTS[eid]['name'] for eid in EVENTS},
        'bunching': all_bunching,
        'panel_did': {k: v for k, v in all_panel.items() if v is not None},
        'pretrend': {k: v for k, v in all_pretrend.items() if v is not None},
    }

    out_path = 'D:/Nexus/nexus/exports/data/mw_epe_lima_results.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved to {out_path}")

    qhaw_path = 'D:/qhawarina/public/assets/data/mw_epe_lima_results.json'
    os.makedirs(os.path.dirname(qhaw_path), exist_ok=True)
    shutil.copy(out_path, qhaw_path)
    print(f"Copied to {qhaw_path}")
