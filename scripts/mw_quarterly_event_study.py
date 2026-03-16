"""
Strategy 2: Quarterly Event Study for Peru MW Increases.
Uses ENAHO quarterly rolling data with tight event windows.

Events:
  A: S/750→S/850 (May 2016)  pre=Q1_2016, post=Q1_2017, Kaitz=750/median_Q1_2016
  B: S/850→S/930 (Apr 2018)  pre=Q1_2018, post=Q1_2019, Kaitz=850/median_Q1_2018
  C: S/930→S/1025 (May 2022) pre=Q1_2022, post=SKIP (no quarterly post data on disk)

Pre-trend tests:
  Event A pre-trend: Q3_2015 → Q4_2015 (no MW change; β should ≈ 0)
  Event B pre-trend: Q1_2016 → Q1_2017 (both post-A, before Event B; β should ≈ 0)

Spec: Y_idt = α_d + γ_t + β(Post_t × Kaitz_d_pre) + Edad + Edad² + Sexo + ε_idt
      Clustered SEs at department level (25 departments).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import json
import os
import shutil
from scipy import stats

QT_DIR = 'D:/Nexus/nexus/data/raw/enaho/quarterly/'

# Quarterly file registry
FILES = {
    'Q3_2015': QT_DIR + 'enaho_qt_Q1_2015a_code484.dta',   # Jul-Sep 2015
    'Q4_2015': QT_DIR + 'enaho_qt_Q4_2015_code494.dta',    # Oct-Dec 2015
    'Q1_2016': QT_DIR + 'enaho_qt_Q1_2016_code510.dta',    # Jan-Mar 2016
    'Q1_2017': QT_DIR + 'enaho_qt_Q1_2017_code551.dta',    # Jan-Mar 2017
    'Q1_2018': QT_DIR + 'enaho_qt_Q1_2018_code607.dta',    # Jan-Mar 2018
    'Q1_2019': QT_DIR + 'enaho_qt_Q1_2019_code641.dta',    # Jan-Mar 2019
    'Q1_2022': QT_DIR + 'enaho_qt_Q1_2022_code765.dta',    # Jan-Mar 2022
}

EVENTS = {
    'A': {
        'name': 'S/750->S/850 (May 2016)',
        'mw_old': 750, 'mw_new': 850,
        'qt_pre': 'Q1_2016', 'qt_post': 'Q1_2017',
        'pretrend_pre': 'Q3_2015', 'pretrend_post': 'Q4_2015',
        'pretrend_label': 'Q3->Q4 2015 (no MW change)',
    },
    'B': {
        'name': 'S/850->S/930 (Apr 2018)',
        'mw_old': 850, 'mw_new': 930,
        'qt_pre': 'Q1_2018', 'qt_post': 'Q1_2019',
        'pretrend_pre': 'Q1_2016', 'pretrend_post': 'Q1_2017',
        'pretrend_label': 'Q1_2016->Q1_2017 (post-Event A, no new increase)',
    },
    # Event C: no quarterly post data available
}


def load_qt(qt_key):
    """Load and standardize a quarterly ENAHO file."""
    path = FILES[qt_key]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    print(f"  Loading {qt_key}: {path.split('/')[-1]}")
    df = pd.read_stata(path, convert_categoricals=False)
    df.columns = [c.lower() for c in df.columns]
    return df


def featurize(df, qt_label):
    """Extract person-level outcomes and controls."""
    out = pd.DataFrame()

    # Department (first 2 digits of ubigeo)
    out['dept'] = df['ubigeo'].astype(str).str.zfill(6).str[:2]

    # Employment (ocu500 == 1)
    out['employed'] = (pd.to_numeric(df['ocu500'], errors='coerce') == 1).astype(float)

    # Formality — ocupinf==2 (2017+), else labor contract p510a1==1
    if 'ocupinf' in df.columns:
        out['formal'] = (pd.to_numeric(df['ocupinf'], errors='coerce') == 2).astype(float)
    elif 'p510a1' in df.columns:
        out['formal'] = (pd.to_numeric(df['p510a1'], errors='coerce') == 1).astype(float)
    else:
        out['formal'] = np.nan

    # Wage (p524a1 = monthly in quarterly files)
    w = pd.to_numeric(df['p524a1'], errors='coerce')
    out['wage'] = w
    out['log_wage'] = np.where((w > 0) & out['employed'].astype(bool), np.log(w), np.nan)

    # Hours (p520)
    if 'p520' in df.columns:
        h = pd.to_numeric(df['p520'], errors='coerce')
        out['hours'] = np.where(out['employed'].astype(bool) & (h > 0) & (h < 100), h, np.nan)
    else:
        out['hours'] = np.nan

    # Controls
    # Age: p208a
    age = pd.to_numeric(df.get('p208a', pd.Series(np.nan, index=df.index)), errors='coerce')
    out['edad'] = age
    out['edad_sq'] = age ** 2

    # Sex: p207 (1=male, 2=female) → male dummy
    sex = pd.to_numeric(df.get('p207', pd.Series(np.nan, index=df.index)), errors='coerce')
    out['male'] = (sex == 1).astype(float)

    # Weight
    out['wt'] = pd.to_numeric(df['fac500'], errors='coerce').fillna(0)

    out['qt'] = qt_label
    return out


def compute_kaitz(df_pre, mw_old):
    """
    Compute department-level Kaitz = MW_old / median_formal_wage_dept.
    Uses all employed workers with positive wage (quarterly data doesn't
    always have good formal_wage variable, so use all employed wage > 0).
    """
    sub = df_pre[df_pre['employed'] == 1].copy()
    sub = sub[sub['wage'] > 0].copy()

    def wmedian(g):
        g = g.sort_values('wage')
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
    print(f"  Kaitz (MW={mw_old}/dept_median): n_depts={kaitz.notna().sum()}, "
          f"range=[{kaitz.min():.3f},{kaitz.max():.3f}], mean={kaitz.mean():.3f}")
    return kaitz


def run_did(df_all, outcomes, event_label):
    """
    Run DiD: Y_idt = alpha_d + gamma_t + beta*(post*kaitz_pre) + edad + edad_sq + male
    Clustered SEs at dept level.
    Returns dict of results per outcome.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("ERROR: statsmodels not available")
        return {}

    results = {}
    for outcome in outcomes:
        sub = df_all.dropna(subset=[outcome, 'kaitz_pre', 'edad', 'male', 'wt']).copy()
        sub = sub[sub['wt'] > 0]

        if len(sub) < 100:
            print(f"  {outcome}: too few obs ({len(sub)}), skip")
            results[outcome] = None
            continue

        n_depts = sub['dept'].nunique()
        n_obs = len(sub)

        try:
            formula = f'{outcome} ~ post + post:kaitz_pre + edad + edad_sq + male + C(dept)'
            mod = smf.wls(formula, data=sub, weights=sub['wt']).fit(
                cov_type='cluster',
                cov_kwds={'groups': sub['dept']}
            )

            # Extract the interaction coefficient
            inter_key = 'post:kaitz_pre'
            beta = float(mod.params.get(inter_key, np.nan))
            se = float(mod.bse.get(inter_key, np.nan))
            pval = float(mod.pvalues.get(inter_key, np.nan))
            ci = mod.conf_int()
            ci_low = float(ci.loc[inter_key, 0]) if inter_key in ci.index else np.nan
            ci_high = float(ci.loc[inter_key, 1]) if inter_key in ci.index else np.nan

            stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.10 else ''))
            print(f"  {outcome:12s}: beta={beta:+.4f} SE={se:.4f} "
                  f"[{ci_low:+.4f},{ci_high:+.4f}] p={pval:.3f}{stars} "
                  f"N={n_obs:,} depts={n_depts}")

            results[outcome] = {
                'beta': beta, 'se': se,
                'ci_low': ci_low, 'ci_high': ci_high,
                'p': pval, 'n_obs': n_obs, 'n_depts': n_depts
            }

        except Exception as e:
            print(f"  {outcome}: regression failed — {e}")
            results[outcome] = None

    return results


def run_event(event_id, ecfg):
    """Run full quarterly DiD for one event."""
    print(f"\n{'='*65}")
    print(f"QUARTERLY EVENT {event_id}: {ecfg['name']}")
    print(f"  Pre: {ecfg['qt_pre']}, Post: {ecfg['qt_post']}")
    print(f"{'='*65}")

    # Load pre and post
    print(f"\nLoading pre ({ecfg['qt_pre']}):")
    df_pre_raw = load_qt(ecfg['qt_pre'])
    df_pre = featurize(df_pre_raw, ecfg['qt_pre'])

    print(f"Loading post ({ecfg['qt_post']}):")
    df_post_raw = load_qt(ecfg['qt_post'])
    df_post = featurize(df_post_raw, ecfg['qt_post'])

    # Compute Kaitz in pre-period
    print(f"\nKaitz computation (MW={ecfg['mw_old']}):")
    kaitz = compute_kaitz(df_pre, ecfg['mw_old'])

    # Build combined dataset
    df_pre['post'] = 0
    df_post['post'] = 1
    df_all = pd.concat([df_pre, df_post], ignore_index=True)
    df_all = df_all.merge(kaitz.reset_index().rename(columns={'index': 'dept'}),
                          on='dept', how='left')

    # Summary
    print(f"\nSample: pre={df_pre['employed'].notna().sum():,}, "
          f"post={df_post['employed'].notna().sum():,}")
    print(f"Employed (pre): {df_pre['employed'].sum():,.0f} weighted="
          f"{(df_pre['employed']*df_pre['wt']).sum()/df_pre['wt'].sum():.3f}")

    # Main DiD
    print(f"\n--- MAIN DiD RESULTS: Event {event_id} ---")
    outcomes = ['employed', 'formal', 'log_wage', 'hours']
    main_results = run_did(df_all, outcomes, ecfg['name'])

    # Pre-trend test
    print(f"\n--- PRE-TREND TEST: {ecfg['pretrend_label']} ---")
    print(f"  (β should ≈ 0 if parallel trends holds)")
    pt_pre_raw = load_qt(ecfg['pretrend_pre'])
    pt_post_raw = load_qt(ecfg['pretrend_post'])
    pt_pre = featurize(pt_pre_raw, ecfg['pretrend_pre'])
    pt_post = featurize(pt_post_raw, ecfg['pretrend_post'])

    # For pre-trend, use same Kaitz as main event (treatment assignment is permanent)
    pt_pre['post'] = 0
    pt_post['post'] = 1
    pt_all = pd.concat([pt_pre, pt_post], ignore_index=True)
    pt_all = pt_all.merge(kaitz.reset_index().rename(columns={'index': 'dept'}),
                          on='dept', how='left')
    pt_results = run_did(pt_all, ['employed', 'formal', 'log_wage'], 'pre-trend')

    return {
        'event_id': event_id,
        'name': ecfg['name'],
        'mw_old': ecfg['mw_old'],
        'mw_new': ecfg['mw_new'],
        'qt_pre': ecfg['qt_pre'],
        'qt_post': ecfg['qt_post'],
        'main': main_results,
        'pretrend': pt_results,
        'pretrend_label': ecfg['pretrend_label'],
    }


def ivw_pool(event_results, outcomes):
    """Inverse-variance weighted pooled estimates across events."""
    pooled = {}
    for outcome in outcomes:
        betas, ses, labels = [], [], []
        for eid, r in event_results.items():
            res = r['main'].get(outcome)
            if res and not np.isnan(res['beta']) and not np.isnan(res['se']) and res['se'] > 0:
                betas.append(res['beta'])
                ses.append(res['se'])
                labels.append(eid)

        if len(betas) < 2:
            pooled[outcome] = None
            continue

        weights = [1 / se**2 for se in ses]
        total_w = sum(weights)
        beta_pool = sum(b * w for b, w in zip(betas, weights)) / total_w
        se_pool = 1 / np.sqrt(total_w)
        z = beta_pool / se_pool
        p_pool = 2 * (1 - stats.norm.cdf(abs(z)))
        ci_low = beta_pool - 1.96 * se_pool
        ci_high = beta_pool + 1.96 * se_pool

        # Cochran's Q
        Q = sum(w * (b - beta_pool)**2 for b, w in zip(betas, weights))
        df_Q = len(betas) - 1
        I2 = max(0.0, (Q - df_Q) / Q * 100) if Q > 0 else 0.0
        p_Q = 1 - stats.chi2.cdf(Q, df_Q)

        pooled[outcome] = {
            'beta': beta_pool, 'se': se_pool,
            'ci_low': ci_low, 'ci_high': ci_high,
            'p': p_pool,
            'Q': Q, 'I2': I2, 'p_Q': p_Q,
            'n_events': len(betas),
            'events': labels
        }
    return pooled


def print_summary(event_results, pooled):
    print(f"\n{'='*75}")
    print("QUARTERLY EVENT STUDY SUMMARY")
    print(f"{'='*75}")

    outcomes = ['employed', 'formal', 'log_wage', 'hours']
    for outcome in outcomes:
        print(f"\n{outcome.upper()}:")
        for eid, r in event_results.items():
            res = r['main'].get(outcome)
            if res:
                stars = '***' if res['p'] < 0.01 else ('**' if res['p'] < 0.05 else ('*' if res['p'] < 0.10 else '   '))
                print(f"  Event {eid} ({r['name']:30s}): "
                      f"beta={res['beta']:+.4f} SE={res['se']:.4f} "
                      f"[{res['ci_low']:+.4f},{res['ci_high']:+.4f}] "
                      f"p={res['p']:.3f}{stars}  N={res['n_obs']:,}")
            else:
                print(f"  Event {eid} ({r['name']:30s}): NOT AVAILABLE")

        pool = pooled.get(outcome)
        if pool:
            stars = '***' if pool['p'] < 0.01 else ('**' if pool['p'] < 0.05 else ('*' if pool['p'] < 0.10 else '   '))
            print(f"  POOLED IVW ({pool['n_events']} events)                    : "
                  f"beta={pool['beta']:+.4f} SE={pool['se']:.4f} "
                  f"[{pool['ci_low']:+.4f},{pool['ci_high']:+.4f}] "
                  f"p={pool['p']:.3f}{stars}  "
                  f"Q={pool['Q']:.2f} I2={pool['I2']:.1f}%")

    print(f"\n{'='*75}")
    print("PRE-TREND TESTS (β ≈ 0 validates parallel trends):")
    print(f"{'='*75}")
    for eid, r in event_results.items():
        print(f"\nEvent {eid}: {r['pretrend_label']}")
        for outcome in ['employed', 'formal', 'log_wage']:
            res = r['pretrend'].get(outcome)
            if res:
                verdict = "PASS" if abs(res['beta']) < 2 * res['se'] else "FAIL"
                print(f"  {outcome:12s}: beta={res['beta']:+.4f} SE={res['se']:.4f} "
                      f"p={res['p']:.3f}  [{verdict}]")


if __name__ == '__main__':
    event_results = {}

    for eid, ecfg in EVENTS.items():
        try:
            result = run_event(eid, ecfg)
            event_results[eid] = result
        except Exception as e:
            print(f"\nERROR on Event {eid}: {e}")
            import traceback
            traceback.print_exc()

    # Pool across events
    outcomes = ['employed', 'formal', 'log_wage', 'hours']
    pooled = ivw_pool(event_results, outcomes)

    # Print summary
    print_summary(event_results, pooled)

    # Serialize (remove non-JSON-serializable items)
    def clean_result(r):
        out = {k: v for k, v in r.items() if k not in ('main', 'pretrend')}
        out['main'] = {}
        out['pretrend'] = {}
        for k, v in r.get('main', {}).items():
            out['main'][k] = v if v is not None else None
        for k, v in r.get('pretrend', {}).items():
            out['pretrend'][k] = v if v is not None else None
        return out

    output = {
        'methodology': 'Quarterly DiD — Kaitz regional variation (Dube-Lester-Reich 2010)',
        'specification': 'Y_idt = alpha_d + gamma_t + beta*(Post_t x Kaitz_d_pre) + Edad + Edad2 + Sexo',
        'data': 'ENAHO quarterly rolling windows',
        'events': {eid: clean_result(r) for eid, r in event_results.items()},
        'pooled': {k: v for k, v in pooled.items()},
    }

    out_path = 'D:/Nexus/nexus/exports/data/mw_quarterly_results.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved to {out_path}")

    qhaw_path = 'D:/qhawarina/public/assets/data/mw_quarterly_results.json'
    os.makedirs(os.path.dirname(qhaw_path), exist_ok=True)
    shutil.copy(out_path, qhaw_path)
    print(f"Copied to {qhaw_path}")
