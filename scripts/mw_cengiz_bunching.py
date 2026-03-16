"""
Cengiz et al. (2019) bunching estimator for Peru MW increases.
Reference: QJE 134(3): 1405-1454.

For each MW event:
  1. Load ENAHO Module 500 for pre-year and post-year
  2. Filter to employed workers (ocu500==1) with wage > 0
  3. Convert wages to MONTHLY if needed (i524a1/12 for annual)
  4. Apply expansion factors
  5. Build wage bins (S/25 width)
  6. Count weighted jobs per bin
  7. Compute missing mass below new MW and excess mass at/above
  8. Report ratio (excess/missing ≈ 1.0 means no net job loss)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import json
import os
import glob
import shutil

# === CONFIGURATION ===
EVENTS = {
    'A': {'name': 'S/750→S/850 (May 2016)', 'mw_old': 750, 'mw_new': 850,
           'pre_year': 2015, 'post_year': 2017},
    'B': {'name': 'S/850→S/930 (Apr 2018)', 'mw_old': 850, 'mw_new': 930,
           'pre_year': 2017, 'post_year': 2019},
    'C': {'name': 'S/930→S/1025 (May 2022)', 'mw_old': 930, 'mw_new': 1025,
           'pre_year': 2021, 'post_year': 2023},
}

BIN_WIDTH = 25  # soles per bin
WAGE_MIN = 0
WAGE_MAX = 5000
BASE_PATH = 'D:/Nexus/nexus/data/raw/enaho'


# === HELPER: Load ENAHO Module 500 for a given year ===
def load_enaho_year(year):
    """Find and load ENAHO Module 500 for a given year."""
    patterns = [
        f'{BASE_PATH}/**/enaho01a-{year}-500.dta',
        f'{BASE_PATH}/**/enaho01a-{year}-500.sav',
        f'{BASE_PATH}/**/*{year}*500*.dta',
        f'{BASE_PATH}/**/*{year}*500*.sav',
        f'{BASE_PATH}/cross_section/{year}/**/*500*.dta',
        f'{BASE_PATH}/cross_section/{year}/**/*500*.sav',
    ]

    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            f = files[0]
            print(f"  Loading {f}")
            if f.endswith('.dta'):
                return pd.read_stata(f, convert_categoricals=False)
            elif f.endswith('.sav'):
                import pyreadstat
                df, meta = pyreadstat.read_sav(f)
                return df

    raise FileNotFoundError(f"Cannot find ENAHO Module 500 for {year}")


def get_monthly_wage(df):
    """Extract monthly wage, handling annual vs monthly variables."""
    cols = [c.lower() for c in df.columns]
    df.columns = cols

    if 'i524a1' in df.columns:
        # Annual income → monthly
        w = pd.to_numeric(df['i524a1'], errors='coerce') / 12.0
        print(f"  Using i524a1/12 (annual→monthly)")
    elif 'p524a1' in df.columns:
        # Already monthly
        w = pd.to_numeric(df['p524a1'], errors='coerce')
        print(f"  Using p524a1 (monthly)")
    elif 'd529t' in df.columns:
        w = pd.to_numeric(df['d529t'], errors='coerce')
        print(f"  Using d529t (total monthly labor income)")
    else:
        raise ValueError("No wage variable found")
    return w


def get_weight(df):
    """Get expansion factor."""
    for wvar in ['fac500a', 'facpob07', 'factor07', 'fac500']:
        if wvar in df.columns:
            w = pd.to_numeric(df[wvar], errors='coerce').fillna(0)
            print(f"  Using weight: {wvar}")
            return w
    print("  WARNING: No weight found, using uniform weights")
    return pd.Series(1.0, index=df.index)


def get_employed(df):
    """Filter to employed workers."""
    for evar in ['ocu500', 'ocupinf']:
        if evar in df.columns:
            emp = pd.to_numeric(df[evar], errors='coerce')
            mask = emp == 1
            print(f"  Employed filter ({evar}==1): {mask.sum():,} workers")
            return mask
    # Fallback: anyone with positive wage
    print("  WARNING: No employment variable, using wage > 0")
    return pd.Series(True, index=df.index)


def cengiz_bunching(event_id, event_config):
    """Run Cengiz bunching for one MW event."""
    print(f"\n{'='*60}")
    print(f"EVENT {event_id}: {event_config['name']}")
    print(f"{'='*60}")

    mw_old = event_config['mw_old']
    mw_new = event_config['mw_new']

    # Load pre and post data
    print(f"\nPre-period ({event_config['pre_year']}):")
    df_pre = load_enaho_year(event_config['pre_year'])
    print(f"\nPost-period ({event_config['post_year']}):")
    df_post = load_enaho_year(event_config['post_year'])

    results = {}
    for label, df in [('pre', df_pre), ('post', df_post)]:
        # Get monthly wages
        wage = get_monthly_wage(df)
        weight = get_weight(df)
        employed = get_employed(df)

        # Filter: employed, positive wage
        mask = employed & (wage > 0) & (wage < WAGE_MAX) & weight.notna()
        w = wage[mask].values
        wt = weight[mask].values

        print(f"  {label}: {mask.sum():,} workers with valid wage")
        print(f"  Wage stats: median={np.median(w):.0f}, "
              f"mean={np.average(w, weights=wt):.0f}, "
              f"p10={np.percentile(w, 10):.0f}, "
              f"p90={np.percentile(w, 90):.0f}")

        # Bin wages
        bin_edges = np.arange(WAGE_MIN, WAGE_MAX + BIN_WIDTH, BIN_WIDTH)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        counts, _ = np.histogram(w, bins=bin_edges, weights=wt)

        # Normalize to shares
        total = counts.sum()
        shares = counts / total

        results[label] = {
            'counts': counts,
            'shares': shares,
            'total': total,
            'bin_centers': bin_centers,
            'bin_edges': bin_edges
        }

    # Compute delta (change in share per bin)
    delta = results['post']['shares'] - results['pre']['shares']
    bin_centers = results['pre']['bin_centers']

    # Define regions relative to NEW MW
    below_new = bin_centers < mw_new
    at_new = (bin_centers >= mw_new) & (bin_centers < mw_new + 4 * BIN_WIDTH)
    above_buffer = bin_centers >= mw_new + 4 * BIN_WIDTH

    # Missing mass: jobs that disappeared below the new MW
    missing = -delta[below_new & (delta < 0)].sum()

    # Excess mass: jobs that appeared at/near the new MW
    excess = delta[at_new & (delta > 0)].sum()

    # Net effect on total employment
    total_pre = results['pre']['total']
    total_post = results['post']['total']
    emp_change = (total_post / total_pre - 1) * 100

    # Ratio
    ratio = excess / missing if missing > 0 else float('inf')

    # Print results
    print(f"\n--- CENGIZ RESULTS: Event {event_id} ---")
    print(f"MW: S/{mw_old} → S/{mw_new}")
    print(f"Pre workers: {total_pre:,.0f}")
    print(f"Post workers: {total_post:,.0f}")
    print(f"Total employment change: {emp_change:+.2f}%")
    print(f"")
    print(f"Missing mass below S/{mw_new}: {missing*100:.3f}pp ({missing*total_pre:,.0f} workers)")
    print(f"Excess mass at S/{mw_new}±{4*BIN_WIDTH}: {excess*100:.3f}pp ({excess*total_pre:,.0f} workers)")
    print(f"RATIO (excess/missing): {ratio:.3f}")
    print(f"")
    if ratio > 0.8:
        print(f"→ INTERPRETATION: No net job destruction. Jobs below MW were pushed up, not eliminated.")
    elif ratio > 0.5:
        print(f"→ INTERPRETATION: Some job loss, but majority of affected workers saw wage gains.")
    else:
        print(f"→ INTERPRETATION: Significant job loss. More jobs disappeared than reappeared.")

    # Print bin-level detail around the MW threshold
    print(f"\nBin detail around S/{mw_new} (±S/200):")
    mask_detail = np.abs(bin_centers - mw_new) <= 200
    print(f"  {'Bin center':>12} {'Δ share (pp)':>14} {'Pre share':>12} {'Post share':>12}")
    for i in np.where(mask_detail)[0]:
        marker = " ←MW" if abs(bin_centers[i] - mw_new) < BIN_WIDTH else ""
        print(f"  {bin_centers[i]:>12.0f} {delta[i]*100:>+13.4f}pp "
              f"{results['pre']['shares'][i]*100:>11.4f}% "
              f"{results['post']['shares'][i]*100:>11.4f}%{marker}")

    return {
        'event_id': event_id,
        'name': event_config['name'],
        'mw_old': mw_old,
        'mw_new': mw_new,
        'pre_year': event_config['pre_year'],
        'post_year': event_config['post_year'],
        'total_pre': float(total_pre),
        'total_post': float(total_post),
        'employment_change_pct': float(emp_change),
        'missing_mass_pp': float(missing * 100),
        'excess_mass_pp': float(excess * 100),
        'ratio': float(ratio),
        'bin_data': {
            'bin_centers': bin_centers.tolist(),
            'delta': delta.tolist(),
            'shares_pre': results['pre']['shares'].tolist(),
            'shares_post': results['post']['shares'].tolist()
        }
    }


# === RUN ALL EVENTS ===
if __name__ == '__main__':
    all_results = {}

    for eid, econfig in EVENTS.items():
        try:
            result = cengiz_bunching(eid, econfig)
            all_results[eid] = result
        except Exception as e:
            print(f"\nERROR on Event {eid}: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY: CENGIZ BUNCHING ACROSS EVENTS")
    print(f"{'='*60}")
    print(f"{'Event':<30} {'Missing':>10} {'Excess':>10} {'Ratio':>8} {'Emp Δ%':>8}")
    print("-" * 70)
    for eid, r in all_results.items():
        print(f"{r['name']:<30} {r['missing_mass_pp']:>9.3f}pp {r['excess_mass_pp']:>9.3f}pp "
              f"{r['ratio']:>8.3f} {r['employment_change_pct']:>+7.2f}%")

    # Save
    output = {
        'methodology': 'Cengiz et al. (2019) bunching estimator',
        'reference': 'QJE 134(3): 1405-1454',
        'description': 'Missing mass below new MW vs excess mass at/above new MW. '
                       'Ratio ≈ 1.0 means no net job destruction.',
        'bin_width': BIN_WIDTH,
        'events': all_results
    }

    out_path = 'D:/Nexus/nexus/exports/data/mw_cengiz_results.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    qhaw_path = 'D:/qhawarina/public/assets/data/mw_cengiz_results.json'
    os.makedirs(os.path.dirname(qhaw_path), exist_ok=True)
    shutil.copy(out_path, qhaw_path)
    print(f"Copied to {qhaw_path}")
