"""
mw_reviewer_analyses.py
=======================
Four reviewer-requested analyses:

1. Cumulative excess curve: R(W) for W in [50,100,...,500]
2. Counterfactual sensitivity for Event B (6 specs)
3. COVID robustness for Event C (pre=2019 instead of 2021)
4. Effective N in affected/excess zones

Saves:
  exports/figures/fig3_cumulative_excess.pdf
  exports/data/mw_reviewer_analyses.json
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, glob, json
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available — figure will be skipped")

ENAHO_CS = 'D:/Nexus/nexus/data/raw/enaho/cross_section'
OUT_FIG  = 'D:/Nexus/nexus/exports/figures'
OUT_DATA = 'D:/Nexus/nexus/exports/data'
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_DATA, exist_ok=True)

BIN_WIDTH = 25
BINS      = np.arange(0, 6025, BIN_WIDTH)
BC        = BINS[:-1] + BIN_WIDTH / 2   # bin centers

EVENTS = {
    'A': {'mw_old': 750,  'mw_new': 850,  'pre_year': 2015, 'post_year': 2017},
    'B': {'mw_old': 850,  'mw_new': 930,  'pre_year': 2017, 'post_year': 2019},
    'C': {'mw_old': 930,  'mw_new': 1025, 'pre_year': 2021, 'post_year': 2023},
}
DEPENDENT = {3.0, 4.0, 6.0}


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_enaho(year):
    patterns = [
        f'{ENAHO_CS}/modulo_05_{year}/enaho01a-{year}-500.dta',
        f'{ENAHO_CS}/modulo_05_{year}/enaho01a-{year}_500.dta',
        f'{ENAHO_CS}/modulo_05_{year}/enaho01a*{year}*500*.dta',
    ]
    for pat in patterns:
        files = glob.glob(pat, recursive=True)
        files = [f for f in files if not any(x in os.path.basename(f).lower()
                 for x in ['ciiu', 'ciuo', 'tabla', 'diccionario', 'doc'])]
        if files:
            path = sorted(files)[0]
            print(f"  Loading {year}: {os.path.basename(path)}")
            df = pd.read_stata(path, convert_categoricals=False)
            df.columns = [c.lower() for c in df.columns]
            return df
    raise FileNotFoundError(f"No ENAHO Module 500 for {year}")


def extract_workers(df, sample='formal_dep'):
    """
    Return (wages, weights, unweighted_n) for requested sample.
    sample: 'formal_dep' | 'informal' | 'all_dep'
    """
    emp = (pd.to_numeric(df.get('ocu500', pd.Series(np.nan, index=df.index)),
                         errors='coerce') == 1)

    # Dependency: cat07p500a1==2 or p507 in {3,4,6}
    if 'cat07p500a1' in df.columns:
        dep_raw = pd.to_numeric(df['cat07p500a1'], errors='coerce')
        dep = dep_raw == 2
    else:
        p507 = pd.to_numeric(df.get('p507', pd.Series(np.nan, index=df.index)),
                             errors='coerce')
        dep = p507.isin(DEPENDENT)

    # Formality: ocupinf==2
    if 'ocupinf' in df.columns:
        ocinf = pd.to_numeric(df['ocupinf'], errors='coerce')
        formal = ocinf == 2
        informal = ocinf == 1
    else:
        formal = pd.Series(False, index=df.index)
        informal = pd.Series(False, index=df.index)

    # Wage: p524a1 (monthly) > i524a1/12
    if 'p524a1' in df.columns:
        w = pd.to_numeric(df['p524a1'], errors='coerce')
        if w.notna().sum() < 100 or w.median() < 100:
            w = pd.to_numeric(df.get('i524a1', pd.Series(np.nan, index=df.index)),
                              errors='coerce') / 12.0
    else:
        w = pd.to_numeric(df.get('i524a1', pd.Series(np.nan, index=df.index)),
                          errors='coerce') / 12.0

    # Weight: factor07i500a > fac500a
    for wvar in ['factor07i500a', 'fac500a', 'facpob07', 'factor07']:
        if wvar in df.columns:
            wt = pd.to_numeric(df[wvar], errors='coerce')
            break
    else:
        wt = pd.Series(1.0, index=df.index)

    wage_ok = (w > 0) & w.notna() & wt.notna() & (wt > 0)

    if sample == 'formal_dep':
        mask = emp & dep & formal & wage_ok
    elif sample == 'informal':
        mask = emp & dep & informal & wage_ok
    elif sample == 'all_dep':
        mask = emp & dep & wage_ok
    else:
        mask = emp & wage_ok

    wages   = w[mask].values
    weights = wt[mask].values
    n_unw   = mask.sum()
    return wages, weights, n_unw


def make_shares(wages, weights):
    counts, _ = np.histogram(wages, bins=BINS, weights=weights)
    total = counts.sum()
    return counts / total if total > 0 else counts


# ─── CANONICAL BUNCHING CORE ──────────────────────────────────────────────────

def canonical_background(delta, mw_new, clean_threshold=2.0):
    """Inv-abs weighted mean of delta in clean zone."""
    clean = BC > clean_threshold * mw_new
    if clean.sum() < 5:
        return 0.0
    return float(np.average(delta[clean],
                            weights=1.0 / (np.abs(delta[clean]) + 1e-8)))


def compute_missing(delta_adj, mw_old, mw_new):
    """Neg-only missing mass from [0.85*mw_old, mw_new)."""
    affected_lo = 0.85 * mw_old
    zone = (BC >= affected_lo) & (BC < mw_new)
    return float(-delta_adj[zone & (delta_adj < 0)].sum())


def compute_excess(delta_adj, mw_new, window):
    """Pos-only excess mass from [mw_new, mw_new+window)."""
    zone = (BC >= mw_new) & (BC < mw_new + window)
    return float(delta_adj[zone & (delta_adj > 0)].sum())


# ─── CACHED DISTRIBUTIONS ─────────────────────────────────────────────────────

_cache = {}

def get_shares(year, sample='formal_dep'):
    key = (year, sample)
    if key not in _cache:
        df = _cache.get(('df', year))
        if df is None:
            df = load_enaho(year)
            _cache[('df', year)] = df
        wages, wts, n_unw = extract_workers(df, sample)
        _cache[key] = (make_shares(wages, wts), n_unw)
    return _cache[key]


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Cumulative Excess Curve — R(W) for W in [50,...,500]
# ═══════════════════════════════════════════════════════════════════════════════

WINDOWS = [50, 100, 150, 200, 250, 300, 400, 500]

def analysis1():
    print('\n' + '='*65)
    print('ANALYSIS 1: Cumulative Excess Curve R(W)')
    print('='*65)

    results = {}
    for eid, ecfg in EVENTS.items():
        mw_old, mw_new = ecfg['mw_old'], ecfg['mw_new']

        pre_shares, _  = get_shares(ecfg['pre_year'],  'formal_dep')
        post_shares, _ = get_shares(ecfg['post_year'], 'formal_dep')

        delta   = post_shares - pre_shares
        bg      = canonical_background(delta, mw_new)
        delta_adj = delta - bg
        missing = compute_missing(delta_adj, mw_old, mw_new)

        row = {}
        for W in WINDOWS:
            excess = compute_excess(delta_adj, mw_new, W)
            row[W] = round(excess / missing, 3) if missing > 0 else None
        results[eid] = row

        print(f"\nEvent {eid} (MW {mw_old}→{mw_new}): missing={missing*100:.3f}pp")

    # Print table
    print(f"\n{'Window W':>10}", end='')
    for eid in ['A', 'B', 'C']:
        print(f"  R({eid})", end='')
    print()
    print('-' * 40)
    for W in WINDOWS:
        print(f"{W:>10}", end='')
        for eid in ['A', 'B', 'C']:
            v = results[eid].get(W)
            print(f"  {v:.3f}" if v is not None else "     —", end='')
        print()

    # Plot Figure 3
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors  = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}
        labels  = {'A': 'Event A (S/750→850)',
                   'B': 'Event B (S/850→930)',
                   'C': 'Event C (S/930→1,025)'}
        for eid in ['A', 'B', 'C']:
            ys = [results[eid][W] for W in WINDOWS]
            ax.plot(WINDOWS, ys, marker='o', linewidth=2,
                    color=colors[eid], label=labels[eid])

        ax.axhline(1.0, color='black', linewidth=1.2, linestyle='--',
                   label='R = 1 (full redistribution)')
        ax.set_xlabel('Excess window width W (S/.)', fontsize=11)
        ax.set_ylabel('Bunching ratio R(W) = excess / missing', fontsize=11)
        ax.set_title('Figure 3: Bunching Ratio as Function of Excess Window Width\n'
                     'Formal dependent workers. Missing mass fixed at canonical [0.85×MW$_{old}$, MW$_{new}$).',
                     fontsize=10)
        ax.set_xticks(WINDOWS)
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, max(1.3, max(v for ev in results.values() for v in ev.values() if v)))
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        for ext in ['pdf', 'png']:
            path = f'{OUT_FIG}/fig3_cumulative_excess.{ext}'
            plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved: {path}")
        plt.close()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Counterfactual Sensitivity — Event B, 6 specs
# ═══════════════════════════════════════════════════════════════════════════════

def poly_background(delta, mw_new, degree=4):
    """Fit polynomial of given degree to delta in clean zone; predict for all bins."""
    clean = BC > 2.0 * mw_new
    if clean.sum() < degree + 2:
        return np.zeros_like(delta)
    try:
        coeffs = np.polyfit(BC[clean], delta[clean], degree)
        return np.polyval(coeffs, BC)
    except Exception:
        return np.zeros_like(delta)


def analysis2():
    print('\n' + '='*65)
    print('ANALYSIS 2: Counterfactual Sensitivity — Event B')
    print('='*65)

    ecfg = EVENTS['B']
    mw_old, mw_new = ecfg['mw_old'], ecfg['mw_new']
    pre_shares, _  = get_shares(ecfg['pre_year'],  'formal_dep')
    post_shares, _ = get_shares(ecfg['post_year'], 'formal_dep')
    delta = post_shares - pre_shares

    # W=250 for all specs (canonical excess window)
    W = 250

    specs = [
        ('Canonical',          'inv-abs (>2×MW)'),
        ('Uniform weights',    'uniform (>2×MW)'),
        ('Polynomial deg 4',   'poly4 (>2×MW)'),
        ('Clean at 1.5×MW',   'inv-abs (>1.5×MW)'),
        ('Clean at 2.5×MW',   'inv-abs (>2.5×MW)'),
        ('No correction',      'none'),
    ]

    rows = []
    for (spec_name, method_label) in specs:
        if spec_name == 'Canonical':
            bg = canonical_background(delta, mw_new, 2.0)
            delta_adj = delta - bg

        elif spec_name == 'Uniform weights':
            clean = BC > 2.0 * mw_new
            bg = float(np.mean(delta[clean])) if clean.sum() > 0 else 0.0
            delta_adj = delta - bg

        elif spec_name == 'Polynomial deg 4':
            bg_poly = poly_background(delta, mw_new, 4)
            delta_adj = delta - bg_poly

        elif spec_name == 'Clean at 1.5×MW':
            bg = canonical_background(delta, mw_new, 1.5)
            delta_adj = delta - bg

        elif spec_name == 'Clean at 2.5×MW':
            bg = canonical_background(delta, mw_new, 2.5)
            delta_adj = delta - bg

        elif spec_name == 'No correction':
            delta_adj = delta

        missing = compute_missing(delta_adj, mw_old, mw_new)
        excess  = compute_excess(delta_adj, mw_new, W)
        ratio   = round(excess / missing, 3) if missing > 0 else None

        rows.append({
            'spec':    spec_name,
            'method':  method_label,
            'missing': round(missing * 100, 3),
            'excess':  round(excess * 100, 3),
            'ratio':   ratio,
        })

    print(f"\n{'Spec':<22} {'Clean zone/weight':<20} {'Missing':>9} {'Excess':>9} {'R':>7}")
    print('-' * 72)
    for r in rows:
        flag = ''
        if r['ratio'] is not None and r['spec'] != 'Canonical':
            diff = abs(r['ratio'] - 0.829)
            flag = '  ← ROBUST' if diff <= 0.05 else f'  ← DIFFERS by {diff:.3f}'
        print(f"  {r['spec']:<22} {r['method']:<20} "
              f"{r['missing']:>8.3f}pp {r['excess']:>8.3f}pp "
              f"{r['ratio']:>7.3f}{flag}")

    canonical_R = rows[0]['ratio']
    all_within = all(
        abs(r['ratio'] - canonical_R) <= 0.05
        for r in rows[1:]
        if r['ratio'] is not None
    )
    print(f"\n→ All specs within ±0.05 of canonical: {all_within}")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: COVID Robustness — Event C with pre=2019
# ═══════════════════════════════════════════════════════════════════════════════

def analysis3():
    print('\n' + '='*65)
    print('ANALYSIS 3: COVID Robustness — Event C, pre=2019 vs pre=2021')
    print('='*65)

    ecfg = EVENTS['C']
    mw_old, mw_new = ecfg['mw_old'], ecfg['mw_new']

    # Canonical: pre=2021
    pre21_shares, n21 = get_shares(2021, 'formal_dep')
    post23_shares, n23 = get_shares(2023, 'formal_dep')

    delta_21 = post23_shares - pre21_shares
    bg_21    = canonical_background(delta_21, mw_new)
    dadj_21  = delta_21 - bg_21
    miss_21  = compute_missing(dadj_21, mw_old, mw_new)
    exc_21   = compute_excess(dadj_21, mw_new, 250)
    R_21     = round(exc_21 / miss_21, 3) if miss_21 > 0 else None

    print(f"\nCanonical (pre=2021, post=2023): missing={miss_21*100:.3f}pp "
          f"excess={exc_21*100:.3f}pp  R={R_21}")

    # COVID robustness: pre=2019
    try:
        pre19_shares, n19 = get_shares(2019, 'formal_dep')

        delta_19 = post23_shares - pre19_shares
        bg_19    = canonical_background(delta_19, mw_new)
        dadj_19  = delta_19 - bg_19
        miss_19  = compute_missing(dadj_19, mw_old, mw_new)
        exc_19   = compute_excess(dadj_19, mw_new, 250)
        R_19     = round(exc_19 / miss_19, 3) if miss_19 > 0 else None

        print(f"COVID-robust (pre=2019, post=2023): missing={miss_19*100:.3f}pp "
              f"excess={exc_19*100:.3f}pp  R={R_19}")

        diff = abs(R_19 - R_21) if R_19 is not None and R_21 is not None else None
        if diff is not None:
            if diff <= 0.05:
                print(f"→ ROBUST: difference = {diff:.3f} (within ±0.05)")
            else:
                print(f"→ SENSITIVE: difference = {diff:.3f} (exceeds ±0.05) — add caveat")

        # N increase check
        print(f"\nN check (formal dep, unweighted):")
        print(f"  2019: {n19:,}  |  2021: {n21:,}  |  2023: {n23:,}")
        n_increase = n23 - n21
        pct_change = (n23 / n21 - 1) * 100 if n21 > 0 else 0
        print(f"  2021→2023 increase: +{n_increase:,} ({pct_change:+.1f}%)")

        # Check N in compression zone [MW_old, MW_new) = [930, 1025)
        df21 = _cache.get(('df', 2021))
        df23 = _cache.get(('df', 2023))
        if df21 is not None and df23 is not None:
            def n_in_zone(df, lo, hi, sample='formal_dep'):
                wages, _, n = extract_workers(df, sample)
                return int(((wages >= lo) & (wages < hi)).sum())

            n21_comp = n_in_zone(df21, mw_old, mw_new)    # [930, 1025)
            n23_exc  = n_in_zone(df23, mw_new, mw_new+250) # [1025, 1275)
            print(f"\n  2021 formal dep in affected zone [S/{mw_old}, S/{mw_new}): "
                  f"N={n21_comp:,}")
            print(f"  2023 formal dep in excess zone  [S/{mw_new}, S/{mw_new+250}): "
                  f"N={n23_exc:,}")
            new_workers = n23 - n21
            if new_workers > 0 and n23_exc > 0:
                max_pct = min(100, round(new_workers / n23_exc * 100, 1))
                print(f"  → New entrants ({new_workers:,}) are {max_pct}% of excess zone N")
                print(f"    (upper bound on new-entrant contamination of excess mass)")

        result = {
            'canonical_pre2021': {'R': R_21, 'missing_pp': round(miss_21*100, 3),
                                  'excess_pp': round(exc_21*100, 3), 'n': n21},
            'covid_robust_pre2019': {'R': R_19, 'missing_pp': round(miss_19*100, 3),
                                     'excess_pp': round(exc_19*100, 3), 'n': n19},
            'n_2023': n23,
        }
    except Exception as e:
        print(f"  ERROR loading 2019 data: {e}")
        result = {'canonical_pre2021': {'R': R_21}, 'error': str(e)}

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: Effective N in affected/excess zones
# ═══════════════════════════════════════════════════════════════════════════════

def analysis4():
    print('\n' + '='*65)
    print('ANALYSIS 4: Effective N in Affected / Excess Zones')
    print('='*65)

    samples = ['formal_dep', 'informal', 'all_dep']
    results = {}

    print(f"\n{'Event':<7} {'Pop':<14} {'N_affected_pre':>16} "
          f"{'N_excess_post':>14} {'N_total_pre':>12}")
    print('-' * 70)

    for eid, ecfg in EVENTS.items():
        mw_old, mw_new = ecfg['mw_old'], ecfg['mw_new']
        aff_lo = 0.85 * mw_old

        df_pre  = _cache.get(('df', ecfg['pre_year']))
        df_post = _cache.get(('df', ecfg['post_year']))

        if df_pre is None:
            df_pre = load_enaho(ecfg['pre_year'])
            _cache[('df', ecfg['pre_year'])] = df_pre
        if df_post is None:
            df_post = load_enaho(ecfg['post_year'])
            _cache[('df', ecfg['post_year'])] = df_post

        results[eid] = {}
        for samp in samples:
            wages_pre,  _, n_pre  = extract_workers(df_pre,  samp)
            wages_post, _, n_post = extract_workers(df_post, samp)

            n_aff  = int(((wages_pre  >= aff_lo) & (wages_pre  < mw_new)).sum())
            n_exc  = int(((wages_post >= mw_new)  & (wages_post < mw_new + 250)).sum())

            results[eid][samp] = {
                'N_affected_pre': n_aff,
                'N_excess_post':  n_exc,
                'N_total_pre':    n_pre,
            }
            print(f"  {eid:<5} {samp:<14} {n_aff:>16,} {n_exc:>14,} {n_pre:>12,}")

    return results


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Pre-load all needed years
    years_needed = set()
    for ecfg in EVENTS.values():
        years_needed.add(ecfg['pre_year'])
        years_needed.add(ecfg['post_year'])
    years_needed.add(2019)   # for Analysis 3

    print("Pre-loading ENAHO data...")
    for yr in sorted(years_needed):
        if ('df', yr) not in _cache:
            try:
                df = load_enaho(yr)
                _cache[('df', yr)] = df
                print(f"  Loaded {yr}: {len(df):,} rows")
            except Exception as e:
                print(f"  WARNING: Could not load {yr}: {e}")

    # Run analyses
    r1 = analysis1()
    r2 = analysis2()
    r3 = analysis3()
    r4 = analysis4()

    # Save JSON
    output = {
        'analysis1_cumulative_excess': r1,
        'analysis2_counterfactual_sensitivity_B': r2,
        'analysis3_covid_robustness_C': r3,
        'analysis4_effective_N': r4,
    }
    def _to_python(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_python(v) for v in obj]
        return obj

    out_path = f'{OUT_DATA}/mw_reviewer_analyses.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(_to_python(output), f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    # Print summary for text edits
    print('\n' + '='*65)
    print('SUMMARY FOR TEXT EDITS')
    print('='*65)
    print('\n[Analysis 1] R(W=50) and R(W=500) for each event:')
    for eid in ['A', 'B', 'C']:
        print(f"  Event {eid}: R(50)={r1[eid][50]:.3f}  R(500)={r1[eid][500]:.3f}  "
              f"  converge={'yes' if r1[eid][500] > r1[eid][50] else 'no'}")

    print('\n[Analysis 2] Sensitivity range for Event B:')
    ratios = [r['ratio'] for r in r2 if r['ratio'] is not None]
    print(f"  Min R = {min(ratios):.3f}  Max R = {max(ratios):.3f}  "
          f"Range = {max(ratios)-min(ratios):.3f}")

    print('\n[Analysis 3] COVID robustness Event C:')
    if 'covid_robust_pre2019' in r3:
        print(f"  Canonical R (pre=2021): {r3['canonical_pre2021']['R']}")
        print(f"  COVID-robust R (pre=2019): {r3['covid_robust_pre2019']['R']}")

    print('\n[Analysis 4] Sample N (formal dep):')
    for eid in ['A', 'B', 'C']:
        d = r4[eid]['formal_dep']
        print(f"  Event {eid}: N_affected_pre={d['N_affected_pre']:,}  "
              f"N_excess_post={d['N_excess_post']:,}  N_total_pre={d['N_total_pre']:,}")
