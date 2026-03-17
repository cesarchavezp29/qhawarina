"""
mw_heterogeneity_bunching.py
============================
Cengiz-style bunching estimator by subgroup: industry, firm size, age, sex.

Events (identical to mw_cengiz_bunching.py):
  A: S/750 -> S/850  (May 2016)  pre=2015, post=2017
  B: S/850 -> S/930  (Apr 2018)  pre=2017, post=2019
  C: S/930 -> S/1025 (May 2022)  pre=2021, post=2023

SAMPLE: employed dependent workers (p507 in {3,4} = empleado/obrero) with
        monthly wage (i524a1/12) > 0, valid weight (fac500a > 0).

SECTOR MAP (CIIU rev4 / p506r4, first 2 digits):
  01-09    Agriculture / Mining / Fishing
  10-33    Manufacturing
  41-43    Construction
  45-47    Commerce
  49-56    Transport & accommodation
  58-83    Finance / Real estate / Professional services
  84       Public administration
  85-88    Education / Health / Social
  90-99    Other services

FIRM SIZE (p512b = number of workers):
  Micro  : p512b <= 10
  Small  : 11 <= p512b <= 50
  Medium+: p512b > 50

AGE (p208a):
  18-24, 25-44, 45-64

SEX (p207):
  Male (1), Female (2)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import json
import os
import glob
import shutil
from datetime import datetime

# ==============================================================================
# CONFIGURATION
# ==============================================================================
EVENTS = {
    'A': {
        'name': 'S/750->S/850 (May 2016)',
        'mw_old': 750,
        'mw_new': 850,
        'pre_year': 2015,
        'post_year': 2017,
        'period_pre': '2015 (anual)',
        'period_post': '2017 (anual)',
    },
    'B': {
        'name': 'S/850->S/930 (Apr 2018)',
        'mw_old': 850,
        'mw_new': 930,
        'pre_year': 2017,
        'post_year': 2019,
        'period_pre': '2017 (anual)',
        'period_post': '2019 (anual)',
    },
    'C': {
        'name': 'S/930->S/1025 (May 2022)',
        'mw_old': 930,
        'mw_new': 1025,
        'pre_year': 2021,
        'post_year': 2023,
        'period_pre': '2021 (anual)',
        'period_post': '2023 (anual)',
    },
}

BIN_WIDTH = 25          # soles per bin
WAGE_MIN = 0
WAGE_MAX = 6000
BASE_PATH = 'D:/Nexus/nexus/data/raw/enaho'
MIN_N_WORKERS = 200     # minimum sample workers (unweighted) to report ratio
OUT_JSON = 'D:/Nexus/nexus/exports/data/mw_heterogeneity.json'
QHAW_JSON = 'D:/qhawarina/public/assets/data/mw_heterogeneity.json'


# ==============================================================================
# SECTOR MAPPING
# ==============================================================================
def ciiu_sector(code):
    """Map 2-digit CIIU rev4 to broad sector string."""
    try:
        c = int(float(code))
    except (ValueError, TypeError):
        return 'other'
    if 1 <= c <= 9:
        return 'agri_mining'
    elif 10 <= c <= 33:
        return 'manufacturing'
    elif 41 <= c <= 43:
        return 'construction'
    elif 45 <= c <= 47:
        return 'commerce'
    elif 49 <= c <= 56:
        return 'transport_accom'
    elif 58 <= c <= 83:
        return 'finance_prof'
    elif c == 84:
        return 'public_admin'
    elif 85 <= c <= 88:
        return 'edu_health'
    elif 90 <= c <= 99:
        return 'other_services'
    else:
        return 'other'


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_year(year):
    """Load ENAHO Module 500 for a given year."""
    patterns = [
        f'{BASE_PATH}/cross_section/modulo_05_{year}/enaho01a-{year}-500.dta',
        f'{BASE_PATH}/**/enaho01a-{year}-500.dta',
        f'{BASE_PATH}/**/*{year}*500*.dta',
    ]
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            f = files[0]
            print(f'  Loading {year}: {f}')
            df = pd.read_stata(f, convert_categoricals=False)
            df.columns = [c.lower() for c in df.columns]
            return df
    raise FileNotFoundError(f'Cannot find ENAHO Module 500 for {year}')


def build_sample(df, year):
    """
    Extract analysis sample: FORMAL dependent employed workers with valid wage.
    Formal = ocupinf==2 (same definition as mw_complete_margins.py formal_dep).
    Returns DataFrame with: wage, weight, sector_name, firm_size, age, age_cat, sex
    """
    print(f'  Building sample for {year}...')

    # Employment filter: ocu500 == 1
    emp = pd.to_numeric(df.get('ocu500', pd.Series(1, index=df.index)), errors='coerce') == 1
    print(f'    Employed (ocu500==1): {emp.sum():,}')

    # Dependent worker: p507 in {3,4,6} — matches mw_complete_margins.py DEPENDENT constant
    if 'p507' in df.columns:
        cat = pd.to_numeric(df['p507'], errors='coerce')
        dep = cat.isin([3, 4, 6])
        print(f'    Dependent (p507 in {{3,4,6}}): {dep.sum():,}')
    else:
        dep = pd.Series(True, index=df.index)
        print('    WARNING: p507 not found, skipping dep filter')

    # Formality: ocupinf==2 (1=informal, 2=formal)
    if 'ocupinf' in df.columns:
        ocinf = pd.to_numeric(df['ocupinf'], errors='coerce')
        formal = (ocinf == 2)
        print(f'    Formal (ocupinf==2): {formal.sum():,}')
    elif 'p510a1' in df.columns:
        # Fallback: p510a1==1 means written contract (proxy for formal)
        p510 = pd.to_numeric(df['p510a1'], errors='coerce')
        formal = (p510 == 1)
        print(f'    Formal from p510a1==1 (contract): {formal.sum():,}  [WARNING: fallback]')
    else:
        formal = pd.Series(True, index=df.index)
        print('    WARNING: No formality variable (ocupinf/p510a1) found — all workers included')

    # Wage: p524a1 (direct monthly cash wage) preferred, fallback to i524a1/12
    # Priority matches mw_complete_margins.py get_monthly_wage()
    if 'p524a1' in df.columns:
        wage_cand = pd.to_numeric(df['p524a1'], errors='coerce')
        if wage_cand.notna().sum() > 1000 and wage_cand.median() > 200:
            wage = wage_cand
            print(f'    Wage: p524a1 (monthly cash wage) median={wage.median():.0f}')
        elif 'i524a1' in df.columns:
            wage = pd.to_numeric(df['i524a1'], errors='coerce') / 12.0
            print(f'    Wage: i524a1/12 (annual deflated/12) median={wage.median():.0f}  [p524a1 sparse]')
        else:
            raise ValueError('No wage variable found')
    elif 'i524a1' in df.columns:
        wage = pd.to_numeric(df['i524a1'], errors='coerce') / 12.0
        print(f'    Wage: i524a1/12 (annual deflated/12) median={wage.median():.0f}')
    else:
        raise ValueError('No wage variable found')

    # Weight: factor07i500a (primary, matches mw_complete_margins), fallback fac500a
    weight = None
    for wv in ['factor07i500a', 'fac500a', 'facpob07', 'factor07', 'fac500']:
        if wv in df.columns:
            weight = pd.to_numeric(df[wv], errors='coerce').fillna(0)
            print(f'    Weight variable: {wv}')
            break
    if weight is None:
        weight = pd.Series(1.0, index=df.index)
        print('    WARNING: No weight found, using uniform')

    # Combined mask: formal + dependent + employed + valid wage/weight
    mask = emp & dep & formal & (wage > 0) & (wage < WAGE_MAX) & (weight > 0)
    n = mask.sum()
    print(f'    Final sample (formal-dep): {n:,} workers')

    out = pd.DataFrame({
        'wage': wage[mask].values,
        'weight': weight[mask].values,
    }, index=df.index[mask])

    # Sector from p506r4 (CIIU rev4) or p506 (rev3 fallback)
    svar = 'p506r4' if 'p506r4' in df.columns else ('p506' if 'p506' in df.columns else None)
    if svar:
        raw = pd.to_numeric(df[svar][mask], errors='coerce')
        ciiu2 = (raw / 100).apply(np.floor)
        out['sector_name'] = ciiu2.apply(ciiu_sector).values
    else:
        out['sector_name'] = 'other'

    # Firm size from p512b (exact count) or p512a (categories)
    if 'p512b' in df.columns:
        p512b = pd.to_numeric(df['p512b'][mask], errors='coerce')
        def size_from_count(x):
            if pd.isna(x):
                return np.nan
            elif x <= 10:
                return 'micro'
            elif x <= 50:
                return 'small'
            else:
                return 'medium_plus'
        out['firm_size'] = p512b.apply(size_from_count).values
    elif 'p512a' in df.columns:
        p512a = pd.to_numeric(df['p512a'][mask], errors='coerce')
        def size_from_cat(x):
            if pd.isna(x):
                return np.nan
            elif x == 1:
                return 'micro'        # up to 20 (use as proxy for micro)
            elif x == 2:
                return 'small'        # 21-50
            else:
                return 'medium_plus'  # 51+
        out['firm_size'] = p512a.apply(size_from_cat).values
    else:
        out['firm_size'] = np.nan

    # Age from p208a
    if 'p208a' in df.columns:
        age = pd.to_numeric(df['p208a'][mask], errors='coerce')
        out['age'] = age.values
        def age_group(a):
            if pd.isna(a):
                return np.nan
            elif 18 <= a <= 24:
                return '18-24'
            elif 25 <= a <= 44:
                return '25-44'
            elif 45 <= a <= 64:
                return '45-64'
            else:
                return 'other_age'
        out['age_cat'] = age.apply(age_group).values
    else:
        out['age'] = np.nan
        out['age_cat'] = np.nan

    # Sex from p207 (1=male, 2=female)
    if 'p207' in df.columns:
        out['sex'] = pd.to_numeric(df['p207'][mask], errors='coerce').values
    else:
        out['sex'] = np.nan

    med = float(np.median(out['wage']))
    print(f'    Median wage (i524a1/12): {med:.0f} soles/month')

    return out


# ==============================================================================
# BUNCHING ESTIMATOR
# ==============================================================================
def cengiz_ratio(wages_pre, weights_pre, wages_post, weights_post, mw_old, mw_new,
                 excess_bins=10, do_counterfactual=True):
    """
    Cengiz revised bunching estimator (matches mw_complete_margins.py cengiz_revised).

    Missing mass  = net outflow from affected zone [0.85*mw_old, mw_new)
    Excess mass   = positive deltas in [mw_new, mw_new + excess_bins*BIN_WIDTH)
    Counterfactual = background delta subtracted using clean zone (> 2*mw_new)
    Ratio = excess / missing_net_outflow

    Returns dict with ratio, counts, stats. Sets insufficient_n=True if N < MIN_N_WORKERS.
    """
    n_pre_raw = len(wages_pre)
    n_post_raw = len(wages_post)

    if n_pre_raw < MIN_N_WORKERS or n_post_raw < MIN_N_WORKERS:
        return {
            'insufficient_n': True,
            'n_pre': n_pre_raw,
            'n_post': n_post_raw,
            'bunching_ratio': None,
        }

    bin_edges = np.arange(WAGE_MIN, WAGE_MAX + BIN_WIDTH, BIN_WIDTH)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    counts_pre, _ = np.histogram(wages_pre, bins=bin_edges, weights=weights_pre)
    counts_post, _ = np.histogram(wages_post, bins=bin_edges, weights=weights_post)

    total_pre = float(counts_pre.sum())
    total_post = float(counts_post.sum())

    if total_pre == 0:
        return {'insufficient_n': True, 'n_pre': 0, 'n_post': int(total_post), 'bunching_ratio': None}

    shares_pre = counts_pre / total_pre
    shares_post = counts_post / total_post if total_post > 0 else np.zeros_like(counts_post, dtype=float)

    delta = shares_post - shares_pre

    # Counterfactual: subtract background shift from clean zone (far above MW)
    if do_counterfactual:
        clean_zone = bin_centers > 2 * mw_new
        n_clean = clean_zone.sum()
        if n_clean > 5:
            background = np.average(delta[clean_zone],
                                    weights=1.0 / (np.abs(delta[clean_zone]) + 1e-8))
            delta_adj = delta - background
        else:
            delta_adj = delta
    else:
        delta_adj = delta

    # Missing mass: net outflow from directly-affected zone [0.85*mw_old, mw_new)
    affected_lo = 0.85 * mw_old
    below_new = (bin_centers >= affected_lo) & (bin_centers < mw_new)
    missing = float(max(-delta_adj[below_new].sum(), 0.0))

    # Excess mass: positive deltas at [mw_new, mw_new + excess_bins*BIN)
    at_new = (bin_centers >= mw_new) & (bin_centers < mw_new + excess_bins * BIN_WIDTH)
    excess = float(delta_adj[at_new & (delta_adj > 0)].sum())

    # Guard: if missing mass is near-zero (<0.3pp), the ratio is unreliable
    # (too few workers in the affected zone — sector wages far above MW)
    MIN_MISSING_PP = 0.003   # 0.3 percentage points
    ratio_reliable = missing >= MIN_MISSING_PP
    ratio = (excess / missing) if (missing > 0 and ratio_reliable) else float('inf')

    # Stats
    median_pre = float(np.median(wages_pre))
    share_below_old = float(np.sum(weights_pre[wages_pre < mw_old]) / np.sum(weights_pre))

    return {
        'insufficient_n': False,
        'ratio_unreliable': not ratio_reliable,   # True = missing mass too small
        'bunching_ratio': round(ratio, 4) if ratio_reliable else None,
        'missing_pp': round(missing * 100, 4),
        'excess_pp': round(excess * 100, 4),
        'n_pre_weighted': int(round(total_pre)),
        'n_post_weighted': int(round(total_post)),
        'n_pre_raw': n_pre_raw,
        'n_post_raw': n_post_raw,
        'median_wage_pre': round(median_pre, 1),
        'share_below_mw_old_pre': round(share_below_old, 4),
    }


def run_subgroup_event(df_pre, df_post, mask_pre, mask_post, event_id, econf, label):
    """Run bunching for one subgroup × event."""
    wages_pre = df_pre.loc[mask_pre, 'wage'].values
    wts_pre = df_pre.loc[mask_pre, 'weight'].values
    wages_post = df_post.loc[mask_post, 'wage'].values
    wts_post = df_post.loc[mask_post, 'weight'].values

    res = cengiz_ratio(wages_pre, wts_pre, wages_post, wts_post,
                       econf['mw_old'], econf['mw_new'])

    if res['insufficient_n']:
        print(f'    [{label} | Event {event_id}] N insuficiente '
              f'(pre={res["n_pre"]}, post={res["n_post"]})')
    else:
        print(f'    [{label} | Event {event_id}] '
              f'ratio={res["bunching_ratio"]:.3f}  '
              f'missing={res["missing_pp"]:.3f}pp  '
              f'excess={res["excess_pp"]:.3f}pp  '
              f'n_pre_raw={res["n_pre_raw"]:,}')
    return res


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print('=' * 72)
    print('MW HETEROGENEITY ANALYSIS — CENGIZ BUNCHING BY SUBGROUP')
    print('Methodology: Cengiz et al. (2019), dependent workers only')
    print('=' * 72)

    # Load all needed years
    years_needed = set()
    for econf in EVENTS.values():
        years_needed.add(econf['pre_year'])
        years_needed.add(econf['post_year'])

    print(f'\nYears needed: {sorted(years_needed)}')

    raw_data = {}
    samples = {}
    for yr in sorted(years_needed):
        print(f'\n--- Year {yr} ---')
        df_raw = load_year(yr)
        raw_data[yr] = df_raw
        samples[yr] = build_sample(df_raw, yr)

    # Define subgroups as (name, label, filter_fn)
    # filter_fn(df_sample) -> boolean array/Series of same length
    subgroups = []

    # ALL workers
    subgroups.append(('ALL', 'Total (referencia)',
                      lambda df: pd.Series(True, index=df.index)))

    # Sectors
    for sname, slabel in [
        ('agri_mining',     'Agricultura / Mineria'),
        ('manufacturing',   'Manufactura'),
        ('construction',    'Construccion'),
        ('commerce',        'Comercio'),
        ('transport_accom', 'Transporte y alojamiento'),
        ('finance_prof',    'Finanzas / Servicios prof.'),
        ('public_admin',    'Administracion publica'),
        ('edu_health',      'Educacion y salud'),
    ]:
        subgroups.append((f'sector_{sname}', slabel,
                          (lambda df, s=sname: pd.Series(df['sector_name'] == s, index=df.index))))

    # Firm size
    for fsize, flabel in [
        ('micro',        'Micro (<=10 trab.)'),
        ('small',        'Pequena (11-50 trab.)'),
        ('medium_plus',  'Mediana/Grande (51+)'),
    ]:
        subgroups.append((f'size_{fsize}', flabel,
                          (lambda df, s=fsize: pd.Series(df['firm_size'].astype(str) == s, index=df.index))))

    # Age groups
    for acat, alabel in [
        ('18-24', 'Edad 18-24'),
        ('25-44', 'Edad 25-44'),
        ('45-64', 'Edad 45-64'),
    ]:
        subgroups.append((f'age_{acat.replace("-","_")}', alabel,
                          (lambda df, a=acat: pd.Series(df['age_cat'].astype(str) == a, index=df.index))))

    # Sex
    for sex_val, sex_name, sex_label in [
        (1, 'male',   'Hombres'),
        (2, 'female', 'Mujeres'),
    ]:
        subgroups.append((f'sex_{sex_name}', sex_label,
                          (lambda df, s=sex_val: pd.Series(
                              pd.to_numeric(df['sex'], errors='coerce') == s,
                              index=df.index))))

    # Reference year for base stats (earliest pre)
    ref_year = min(yr for econf in EVENTS.values() for yr in [econf['pre_year']])
    df_ref = samples[ref_year]
    ref_mw = min(econf['mw_old'] for econf in EVENTS.values())
    print(f'\nReference year for base stats: {ref_year}, MW_old reference: {ref_mw}')

    # Run all subgroups
    all_subgroup_results = []

    for sg_name, sg_label, sg_filter in subgroups:
        print(f'\n{"─"*60}')
        print(f'SUBGROUP: {sg_label}')

        # Base stats from reference year
        m_ref = sg_filter(df_ref)
        n_ref = int(m_ref.sum())
        if n_ref >= MIN_N_WORKERS:
            med_ref = float(np.median(df_ref.loc[m_ref, 'wage']))
            share_below_ref = float((df_ref.loc[m_ref, 'wage'] < ref_mw).mean())
        else:
            med_ref = None
            share_below_ref = None

        med_disp = f'{med_ref:.0f}' if med_ref is not None else 'N/A'
        sh_disp = f'{share_below_ref:.3f}' if share_below_ref is not None else 'N/A'
        print(f'  n_pre_{ref_year}={n_ref:,}  '
              f'median_wage={med_disp}  '
              f'share<{ref_mw}={sh_disp}')

        sg_result = {
            'name': sg_name,
            'label': sg_label,
            f'n_pre_{ref_year}': n_ref,
            'median_wage_pre': round(med_ref, 1) if med_ref is not None else None,
            'share_below_mw_pre': round(share_below_ref, 4) if share_below_ref is not None else None,
        }

        # Per-event
        for eid, econf in EVENTS.items():
            df_pre = samples[econf['pre_year']]
            df_post = samples[econf['post_year']]
            m_pre = sg_filter(df_pre)
            m_post = sg_filter(df_post)
            ev_res = run_subgroup_event(df_pre, df_post, m_pre, m_post, eid, econf, sg_label)
            sg_result[f'event_{eid}'] = ev_res

        # Average ratio across events with valid AND reliable data
        valid_ratios = [
            sg_result[f'event_{eid}']['bunching_ratio']
            for eid in EVENTS
            if not sg_result[f'event_{eid}'].get('insufficient_n')
               and not sg_result[f'event_{eid}'].get('ratio_unreliable')
               and sg_result[f'event_{eid}'].get('bunching_ratio') is not None
        ]
        sg_result['avg_ratio'] = round(float(np.mean(valid_ratios)), 4) if valid_ratios else None

        all_subgroup_results.append(sg_result)

    # ===========================================================================
    # PRINT SUMMARY TABLE
    # ===========================================================================
    print(f'\n\n{"="*106}')
    print('MW EFFECTS HETEROGENEITY — BUNCHING RATIO (formal-dep workers, i524a1/12)')
    print('Higher ratio = more redistribution, less displacement.')
    print(f'{"="*106}')
    header = (f'{"Subgroup":<32} {"N_pre":>7} {"Med_wage":>9} {"Sh<MW":>7} '
              f'{"Event A":>9} {"Event B":>9} {"Event C":>9} {"Avg":>7}')
    print(header)
    print('-' * 106)

    for sg_result in all_subgroup_results:
        label = sg_result['label']
        n_pre = sg_result.get(f'n_pre_{ref_year}', 0)
        med = sg_result.get('median_wage_pre')
        sh = sg_result.get('share_below_mw_pre')
        avg = sg_result.get('avg_ratio')

        def fmt_ratio(eid):
            ev = sg_result.get(f'event_{eid}', {})
            if ev.get('insufficient_n'):
                return '  insuf. '
            if ev.get('ratio_unreliable'):
                return '  [<MW] '
            br = ev.get('bunching_ratio')
            return f'   {br:.3f} ' if br is not None else '    ---  '

        med_str = f'{med:>9.0f}' if med is not None else f'{"N/A":>9}'
        sh_str = f'{sh:>7.3f}' if sh is not None else f'{"N/A":>7}'
        avg_str = f'  {avg:.3f}' if avg is not None else '   ---'

        print(f'{label:<32} {n_pre:>7,} {med_str} {sh_str} '
              f'{fmt_ratio("A")}{fmt_ratio("B")}{fmt_ratio("C")}{avg_str}')

    # ===========================================================================
    # SAVE JSON
    # ===========================================================================
    output = {
        'generated_at': datetime.now().isoformat(),
        'methodology': (
            'Cengiz et al. (2019) revised bunching estimator. '
            'Sample: empleados/obreros (p507 in {3,4}), '
            'wage = i524a1/12 (annualized deflated monthly soles), fac500a > 0. '
            'Missing mass = net outflow from affected zone [0.85*mw_old, mw_new). '
            'Excess mass = positive share changes in [mw_new, mw_new + 10*BIN]. '
            'Counterfactual: background shift from clean zone (>2*mw_new) subtracted. '
            'Bunching ratio = excess / missing_net_outflow.'
        ),
        'bin_width_soles': BIN_WIDTH,
        'min_n_threshold': MIN_N_WORKERS,
        'events': {
            eid: {
                'name': econf['name'],
                'mw_old': econf['mw_old'],
                'mw_new': econf['mw_new'],
                'period_pre': econf['period_pre'],
                'period_post': econf['period_post'],
                'pre_year': econf['pre_year'],
                'post_year': econf['post_year'],
            }
            for eid, econf in EVENTS.items()
        },
        'subgroups': all_subgroup_results,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\nSaved: {OUT_JSON}')

    os.makedirs(os.path.dirname(QHAW_JSON), exist_ok=True)
    shutil.copy(OUT_JSON, QHAW_JSON)
    print(f'Copied: {QHAW_JSON}')

    return output


if __name__ == '__main__':
    main()
