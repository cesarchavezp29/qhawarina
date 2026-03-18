"""
mw_ethnicity_heterogeneity.py
==============================
Cengiz bunching by ethnicity, education, geography, and employer sector.

Events (matching canonical mw_complete_margins.py):
  A: S/750 -> S/850 (May 2016)   pre=2015, post=2017
  B: S/850 -> S/930 (Apr 2018)   pre=2017, post=2019
  C: S/930 -> S/1025 (May 2022)  pre=2021, post=2023

ETHNICITY SOURCES (bridged variable):
  Years 2015-2019: Module 300 p300a (mother tongue), numeric codes:
    1=quechua, 2=aymara, 3=otra nativa -> indigenous
    4=castellano -> spanish
    other -> other
  Years 2021-2023: Module 500 p558c (ethnic self-ID), numeric codes:
    1=quechua, 2=aimara, 3=nativo amazonia, 9=otro indigena -> indigenous
    4=negro/afro -> afro
    5=blanco -> blanco
    6=mestizo -> mestizo
    7=otro -> other;  8=no sabe -> excluded
  NOTE: 2015-2019 ethnicty from mother tongue; 2021-2023 from self-ID.
  Both identify indigenous/non-indigenous consistently. Afro/blanco/mestizo
  cells for Events A-B are combined into 'spanish/non-indigenous'.

EDUCATION (p301a from Module 500, all years):
  {1,2,3,4,12} -> primary_less   (sin nivel, inicial, primaria, basica especial)
  {5,6}        -> secondary       (secundaria incompleta/completa)
  {7..11}      -> superior        (superior no-univ/univ/postgrado)

GEOGRAPHY (Module 500, all years):
  dominio==8          -> lima
  dominio in {1..7}   -> rest
  estrato in {1..6}   -> urban   (500+ habitantes)
  estrato in {7,8}    -> rural   (AER)

EMPLOYER SECTOR (p510 from Module 500, all years):
  p510 in {1,2,3} -> public   (FF.AA./PNP, admin publica, empresa publica)
  p510 in {5,6}   -> private  (empresa privada, empresas especiales/service)
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
    'A': {'mw_old': 750,  'mw_new': 850,  'pre_year': 2015, 'post_year': 2017,
          'name': 'S/750->S/850 (May 2016)'},
    'B': {'mw_old': 850,  'mw_new': 930,  'pre_year': 2017, 'post_year': 2019,
          'name': 'S/850->S/930 (Apr 2018)'},
    'C': {'mw_old': 930,  'mw_new': 1025, 'pre_year': 2021, 'post_year': 2023,
          'name': 'S/930->S/1025 (May 2022)'},
}

BIN_WIDTH   = 25
WAGE_MIN    = 0
WAGE_MAX    = 6000
MIN_N       = 200          # minimum unweighted N to report bunching ratio
BASE_PATH   = 'D:/Nexus/nexus/data/raw/enaho'
OUT_JSON    = 'D:/Nexus/nexus/exports/data/mw_ethnicity_heterogeneity.json'
QHAW_JSON   = 'D:/qhawarina/public/assets/data/mw_ethnicity_heterogeneity.json'

# Numeric code mappings (confirmed from pyreadstat label check)
P300A_INDIGENOUS = {1, 2, 3}   # quechua, aymara, otra nativa
P300A_SPANISH    = {4}
P558C_INDIGENOUS = {1, 2, 3, 9}  # quechua, aimara, nativo amazonia, otro indigena
P558C_AFRO       = {4}
P558C_BLANCO     = {5}
P558C_MESTIZO    = {6}
P558C_NOSABE     = {8}          # excluded

P301A_PRIMARY    = {1, 2, 3, 4, 12}
P301A_SECONDARY  = {5, 6}
P301A_SUPERIOR   = {7, 8, 9, 10, 11}

DOMINIO_LIMA     = {8}
ESTRATO_RURAL    = {7, 8}

P510_PUBLIC      = {1, 2, 3}
P510_PRIVATE     = {5, 6}


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_module500(year):
    """Load ENAHO Module 500 for a given year (numeric codes)."""
    patterns = [
        f'{BASE_PATH}/cross_section/modulo_05_{year}/enaho01a-{year}-500.dta',
        f'{BASE_PATH}/**/enaho01a-{year}-500.dta',
    ]
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            print(f'  Loading Module 500 {year}: {files[0]}')
            df = pd.read_stata(files[0], convert_categoricals=False)
            df.columns = [c.lower() for c in df.columns]
            return df
    raise FileNotFoundError(f'Cannot find ENAHO Module 500 for {year}')


def load_module300(year):
    """
    Load ENAHO Module 300 and return merge-key + p300a (mother tongue, numeric).
    Available for 2015-2019 only.
    Returns None if unavailable.
    """
    patterns = [
        f'{BASE_PATH}/cross_section/modulo_03_{year}/enaho01a-{year}-300.dta',
        f'{BASE_PATH}/**/enaho01a-{year}-300.dta',
    ]
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            print(f'  Loading Module 300 {year}: {files[0]}')
            df = pd.read_stata(files[0], convert_categoricals=False)
            df.columns = [c.lower() for c in df.columns]
            merge_keys = ['conglome', 'vivienda', 'hogar', 'codperso']
            if 'p300a' not in df.columns:
                print(f'    WARNING: p300a not found in Module 300 {year}')
                return None
            keep = [k for k in merge_keys if k in df.columns] + ['p300a']
            out = df[keep].drop_duplicates(subset=[k for k in merge_keys if k in df.columns])
            print(f'    Module 300 {year}: {len(out):,} persons with p300a')
            return out
    print(f'  Module 300 {year}: not available (no file found)')
    return None


# ==============================================================================
# SAMPLE CONSTRUCTION
# ==============================================================================
def build_sample(df_raw, year, m300=None):
    """
    Extract formal dependent workers with valid wage. Adds extended variables:
    ethnic_grp, educ_cat, lima, urban, public_sector.

    ethnic_grp values:
      'indigenous'  - quechua/aymara/nativa or self-ID indigenous
      'afro'        - afro-descendiente (2021+ only from p558c)
      'blanco'      - blanco (2021+ only)
      'mestizo'     - mestizo (2021+ only)
      'spanish'     - castellano mother tongue (2015-19)
      'other'       - other / unknown / NaN
    """
    df = df_raw.copy()
    print(f'  Building sample for {year}...')

    # Employment filter: ocu500==1
    emp = pd.to_numeric(df.get('ocu500', pd.Series(1, index=df.index)), errors='coerce') == 1

    # Dependent worker: p507 in {3,4,6}
    if 'p507' in df.columns:
        cat = pd.to_numeric(df['p507'], errors='coerce')
        dep = cat.isin([3, 4, 6])
    else:
        dep = pd.Series(True, index=df.index)

    # Formality: ocupinf==2
    if 'ocupinf' in df.columns:
        formal = (pd.to_numeric(df['ocupinf'], errors='coerce') == 2)
    else:
        formal = pd.Series(True, index=df.index)

    # Wage: column-level fallback (canonical spec matching mw_complete_margins.py)
    if 'p524a1' in df.columns:
        wage_cand = pd.to_numeric(df['p524a1'], errors='coerce')
        if wage_cand.notna().sum() > 1000 and wage_cand.median() > 200:
            wage = wage_cand
        elif 'i524a1' in df.columns:
            wage = pd.to_numeric(df['i524a1'], errors='coerce') / 12.0
        else:
            raise ValueError(f'No usable wage for {year}')
    elif 'i524a1' in df.columns:
        wage = pd.to_numeric(df['i524a1'], errors='coerce') / 12.0
    else:
        raise ValueError(f'No wage variable for {year}')

    # Weight
    weight = None
    for wv in ['factor07i500a', 'fac500a', 'facpob07', 'factor07', 'fac500']:
        if wv in df.columns:
            weight = pd.to_numeric(df[wv], errors='coerce').fillna(0)
            break
    if weight is None:
        weight = pd.Series(1.0, index=df.index)

    mask = emp & dep & formal & (wage > 0) & (wage < WAGE_MAX) & (weight > 0)
    n = mask.sum()
    print(f'    Final formal-dep sample: {n:,}')

    out = pd.DataFrame({
        'wage':   wage[mask].values,
        'weight': weight[mask].values,
    }, index=df.index[mask])

    # ----- ETHNICITY -----
    # For 2021+: use p558c from Module 500
    # For 2015-2019: merge Module 300 p300a
    if year >= 2021 and 'p558c' in df.columns:
        p558c = pd.to_numeric(df['p558c'][mask], errors='coerce')
        def ethnic_from_558c(x):
            if pd.isna(x) or int(x) in P558C_NOSABE:
                return 'other'
            xi = int(x)
            if xi in P558C_INDIGENOUS: return 'indigenous'
            if xi in P558C_AFRO:       return 'afro'
            if xi in P558C_BLANCO:     return 'blanco'
            if xi in P558C_MESTIZO:    return 'mestizo'
            return 'other'
        out['ethnic_grp'] = p558c.apply(ethnic_from_558c).values
        counts = out['ethnic_grp'].value_counts().to_dict()
        print(f'    Ethnicity (p558c): {counts}')
    elif m300 is not None:
        # Merge Module 300 on conglome+vivienda+hogar+codperso
        merge_keys = [k for k in ['conglome', 'vivienda', 'hogar', 'codperso'] if k in df.columns]
        merge_right = m300[[k for k in merge_keys if k in m300.columns] + ['p300a']].copy()
        # Convert merge keys to same dtype
        df_sub = df.loc[mask, merge_keys].copy()
        for k in merge_keys:
            df_sub[k] = pd.to_numeric(df_sub[k], errors='coerce')
            merge_right[k] = pd.to_numeric(merge_right[k], errors='coerce')
        merged = df_sub.merge(merge_right, on=merge_keys, how='left')
        p300a = pd.to_numeric(merged['p300a'].values, errors='coerce')
        def ethnic_from_300a(x):
            if pd.isna(x): return 'other'
            xi = int(x)
            if xi in P300A_INDIGENOUS: return 'indigenous'
            if xi in P300A_SPANISH:    return 'spanish'
            return 'other'
        out['ethnic_grp'] = np.vectorize(ethnic_from_300a)(p300a)
        counts = out['ethnic_grp'].value_counts().to_dict()
        match_rate = float((~pd.isna(p300a)).mean())
        print(f'    Ethnicity (p300a merge): {counts}  merge_rate={match_rate:.1%}')
    else:
        out['ethnic_grp'] = 'other'
        print(f'    Ethnicity: NOT AVAILABLE for {year}')

    # ----- EDUCATION -----
    if 'p301a' in df.columns:
        p301 = pd.to_numeric(df['p301a'][mask], errors='coerce')
        def educ_cat(x):
            if pd.isna(x): return 'unknown'
            xi = int(x)
            if xi in P301A_PRIMARY:   return 'primary_less'
            if xi in P301A_SECONDARY: return 'secondary'
            if xi in P301A_SUPERIOR:  return 'superior'
            return 'unknown'
        out['educ_cat'] = p301.apply(educ_cat).values
    else:
        out['educ_cat'] = 'unknown'

    # ----- GEOGRAPHY -----
    if 'dominio' in df.columns:
        dom = pd.to_numeric(df['dominio'][mask], errors='coerce')
        out['lima'] = dom.isin(DOMINIO_LIMA).values
    else:
        out['lima'] = False

    if 'estrato' in df.columns:
        est = pd.to_numeric(df['estrato'][mask], errors='coerce')
        out['urban'] = (~est.isin(ESTRATO_RURAL)).values
    else:
        out['urban'] = True

    # ----- EMPLOYER SECTOR -----
    if 'p510' in df.columns:
        p510 = pd.to_numeric(df['p510'][mask], errors='coerce')
        out['employer_public']  = p510.isin(P510_PUBLIC).values
        out['employer_private'] = p510.isin(P510_PRIVATE).values
    else:
        out['employer_public']  = False
        out['employer_private'] = False

    return out


# ==============================================================================
# BUNCHING ESTIMATOR (canonical Cengiz spec)
# ==============================================================================
def cengiz_ratio(wages_pre, wts_pre, wages_post, wts_post, mw_old, mw_new,
                 excess_bins=10):
    """
    Canonical Cengiz et al. (2019) revised bunching.
    Missing mass: neg-only outflow from [0.85*mw_old, mw_new).
    Excess mass:  pos share changes in [mw_new, mw_new + excess_bins*BIN].
    Counterfactual: inverse-abs-weighted background from clean zone (>2*mw_new).
    Ratio = excess / missing_net_outflow.
    """
    if len(wages_pre) < MIN_N or len(wages_post) < MIN_N:
        return {'insufficient_n': True, 'n_pre': len(wages_pre), 'n_post': len(wages_post),
                'bunching_ratio': None}

    bin_edges   = np.arange(WAGE_MIN, WAGE_MAX + BIN_WIDTH, BIN_WIDTH)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    h_pre,  _ = np.histogram(wages_pre,  bins=bin_edges, weights=wts_pre)
    h_post, _ = np.histogram(wages_post, bins=bin_edges, weights=wts_post)

    tp = float(h_pre.sum())
    tq = float(h_post.sum())
    if tp == 0:
        return {'insufficient_n': True, 'n_pre': 0, 'n_post': int(tq), 'bunching_ratio': None}

    sp = h_pre  / tp
    sq = h_post / tq if tq > 0 else np.zeros_like(h_post, dtype=float)
    delta = sq - sp

    # Counterfactual background correction
    clean = bin_centers > 2 * mw_new
    if clean.sum() > 5:
        background = np.average(delta[clean], weights=1.0 / (np.abs(delta[clean]) + 1e-8))
        delta_adj = delta - background
    else:
        delta_adj = delta

    # Missing mass: neg-only in [0.85*mw_old, mw_new)
    lo = 0.85 * mw_old
    zone_below = (bin_centers >= lo) & (bin_centers < mw_new)
    missing = float(-delta_adj[zone_below & (delta_adj < 0)].sum())

    # Excess mass: pos in [mw_new, mw_new + 10*BIN)
    zone_above = (bin_centers >= mw_new) & (bin_centers < mw_new + excess_bins * BIN_WIDTH)
    excess = float(delta_adj[zone_above & (delta_adj > 0)].sum())

    MIN_MISSING = 0.003  # 0.3pp minimum for reliable ratio
    reliable = missing >= MIN_MISSING
    ratio = (excess / missing) if (missing > 0 and reliable) else None

    return {
        'insufficient_n':   False,
        'ratio_unreliable': not reliable,
        'bunching_ratio':   round(ratio, 4) if reliable else None,
        'missing_pp':       round(missing * 100, 4),
        'excess_pp':        round(excess * 100, 4),
        'n_pre_raw':        len(wages_pre),
        'n_post_raw':       len(wages_post),
        'n_pre_weighted':   int(round(tp)),
        'n_post_weighted':  int(round(tq)),
        'median_wage_pre':  round(float(np.median(wages_pre)), 1),
    }


def run_event(df_pre, df_post, mask_pre, mask_post, eid, ecfg, label):
    """Run bunching for one subgroup × event. Returns result dict."""
    w_pre  = df_pre.loc[mask_pre,  'wage'].values
    wt_pre = df_pre.loc[mask_pre,  'weight'].values
    w_post = df_post.loc[mask_post, 'wage'].values
    wt_post= df_post.loc[mask_post, 'weight'].values
    res = cengiz_ratio(w_pre, wt_pre, w_post, wt_post, ecfg['mw_old'], ecfg['mw_new'])
    if res['insufficient_n']:
        print(f'      [{label} | Event {eid}] N insufficient (pre={res["n_pre"]}, post={res["n_post"]})')
    else:
        r_str = f'{res["bunching_ratio"]:.3f}' if res.get('bunching_ratio') is not None else 'unreliable'
        print(f'      [{label} | Event {eid}] ratio={r_str}  '
              f'miss={res["missing_pp"]:.3f}pp  exc={res["excess_pp"]:.3f}pp  '
              f'n_pre={res["n_pre_raw"]:,}')
    return res


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print('=' * 76)
    print('MW ETHNICITY & EXTENDED HETEROGENEITY — CENGIZ BUNCHING BY SUBGROUP')
    print('=' * 76)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    years_m500 = sorted({ec['pre_year'] for ec in EVENTS.values()} |
                        {ec['post_year'] for ec in EVENTS.values()})
    print(f'\nModule 500 years needed: {years_m500}')

    # Load Module 300 for 2015-2019 years
    m300_cache = {}
    for yr in years_m500:
        if yr <= 2019:
            m300_cache[yr] = load_module300(yr)

    # Load Module 500 and build extended samples
    raw_data = {}
    samples  = {}
    for yr in years_m500:
        print(f'\n--- Year {yr} ---')
        df_raw = load_module500(yr)
        raw_data[yr] = df_raw
        samples[yr]  = build_sample(df_raw, yr, m300=m300_cache.get(yr))

    # ------------------------------------------------------------------
    # 2. Define subgroups
    #    Each entry: (name, label, filter_fn(df_sample) -> bool Series/array)
    # ------------------------------------------------------------------
    # Helper: make a flag-based filter
    def col_eq(col, vals):
        return lambda df: pd.Series(
            df[col].isin(vals) if hasattr(df[col], 'isin')
            else np.isin(df[col].values, list(vals)),
            index=df.index)

    subgroups = [
        # --- Total ---
        ('ALL', 'TOTAL (referencia)',
         lambda df: pd.Series(True, index=df.index)),

        # ── Panel A: Ethnicity ──────────────────────────────────────────
        ('ethn_indigenous', 'A. Indigena (quechua/aymara/nativo)',
         lambda df: pd.Series(df['ethnic_grp'] == 'indigenous', index=df.index)),

        ('ethn_afro', 'A. Afrodescendiente (2021+ solo)',
         lambda df: pd.Series(df['ethnic_grp'] == 'afro', index=df.index)),

        ('ethn_mestizo', 'A. Mestizo (2021+ solo)',
         lambda df: pd.Series(df['ethnic_grp'] == 'mestizo', index=df.index)),

        ('ethn_blanco', 'A. Blanco (2021+ solo)',
         lambda df: pd.Series(df['ethnic_grp'] == 'blanco', index=df.index)),

        ('ethn_spanish', 'A. Castellanohablante (2015-19 solo)',
         lambda df: pd.Series(df['ethnic_grp'] == 'spanish', index=df.index)),

        ('ethn_non_indigenous', 'A. No indigena (castellano/mestizo/blanco)',
         lambda df: pd.Series(
             df['ethnic_grp'].isin(['spanish', 'mestizo', 'blanco']), index=df.index)),

        # ── Panel B: Education ──────────────────────────────────────────
        ('educ_primary', 'B. Educacion: primaria o menos',
         lambda df: pd.Series(df['educ_cat'] == 'primary_less', index=df.index)),

        ('educ_secondary', 'B. Educacion: secundaria',
         lambda df: pd.Series(df['educ_cat'] == 'secondary', index=df.index)),

        ('educ_superior', 'B. Educacion: superior',
         lambda df: pd.Series(df['educ_cat'] == 'superior', index=df.index)),

        # ── Panel C: Geography ──────────────────────────────────────────
        ('geo_lima', 'C. Lima Metropolitana',
         lambda df: pd.Series(df['lima'].astype(bool), index=df.index)),

        ('geo_rest', 'C. Resto del pais',
         lambda df: pd.Series(~df['lima'].astype(bool), index=df.index)),

        ('geo_urban', 'C. Area urbana (500+ hab.)',
         lambda df: pd.Series(df['urban'].astype(bool), index=df.index)),

        ('geo_rural', 'C. Area rural (AER)',
         lambda df: pd.Series(~df['urban'].astype(bool), index=df.index)),

        # ── Panel D: Employer sector ────────────────────────────────────
        ('sector_public', 'D. Sector publico (admin/FFAA/emp.pub.)',
         lambda df: pd.Series(df['employer_public'].astype(bool), index=df.index)),

        ('sector_private', 'D. Sector privado (empresa privada)',
         lambda df: pd.Series(df['employer_private'].astype(bool), index=df.index)),
    ]

    # ------------------------------------------------------------------
    # 3. Run analysis
    # ------------------------------------------------------------------
    all_results = []
    ref_year = 2015  # reference year for base stats

    for sg_name, sg_label, sg_filter in subgroups:
        print(f'\n{"─"*68}')
        print(f'SUBGROUP: {sg_label}')

        sg_res = {'name': sg_name, 'label': sg_label}

        # Base stats from reference year
        df_ref = samples[ref_year]
        mask_ref = sg_filter(df_ref)
        n_ref = int(mask_ref.sum())
        if n_ref >= MIN_N:
            wages_ref = df_ref.loc[mask_ref, 'wage'].values
            sg_res['n_ref_2015']       = n_ref
            sg_res['median_wage_2015'] = round(float(np.median(wages_ref)), 1)
            sg_res['share_below_750']  = round(float((wages_ref < 750).mean()), 4)
        else:
            sg_res['n_ref_2015']       = n_ref
            sg_res['median_wage_2015'] = None
            sg_res['share_below_750']  = None
        print(f'  n_2015={n_ref:,}  '
              f'median={sg_res["median_wage_2015"]}  '
              f'share<750={sg_res["share_below_750"]}')

        # Per-event bunching
        for eid, ecfg in EVENTS.items():
            df_pre  = samples[ecfg['pre_year']]
            df_post = samples[ecfg['post_year']]
            m_pre   = sg_filter(df_pre)
            m_post  = sg_filter(df_post)
            ev_res  = run_event(df_pre, df_post, m_pre, m_post, eid, ecfg, sg_label)
            sg_res[f'event_{eid}'] = ev_res

        # Average ratio across events with valid+reliable data
        valid_ratios = [
            sg_res[f'event_{eid}']['bunching_ratio']
            for eid in EVENTS
            if not sg_res[f'event_{eid}'].get('insufficient_n')
               and not sg_res[f'event_{eid}'].get('ratio_unreliable')
               and sg_res[f'event_{eid}'].get('bunching_ratio') is not None
        ]
        sg_res['avg_ratio'] = round(float(np.mean(valid_ratios)), 4) if valid_ratios else None

        all_results.append(sg_res)

    # ------------------------------------------------------------------
    # 4. Print formatted table
    # ------------------------------------------------------------------
    W_LBL  = 40
    W_N    = 8
    W_MED  = 9
    W_SH   = 7
    W_EV   = 9

    hdr = (f'{"Subgroup":<{W_LBL}} {"N_2015":>{W_N}} {"Median":>{W_MED}} '
           f'{"Sh<750":>{W_SH}} {"Ev A":>{W_EV}} {"Ev B":>{W_EV}} {"Ev C":>{W_EV}} {"Avg":>{W_EV}}')
    sep = '─' * (W_LBL + W_N + W_MED + W_SH + 3 * W_EV + W_EV + 10)

    print(f'\n\n{"=" * len(sep)}')
    print('MW EFFECTS — EXTENDED HETEROGENEITY (Cengiz bunching, formal-dep workers)')
    print(f'{"=" * len(sep)}')
    print(hdr)
    print(sep)

    current_panel = ''
    for sg in all_results:
        # Panel header
        panel_letter = sg['label'][0] if sg['label'][0] in 'ABCDT' else ''
        if panel_letter != current_panel and panel_letter in 'ABCD':
            print()
            current_panel = panel_letter

        label = sg['label']
        n_ref = sg.get('n_ref_2015', 0)
        med   = sg.get('median_wage_2015')
        sh    = sg.get('share_below_750')
        avg   = sg.get('avg_ratio')

        def fmt(eid):
            ev = sg.get(f'event_{eid}', {})
            if ev.get('insufficient_n'):
                return f'{"insuf.":>{W_EV}}'
            if ev.get('ratio_unreliable'):
                return f'{"[<MW]":>{W_EV}}'
            br = ev.get('bunching_ratio')
            return f'{br:>{W_EV}.3f}' if br is not None else f'{"---":>{W_EV}}'

        med_s = f'{med:>{W_MED}.0f}' if med is not None else f'{"N/A":>{W_MED}}'
        sh_s  = f'{sh:>{W_SH}.3f}'   if sh  is not None else f'{"N/A":>{W_SH}}'
        avg_s = f'{avg:>{W_EV}.3f}'  if avg is not None else f'{"---":>{W_EV}}'

        print(f'{label:<{W_LBL}} {n_ref:>{W_N},} {med_s} {sh_s} '
              f'{fmt("A")}{fmt("B")}{fmt("C")}{avg_s}')

    print(sep)
    print('\nNotes:')
    print('  Ethnicity Events A & B: from mother tongue (Module 300 p300a, merge).')
    print('  Ethnicity Event C:      from self-ID (Module 500 p558c, 2021/2023).')
    print('  Afro/Mestizo/Blanco cells have no Events A/B data (source mismatch).')
    print('  insuf. = N < 200 unweighted workers. [<MW] = missing mass < 0.3pp.')

    # ------------------------------------------------------------------
    # 5. Save JSON
    # ------------------------------------------------------------------
    output = {
        'generated_at': datetime.now().isoformat(),
        'description': 'Cengiz bunching by ethnicity, education, geography, employer sector',
        'ethnicity_note': (
            'Events A/B: mother tongue from Module 300 (p300a); '
            'Event C: ethnic self-ID from Module 500 (p558c). '
            'Indigenous = quechua+aymara+nativa for both sources.'
        ),
        'events': {eid: {'name': ec['name'], 'mw_old': ec['mw_old'], 'mw_new': ec['mw_new'],
                         'pre_year': ec['pre_year'], 'post_year': ec['post_year']}
                   for eid, ec in EVENTS.items()},
        'subgroups': all_results,
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
