"""
DEFINITIVE MW ANALYSIS — Complete Margins Decomposition

PART 1: Revised Cengiz Bunching
  - Three worker populations: all, formal-dependent, informal
  - Counterfactual subtraction (above-2×MW as control zone)
  - Expanded excess window (+10 bins = S/250)

PART 2: Near-MW Worker Decomposition
  - ENAHO Panel 978 (2021→2023) for Event C
  - EPE Lima panel (pid) for Events A, B, C
  - Treatment: formal workers [0.85×MW_old, MW_new]
  - Control: formal workers [1.2×MW_new, 2.5×MW_new]
  - Outcomes: employment, formal status, transition matrix,
              hours change, wage change

PART 3: Lighthouse Effect (informal wages vs Kaitz)

PART 4: Wage Compression

PART 5: Informal Sector Decomposition (dependent vs self-employed)
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
try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False
    print("WARNING: statsmodels not available — regression parts will be skipped")

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

EVENTS = {
    'A': {'mw_old': 750,  'mw_new': 850,  'pre_year': 2015, 'post_year': 2017,
          'name': 'S/750→S/850 (May 2016)'},
    'B': {'mw_old': 850,  'mw_new': 930,  'pre_year': 2017, 'post_year': 2019,
          'name': 'S/850→S/930 (Apr 2018)'},
    'C': {'mw_old': 930,  'mw_new': 1025, 'pre_year': 2021, 'post_year': 2023,
          'name': 'S/930→S/1025 (May 2022)'},
}

BIN_WIDTH   = 25
WAGE_MAX    = 6000
ENAHO_CS    = 'D:/Nexus/nexus/data/raw/enaho/cross_section'
QT_DIR      = 'D:/Nexus/nexus/data/raw/enaho/quarterly'
PANEL_CSV   = 'D:/Nexus/nexus/data/raw/enaho/978_panel/enaho01a-2020-2024-500-panel.csv'
EPE_CSV_DIR = 'D:/Nexus/nexus/data/raw/epe/csv'

# p507 employment category codes
DEPENDENT   = {3.0, 4.0, 6.0}   # empleado, obrero, trabajador del hogar
INDEPENDENT = {2.0}              # cuenta propia / trabajador independiente
EMPLOYER    = {1.0}              # patron / empleador

STARS = lambda p: '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else '   '))


# ════════════════════════════════════════════════════════════════════════════════
# LOADERS
# ════════════════════════════════════════════════════════════════════════════════

def load_enaho_cs(year):
    """Load ENAHO Module 500 cross-section for a given year."""
    # Try specific patterns first to avoid loading CIIU/CIUO reference tables
    patterns = [
        f'{ENAHO_CS}/modulo_05_{year}/enaho01a-{year}-500.dta',
        f'{ENAHO_CS}/modulo_05_{year}/enaho01a-{year}_500.dta',
        f'{ENAHO_CS}/modulo_05_{year}/enaho01a*{year}*500*.dta',
    ]
    for pat in patterns:
        files = glob.glob(pat, recursive=True)
        # Filter out reference/lookup tables
        files = [f for f in files if not any(x in os.path.basename(f) for x in
                 ['ciiu', 'ciuo', 'tabla', 'diccionario', 'doc'])]
        if files:
            f = sorted(files)[0]
            print(f"  Loading: {os.path.basename(f)}")
            df = pd.read_stata(f, convert_categoricals=False)
            df.columns = [c.lower() for c in df.columns]
            return df
    raise FileNotFoundError(f"No ENAHO Module 500 for {year}")


def load_qt(qt_key):
    """Load quarterly ENAHO file."""
    file_map = {
        'Q3_2015': QT_DIR + '/enaho_qt_Q3_2015_code484.dta',
        'Q4_2015': QT_DIR + '/enaho_qt_Q4_2015_code494.dta',
        'Q1_2016': QT_DIR + '/enaho_qt_Q1_2016_code510.dta',
        'Q1_2017': QT_DIR + '/enaho_qt_Q1_2017_code551.dta',
        'Q1_2018': QT_DIR + '/enaho_qt_Q1_2018_code607.dta',
        'Q1_2019': QT_DIR + '/enaho_qt_Q1_2019_code641.dta',
        'Q1_2022': QT_DIR + '/enaho_qt_Q1_2022_code765.dta',
    }
    path = file_map.get(qt_key)
    if path and os.path.exists(path):
        df = pd.read_stata(path, convert_categoricals=False)
        df.columns = [c.lower() for c in df.columns]
        print(f"  Loaded {qt_key}: {df.shape[0]:,} rows")
        return df
    raise FileNotFoundError(f"Quarterly file not found: {qt_key}")


def load_epe(dir_name):
    """Load EPE Lima quarterly CSV."""
    path = os.path.join(EPE_CSV_DIR, dir_name)
    csvs = glob.glob(os.path.join(path, '*.csv'))
    if not csvs:
        raise FileNotFoundError(f"No CSV in {path}")
    df = pd.read_csv(csvs[0], encoding='latin-1')
    df.columns = [c.lower() for c in df.columns]
    print(f"  Loaded EPE {dir_name}: {df.shape[0]:,} rows")
    return df


# ════════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def get_monthly_wage(df, label=''):
    """Monthly wage from ENAHO cross-section (priority: p524a1 > i524a1/12 > d529t)."""
    if 'p524a1' in df.columns:
        w = pd.to_numeric(df['p524a1'], errors='coerce')
        if w.notna().sum() > 1000 and w.median() > 200:
            print(f"    {label} wage: p524a1 (monthly) median={w.median():.0f}")
            return w
    if 'i524a1' in df.columns:
        w = pd.to_numeric(df['i524a1'], errors='coerce') / 12.0
        print(f"    {label} wage: i524a1/12 (annual→monthly) median={w.median():.0f}")
        return w
    if 'd529t' in df.columns:
        w = pd.to_numeric(df['d529t'], errors='coerce')
        print(f"    {label} wage: d529t (total monthly)")
        return w
    raise ValueError(f"No wage variable found for {label}")


def get_enaho_features(df, label='', source='cross_section'):
    """
    Extract person-level features from ENAHO Module 500.
    PRINTS full variable audit.
    """
    print(f"\n  {'─'*50}")
    print(f"  VARIABLE AUDIT: {label} (n={len(df):,})")
    print(f"  {'─'*50}")

    out = pd.DataFrame(index=df.index)

    # Department
    if 'ubigeo' in df.columns:
        out['dept'] = df['ubigeo'].astype(str).str.zfill(6).str[:2].astype(int, errors='ignore')
    else:
        out['dept'] = np.nan

    # Employment
    emp_raw = pd.to_numeric(df.get('ocu500', pd.Series(np.nan, index=df.index)), errors='coerce')
    out['employed'] = (emp_raw == 1).astype(float)
    print(f"  employed (ocu500==1): {out['employed'].sum():,.0f} / {len(df):,} = {out['employed'].mean():.3f}")

    # Employment category (p507)
    p507 = pd.to_numeric(df.get('p507', pd.Series(np.nan, index=df.index)), errors='coerce')
    out['p507'] = p507
    out['dependent']   = p507.isin(DEPENDENT).astype(float)
    out['independent'] = p507.isin(INDEPENDENT).astype(float)
    dist507 = dict(p507[out['employed']==1].value_counts().sort_index())
    print(f"  p507 (among employed): {dist507}")
    print(f"  dependent workers (3/4/6): {out['dependent'][out['employed']==1].sum():,.0f}")
    print(f"  independent workers (2):   {out['independent'][out['employed']==1].sum():,.0f}")

    # Formality (ocupinf: 1=informal, 2=formal)
    if 'ocupinf' in df.columns:
        ocinf = pd.to_numeric(df['ocupinf'], errors='coerce')
        out['formal'] = (ocinf == 2).astype(float)
        out['informal_worker'] = (ocinf == 1).astype(float)
        dist_inf = dict(ocinf[out['employed']==1].value_counts().sort_index())
        print(f"  ocupinf (among employed, 1=informal 2=formal): {dist_inf}")
    elif 'p510a1' in df.columns:
        # Fallback: labor contract
        p510 = pd.to_numeric(df['p510a1'], errors='coerce')
        out['formal'] = ((p510 == 1) & out['dependent'].astype(bool)).astype(float)
        out['informal_worker'] = ((p510 != 1) & out['dependent'].astype(bool)).astype(float)
        print(f"  Formality from p510a1 (contract==1): {out['formal'][out['employed']==1].sum():,.0f}")
    else:
        out['formal'] = np.nan
        out['informal_worker'] = np.nan
        print("  WARNING: No formality variable available")

    # Set formal/informal to NaN for non-employed
    out.loc[out['employed'] == 0, ['formal', 'informal_worker']] = np.nan
    frm = out['formal'][out['employed']==1]
    print(f"  formal rate (among employed): {frm.mean():.3f}")

    # Wage
    wage = get_monthly_wage(df, label)
    out['wage'] = np.where(out['employed'] == 1, wage, np.nan)
    out['log_wage'] = np.where((out['employed']==1) & (out['wage']>0), np.log(out['wage']), np.nan)

    # Hours (p520 = weekly hours)
    if 'p520' in df.columns:
        h = pd.to_numeric(df['p520'], errors='coerce')
        out['hours'] = np.where((out['employed']==1) & (h>0) & (h<100), h, np.nan)
    else:
        out['hours'] = np.nan

    # Controls
    out['edad'] = pd.to_numeric(df.get('p208a', pd.Series(np.nan, index=df.index)), errors='coerce')
    out['edad_sq'] = out['edad'] ** 2
    sex = pd.to_numeric(df.get('p207', pd.Series(np.nan, index=df.index)), errors='coerce')
    out['male'] = (sex == 1).astype(float)

    # Weight
    for wv in ['fac500a', 'facpob07', 'factor07', 'fac500']:
        if wv in df.columns:
            out['wt'] = pd.to_numeric(df[wv], errors='coerce').fillna(0)
            print(f"  weight: {wv}  sum={out['wt'].sum():,.0f}")
            break
    else:
        out['wt'] = 1.0

    # Wage summary among formal dependent workers
    fd = out[(out['employed']==1) & (out['dependent']==1) & (out['formal']==1) & (out['wage']>0)]
    if len(fd) > 10:
        print(f"  Formal dependent wages: n={len(fd):,}  "
              f"median={fd['wage'].median():.0f}  p10={fd['wage'].quantile(0.1):.0f}  "
              f"p25={fd['wage'].quantile(0.25):.0f}  p75={fd['wage'].quantile(0.75):.0f}  "
              f"p90={fd['wage'].quantile(0.9):.0f}")

    return out


# ════════════════════════════════════════════════════════════════════════════════
# PART 1: REVISED CENGIZ BUNCHING
# ════════════════════════════════════════════════════════════════════════════════

def cengiz_revised(df_pre, df_post, mw_old, mw_new, event_name,
                   sample='all', do_counterfactual=True, excess_bins=10):
    """
    Revised Cengiz bunching.
    sample: 'all' | 'formal_dep' | 'informal' | 'dependent'
    Counterfactual: subtract background change in bins > 2*MW_new
    Excess window: mw_new to mw_new + excess_bins * BIN_WIDTH
    """
    def get_sample_mask(feat, sample_type):
        emp = feat['employed'] == 1
        wage_pos = feat['wage'] > 0
        wage_ok  = feat['wage'] < WAGE_MAX
        wt_ok    = feat['wt'] > 0
        base = emp & wage_pos & wage_ok & wt_ok
        if sample_type == 'all':
            return base
        elif sample_type == 'formal_dep':
            return base & (feat['formal'] == 1) & (feat['dependent'] == 1)
        elif sample_type == 'informal':
            return base & (feat['informal_worker'] == 1)
        elif sample_type == 'dependent':
            return base & (feat['dependent'] == 1)
        elif sample_type == 'independent':
            return base & (feat['independent'] == 1)
        return base

    results = {}
    for label, feat in [('pre', df_pre), ('post', df_post)]:
        mask = get_sample_mask(feat, sample)
        if mask.sum() < 50:
            print(f"    {sample} {label}: only {mask.sum()} obs — skip")
            return None
        w   = feat.loc[mask, 'wage'].values
        wt  = feat.loc[mask, 'wt'].values
        n   = mask.sum()
        print(f"    {sample} {label}: n={n:,}  median={np.median(w):.0f}  "
              f"p10={np.percentile(w,10):.0f}  p25={np.percentile(w,25):.0f}  "
              f"p75={np.percentile(w,75):.0f}  p90={np.percentile(w,90):.0f}")

        bin_edges   = np.arange(0, WAGE_MAX + BIN_WIDTH, BIN_WIDTH)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        counts, _   = np.histogram(w, bins=bin_edges, weights=wt)
        total       = counts.sum()
        shares      = counts / total if total > 0 else np.zeros_like(counts, dtype=float)
        results[label] = {'shares': shares, 'total': float(total), 'bin_centers': bin_centers}

    delta = results['post']['shares'] - results['pre']['shares']
    bc    = results['pre']['bin_centers']

    # Counterfactual correction: subtract mean delta in clean zone (> 2*MW_new)
    if do_counterfactual:
        clean_zone   = bc > 2 * mw_new
        n_clean      = clean_zone.sum()
        if n_clean > 5:
            background = np.average(delta[clean_zone],
                                    weights=1.0 / (np.abs(delta[clean_zone]) + 1e-8))
            delta_adj  = delta - background
            print(f"    Counterfactual: background={background*100:+.4f}pp "
                  f"from {n_clean} clean bins above S/{2*mw_new:.0f}")
        else:
            delta_adj = delta
            print(f"    Counterfactual: not enough clean bins ({n_clean})")
    else:
        delta_adj = delta

    # Missing mass: ONLY the directly-affected window [0.85*mw_old, mw_new)
    # Workers below 0.85*mw_old are not directly affected by the MW increase
    # and their rightward movement from wage growth contaminates the "missing" count.
    # This is the key fix: narrow the affected zone to just the notch.
    affected_lo = 0.85 * mw_old
    below_new   = (bc >= affected_lo) & (bc < mw_new)
    missing     = -delta_adj[below_new & (delta_adj < 0)].sum()
    # Also add any negative delta between mw_old and mw_new (where workers
    # should have been pushed up regardless of sign, they "disappeared" from this zone)
    # Use total mass change in affected zone as true missing mass estimate
    missing_total = -delta_adj[below_new].sum()   # net outflow from notch
    missing_total = max(missing_total, 0.0)

    # Excess mass (new MW to new MW + excess_bins bins, delta > 0)
    at_new = (bc >= mw_new) & (bc < mw_new + excess_bins * BIN_WIDTH)
    excess = delta_adj[at_new & (delta_adj > 0)].sum()

    print(f"    Affected zone: [S/{affected_lo:.0f}, S/{mw_new:.0f}) "
          f"({below_new.sum()} bins)  net outflow={missing_total*100:.3f}pp  "
          f"neg-only missing={missing*100:.3f}pp")

    ratio       = excess / missing if missing > 0 else float('inf')
    ratio_total = excess / missing_total if missing_total > 0 else float('inf')

    total_pre  = results['pre']['total']
    total_post = results['post']['total']
    emp_chg    = (total_post / total_pre - 1) * 100 if total_pre > 0 else 0.0

    print(f"\n    [{sample}] Missing(neg-only)={missing*100:.3f}pp  "
          f"Missing(net outflow)={missing_total*100:.3f}pp  "
          f"Excess={excess*100:.3f}pp")
    print(f"    Ratio(neg-only)={ratio:.3f}  Ratio(net outflow)={ratio_total:.3f}  "
          f"EmpChg={emp_chg:+.2f}%")

    return {
        'sample': sample,
        'event': event_name,
        'missing_mass_pp':       float(missing * 100),
        'missing_net_outflow_pp': float(missing_total * 100),
        'excess_mass_pp':         float(excess * 100),
        'ratio':                  float(ratio),
        'ratio_net_outflow':      float(ratio_total),
        'employment_change_pct': float(emp_chg),
        'bin_data': {
            'bin_centers': bc.tolist(),
            'delta_raw':   delta.tolist(),
            'delta_adj':   delta_adj.tolist(),
            'shares_pre':  results['pre']['shares'].tolist(),
            'shares_post': results['post']['shares'].tolist(),
        }
    }


def run_part1_bunching():
    """Part 1: Revised Cengiz bunching for all events, all sample types."""
    print('\n' + '=' * 70)
    print('PART 1: REVISED CENGIZ BUNCHING')
    print('=' * 70)
    print('Samples: all / formal_dep / informal / dependent')
    print('Counterfactual: background trend from bins > 2×MW_new subtracted')
    print('Excess window: mw_new to mw_new + 10 bins (S/250)')

    all_results = {}
    samples = ['all', 'formal_dep', 'informal', 'dependent']

    for eid, ecfg in EVENTS.items():
        print(f"\n{'─'*65}")
        print(f"EVENT {eid}: {ecfg['name']}")
        print(f"  Pre year: {ecfg['pre_year']}  Post year: {ecfg['post_year']}")
        print(f"{'─'*65}")

        try:
            df_pre_raw  = load_enaho_cs(ecfg['pre_year'])
            df_post_raw = load_enaho_cs(ecfg['post_year'])

            print(f"\n  Pre ({ecfg['pre_year']}):")
            feat_pre  = get_enaho_features(df_pre_raw, f"Pre {ecfg['pre_year']}")

            print(f"\n  Post ({ecfg['post_year']}):")
            feat_post = get_enaho_features(df_post_raw, f"Post {ecfg['post_year']}")

        except Exception as e:
            print(f"  ERROR loading Event {eid}: {e}")
            continue

        event_results = {}
        print(f"\n  BUNCHING RESULTS (MW {ecfg['mw_old']}→{ecfg['mw_new']}):")

        for samp in samples:
            res = cengiz_revised(feat_pre, feat_post,
                                 ecfg['mw_old'], ecfg['mw_new'],
                                 ecfg['name'], sample=samp)
            event_results[samp] = res

        # Print comparison table
        print(f"\n  ┌{'─'*55}┐")
        print(f"  │ {'Sample':<20} {'Missing':>8} {'Excess':>8} {'Ratio':>7} │")
        print(f"  ├{'─'*55}┤")
        for samp in samples:
            r = event_results.get(samp)
            if r:
                print(f"  │ {samp:<20} {r['missing_mass_pp']:>7.3f}pp "
                      f"{r['excess_mass_pp']:>7.3f}pp {r['ratio']:>7.3f} │")
        print(f"  └{'─'*55}┘")

        all_results[eid] = event_results

    return all_results


# ════════════════════════════════════════════════════════════════════════════════
# PART 2: NEAR-MW DECOMPOSITION — ENAHO PANEL (Event C, 2021→2023)
# ════════════════════════════════════════════════════════════════════════════════

def run_part2a_enaho_panel():
    """
    Near-MW decomposition using ENAHO panel 978 (2020-2024 wide format).
    Event C: pre=2021, post=2023.
    Treatment: formal dependent workers earning [0.85*930, 1025] in 2021.
    Control:   formal dependent workers earning [1.2*1025, 2.5*930] in 2021.
    """
    print('\n' + '=' * 70)
    print('PART 2A: NEAR-MW DECOMPOSITION — ENAHO PANEL (Event C, 2021→2023)')
    print('=' * 70)

    MW_OLD, MW_NEW = 930, 1025
    TREAT_LO = 0.85 * MW_OLD   # 790.5
    TREAT_HI = MW_NEW           # 1025
    CTRL_LO  = 1.2  * MW_NEW   # 1230
    CTRL_HI  = 2.5  * MW_OLD   # 2325

    print(f"\n  Treatment band: [S/{TREAT_LO:.0f}, S/{TREAT_HI:.0f})")
    print(f"  Control band:   [S/{CTRL_LO:.0f}, S/{CTRL_HI:.0f})")

    if not os.path.exists(PANEL_CSV):
        print(f"  SKIP: Panel CSV not found at {PANEL_CSV}")
        return None

    print(f"\n  Loading panel CSV (selected columns only)…")
    # Only load the columns we need to avoid the 7,337-column full load
    NEEDED_SUFFIXES = ['21', '22', '23']
    NEEDED_VARS     = ['ocu500', 'ocupinf', 'p507', 'p524a1', 'p520',
                       'facpob07', 'factor07', 'ubigeo']
    needed_cols = (
        ['numper', 'conglome', 'vivienda'] +
        [f'{v}_{s}' for v in NEEDED_VARS for s in NEEDED_SUFFIXES]
    )
    # Verify which columns exist
    header = pd.read_csv(PANEL_CSV, encoding='latin-1', nrows=0)
    header.columns = [c.lower() for c in header.columns]
    available = [c for c in needed_cols if c in header.columns]
    print(f"  Loading {len(available)} of {len(header.columns)} columns")
    df = pd.read_csv(PANEL_CSV, encoding='latin-1', usecols=available, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    print(f"  Shape: {df.shape}")

    # Pre-period = 2021 columns (suffix _21)
    # Post-period = 2023 columns (suffix _23)

    def get_col(col, yr):
        key = f"{col}_{yr}"
        if key in df.columns:
            return pd.to_numeric(df[key], errors='coerce')
        return pd.Series(np.nan, index=df.index)

    # Pre 2021
    emp21    = get_col('ocu500', '21')
    ocinf21  = get_col('ocupinf', '21')
    p507_21  = get_col('p507', '21')
    wage21   = get_col('p524a1', '21')
    hours21  = get_col('p520', '21')
    wt21     = get_col('facpob07', '21')
    wt21     = wt21.where(wt21 > 0, get_col('factor07', '21'))

    # Post 2023
    emp23   = get_col('ocu500', '23')
    ocinf23 = get_col('ocupinf', '23')
    p507_23 = get_col('p507', '23')
    wage23  = get_col('p524a1', '23')
    hours23 = get_col('p520', '23')

    # Variable audit
    print(f"\n  === VARIABLE AUDIT: ENAHO Panel 978 ===")
    print(f"  Total rows: {len(df):,}")
    print(f"  ocu500_21 dist: {dict(emp21.value_counts().sort_index())}")
    print(f"  ocu500_23 dist: {dict(emp23.value_counts().sort_index())}")
    print(f"  ocupinf_21 (1=inf,2=formal) dist: {dict(ocinf21.value_counts().sort_index())}")
    print(f"  p507_21 dist: {dict(p507_21.value_counts().sort_index())}")
    print(f"  p524a1_21 among employed+pos: "
          f"median={wage21[(emp21==1)&(wage21>0)].median():.0f}  "
          f"p25={wage21[(emp21==1)&(wage21>0)].quantile(0.25):.0f}  "
          f"p75={wage21[(emp21==1)&(wage21>0)].quantile(0.75):.0f}")
    n_both = ((emp21.notna()) & (emp23.notna())).sum()
    print(f"  Both 2021+2023 non-null: {n_both:,}")

    # Build pre-sample: formal dependent workers in 2021
    formal_dep_21 = (
        (emp21 == 1) &
        (ocinf21 == 2) &
        (p507_21.isin(DEPENDENT)) &
        (wage21 > 0) &
        (wt21 > 0)
    )
    print(f"\n  Formal dependent employed in 2021: {formal_dep_21.sum():,}")

    # Assign treatment / control
    treat_mask = formal_dep_21 & (wage21 >= TREAT_LO) & (wage21 < TREAT_HI)
    ctrl_mask  = formal_dep_21 & (wage21 >= CTRL_LO)  & (wage21 <= CTRL_HI)
    print(f"  Treatment (pre-MW zone): {treat_mask.sum():,}")
    print(f"  Control (above MW zone): {ctrl_mask.sum():,}")

    if treat_mask.sum() < 30 or ctrl_mask.sum() < 30:
        print("  SKIP: insufficient sample")
        return None

    # Post-period outcomes
    def fmt_pct(x): return f"{x*100:.1f}%"

    results = {}

    for group_name, mask in [('treatment', treat_mask), ('control', ctrl_mask)]:
        sub = df.index[mask]
        n   = mask.sum()

        emp_post    = (emp23.loc[mask] == 1).astype(float)
        formal_post = ((emp23.loc[mask]==1) & (ocinf23.loc[mask]==2)).astype(float)
        dep_post    = ((emp23.loc[mask]==1) & (p507_23.loc[mask].isin(DEPENDENT))).astype(float)
        inf_post    = ((emp23.loc[mask]==1) & (ocinf23.loc[mask]==1)).astype(float)

        # Log wage change (among those employed in both periods with positive wage)
        stayers = mask & (emp21==1) & (emp23==1) & (wage21>0) & (wage23>0)
        dlw = (np.log(wage23.loc[stayers]) - np.log(wage21.loc[stayers]))

        # Hours change
        h_both = mask & (emp21==1) & (emp23==1) & (hours21>0) & (hours21<100) & (hours23>0) & (hours23<100)
        dh = (hours23.loc[h_both] - hours21.loc[h_both])

        results[group_name] = {
            'n': int(n),
            'emp_retention':  float(emp_post.mean()),
            'formal_retention': float(formal_post.mean()),
            'formal→informal': float(inf_post.mean()),
            'formal→not_emp':  float((1 - emp_post).mean()),
            'n_stayers': int(stayers.sum()),
            'dlw_mean':  float(dlw.mean()) if len(dlw) > 0 else np.nan,
            'dlw_pct':   float(np.exp(dlw.mean())-1)*100 if len(dlw) > 0 else np.nan,
            'dh_mean':   float(dh.mean()) if len(dh) > 0 else np.nan,
            'hours_reduction_pct': float((dh < -4).mean()) if len(dh) > 0 else np.nan,
        }

    # DiD estimates
    print(f"\n  === RESULTS: Event C Near-MW Decomposition (ENAHO Panel) ===")
    print(f"  N treatment: {results['treatment']['n']:,}  "
          f"N control: {results['control']['n']:,}")

    metrics = [
        ('Employment retention', 'emp_retention'),
        ('Formal retention',     'formal_retention'),
        ('formal→informal',      'formal→informal'),
        ('formal→not employed',  'formal→not_emp'),
        ('Δ log wage (stayers)', 'dlw_pct'),
        ('Δ weekly hours',       'dh_mean'),
        ('Hours reduction >4h',  'hours_reduction_pct'),
    ]

    did_results = {}
    print(f"\n  {'Metric':<28} {'Treatment':>12} {'Control':>12} {'DiD':>10}")
    print(f"  {'─'*68}")
    for label, key in metrics:
        t = results['treatment'].get(key, np.nan)
        c = results['control'].get(key, np.nan)
        if np.isnan(t) or np.isnan(c):
            print(f"  {label:<28} {'N/A':>12} {'N/A':>12} {'N/A':>10}")
            continue
        did = t - c
        unit = '%' if 'pct' in key or 'retention' in key or '→' in key or 'Reduction' in key else ''
        scale = 100 if 'pct' in key or 'retention' in key or '→' in key or 'Reduction' in key else 1
        print(f"  {label:<28} {t*scale:>11.1f}{unit} {c*scale:>11.1f}{unit} {did*scale:>+9.1f}{unit}")
        did_results[key] = {'treatment': float(t), 'control': float(c), 'did': float(did)}

    # Transition matrix (treatment group)
    print(f"\n  TRANSITION MATRIX (treatment group, Event C 2021→2023):")
    print(f"  (All workers who were formal-dependent near MW in 2021)")

    formal_pre_mask  = treat_mask & (ocinf21 == 2)
    informal_pre_mask = treat_mask & (ocinf21 == 1)

    def transition_row(pre_mask, label):
        if pre_mask.sum() == 0:
            print(f"  {label}: 0 workers")
            return
        n = pre_mask.sum()
        formal_post  = ((emp23.loc[pre_mask]==1) & (ocinf23.loc[pre_mask]==2)).mean() * 100
        inform_post  = ((emp23.loc[pre_mask]==1) & (ocinf23.loc[pre_mask]==1)).mean() * 100
        not_emp_post = (emp23.loc[pre_mask] != 1).mean() * 100
        nan_post     = emp23.loc[pre_mask].isna().mean() * 100
        print(f"  {label:<22} n={n:>5,}  "
              f"→formal:{formal_post:>5.1f}%  "
              f"→informal:{inform_post:>5.1f}%  "
              f"→not_emp:{not_emp_post:>5.1f}%  "
              f"→missing:{nan_post:>5.1f}%")

    transition_row(formal_pre_mask,  "formal_pre")
    transition_row(informal_pre_mask, "informal_pre")
    # Also full treatment group
    transition_row(treat_mask, "all_treat_pre")

    return {'enaho_panel_event_C': results, 'did': did_results}


# ════════════════════════════════════════════════════════════════════════════════
# PART 2B: NEAR-MW DECOMPOSITION — EPE LIMA PANEL (Events A, B, C)
# ════════════════════════════════════════════════════════════════════════════════

EPE_EVENTS = {
    'A': {
        'mw_old': 750, 'mw_new': 850,
        'pre_dir':  'epe_2016_ene_feb_mar',
        'post_dir': 'epe_2016_jun_jul_ago',
        'name': 'S/750→S/850 (May 2016)',
    },
    'B': {
        'mw_old': 850, 'mw_new': 930,
        'pre_dir':  'epe_2018_ene_feb_mar',
        'post_dir': 'epe_2018_jun_jul_ago',
        'name': 'S/850→S/930 (Apr 2018)',
    },
    'C': {
        'mw_old': 930, 'mw_new': 1025,
        'pre_dir':  'epe_2022_ene_feb_mar',
        'post_dir': 'epe_2022_jun_jul_ago',
        'name': 'S/930→S/1025 (May 2022)',
    },
}
EPE_FORMAL_CODES = {1.0, 2.0, 3.0}  # EsSalud, Seguro Privado, Ambos


def featurize_epe(df):
    """Extract features from EPE Lima CSV with full audit."""
    out = pd.DataFrame(index=df.index)
    out['pid'] = (df['conglome'].astype(str).str.strip() + '_' +
                  df['vivienda'].astype(str).str.strip() + '_' +
                  df['hogar'].astype(str).str.strip() + '_' +
                  df['codperso'].astype(str).str.strip())
    out['employed'] = (pd.to_numeric(df['ocu200'], errors='coerce') == 1).astype(float)
    out['wage']     = pd.to_numeric(df['ingprin'], errors='coerce')
    out['log_wage'] = np.where((out['employed']==1) & (out['wage']>0), np.log(out['wage']), np.nan)
    p222 = pd.to_numeric(df['p222'], errors='coerce')
    out['formal'] = p222.isin(EPE_FORMAL_CODES).astype(float)
    out.loc[out['employed'] == 0, 'formal'] = np.nan
    # Hours
    if 'p214' in df.columns:
        h = pd.to_numeric(df['p214'], errors='coerce')
        out['hours'] = np.where((out['employed']==1) & (h>0) & (h<100), h, np.nan)
    elif 'p220' in df.columns:
        h = pd.to_numeric(df['p220'], errors='coerce')
        out['hours'] = np.where((out['employed']==1) & (h>0) & (h<100), h, np.nan)
    else:
        out['hours'] = np.nan
    # Weight
    fa_cols = [c for c in df.columns if c.startswith('fa_')]
    if fa_cols:
        out['wt'] = pd.to_numeric(df[fa_cols[0]], errors='coerce').fillna(0)
    else:
        out['wt'] = 1.0
    return out


def run_part2b_epe_panel():
    """Near-MW decomposition using EPE Lima panel matching."""
    print('\n' + '=' * 70)
    print('PART 2B: NEAR-MW DECOMPOSITION — EPE LIMA PANEL (Events A, B, C)')
    print('=' * 70)
    print('Formality = p222 ∈ {1,2,3} (EsSalud/SegPrivado/Ambos)')
    print('Panel ID  = conglome+vivienda+hogar+codperso')

    all_results = {}

    for eid, ecfg in EPE_EVENTS.items():
        mw_old, mw_new = ecfg['mw_old'], ecfg['mw_new']
        TREAT_LO = 0.85 * mw_old
        TREAT_HI = mw_new
        CTRL_LO  = 1.2  * mw_new
        CTRL_HI  = 2.5  * mw_old

        print(f"\n{'─'*65}")
        print(f"EVENT {eid}: {ecfg['name']}")
        print(f"  Treatment: [S/{TREAT_LO:.0f}, S/{TREAT_HI:.0f})  "
              f"Control: [S/{CTRL_LO:.0f}, S/{CTRL_HI:.0f}]")

        try:
            df_pre_raw  = load_epe(ecfg['pre_dir'])
            df_post_raw = load_epe(ecfg['post_dir'])
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        # Variable audit
        print(f"\n  PRE variable audit ({ecfg['pre_dir']}):")
        fa_cols = [c for c in df_pre_raw.columns if c.startswith('fa_')]
        p222_dist = dict(pd.to_numeric(df_pre_raw.get('p222', pd.Series()), errors='coerce')
                           .value_counts().sort_index())
        w_pre = pd.to_numeric(df_pre_raw['ingprin'], errors='coerce')
        emp_pre = (pd.to_numeric(df_pre_raw['ocu200'], errors='coerce') == 1)
        print(f"  ocu200==1: {emp_pre.sum():,}  ingprin median (employed): "
              f"{w_pre[emp_pre].median():.0f}  p222 dist: {p222_dist}  wt: {fa_cols[0] if fa_cols else 'uniform'}")

        feat_pre  = featurize_epe(df_pre_raw)
        feat_post = featurize_epe(df_post_raw)

        # Panel match on pid
        post_sub = feat_post[['pid', 'employed', 'formal', 'wage', 'log_wage', 'hours']].copy()
        post_sub.columns = ['pid', 'emp_post', 'formal_post', 'wage_post', 'lw_post', 'hours_post']

        panel = feat_pre[feat_pre['employed'] == 1].merge(post_sub, on='pid', how='inner')
        n_pre  = (feat_pre['employed'] == 1).sum()
        n_matched = len(panel)
        match_rate = n_matched / n_pre * 100 if n_pre > 0 else 0
        print(f"\n  Panel match: {n_matched:,} / {n_pre:,} ({match_rate:.1f}%)")

        if n_matched < 50:
            print("  SKIP: too few matched workers")
            continue

        # Assign treatment/control in pre-period
        panel['treat']   = ((panel['wage'] >= TREAT_LO) & (panel['wage'] < TREAT_HI)).astype(int)
        panel['control'] = ((panel['wage'] >= CTRL_LO)  & (panel['wage'] <= CTRL_HI)).astype(int)
        sample = panel[(panel['treat']==1) | (panel['control']==1)].copy()

        n_treat = (sample['treat']==1).sum()
        n_ctrl  = (sample['control']==1).sum()
        print(f"  N treatment: {n_treat:,}  N control: {n_ctrl:,}")

        if n_treat < 20 or n_ctrl < 20:
            print("  SKIP: too few treat/ctrl obs")
            continue

        # Compute outcomes
        t = sample[sample['treat']==1]
        c = sample[sample['control']==1]

        metrics_dict = {}
        rows = []
        for grp_name, grp in [('treatment', t), ('control', c)]:
            emp_r   = grp['emp_post'].mean()
            frm_r   = (grp['emp_post'] == 1) & (grp['formal_post'] == 1)
            frm_r   = frm_r.mean()
            inf_r   = ((grp['emp_post']==1) & (grp['formal_post']==0)).mean()
            not_e   = (grp['emp_post'] != 1).mean()
            # Wage change (stayers with positive wage in both)
            stayers = (grp['emp_post']==1) & (grp['wage']>0) & (grp['wage_post']>0)
            dlw     = (np.log(grp.loc[stayers,'wage_post']) - np.log(grp.loc[stayers,'wage']))
            # Hours change
            h_stayers = stayers & grp['hours'].notna() & grp['hours_post'].notna()
            dh = (grp.loc[h_stayers,'hours_post'] - grp.loc[h_stayers,'hours'])

            metrics_dict[grp_name] = {
                'n':              int(len(grp)),
                'emp_retention':  float(emp_r),
                'formal_retention': float(frm_r),
                'formal→informal':  float(inf_r),
                'formal→not_emp':   float(not_e),
                'n_stayers':      int(stayers.sum()),
                'dlw_pct':        float((np.exp(dlw.mean())-1)*100) if len(dlw)>0 else np.nan,
                'dh_mean':        float(dh.mean()) if len(dh)>0 else np.nan,
                'hours_red_pct':  float((dh<-4).mean()*100) if len(dh)>0 else np.nan,
            }

        # Print results table
        print(f"\n  {'Metric':<28} {'Treatment':>12} {'Control':>12} {'DiD':>10}")
        print(f"  {'─'*68}")
        metric_labels = [
            ('Employment retention',    'emp_retention',  100, '%'),
            ('Formal retention',        'formal_retention', 100, '%'),
            ('formal→informal',         'formal→informal', 100, '%'),
            ('formal→not_emp',          'formal→not_emp',  100, '%'),
            ('Δ log wage (stayers %)',   'dlw_pct',          1,  '%'),
            ('Δ weekly hours',           'dh_mean',          1, 'h'),
            ('Hours reduction >4h',     'hours_red_pct',     1,  '%'),
        ]
        for label, key, scale, unit in metric_labels:
            tv = metrics_dict['treatment'].get(key, np.nan)
            cv = metrics_dict['control'].get(key, np.nan)
            if np.isnan(tv) or np.isnan(cv):
                print(f"  {label:<28} {'N/A':>12} {'N/A':>12} {'N/A':>10}")
                continue
            did = tv - cv
            print(f"  {label:<28} {tv*scale:>11.1f}{unit} {cv*scale:>11.1f}{unit} "
                  f"{did*scale:>+9.1f}{unit}")

        # Transition matrix (treatment only, formal pre vs informal pre)
        print(f"\n  TRANSITION MATRIX (treatment group):")
        print(f"  {'Pre status':<22} {'→formal':>9} {'→informal':>11} "
              f"{'→not_emp':>10} {'→missing':>10}")
        for pre_label, pre_filter in [
            ('formal_pre (p222∈{1,2,3})',   t['formal']==1),
            ('informal_pre (p222∉{1,2,3})', t['formal']==0),
        ]:
            grp = t[pre_filter]
            if len(grp) == 0:
                continue
            fml = ((grp['emp_post']==1) & (grp['formal_post']==1)).mean()*100
            inf = ((grp['emp_post']==1) & (grp['formal_post']==0)).mean()*100
            noe = (grp['emp_post'] != 1).mean()*100
            mis = grp['emp_post'].isna().mean()*100
            print(f"  {pre_label:<22} {fml:>8.1f}%  {inf:>10.1f}%  "
                  f"{noe:>9.1f}%  {mis:>9.1f}%  n={len(grp):,}")

        all_results[eid] = metrics_dict

    return all_results


# ════════════════════════════════════════════════════════════════════════════════
# PART 3: LIGHTHOUSE EFFECT (Informal wages vs Kaitz, dept-level DiD)
# ════════════════════════════════════════════════════════════════════════════════

def run_part3_lighthouse():
    """
    Lighthouse: informal workers in high-Kaitz vs low-Kaitz departments.
    Using ENAHO annual cross-section. Event C: pre=2021, post=2023.
    Informal workers earning [0.5×MW_old, 1.2×MW_old] in pre-period.
    Control: informal workers earning [1.5×MW_new, 2.5×MW_new] in pre-period.
    """
    print('\n' + '=' * 70)
    print('PART 3: LIGHTHOUSE EFFECT')
    print('Informal wages → do they rise more in high-Kaitz departments?')
    print('=' * 70)
    print('Reference: Derenoncourt et al. (2025) for Brazil')

    results = {}

    for eid, ecfg in EVENTS.items():
        mw_old, mw_new = ecfg['mw_old'], ecfg['mw_new']
        print(f"\n{'─'*60}")
        print(f"Event {eid}: {ecfg['name']}")

        try:
            df_pre_raw  = load_enaho_cs(ecfg['pre_year'])
            df_post_raw = load_enaho_cs(ecfg['post_year'])
            feat_pre  = get_enaho_features(df_pre_raw,  f"Pre {ecfg['pre_year']}")
            feat_post = get_enaho_features(df_post_raw, f"Post {ecfg['post_year']}")
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        # Compute Kaitz from pre-period (among formal dependent workers)
        fd_pre = feat_pre[(feat_pre['formal']==1) & (feat_pre['dependent']==1) &
                          (feat_pre['employed']==1) & (feat_pre['wage']>0) & (feat_pre['wt']>0)]

        def wmedian(g):
            g = g.dropna(subset=['wage']).sort_values('wage')
            cum = g['wt'].cumsum(); tot = g['wt'].sum()
            if tot == 0 or len(g) == 0: return np.nan
            return float(g.loc[(cum >= tot/2).idxmax(), 'wage'])

        try:
            med = fd_pre.groupby('dept').apply(wmedian, include_groups=False)
        except TypeError:
            med = fd_pre.groupby('dept').apply(wmedian)
        med = pd.to_numeric(med, errors='coerce')
        kaitz = (mw_old / med).rename('kaitz_pre')

        print(f"  Kaitz: n_depts={kaitz.notna().sum()}  "
              f"range=[{kaitz.min():.3f},{kaitz.max():.3f}]  mean={kaitz.mean():.3f}")

        # Split Kaitz into high / low (median split)
        kaitz_med = kaitz.median()
        high_kaitz_depts = set(kaitz[kaitz >= kaitz_med].index.astype(str))
        low_kaitz_depts  = set(kaitz[kaitz <  kaitz_med].index.astype(str))

        # Informal workers near old MW
        INFORM_LO = 0.5  * mw_old
        INFORM_HI = 1.2  * mw_old
        CTRL_LO   = 1.5  * mw_new
        CTRL_HI   = 2.5  * mw_new

        def get_informal_lw(feat, wage_lo, wage_hi):
            mask = (
                (feat['employed']==1) &
                (feat['informal_worker']==1) &
                (feat['wage'] >= wage_lo) &
                (feat['wage'] <= wage_hi) &
                (feat['wt'] > 0)
            )
            return feat.loc[mask, ['dept', 'log_wage', 'wt']]

        inf_pre_near  = get_informal_lw(feat_pre,  INFORM_LO, INFORM_HI)
        inf_post_near = get_informal_lw(feat_post, INFORM_LO, INFORM_HI)
        inf_pre_ctrl  = get_informal_lw(feat_pre,  CTRL_LO, CTRL_HI)
        inf_post_ctrl = get_informal_lw(feat_post, CTRL_LO, CTRL_HI)

        def wmean_lw(group):
            if len(group) == 0: return np.nan
            return np.average(group['log_wage'].dropna(),
                              weights=group.loc[group['log_wage'].notna(), 'wt'])

        def lw_by_kaitz(df_pre_g, df_post_g, dept_set):
            pre  = df_pre_g[df_pre_g['dept'].astype(str).isin(dept_set)]
            post = df_post_g[df_post_g['dept'].astype(str).isin(dept_set)]
            lw_pre  = wmean_lw(pre)
            lw_post = wmean_lw(post)
            dlw = lw_post - lw_pre if not np.isnan(lw_pre) and not np.isnan(lw_post) else np.nan
            return lw_pre, lw_post, dlw, len(pre), len(post)

        near_hi = lw_by_kaitz(inf_pre_near, inf_post_near, high_kaitz_depts)
        near_lo = lw_by_kaitz(inf_pre_near, inf_post_near, low_kaitz_depts)
        ctrl_hi = lw_by_kaitz(inf_pre_ctrl, inf_post_ctrl, high_kaitz_depts)
        ctrl_lo = lw_by_kaitz(inf_pre_ctrl, inf_post_ctrl, low_kaitz_depts)

        # DiD: (high_kaitz_near - low_kaitz_near) - (high_kaitz_ctrl - low_kaitz_ctrl)
        lighthouse_did = np.nan
        if not any(np.isnan(x[2]) for x in [near_hi, near_lo, ctrl_hi, ctrl_lo]):
            treat_diff  = near_hi[2] - near_lo[2]
            control_diff = ctrl_hi[2] - ctrl_lo[2]
            lighthouse_did = treat_diff - control_diff

        print(f"\n  Informal workers near MW (S/{INFORM_LO:.0f}–S/{INFORM_HI:.0f}):")
        print(f"    High-Kaitz depts: Δlog_wage = {near_hi[2]:+.4f} "
              f"(n_pre={near_hi[3]:,}, n_post={near_hi[4]:,})")
        print(f"    Low-Kaitz depts:  Δlog_wage = {near_lo[2]:+.4f} "
              f"(n_pre={near_lo[3]:,}, n_post={near_lo[4]:,})")
        print(f"  Informal workers far above MW (S/{CTRL_LO:.0f}–S/{CTRL_HI:.0f}) — control:")
        print(f"    High-Kaitz depts: Δlog_wage = {ctrl_hi[2]:+.4f}")
        print(f"    Low-Kaitz depts:  Δlog_wage = {ctrl_lo[2]:+.4f}")
        if not np.isnan(lighthouse_did):
            print(f"\n  LIGHTHOUSE DiD: {lighthouse_did:+.4f} "
                  f"({(np.exp(lighthouse_did)-1)*100:+.2f}%)")
            if lighthouse_did > 0:
                print(f"  → EVIDENCE of lighthouse effect: informal wages rose more "
                      f"in high-MW-binding departments")
            else:
                print(f"  → No lighthouse effect detected (DiD ≤ 0)")
        else:
            print(f"  Lighthouse DiD: insufficient data")

        # Regression: Δlog_wage_informal_dept ~ Kaitz + controls
        if HAS_SM:
            feat_pre['kaitz']  = feat_pre['dept'].map(kaitz).values
            feat_post['kaitz'] = feat_post['dept'].map(kaitz).values

            # Dept-level change in informal near-MW log wages
            def dept_mean_lw(feat, wage_lo, wage_hi):
                mask = ((feat['employed']==1) & (feat['informal_worker']==1) &
                        (feat['wage']>=wage_lo) & (feat['wage']<=wage_hi) & (feat['wt']>0))
                return feat.loc[mask].groupby('dept').apply(
                    lambda g: np.average(g['log_wage'].dropna(),
                                         weights=g.loc[g['log_wage'].notna(),'wt'])
                    if g['log_wage'].notna().sum() > 0 else np.nan
                )
            lw_pre_d  = dept_mean_lw(feat_pre,  INFORM_LO, INFORM_HI)
            lw_post_d = dept_mean_lw(feat_post, INFORM_LO, INFORM_HI)
            dlw_d     = (lw_post_d - lw_pre_d).dropna()
            kaitz_d   = kaitz.reindex(dlw_d.index)
            reg_data  = pd.DataFrame({'dlw': dlw_d, 'kaitz': kaitz_d}).dropna()

            if len(reg_data) >= 10:
                try:
                    mod = sm.OLS(reg_data['dlw'], sm.add_constant(reg_data['kaitz'])).fit(
                        cov_type='HC1')
                    b  = float(mod.params.get('kaitz', np.nan))
                    se = float(mod.bse.get('kaitz', np.nan))
                    p  = float(mod.pvalues.get('kaitz', np.nan))
                    print(f"\n  OLS: Δlog_wage_informal ~ Kaitz (dept-level, n={len(reg_data)})")
                    print(f"  β_kaitz = {b:+.4f}  SE={se:.4f}  p={p:.3f}{STARS(p)}")
                    if b > 0 and p < 0.1:
                        print(f"  → Statistically significant lighthouse effect")
                except Exception as e:
                    print(f"  OLS failed: {e}")

        results[eid] = {
            'kaitz_median_split': float(kaitz_med),
            'near_high': {'dlw': float(near_hi[2]) if not np.isnan(near_hi[2]) else None, 'n_pre': near_hi[3]},
            'near_low':  {'dlw': float(near_lo[2]) if not np.isnan(near_lo[2]) else None, 'n_pre': near_lo[3]},
            'lighthouse_did': float(lighthouse_did) if not np.isnan(lighthouse_did) else None,
        }

    return results


# ════════════════════════════════════════════════════════════════════════════════
# PART 4: WAGE COMPRESSION
# ════════════════════════════════════════════════════════════════════════════════

def run_part4_compression():
    """
    Wage compression: formal workers at [MW_new, 1.5×MW_new] vs [2×MW_new, 3×MW_new].
    Using ENAHO annual cross-section (pseudo-panel: compare same wage bands pre vs post).
    """
    print('\n' + '=' * 70)
    print('PART 4: WAGE COMPRESSION')
    print('Compression group: formal workers [MW_new, 1.5×MW_new]')
    print('High-wage group:   formal workers [2×MW_new, 3×MW_new]')
    print('=' * 70)

    results = {}

    for eid, ecfg in EVENTS.items():
        mw_old, mw_new = ecfg['mw_old'], ecfg['mw_new']
        COMP_LO = mw_new
        COMP_HI = 1.5 * mw_new
        HIGH_LO = 2.0 * mw_new
        HIGH_HI = 3.0 * mw_new

        print(f"\n{'─'*60}")
        print(f"Event {eid}: {ecfg['name']}")
        print(f"  Compression zone: S/{COMP_LO:.0f} – S/{COMP_HI:.0f}")
        print(f"  High-wage zone:   S/{HIGH_LO:.0f} – S/{HIGH_HI:.0f}")

        try:
            df_pre_raw  = load_enaho_cs(ecfg['pre_year'])
            df_post_raw = load_enaho_cs(ecfg['post_year'])
            feat_pre  = get_enaho_features(df_pre_raw,  f"Pre {ecfg['pre_year']}")
            feat_post = get_enaho_features(df_post_raw, f"Post {ecfg['post_year']}")
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        def mean_lw_formal_dep(feat, lo, hi):
            mask = (
                (feat['employed']==1) & (feat['formal']==1) & (feat['dependent']==1) &
                (feat['wage'] >= lo) & (feat['wage'] <= hi) &
                (feat['wage'] > 0) & (feat['wt'] > 0)
            )
            sub = feat.loc[mask, ['log_wage', 'wt']]
            if sub['log_wage'].notna().sum() < 10:
                return np.nan, 0
            return (float(np.average(sub['log_wage'].dropna(),
                                     weights=sub.loc[sub['log_wage'].notna(),'wt'])),
                    int(mask.sum()))

        comp_pre,  n_comp_pre  = mean_lw_formal_dep(feat_pre,  COMP_LO, COMP_HI)
        comp_post, n_comp_post = mean_lw_formal_dep(feat_post, COMP_LO, COMP_HI)
        high_pre,  n_high_pre  = mean_lw_formal_dep(feat_pre,  HIGH_LO, HIGH_HI)
        high_post, n_high_post = mean_lw_formal_dep(feat_post, HIGH_LO, HIGH_HI)

        dlw_comp = comp_post - comp_pre if not any(np.isnan([comp_pre, comp_post])) else np.nan
        dlw_high = high_post - high_pre if not any(np.isnan([high_pre, high_post])) else np.nan
        comp_did = dlw_comp - dlw_high if not any(np.isnan([dlw_comp, dlw_high])) else np.nan

        print(f"\n  Compression zone [S/{COMP_LO:.0f}–S/{COMP_HI:.0f}]:")
        print(f"    Pre log_wage = {comp_pre:.4f} (n={n_comp_pre:,})  "
              f"Post = {comp_post:.4f} (n={n_comp_post:,})")
        print(f"    Δlog_wage = {dlw_comp:+.4f} ({(np.exp(dlw_comp)-1)*100:+.1f}%)" if not np.isnan(dlw_comp) else "    N/A")

        print(f"  High-wage zone [S/{HIGH_LO:.0f}–S/{HIGH_HI:.0f}]:")
        print(f"    Pre log_wage = {high_pre:.4f} (n={n_high_pre:,})  "
              f"Post = {high_post:.4f} (n={n_high_post:,})")
        print(f"    Δlog_wage = {dlw_high:+.4f} ({(np.exp(dlw_high)-1)*100:+.1f}%)" if not np.isnan(dlw_high) else "    N/A")

        if not np.isnan(comp_did):
            print(f"\n  COMPRESSION DiD: {comp_did:+.4f} "
                  f"({(np.exp(dlw_comp)-1)*100:+.1f}% vs {(np.exp(dlw_high)-1)*100:+.1f}%)")
            if comp_did < -0.02:
                print(f"  → WAGE COMPRESSION DETECTED: workers just above MW "
                      f"grew wages slower than higher earners")
            elif comp_did > 0.02:
                print(f"  → INVERSE: workers just above MW grew faster (spillover effects)")
            else:
                print(f"  → No significant compression (DiD ≈ 0)")

        results[eid] = {
            'dlw_compression': float(dlw_comp) if not np.isnan(dlw_comp) else None,
            'dlw_highwage':    float(dlw_high) if not np.isnan(dlw_high) else None,
            'compression_did': float(comp_did) if not np.isnan(comp_did) else None,
            'pct_comp': float((np.exp(dlw_comp)-1)*100) if not np.isnan(dlw_comp) else None,
            'pct_high': float((np.exp(dlw_high)-1)*100) if not np.isnan(dlw_high) else None,
        }

    return results


# ════════════════════════════════════════════════════════════════════════════════
# PART 5: INFORMAL SECTOR DECOMPOSITION
# ════════════════════════════════════════════════════════════════════════════════

def run_part5_informal_decomp():
    """
    Decompose informal workers into:
    a) Dependent informals (asalariados sin beneficios): p507 ∈ {3,4,6} AND ocupinf==1
    b) Self-employed: p507==2 AND ocupinf==1
    Run separate Cengiz bunching for each group.
    """
    print('\n' + '=' * 70)
    print('PART 5: INFORMAL SECTOR DECOMPOSITION')
    print('a) Dependent informals (asalariados sin beneficios)')
    print('b) Self-employed (cuenta propia)')
    print('=' * 70)

    results = {}

    for eid, ecfg in EVENTS.items():
        mw_old, mw_new = ecfg['mw_old'], ecfg['mw_new']
        print(f"\n{'─'*60}")
        print(f"Event {eid}: {ecfg['name']}")

        try:
            df_pre_raw  = load_enaho_cs(ecfg['pre_year'])
            df_post_raw = load_enaho_cs(ecfg['post_year'])
            feat_pre  = get_enaho_features(df_pre_raw,  f"Pre {ecfg['pre_year']}")
            feat_post = get_enaho_features(df_post_raw, f"Post {ecfg['post_year']}")
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        # For self-employed, p524a1 (wage-earner variable) is NaN.
        # Build an alternative wage from i524a1/12 or d529t for independents only.
        def get_self_emp_wage(df_raw):
            """Get best monthly wage for self-employed workers."""
            df = df_raw.copy()
            df.columns = [c.lower() for c in df.columns]
            p507 = pd.to_numeric(df.get('p507', pd.Series(np.nan, index=df.index)), errors='coerce')
            ind_mask = p507.isin(INDEPENDENT)
            # Use i524a1/12 (annual labor income → monthly) or d529t (monthly labor income)
            if 'i524a1' in df.columns:
                w = pd.to_numeric(df['i524a1'], errors='coerce') / 12.0
                if w[ind_mask].notna().sum() > 100:
                    return w
            if 'd529t' in df.columns:
                w = pd.to_numeric(df['d529t'], errors='coerce')
                if w[ind_mask].notna().sum() > 100:
                    return w
            return pd.Series(np.nan, index=df.index)

        # Build a composite wage: p524a1 for dependents, i524a1/12 for self-employed
        se_wage_pre  = get_self_emp_wage(df_pre_raw)
        se_wage_post = get_self_emp_wage(df_post_raw)
        feat_pre_se  = feat_pre.copy()
        feat_post_se = feat_post.copy()
        # Fill in self-employed wage where p524a1 is missing/zero
        ind_pre  = feat_pre['independent'] == 1
        ind_post = feat_post['independent'] == 1
        feat_pre_se.loc[ind_pre,  'wage'] = se_wage_pre.loc[ind_pre].values
        feat_post_se.loc[ind_post, 'wage'] = se_wage_post.loc[ind_post].values

        event_res = {}
        for samp_name in ['dep_informal', 'self_employed']:
            def make_mask(feat, s):
                base = (feat['employed']==1) & (feat['wage']>0) & (feat['wage']<WAGE_MAX) & (feat['wt']>0)
                if s == 'dep_informal':
                    return base & (feat['informal_worker']==1) & (feat['dependent']==1)
                elif s == 'self_employed':
                    return base & (feat['independent']==1)
                return base

            # Use composite wage for self-employed
            fp = feat_pre_se  if samp_name == 'self_employed' else feat_pre
            fp2 = feat_post_se if samp_name == 'self_employed' else feat_post

            print(f"\n  Sample: {samp_name}")
            if samp_name == 'self_employed':
                # Self-employed income (p523/i524a1/d529t) is in Module 34, not Module 500.
                # Module 500 records p524a1 (wage) only for dependent workers.
                # Bunching for self-employed requires Module 34 merge — not implemented.
                print("    SKIP: self-employed income not in Module 500 (use Module 34).")
                print("    Module 34 data not available for this run.")
                continue
            res_parts = {}
            for period, feat in [('pre', fp), ('post', fp2)]:
                m = make_mask(feat, samp_name)
                if m.sum() < 50:
                    print(f"    {period}: only {m.sum()} obs")
                    res_parts[period] = None
                    continue
                w   = feat.loc[m, 'wage'].values
                wt  = feat.loc[m, 'wt'].values
                print(f"    {period}: n={m.sum():,}  median={np.median(w):.0f}  "
                      f"p10={np.percentile(w,10):.0f}  p25={np.percentile(w,25):.0f}  "
                      f"p75={np.percentile(w,75):.0f}  p90={np.percentile(w,90):.0f}")
                bin_edges   = np.arange(0, WAGE_MAX + BIN_WIDTH, BIN_WIDTH)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                counts, _   = np.histogram(w, bins=bin_edges, weights=wt)
                total       = counts.sum()
                shares      = counts / total if total > 0 else np.zeros_like(counts, float)
                res_parts[period] = {'shares': shares, 'total': float(total)}

            if not res_parts.get('pre') or not res_parts.get('post'):
                continue

            delta = res_parts['post']['shares'] - res_parts['pre']['shares']
            bc    = results.get('bin_centers', np.arange(BIN_WIDTH/2, WAGE_MAX, BIN_WIDTH))

            bin_edges   = np.arange(0, WAGE_MAX + BIN_WIDTH, BIN_WIDTH)
            bc          = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Counterfactual
            clean = bc > 2 * mw_new
            if clean.sum() > 5:
                bg    = delta[clean].mean()
                delta = delta - bg

            # Narrow affected zone: [0.85*mw_old, mw_new)
            affected_lo = 0.85 * ecfg['mw_old']
            below_new   = (bc >= affected_lo) & (bc < mw_new)
            missing_net = max(-delta[below_new].sum(), 0.0)  # net outflow from notch
            missing_neg = -delta[below_new & (delta < 0)].sum()  # neg-only
            at_new      = (bc >= mw_new) & (bc < mw_new + 10 * BIN_WIDTH)
            excess      = delta[at_new & (delta > 0)].sum()
            ratio       = excess / missing_neg if missing_neg > 0 else float('inf')
            ratio_net   = excess / missing_net  if missing_net  > 0 else float('inf')
            emp_chg     = (res_parts['post']['total'] / res_parts['pre']['total'] - 1) * 100

            print(f"    → Missing(neg)={missing_neg*100:.3f}pp  "
                  f"Missing(net)={missing_net*100:.3f}pp  "
                  f"Excess={excess*100:.3f}pp  "
                  f"Ratio(neg)={ratio:.3f}  Ratio(net)={ratio_net:.3f}  EmpChg={emp_chg:+.2f}%")
            event_res[samp_name] = {
                'missing_mass_pp': float(missing_neg*100),
                'missing_net_pp':  float(missing_net*100),
                'excess_mass_pp':  float(excess*100),
                'ratio':           float(ratio),
                'ratio_net':       float(ratio_net),
                'emp_chg':         float(emp_chg)
            }

        results[eid] = event_res

    return results


# ════════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════════

def print_final_summary(bunching, near_mw_panel, near_mw_epe, lighthouse, compression, informal):
    print('\n\n' + '=' * 70)
    print('COMPLETE MW EFFECTS DECOMPOSITION — FINAL SUMMARY')
    print('=' * 70)

    for eid, ecfg in EVENTS.items():
        mw_old, mw_new = ecfg['mw_old'], ecfg['mw_new']
        print(f"\n{'━'*70}")
        print(f"EVENT {eid}: {ecfg['name']}")
        print(f"{'━'*70}")

        # Bunching
        print(f"\n  BUNCHING (Cengiz revised, excess window = mw_new + S/250):")
        b = bunching.get(eid, {})
        for samp in ['all', 'formal_dep', 'informal']:
            r = b.get(samp)
            if r:
                print(f"    {samp:<20} missing={r['missing_mass_pp']:.3f}pp  "
                      f"excess={r['excess_mass_pp']:.3f}pp  ratio={r['ratio']:.3f}")

        # Near-MW panel (EPE Lima)
        epe = near_mw_epe.get(eid, {})
        if epe:
            t = epe.get('treatment', {})
            c_g = epe.get('control', {})
            print(f"\n  NEAR-MW DECOMPOSITION (EPE Lima, n_treat={t.get('n',0):,} / n_ctrl={c_g.get('n',0):,}):")
            metrics_print = [
                ('Employment retention', 'emp_retention', 100, '%'),
                ('Formal retention',     'formal_retention', 100, '%'),
                ('formal→informal',      'formal→informal', 100, '%'),
                ('formal→not_emp',       'formal→not_emp',  100, '%'),
                ('Δ log wage (stayers)', 'dlw_pct',          1,  '%'),
                ('Δ weekly hours',       'dh_mean',          1, ' hrs'),
            ]
            print(f"  {'Metric':<26} {'Treatment':>10} {'Control':>10} {'DiD':>8}")
            print(f"  {'─'*58}")
            for lbl, key, scale, unit in metrics_print:
                tv = t.get(key, np.nan)
                cv = c_g.get(key, np.nan)
                if isinstance(tv, float) and isinstance(cv, float) and not np.isnan(tv) and not np.isnan(cv):
                    print(f"  {lbl:<26} {tv*scale:>9.1f}{unit} {cv*scale:>9.1f}{unit} "
                          f"{(tv-cv)*scale:>+7.1f}{unit}")

        # Panel (ENAHO 978 for Event C)
        if eid == 'C' and near_mw_panel:
            t = near_mw_panel.get('enaho_panel_event_C', {}).get('treatment', {})
            c_g = near_mw_panel.get('enaho_panel_event_C', {}).get('control', {})
            if t:
                print(f"\n  NEAR-MW (ENAHO Panel 978, n_treat={t.get('n',0):,} / "
                      f"n_ctrl={c_g.get('n',0):,}):")
                for lbl, key, scale, unit in metrics_print:
                    tv = t.get(key, np.nan)
                    cv = c_g.get(key, np.nan)
                    if isinstance(tv, float) and isinstance(cv, float) and not np.isnan(tv) and not np.isnan(cv):
                        print(f"  {lbl:<26} {tv*scale:>9.1f}{unit} {cv*scale:>9.1f}{unit} "
                              f"{(tv-cv)*scale:>+7.1f}{unit}")

        # Lighthouse
        lh = lighthouse.get(eid, {})
        if lh.get('lighthouse_did') is not None:
            did = lh['lighthouse_did']
            near_hi = lh.get('near_high', {}).get('dlw')
            near_lo = lh.get('near_low', {}).get('dlw')
            print(f"\n  LIGHTHOUSE EFFECT:")
            if near_hi is not None:
                print(f"    High-Kaitz depts: Δlog_wage = {near_hi:+.4f}  "
                      f"Low-Kaitz: {near_lo:+.4f}  DiD = {did:+.4f} "
                      f"({'LIGHTHOUSE ✓' if did>0 else 'no effect'})")

        # Compression
        comp = compression.get(eid, {})
        if comp.get('compression_did') is not None:
            print(f"\n  WAGE COMPRESSION:")
            print(f"    [MW_new, 1.5×MW]: {comp.get('pct_comp', np.nan):+.1f}%  "
                  f"[2×, 3×MW]: {comp.get('pct_high', np.nan):+.1f}%  "
                  f"DiD = {comp['compression_did']:+.4f} "
                  f"({'COMPRESSION ✓' if comp['compression_did']<-0.02 else 'spillover' if comp['compression_did']>0.02 else 'none'})")

        # Informal decomp bunching
        inf = informal.get(eid, {})
        if inf:
            print(f"\n  INFORMAL DECOMPOSITION (Cengiz bunching):")
            for samp_name in ['dep_informal', 'self_employed']:
                r = inf.get(samp_name)
                if r:
                    print(f"    {samp_name:<20} ratio={r['ratio']:.3f}  "
                          f"missing={r['missing_mass_pp']:.3f}pp  emp_chg={r['emp_chg']:+.2f}%")
            if not inf.get('self_employed'):
                print(f"    self_employed:       N/A (income in Module 34, not Module 500)")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=' * 70)
    print('MW COMPLETE MARGINS ANALYSIS')
    print('Parts 1–5: Bunching, Near-MW Decomp, Lighthouse, Compression, Informality')
    print('=' * 70)

    all_output = {}

    # ── PART 1 ───────────────────────────────────────────────────────────────
    try:
        bunching = run_part1_bunching()
        all_output['bunching_revised'] = bunching
    except Exception as e:
        print(f"\nPART 1 ERROR: {e}")
        import traceback; traceback.print_exc()
        bunching = {}

    # ── PART 2A: ENAHO Panel ──────────────────────────────────────────────────
    try:
        near_mw_panel = run_part2a_enaho_panel()
        all_output['near_mw_panel'] = near_mw_panel
    except Exception as e:
        print(f"\nPART 2A ERROR: {e}")
        import traceback; traceback.print_exc()
        near_mw_panel = None

    # ── PART 2B: EPE Lima ─────────────────────────────────────────────────────
    try:
        near_mw_epe = run_part2b_epe_panel()
        all_output['near_mw_epe'] = near_mw_epe
    except Exception as e:
        print(f"\nPART 2B ERROR: {e}")
        import traceback; traceback.print_exc()
        near_mw_epe = {}

    # ── PART 3: Lighthouse ────────────────────────────────────────────────────
    try:
        lighthouse = run_part3_lighthouse()
        all_output['lighthouse'] = lighthouse
    except Exception as e:
        print(f"\nPART 3 ERROR: {e}")
        import traceback; traceback.print_exc()
        lighthouse = {}

    # ── PART 4: Compression ───────────────────────────────────────────────────
    try:
        compression = run_part4_compression()
        all_output['wage_compression'] = compression
    except Exception as e:
        print(f"\nPART 4 ERROR: {e}")
        import traceback; traceback.print_exc()
        compression = {}

    # ── PART 5: Informal Decomp ───────────────────────────────────────────────
    try:
        informal = run_part5_informal_decomp()
        all_output['informal_decomp'] = informal
    except Exception as e:
        print(f"\nPART 5 ERROR: {e}")
        import traceback; traceback.print_exc()
        informal = {}

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print_final_summary(bunching, near_mw_panel, near_mw_epe, lighthouse, compression, informal)

    # ── SAVE ──────────────────────────────────────────────────────────────────
    def clean_json(v):
        if isinstance(v, dict):  return {k: clean_json(x) for k, x in v.items()}
        if isinstance(v, list):  return [clean_json(x) for x in v]
        if isinstance(v, float) and np.isnan(v): return None
        if isinstance(v, (np.integer,)):  return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        return v

    out_path = 'D:/Nexus/nexus/exports/data/mw_complete_margins.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(clean_json(all_output), f, indent=2, ensure_ascii=False)
    print(f'\nSaved to {out_path}')

    qhaw_path = 'D:/qhawarina/public/assets/data/mw_complete_margins.json'
    os.makedirs(os.path.dirname(qhaw_path), exist_ok=True)
    shutil.copy(out_path, qhaw_path)
    print(f'Copied to {qhaw_path}')
