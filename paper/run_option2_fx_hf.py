#!/usr/bin/env python3
"""
run_option2_fx_hf.py
Option 2: FX High-Frequency Monetary Policy Surprise Instrument
================================================================
Constructs an FX-based surprise measure around BCRP announcement dates
and uses it as a proxy-SVAR / LP-IV instrument for identification.

DATA SITUATION (reported at run time):
  - Daily PEN/USD data: NOT AVAILABLE in the Nexus data store.
    Only monthly average PN01246PM is present.
  - EMBIG / sovereign spread: NOT AVAILABLE in the Nexus data store.
  - Fallback: use the change in monthly FX around announcement months
    as a coarser instrument (z_monthly).

Strategy implemented:
  1. Report data availability findings in full.
  2. Build BCRP announcement calendar (first Thursday of each month,
     2004-2025).
  3. Construct monthly FX surprise: change in monthly avg FX in the
     announcement month vs the prior month  (coarser but available).
  4. Aggregate to quarterly: sum within each quarter.
  5. Load VAR data (identical to run_robustness.py load_var_data()).
  6. Proxy-SVAR diagnostics:
     a. Estimate baseline VAR(1) [tot, gdp, cpi, fx, rate].
     b. Extract rate equation residual u_rate.
     c. First-stage F-stat: regress u_rate on z_q.
     d. Report F, R^2, partial R^2.
  7. If F > 10: LP-IV using z_q as instrument for the rate shock.
  8. LP-OLS (for comparison).
  9. Save figure to paper/output/robustness/option2_fx_hf.pdf.
"""

import sys, io, warnings, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT    = Path('D:/Nexus/nexus')
PANEL   = ROOT / 'data/processed/national/panel_national_monthly.parquet'
RAW_DIR = ROOT / 'data/raw/bcrp'
OUT_DIR = ROOT / 'paper/output/robustness'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
C_BLACK = '#000000'
C_DARK  = '#404040'
C_MED   = '#808080'
C_LIGHT = '#c0c0c0'
C_BAND1 = '#d0d0d0'

plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Palatino', 'Georgia', 'Times New Roman', 'DejaVu Serif'],
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
})

BASELINE_PEAK_GDP = -0.195

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: DATA AVAILABILITY AUDIT
# ═══════════════════════════════════════════════════════════════════════════════

def audit_data_availability():
    print("=" * 70)
    print("OPTION 2: FX HIGH-FREQUENCY MONETARY POLICY SURPRISE INSTRUMENT")
    print("=" * 70)
    print()
    print("─" * 70)
    print("SECTION 1: DATA AVAILABILITY AUDIT")
    print("─" * 70)

    findings = {}

    # ── Check panel for daily FX ──────────────────────────────────────────
    raw = pd.read_parquet(PANEL)
    raw['date'] = pd.to_datetime(raw['date'])

    target_ids = ['PD04637PD', 'PD04639PD']
    found_daily = []
    for sid in target_ids:
        hits = raw[raw['series_id'] == sid]
        if len(hits):
            found_daily.append(sid)
            print(f"  [FOUND]   Panel series {sid}: {len(hits)} rows, "
                  f"freq={hits['frequency_original'].iloc[0]}")
        else:
            print(f"  [MISSING] Panel series {sid} (daily PEN/USD) — not in panel_national_monthly.parquet")

    findings['daily_fx_in_panel'] = len(found_daily) > 0

    # ── Check monthly FX ──────────────────────────────────────────────────
    monthly_fx_id = 'PN01246PM'
    mfx = raw[raw['series_id'] == monthly_fx_id]
    if len(mfx):
        print(f"  [FOUND]   Monthly FX {monthly_fx_id}: {len(mfx)} rows, "
              f"{mfx['date'].min().date()} to {mfx['date'].max().date()}, "
              f"freq={mfx['frequency_original'].iloc[0]}")
        findings['monthly_fx'] = True
        findings['monthly_fx_data'] = mfx[['date','value_raw']].set_index('date').sort_index()
    else:
        print(f"  [MISSING] Monthly FX {monthly_fx_id}")
        findings['monthly_fx'] = False

    # ── Check raw bcrp directory for daily CSV files ──────────────────────
    raw_dir = Path('D:/Nexus/nexus/data/raw/bcrp')
    if raw_dir.exists():
        daily_csvs = []
        for root, dirs, files in os.walk(raw_dir):
            for f in files:
                if f.endswith('.csv'):
                    fp = Path(root) / f
                    daily_csvs.append(str(fp))
        if daily_csvs:
            print(f"  [INFO]    Found {len(daily_csvs)} CSV files in raw/bcrp:")
            for f in daily_csvs[:5]:
                print(f"            {f}")
            findings['raw_csv_files'] = daily_csvs
        else:
            print("  [MISSING] No CSV files in D:/Nexus/nexus/data/raw/bcrp/")
            findings['raw_csv_files'] = []
    else:
        print("  [MISSING] D:/Nexus/nexus/data/raw/bcrp/ not found")
        findings['raw_csv_files'] = []

    # ── Check for EMBIG / sovereign spread ───────────────────────────────
    embig_series = []
    all_panel_ids = raw['series_id'].unique().tolist()
    embig_keywords = ['EMBI', 'EMBIG', 'SPREAD', 'CDS', 'SOVEREIGN', 'RIESGO_PAIS']
    for sid in all_panel_ids:
        if any(kw in sid.upper() for kw in embig_keywords):
            embig_series.append(sid)
    # Also check series_name
    embig_by_name = raw[raw['series_name'].str.contains(
        'EMBIG|embi|riesgo pais|country risk|spread', case=False, na=False
    )]['series_id'].unique().tolist()
    all_embig = list(set(embig_series + embig_by_name))
    if all_embig:
        print(f"  [FOUND]   EMBIG/risk spread series: {all_embig}")
        findings['embig'] = True
    else:
        print("  [MISSING] No EMBIG / sovereign spread / CDS data in panel")
        findings['embig'] = False

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("  SUMMARY OF DATA AVAILABILITY:")
    print(f"    Daily PEN/USD (PD04637PD, PD04639PD): {'YES' if findings['daily_fx_in_panel'] else 'NO — not available'}")
    print(f"    Monthly PEN/USD (PN01246PM):            {'YES' if findings['monthly_fx'] else 'NO'}")
    print(f"    Daily FX CSV in raw/bcrp/:              {len(findings.get('raw_csv_files', []))} file(s)")
    print(f"    EMBIG / sovereign spread:               {'YES' if findings['embig'] else 'NO — not available'}")
    print()
    print("  WHAT DAILY DATA WOULD BE NEEDED:")
    print("    For a true HF instrument, the ideal source is BCRP's own")
    print("    daily FX fixing data (series PD04637PD = buying rate,")
    print("    PD04639PD = selling rate) downloaded at daily frequency.")
    print("    These are published by BCRP at estadisticas.bcrp.gob.pe")
    print("    as daily time series. The Nexus pipeline currently only")
    print("    ingests them at monthly aggregation (PN01246PM).")
    print()
    print("  FALLBACK APPROACH (implemented below):")
    print("    Use monthly FX change in announcement month as coarser proxy.")
    print("    Instrument: z_m = ln(FX_m) - ln(FX_{m-1}), where m = announcement month.")
    print("    Aggregate to quarterly: z_q = sum of z_m within each quarter.")
    print("    This is less clean than a ±1 day window but still captures")
    print("    the direction of FX response in months where BCRP acts.")

    return findings


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: BCRP ANNOUNCEMENT CALENDAR
# ═══════════════════════════════════════════════════════════════════════════════

def build_bcrp_calendar(start_year=2004, end_year=2025):
    """
    BCRP announces on a pre-set monthly calendar.
    Approximate rule: first Thursday of each month.
    Returns a list of dates.
    """
    print("─" * 70)
    print("SECTION 2: BCRP ANNOUNCEMENT CALENDAR")
    print("─" * 70)

    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Find first Thursday (weekday=3)
            d = pd.Timestamp(year, month, 1)
            # Move forward to Thursday
            days_to_thu = (3 - d.weekday()) % 7
            first_thu = d + pd.Timedelta(days=days_to_thu)
            dates.append(first_thu)

    dates = pd.DatetimeIndex(dates)
    print(f"  Constructed {len(dates)} announcement dates from "
          f"{dates[0].date()} to {dates[-1].date()}")
    print(f"  First 6 dates: {[str(d.date()) for d in dates[:6]]}")

    # Map to year-month for merging with monthly data
    ann_months = pd.Series(1, index=dates, name='announcement')
    ann_months_df = pd.DataFrame({
        'year':  dates.year,
        'month': dates.month,
        'ann_date': dates
    })
    return dates, ann_months_df


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: CONSTRUCT MONTHLY FX SURPRISE INSTRUMENT
# ═══════════════════════════════════════════════════════════════════════════════

def build_monthly_fx_instrument(monthly_fx_data, ann_months_df):
    """
    Monthly FX surprise:
      z_m = ln(FX_m) - ln(FX_{m-1})
    where m is an announcement month.

    Also compute the FX change in the month AFTER the announcement
    (for alternative specification).
    """
    print("─" * 70)
    print("SECTION 3: MONTHLY FX SURPRISE INSTRUMENT")
    print("─" * 70)

    fx = monthly_fx_data.copy()
    fx.index = pd.to_datetime(fx.index)
    fx = fx.sort_index()
    fx.columns = ['fx_level']

    # Log changes
    fx['log_fx'] = np.log(fx['fx_level'])
    fx['dlog_fx'] = fx['log_fx'].diff()   # ln(FX_m) - ln(FX_{m-1})

    # Add year/month
    fx['year']  = fx.index.year
    fx['month'] = fx.index.month

    # Merge with announcement calendar
    ann = ann_months_df[['year','month']].copy()
    ann['is_announcement'] = 1

    fx_merged = fx.merge(ann, on=['year','month'], how='left')
    fx_merged['is_announcement'] = fx_merged['is_announcement'].fillna(0)

    # z_m = dlog_fx only in announcement months
    fx_merged['z_m'] = np.where(fx_merged['is_announcement'] == 1,
                                fx_merged['dlog_fx'], np.nan)

    n_ann = fx_merged['is_announcement'].sum()
    n_z   = fx_merged['z_m'].notna().sum()
    print(f"  Announcement months identified: {int(n_ann)}")
    print(f"  z_m observations (non-NaN): {n_z}")
    print(f"  z_m mean:  {fx_merged['z_m'].mean():.6f}")
    print(f"  z_m std:   {fx_merged['z_m'].std():.6f}")
    print(f"  z_m range: [{fx_merged['z_m'].min():.6f}, {fx_merged['z_m'].max():.6f}]")

    # Reindex to DatetimeIndex for resampling
    fx_merged_idx = fx_merged.copy()
    fx_merged_idx.index = pd.to_datetime(
        dict(year=fx_merged_idx['year'], month=fx_merged_idx['month'], day=1)
    )

    print()
    print("  Sample z_m values (announcement months):")
    sample = fx_merged_idx[fx_merged_idx['is_announcement']==1][['fx_level','dlog_fx','z_m']].head(10)
    print(sample.round(6).to_string())

    return fx_merged_idx


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: AGGREGATE TO QUARTERLY
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_to_quarterly(fx_monthly):
    """
    Aggregate monthly z_m to quarterly:
      z_q = sum of z_m within quarter (treating NaN as 0 for quarters
            with no announcement — i.e., keep NaN if no announcement in Q)
    """
    print("─" * 70)
    print("SECTION 4: QUARTERLY AGGREGATION")
    print("─" * 70)

    # Sum of z_m per quarter (NaN-aware: if no announcement, sum = NaN)
    z_q = fx_monthly['z_m'].resample('QE').sum(min_count=1)
    z_q.name = 'z_q'

    # Also build a "count" — how many announcements per quarter
    count_q = fx_monthly['is_announcement'].resample('QE').sum()
    count_q.name = 'n_ann_q'

    result = pd.DataFrame({'z_q': z_q, 'n_ann_q': count_q})
    result = result.dropna(subset=['z_q'])

    print(f"  Quarterly z_q observations: {len(result)}")
    print(f"  Date range: {result.index.min().date()} to {result.index.max().date()}")
    print(f"  z_q mean:   {result['z_q'].mean():.6f}")
    print(f"  z_q std:    {result['z_q'].std():.6f}")
    print(f"  Quarters per year with ≥1 announcement: "
          f"{(result['n_ann_q'] >= 1).sum()} of {len(result)}")
    print()
    print("  First 8 quarterly z_q values:")
    print(result.head(8).round(6).to_string())

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5: LOAD VAR DATA (identical to run_robustness.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_var_data():
    print("─" * 70)
    print("SECTION 5: VAR DATA LOADING")
    print("─" * 70)

    raw = pd.read_parquet(PANEL)
    raw['date'] = pd.to_datetime(raw['date'])

    series_map = {
        'PD04722MM': 'rate_raw',
        'PN01731AM': 'gdp_m',
        'PN01271PM': 'cpi_m',
        'PN01246PM': 'fx_m',
        'PN38923BM': 'tot_m',
    }

    frames = []
    for sid, col in series_map.items():
        s = raw[raw['series_id'] == sid][['date', 'value_raw']].copy()
        s = s.rename(columns={'value_raw': col})
        s = s.set_index('date').sort_index()
        frames.append(s)

    monthly = frames[0].join(frames[1:], how='outer')
    monthly = monthly.sort_index()

    q = pd.DataFrame()
    q['rate_level'] = monthly['rate_raw'].resample('QE').mean()
    q['d_rate']     = q['rate_level'].diff()
    q['gdp']        = monthly['gdp_m'].resample('QE').sum()
    q['cpi']        = monthly['cpi_m'].resample('QE').sum()
    q['fx']         = monthly['fx_m'].resample('QE').mean().pct_change() * 100
    q['tot']        = monthly['tot_m'].resample('QE').mean().pct_change() * 100
    q = q.dropna()
    q = q.loc['2004-04-01':'2025-09-30']

    # FWL COVID dummies
    covid_dummy = pd.Series(0, index=q.index, name='covid')
    for cq in [pd.Timestamp('2020-03-31'), pd.Timestamp('2020-06-30')]:
        if cq in covid_dummy.index:
            covid_dummy.loc[cq] = 1

    def fwl_partial(series, dummy):
        X = sm.add_constant(dummy)
        res = sm.OLS(series, X).fit()
        return series - res.fittedvalues + series.mean()

    for col in ['d_rate', 'gdp', 'cpi', 'fx', 'tot', 'rate_level']:
        if col in q.columns:
            q[col] = fwl_partial(q[col], covid_dummy)

    var_df = q[['tot', 'gdp', 'cpi', 'fx', 'd_rate']].copy()
    var_df.columns = ['tot', 'gdp', 'cpi', 'fx', 'rate']

    t0 = var_df.index[0]
    t1 = var_df.index[-1]
    T  = len(var_df)
    print(f"  VAR data: {t0.year}Q{t0.quarter} to {t1.year}Q{t1.quarter}, T={T}")
    print(f"  Columns: {list(var_df.columns)}")
    print(f"  Means:   {var_df.mean().round(3).to_dict()}")

    return var_df, q


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6: PROXY-SVAR DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def run_proxy_svar(var_df, z_quarterly):
    print("─" * 70)
    print("SECTION 6: PROXY-SVAR DIAGNOSTICS")
    print("─" * 70)

    # ── 6a: Estimate baseline VAR(1) ─────────────────────────────────────
    var1_model  = VAR(var_df)
    var1_result = var1_model.fit(1, trend='c')
    max_eig = float(max(abs(np.linalg.eigvals(var1_result.coefs[0]))))
    print(f"  Baseline VAR(1) estimated. Max eigenvalue: {max_eig:.4f}")
    print(f"  T={var1_result.nobs}, K={var1_result.neqs}")
    print()

    # ── 6b: Extract rate equation residual ───────────────────────────────
    # Rate is the last variable (index 4)
    resids = var1_result.resid   # DataFrame, columns = var names
    u_rate = resids['rate'].copy()
    u_rate.name = 'u_rate'

    print(f"  Rate equation residual u_rate:")
    print(f"    Mean:  {u_rate.mean():.6f}")
    print(f"    Std:   {u_rate.std():.6f}")
    print(f"    N obs: {len(u_rate)}")
    print()

    # ── 6c: Merge residuals with quarterly instrument ─────────────────────
    # Align on index (both should be quarterly QE)
    z_q = z_quarterly['z_q'].copy()

    # Align
    merged = pd.DataFrame({'u_rate': u_rate, 'z_q': z_q}).dropna()
    N_merged = len(merged)
    print(f"  Merged (u_rate, z_q): N = {N_merged} overlapping quarters")
    if N_merged == 0:
        print("  ERROR: No overlapping observations — check index alignment.")
        print("  u_rate index sample:", u_rate.index[:5].tolist())
        print("  z_q index sample:   ", z_q.index[:5].tolist())
        return None, None, None

    print(f"  u_rate sample (first 5):\n    {u_rate.head(5).round(4).to_dict()}")
    print(f"  z_q sample (first 5):\n    {z_q.head(5).round(6).to_dict()}")
    print()

    # ── First-stage regression: u_rate ~ z_q ─────────────────────────────
    X_fs = sm.add_constant(merged['z_q'])
    y_fs = merged['u_rate']
    fs   = sm.OLS(y_fs, X_fs).fit()

    F_stat  = fs.fvalue
    F_pval  = fs.f_pvalue
    R2_fs   = fs.rsquared
    coef_z  = fs.params['z_q']
    se_z    = fs.bse['z_q']
    t_stat  = fs.tvalues['z_q']

    # Partial R^2 (same as R^2 here since only one regressor besides const)
    partial_R2 = R2_fs

    print("  FIRST-STAGE REGRESSION: u_rate ~ const + z_q")
    print(f"    N observations:     {N_merged}")
    print(f"    Coefficient z_q:    {coef_z:.6f}")
    print(f"    Std Error:          {se_z:.6f}")
    print(f"    t-statistic:        {t_stat:.4f}")
    print(f"    F-statistic:        {F_stat:.4f}")
    print(f"    F p-value:          {F_pval:.6f}")
    print(f"    R-squared:          {R2_fs:.6f}")
    print(f"    Partial R-squared:  {partial_R2:.6f}")
    print()

    if F_stat > 10:
        print(f"  STRONG INSTRUMENT (F={F_stat:.2f} > 10): proceeding to LP-IV.")
    elif F_stat > 5:
        print(f"  WEAK INSTRUMENT WARNING (F={F_stat:.2f}, 5 < F < 10): LP-IV results unreliable.")
    else:
        print(f"  VERY WEAK INSTRUMENT (F={F_stat:.2f} < 5): instrument is invalid for IV.")
        print("  Reporting LP-IV for completeness; results should NOT be interpreted causally.")

    print()

    # Instrument quality assessment
    print("  INSTRUMENT QUALITY ASSESSMENT:")
    print(f"    Sign of z_q coefficient: {'+' if coef_z > 0 else '-'}")
    print("    Economic interpretation: a positive z_q (FX depreciation in")
    print("    announcement month) should be associated with a rate CUT")
    print("    residual (BCRP cutting rates to defend against depreciation?).")
    print("    OR: rate hike → appreciation → negative z_q, negative coef.")
    if coef_z < 0:
        print("    Observed: negative coef — rate hike associated with FX appreciation. [EXPECTED]")
    else:
        print("    Observed: positive coef — rate hike associated with FX depreciation. [UNEXPECTED]")

    return fs, merged, max_eig


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7: LOCAL PROJECTIONS (OLS and IV)
# ═══════════════════════════════════════════════════════════════════════════════

def run_lp(var_df, merged_iv, F_stat, horizon=9, n_boot=500):
    print("─" * 70)
    print("SECTION 7: LOCAL PROJECTIONS — OLS AND IV")
    print("─" * 70)

    RESPONSES = ['gdp', 'cpi', 'fx']
    results   = {}

    # Controls: lags of all variables
    controls = var_df[['tot', 'gdp', 'cpi', 'fx', 'rate']].copy()

    for resp_var in RESPONSES:
        print(f"\n  Response variable: {resp_var.upper()}")
        lp_ols  = np.full(horizon, np.nan)
        lp_iv   = np.full(horizon, np.nan)
        ci_lo_ols = np.full(horizon, np.nan)
        ci_hi_ols = np.full(horizon, np.nan)
        ci_lo_iv  = np.full(horizon, np.nan)
        ci_hi_iv  = np.full(horizon, np.nan)

        for h in range(horizon):
            # Dependent variable: h-step ahead change (cumulative from h=0)
            y_h = var_df[resp_var].shift(-h) - var_df[resp_var]
            y_h.name = f'{resp_var}_h{h}'

            # Contemporaneous controls: lag-1 of all vars
            X_ctrl = pd.DataFrame({
                'const':   1,
                'tot_l1':  controls['tot'].shift(1),
                'gdp_l1':  controls['gdp'].shift(1),
                'cpi_l1':  controls['cpi'].shift(1),
                'fx_l1':   controls['fx'].shift(1),
                'rate_l1': controls['rate'].shift(1),
            })

            # Endogenous: rate (contemporaneous)
            endog = controls['rate'].copy()

            # Instrument: z_q aligned
            z_q = merged_iv['z_q'].copy()

            # Build combined frame
            df_h = pd.concat([y_h, endog, z_q, X_ctrl], axis=1).dropna()
            if len(df_h) < 20:
                continue

            y_vals    = df_h.iloc[:, 0]
            rate_vals = df_h['rate']
            z_vals    = df_h['z_q']
            ctrl_cols = [c for c in X_ctrl.columns]
            ctrl_vals = df_h[ctrl_cols]

            # ── LP-OLS ────────────────────────────────────────────────────
            X_ols = pd.concat([ctrl_vals, rate_vals], axis=1)
            try:
                ols = sm.OLS(y_vals, X_ols).fit(cov_type='HC1')
                lp_ols[h]    = ols.params['rate']
                ci_lo_ols[h] = ols.conf_int(alpha=0.10).loc['rate', 0]
                ci_hi_ols[h] = ols.conf_int(alpha=0.10).loc['rate', 1]
            except Exception as e:
                print(f"    OLS h={h} failed: {e}")

            # ── LP-IV (2SLS via manual first stage) ───────────────────────
            if F_stat >= 5:
                try:
                    # First stage
                    X_fs2 = pd.concat([ctrl_vals, z_vals], axis=1)
                    fs2   = sm.OLS(rate_vals, X_fs2).fit()
                    rate_hat = fs2.fittedvalues

                    # Second stage
                    X_ss = pd.concat([ctrl_vals,
                                      pd.Series(rate_hat, index=ctrl_vals.index, name='rate')],
                                     axis=1)
                    ss   = sm.OLS(y_vals, X_ss).fit(cov_type='HC1')
                    lp_iv[h] = ss.params['rate']

                    # Bootstrap CI for IV
                    boot_iv = []
                    n_obs   = len(df_h)
                    for _ in range(n_boot):
                        idx = np.random.randint(0, n_obs, n_obs)
                        df_b = df_h.iloc[idx].copy()
                        y_b, rate_b, z_b = (df_b.iloc[:,0],
                                             df_b['rate'], df_b['z_q'])
                        ctrl_b = df_b[ctrl_cols]
                        try:
                            X_fs_b  = pd.concat([ctrl_b, z_b], axis=1)
                            fs_b    = sm.OLS(rate_b, X_fs_b).fit()
                            rate_hb = fs_b.fittedvalues
                            X_ss_b  = pd.concat([ctrl_b,
                                                 pd.Series(rate_hb, index=ctrl_b.index, name='rate')],
                                                axis=1)
                            ss_b = sm.OLS(y_b, X_ss_b).fit()
                            boot_iv.append(ss_b.params['rate'])
                        except Exception:
                            pass
                    if len(boot_iv) > 10:
                        ci_lo_iv[h] = np.percentile(boot_iv, 5)
                        ci_hi_iv[h] = np.percentile(boot_iv, 95)
                except Exception as e:
                    print(f"    IV h={h} failed: {e}")

        # Report point estimates
        valid_h = [h for h in range(horizon) if not np.isnan(lp_ols[h])]
        print(f"    Horizons with estimates: {valid_h}")
        print(f"    LP-OLS coefficients: "
              f"{[round(float(lp_ols[h]),4) for h in valid_h]}")
        if F_stat >= 5:
            print(f"    LP-IV  coefficients: "
                  f"{[round(float(lp_iv[h]),4) if not np.isnan(lp_iv[h]) else 'NaN' for h in valid_h]}")
        if len(valid_h) >= 3:
            peak_ols = lp_ols[valid_h[np.argmin(np.abs(np.array(valid_h) - 3))]]
            print(f"    LP-OLS GDP response at h~3: {peak_ols:.4f} pp")
            print(f"    Baseline VAR(1) GDP at h=3: {BASELINE_PEAK_GDP:.4f} pp")

        results[resp_var] = {
            'lp_ols': lp_ols, 'lp_iv': lp_iv,
            'ci_lo_ols': ci_lo_ols, 'ci_hi_ols': ci_hi_ols,
            'ci_lo_iv': ci_lo_iv, 'ci_hi_iv': ci_hi_iv,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 8: ADDITIONAL DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def run_diagnostics(merged_iv, var_df):
    print()
    print("─" * 70)
    print("SECTION 8: INSTRUMENT VALIDITY DIAGNOSTICS")
    print("─" * 70)

    z_q   = merged_iv['z_q']
    u_rate = merged_iv['u_rate']

    # ── Correlation with VAR residuals (exogeneity checks) ────────────────
    print()
    print("  Correlation of z_q with VAR residuals (should be ~0 for non-rate):")

    # Re-estimate VAR to get all residuals
    var1_model  = VAR(var_df)
    var1_result = var1_model.fit(1, trend='c')
    resids = var1_result.resid

    corr_table = {}
    for col in resids.columns:
        u_col = resids[col]
        merged_check = pd.DataFrame({'z_q': z_q, 'u': u_col}).dropna()
        if len(merged_check) < 5:
            continue
        r, p = stats.pearsonr(merged_check['z_q'], merged_check['u'])
        corr_table[col] = (r, p)
        flag = " *** VIOLATION?" if (col != 'rate' and abs(r) > 0.3) else ""
        print(f"    corr(z_q, u_{col:5s}) = {r:+.4f}  (p={p:.4f}){flag}")

    print()

    # ── Hausman-type endogeneity test ─────────────────────────────────────
    # Test if OLS and IV give different results for GDP at h=3
    print("  Hausman-type test for endogeneity (GDP at h=3):")
    h = 3
    y_h = var_df['gdp'].shift(-h) - var_df['gdp']
    X_ctrl = pd.DataFrame({
        'const':   1,
        'tot_l1':  var_df['tot'].shift(1),
        'gdp_l1':  var_df['gdp'].shift(1),
        'cpi_l1':  var_df['cpi'].shift(1),
        'fx_l1':   var_df['fx'].shift(1),
        'rate_l1': var_df['rate'].shift(1),
    })
    endog = var_df['rate'].copy()
    df_h  = pd.concat([y_h, endog, z_q, X_ctrl], axis=1).dropna()

    if len(df_h) >= 20:
        ctrl_cols = [c for c in X_ctrl.columns]
        y_v = df_h.iloc[:,0]
        r_v = df_h['rate']
        z_v = df_h['z_q']
        c_v = df_h[ctrl_cols]

        # First stage residuals
        X_fs = pd.concat([c_v, z_v], axis=1)
        fs   = sm.OLS(r_v, X_fs).fit()
        v_hat = fs.resid

        # Augmented regression (Durbin-Wu-Hausman)
        X_aug = pd.concat([c_v, r_v,
                           pd.Series(v_hat, index=c_v.index, name='v_hat')], axis=1)
        aug   = sm.OLS(y_v, X_aug).fit()
        t_dw  = aug.tvalues['v_hat']
        p_dw  = aug.pvalues['v_hat']
        print(f"    DWH test: t(v_hat) = {t_dw:.4f}, p = {p_dw:.4f}")
        if p_dw < 0.10:
            print("    --> Reject exogeneity (p<0.10): rate IS endogenous. IV needed.")
        else:
            print("    --> Cannot reject exogeneity (p≥0.10): OLS may be consistent.")
    else:
        print("    Insufficient observations for DWH test.")

    return corr_table


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 9: FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure(lp_results, fs_result, merged_iv, F_stat, R2_fs, max_eig):
    print()
    print("─" * 70)
    print("SECTION 9: FIGURE")
    print("─" * 70)

    horizon = 9
    h_axis  = np.arange(horizon)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        'Option 2: FX Surprise Instrument — Proxy-SVAR Diagnostics\n'
        '(Monthly PEN/USD fallback; daily FX data not available)',
        fontsize=11, y=0.98
    )

    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # ── Panel 1: First-stage scatter ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(merged_iv['z_q'], merged_iv['u_rate'],
                s=18, color=C_DARK, alpha=0.6, zorder=3)
    # Fit line
    x_line = np.linspace(merged_iv['z_q'].min(), merged_iv['z_q'].max(), 100)
    y_line = (fs_result.params['const'] +
               fs_result.params['z_q'] * x_line)
    ax1.plot(x_line, y_line, color=C_BLACK, lw=1.5, zorder=4)
    ax1.axhline(0, color=C_LIGHT, lw=0.8)
    ax1.axvline(0, color=C_LIGHT, lw=0.8)
    ax1.set_xlabel('z_q (quarterly FX log-change)')
    ax1.set_ylabel('u_rate (VAR rate residual)')
    ax1.set_title(f'First Stage\nF={F_stat:.2f}, R²={R2_fs:.4f}', fontsize=9)

    # ── Panel 2: z_q time series ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(merged_iv.index, merged_iv['z_q'],
            width=60, color=C_MED, alpha=0.7)
    ax2.axhline(0, color=C_BLACK, lw=0.8)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('z_q')
    ax2.set_title('Quarterly FX Surprise Instrument', fontsize=9)

    # ── Panel 3: u_rate time series ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(merged_iv.index, merged_iv['u_rate'],
             color=C_DARK, lw=1.2)
    ax3.axhline(0, color=C_LIGHT, lw=0.8)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('u_rate (pp)')
    ax3.set_title('Rate Equation Residual', fontsize=9)

    # ── Panels 4-6: LP-OLS and LP-IV for GDP, CPI, FX ───────────────────
    resp_labels = {'gdp': 'GDP (%)', 'cpi': 'CPI (%)', 'fx': 'FX (%)'}
    for col_i, resp_var in enumerate(['gdp', 'cpi', 'fx']):
        ax = fig.add_subplot(gs[1, col_i])
        res = lp_results[resp_var]

        ols = res['lp_ols']
        iv  = res['lp_iv']
        ci_lo_ols = res['ci_lo_ols']
        ci_hi_ols = res['ci_hi_ols']
        ci_lo_iv  = res['ci_lo_iv']
        ci_hi_iv  = res['ci_hi_iv']

        valid = ~np.isnan(ols)
        if valid.any():
            ax.fill_between(h_axis[valid], ci_lo_ols[valid], ci_hi_ols[valid],
                            alpha=0.25, color=C_BAND1, label='OLS 90% CI')
            ax.plot(h_axis[valid], ols[valid], color=C_DARK, lw=1.8,
                    label='LP-OLS')

        iv_valid = ~np.isnan(iv)
        if iv_valid.any() and F_stat >= 5:
            ax.fill_between(h_axis[iv_valid], ci_lo_iv[iv_valid], ci_hi_iv[iv_valid],
                            alpha=0.20, color='#aaaaff', label='IV 90% CI')
            ax.plot(h_axis[iv_valid], iv[iv_valid], color='#0000cc', lw=1.5,
                    ls='--', label='LP-IV')

        ax.axhline(0, color=C_LIGHT, lw=0.8)
        ax.set_xlabel('Quarters after shock')
        ax.set_ylabel(resp_labels[resp_var])
        ax.set_title(f'LP Response: {resp_var.upper()}', fontsize=9)
        if col_i == 0:
            ax.legend(fontsize=7, loc='lower left')

    # ── Panel 7-9: Correlation of z_q with all residuals ─────────────────
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis('off')

    textlines = [
        "DATA AVAILABILITY FINDING:",
        "  Daily PEN/USD (PD04637PD, PD04639PD): NOT FOUND in Nexus data store.",
        "  EMBIG / sovereign spread: NOT FOUND in Nexus data store.",
        "  Fallback: monthly FX log-change in BCRP announcement months.",
        "",
        f"INSTRUMENT QUALITY:  F = {F_stat:.2f}  |  R² = {R2_fs:.4f}  |  Max VAR eigenvalue = {max_eig:.4f}",
        f"{'STRONG INSTRUMENT (F>10)' if F_stat>10 else 'WEAK INSTRUMENT (F<10) — results are illustrative only'}",
        "",
        "NOTE: With only monthly data, the FX surprise is measured over a full month,",
        "not a narrow ±1-day window. This introduces noise from non-policy factors.",
        "True HF identification requires daily PEN/USD from BCRP estadisticas portal.",
    ]

    y_pos = 0.95
    for line in textlines:
        weight = 'bold' if line.startswith('DATA') or line.startswith('INSTRUMENT') else 'normal'
        fontsize = 9 if not line.startswith('  ') else 8
        ax_text.text(0.01, y_pos, line, transform=ax_text.transAxes,
                     fontsize=fontsize, verticalalignment='top',
                     fontweight=weight, color=C_BLACK,
                     fontfamily='monospace')
        y_pos -= 0.09

    out_path = OUT_DIR / 'option2_fx_hf.pdf'
    fig.savefig(out_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure: {out_path}")
    return str(out_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Section 1: Data audit ────────────────────────────────────────────
    findings = audit_data_availability()

    if not findings['monthly_fx']:
        print("FATAL: Monthly FX data (PN01246PM) not found. Cannot proceed.")
        return

    # ── Section 2: BCRP announcement calendar ────────────────────────────
    ann_dates, ann_months_df = build_bcrp_calendar(2004, 2025)

    # ── Section 3: Build monthly FX surprise instrument ───────────────────
    fx_monthly = build_monthly_fx_instrument(
        findings['monthly_fx_data'], ann_months_df
    )

    # ── Section 4: Aggregate to quarterly ────────────────────────────────
    z_quarterly = aggregate_to_quarterly(fx_monthly)

    # ── Section 5: Load VAR data ──────────────────────────────────────────
    var_df, q_full = load_var_data()

    # ── Section 6: Proxy-SVAR diagnostics ────────────────────────────────
    fs_result, merged_iv, max_eig = run_proxy_svar(var_df, z_quarterly)

    if fs_result is None:
        print("Cannot proceed: no overlapping observations between instrument and VAR.")
        return

    F_stat = fs_result.fvalue
    R2_fs  = fs_result.rsquared

    # ── Section 7: Local projections ─────────────────────────────────────
    lp_results = run_lp(var_df, merged_iv, F_stat)

    # ── Section 8: Diagnostics ────────────────────────────────────────────
    corr_table = run_diagnostics(merged_iv, var_df)

    # ── Section 9: Figure ────────────────────────────────────────────────
    fig_path = make_figure(lp_results, fs_result, merged_iv,
                           F_stat, R2_fs, max_eig)

    # ── Final Summary ─────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("OPTION 2 FINAL SUMMARY")
    print("=" * 70)
    print()
    print("DATA AVAILABILITY:")
    print("  Daily PEN/USD (PD04637PD / PD04639PD): NOT AVAILABLE")
    print("  EMBIG / sovereign spread:               NOT AVAILABLE")
    print("  Fallback used: monthly FX log-change in announcement months")
    print()
    print("BCRP ANNOUNCEMENT CALENDAR:")
    print(f"  {len(ann_dates)} dates constructed (first Thursday of each month, 2004-2025)")
    print()
    print("INSTRUMENT CONSTRUCTION:")
    print(f"  z_m = ln(FX_m) - ln(FX_{{m-1}}) for announcement months")
    print(f"  z_q = sum of z_m within each quarter")
    n_z = z_quarterly['z_q'].notna().sum()
    print(f"  Quarterly observations: {n_z}")
    print(f"  z_q mean: {z_quarterly['z_q'].mean():.6f}")
    print(f"  z_q std:  {z_quarterly['z_q'].std():.6f}")
    print()
    print("PROXY-SVAR FIRST STAGE:")
    print(f"  F-statistic:        {F_stat:.4f}")
    print(f"  R-squared:          {R2_fs:.6f}")
    print(f"  Coefficient (z_q):  {fs_result.params['z_q']:.6f}")
    print(f"  t-statistic:        {fs_result.tvalues['z_q']:.4f}")
    print(f"  p-value:            {fs_result.pvalues['z_q']:.6f}")
    print(f"  Instrument strength: {'STRONG (F>10)' if F_stat>10 else 'WEAK (F<10)'}")
    print()
    print("LP-OLS GDP RESPONSE (baseline comparison):")
    ols_gdp = lp_results['gdp']['lp_ols']
    iv_gdp  = lp_results['gdp']['lp_iv']
    for h in range(9):
        iv_str = f"{iv_gdp[h]:.4f}" if not np.isnan(iv_gdp[h]) else "n/a"
        ols_str = f"{ols_gdp[h]:.4f}" if not np.isnan(ols_gdp[h]) else "n/a"
        print(f"  h={h}: OLS={ols_str}  IV={iv_str}")
    print()
    print(f"  Baseline VAR(1) peak GDP (h=3): {BASELINE_PEAK_GDP:.4f} pp")
    print()
    print("VALIDITY ASSESSMENT:")
    if F_stat > 10:
        print("  F > 10: Instrument passes relevance threshold.")
        print("  However, note that monthly FX changes do NOT isolate")
        print("  policy surprises — they also capture global risk sentiment,")
        print("  commodity price movements, and political events.")
        print("  The instrument is relevant but excludability is questionable.")
    elif F_stat > 5:
        print("  5 < F < 10: Weak instrument. 2SLS estimates are biased.")
        print("  Results are indicative only.")
    else:
        print("  F < 5: Very weak instrument. Monthly FX does not identify")
        print("  monetary policy shocks. Daily HF data is required.")
    print()
    print("WHAT IS NEEDED FOR A VALID HF INSTRUMENT:")
    print("  1. Daily PEN/USD from BCRP (series PD04637PD or PD04639PD),")
    print("     available at estadisticas.bcrp.gob.pe at daily frequency.")
    print("  2. Narrow event window: z_t = ln(FX_{d+1}) - ln(FX_{d-1})")
    print("     where d = announcement date. This isolates the ±1-day response.")
    print("  3. The 2-day window: z2_t = ln(FX_{d+2}) - ln(FX_{d-2})")
    print("     captures slower incorporation of policy surprises.")
    print("  4. With daily data, typical first-stage F statistics for Peru")
    print("     HF instruments range from 8-15 in the literature.")
    print()
    print(f"Figure saved: {fig_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
