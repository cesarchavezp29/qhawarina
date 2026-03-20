#!/usr/bin/env python3
"""
run_option1_taylor.py — Option 1: Taylor-Rule Residual as Monetary Policy Shock Instrument
Peru Monetary Policy Paper — Proxy-SVAR with Taylor-rule residuals
"""
import sys, io, warnings, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.filters.hp_filter import hpfilter
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT    = Path('D:/Nexus/nexus')
PANEL   = ROOT / 'data/processed/national/panel_national_monthly.parquet'
OUT_DIR = ROOT / 'paper/output/robustness'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Style ──────────────────────────────────────────────────────────────────
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

# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING  (copied from run_robustness.py lines 91-173)
# ════════════════════════════════════════════════════════════════════════════

def load_var_data():
    """
    Build quarterly VAR dataset from monthly panel.
    Returns DataFrame with columns [tot, gdp, cpi, fx, rate] and DatetimeIndex.
    Also returns rate_level (not differenced).
    """
    raw = pd.read_parquet(PANEL)
    raw['date'] = pd.to_datetime(raw['date'])

    series_map = {
        'PD04722MM': 'rate_raw',   # level
        'PN01731AM': 'gdp_m',      # monthly % change SA
        'PN01271PM': 'cpi_m',      # monthly % change SA
        'PN01246PM': 'fx_m',       # PEN/USD level
        'PN38923BM': 'tot_m',      # ToT index
    }

    # Extract each series
    frames = []
    for sid, col in series_map.items():
        s = raw[raw['series_id'] == sid][['date', 'value_raw']].copy()
        s = s.rename(columns={'value_raw': col})
        s = s.set_index('date').sort_index()
        frames.append(s)

    monthly = frames[0].join(frames[1:], how='outer')
    monthly = monthly.sort_index()

    # ── Quarterly aggregation ──────────────────────────────────────────────
    q = pd.DataFrame()
    # rate: quarterly mean of level, then diff
    q['rate_level'] = monthly['rate_raw'].resample('QE').mean()
    q['d_rate']     = q['rate_level'].diff()

    # gdp: sum of monthly % changes
    q['gdp'] = monthly['gdp_m'].resample('QE').sum()

    # cpi: sum of monthly % changes
    q['cpi'] = monthly['cpi_m'].resample('QE').sum()

    # fx: quarterly mean, pct_change*100
    q['fx'] = monthly['fx_m'].resample('QE').mean().pct_change() * 100

    # tot: quarterly mean, pct_change*100
    q['tot'] = monthly['tot_m'].resample('QE').mean().pct_change() * 100

    # Drop first row (NaN from diff/pct_change)
    q = q.dropna()

    # Keep 2004Q2 through 2025Q3
    q = q.loc['2004-04-01':'2025-09-30']

    # ── FWL COVID treatment ───────────────────────────────────────────────
    covid_dummy = pd.Series(0, index=q.index, name='covid')
    covid_quarters = [pd.Timestamp('2020-03-31'), pd.Timestamp('2020-06-30')]
    for cq in covid_quarters:
        if cq in covid_dummy.index:
            covid_dummy.loc[cq] = 1

    def fwl_partial(series, dummy):
        X = sm.add_constant(dummy)
        res = sm.OLS(series, X).fit()
        return series - res.fittedvalues + series.mean()

    cols_to_fwl = ['d_rate', 'gdp', 'cpi', 'fx', 'tot', 'rate_level']
    for col in cols_to_fwl:
        if col in q.columns:
            q[col] = fwl_partial(q[col], covid_dummy)

    # Build final VAR DataFrame: [tot, gdp, cpi, fx, d_rate]
    var_df = q[['tot', 'gdp', 'cpi', 'fx', 'd_rate']].copy()
    var_df.columns = ['tot', 'gdp', 'cpi', 'fx', 'rate']

    t0 = var_df.index[0]
    t1 = var_df.index[-1]
    print(f"VAR data: {t0.year}Q{t0.quarter} to {t1.year}Q{t1.quarter}, T={len(var_df)}")
    print(f"  Columns: {list(var_df.columns)}")
    print(f"  Means:   {var_df.mean().round(3).to_dict()}")

    # Also return rate_level separately
    rate_level = q['rate_level'].copy()

    return var_df, rate_level, covid_dummy, q


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("OPTION 1: TAYLOR-RULE RESIDUAL AS MONETARY POLICY SHOCK")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    var_df, rate_level, covid_dummy, q_full = load_var_data()
    T = len(var_df)
    print(f"T = {T} quarters")

    # ── Check for survey inflation expectations series ────────────────────
    print("\n" + "─"*60)
    print("STEP 0: Inflation Expectations")
    raw = pd.read_parquet(PANEL)
    raw['date'] = pd.to_datetime(raw['date'])
    survey_series_id = 'PN01270PM'
    survey_rows = raw[raw['series_id'] == survey_series_id]
    survey_found = len(survey_rows) > 0
    print(f"  Survey series '{survey_series_id}' found: {survey_found}")

    # ─── Build inflation expectations ─────────────────────────────────────
    # Use quarterly CPI from q_full (already FWL-partialled)
    cpi_q = q_full['cpi'].copy()  # quarterly % change

    if survey_found:
        print(f"  Using survey series (rows: {len(survey_rows)})")
        # Monthly survey -> quarterly mean
        survey_m = survey_rows[['date', 'value_raw']].set_index('date').sort_index()
        survey_m = survey_m.rename(columns={'value_raw': 'infl_exp'})
        infl_exp_q = survey_m['infl_exp'].resample('QE').mean()
        infl_exp_q = infl_exp_q.reindex(var_df.index)
    else:
        print(f"  Using adaptive expectation: rolling 4-quarter mean of lagged CPI")
        # E_t[pi_{t+4}] = rolling mean of pi_{t-3}..pi_t  (4-quarter backward window)
        infl_exp_q = cpi_q.rolling(window=4, min_periods=4).mean()
        infl_exp_q = infl_exp_q.reindex(var_df.index)

    print(f"  Inflation expectations: {infl_exp_q.dropna().shape[0]} non-NaN obs")
    print(f"  Mean E[pi]: {infl_exp_q.mean():.4f}")

    # ─── STEP 1: HP Output Gap ────────────────────────────────────────────
    print("\n" + "─"*60)
    print("STEP 1: HP Output Gap")

    # Reconstruct log-level of GDP from QoQ % changes
    gdp_growth = q_full['gdp'].copy()   # quarterly % change (sum of monthly %)
    # Cumulative sum gives log-level (approximately, since % changes are ~dlog)
    gdp_log = gdp_growth.cumsum()

    # HP filter with lambda=1600 (standard for quarterly data)
    gdp_cycle, gdp_trend = hpfilter(gdp_log.values, lamb=1600)
    output_gap = pd.Series(gdp_cycle, index=gdp_log.index, name='output_gap')

    print(f"  GDP log-level range: [{gdp_log.min():.4f}, {gdp_log.max():.4f}]")
    print(f"  Output gap std: {output_gap.std():.4f}")
    print(f"  Output gap range: [{output_gap.min():.4f}, {output_gap.max():.4f}]")

    # Align with var_df
    output_gap_aligned = output_gap.reindex(var_df.index)

    # ─── STEP 2: Forward-looking Taylor Rule OLS ──────────────────────────
    print("\n" + "─"*60)
    print("STEP 2: Forward-Looking Taylor Rule OLS")

    # Build Taylor rule regressors (raw, before FWL on covid)
    # delta_rate_t = alpha + beta1*(cpi_t - 2) + beta2*(E[pi_{t+4}] - 2)
    #               + beta3*output_gap_t + beta4*rate_level_{t-1} + rho*delta_rate_{t-1} + eps_t

    d_rate    = var_df['rate'].values          # FWL-partialled delta_rate
    cpi_dev   = (var_df['cpi'] - 2.0).values  # CPI deviation from 2% target
    rate_lvl  = rate_level.reindex(var_df.index).values  # rate level (FWL-partialled)

    # Lag vectors
    d_rate_lag = np.roll(d_rate, 1);   d_rate_lag[0] = np.nan
    rate_lvl_lag = np.roll(rate_lvl, 1); rate_lvl_lag[0] = np.nan

    # Inflation expectations deviation
    infl_exp_aligned = infl_exp_q.reindex(var_df.index)
    infl_exp_dev = (infl_exp_aligned - 2.0).values

    # Output gap
    ogap = output_gap_aligned.values

    # Stack regressors
    N = len(d_rate)
    data_matrix = pd.DataFrame({
        'd_rate':       d_rate,
        'cpi_dev':      cpi_dev,
        'infl_exp_dev': infl_exp_dev,
        'ogap':         ogap,
        'rate_lvl_lag': rate_lvl_lag,
        'd_rate_lag':   d_rate_lag,
    }, index=var_df.index)

    # Drop NaNs
    data_matrix = data_matrix.dropna()
    print(f"  Taylor regression obs after dropping NaN: {len(data_matrix)}")

    # FWL-partial out COVID dummies (replicate baseline treatment)
    # COVID quarters: 2020Q1 (2020-03-31) and 2020Q2 (2020-06-30)
    covid_col = covid_dummy.reindex(data_matrix.index).fillna(0)

    def fwl_col(series, dummy):
        X = sm.add_constant(dummy.values)
        res = sm.OLS(series.values, X).fit()
        return series.values - res.fittedvalues + series.mean()

    for col in data_matrix.columns:
        data_matrix[col] = fwl_col(data_matrix[col], covid_col)

    Y_taylor = data_matrix['d_rate'].values
    X_taylor = sm.add_constant(data_matrix[['cpi_dev', 'infl_exp_dev', 'ogap',
                                            'rate_lvl_lag', 'd_rate_lag']].values)
    col_names_taylor = ['const', 'beta1_cpi', 'beta2_infl_exp', 'beta3_ogap',
                        'beta4_rate_lag', 'rho_drate_lag']

    res_taylor = sm.OLS(Y_taylor, X_taylor).fit(cov_type='HC3')

    print("\n  Taylor Rule OLS Results (HC3 SEs):")
    print(f"  {'Param':<18} {'Coeff':>10} {'SE':>10} {'t-stat':>10} {'p-val':>10}")
    print(f"  {'-'*58}")
    for name, coef, se, tstat, pval in zip(
            col_names_taylor,
            res_taylor.params,
            res_taylor.bse,
            res_taylor.tvalues,
            res_taylor.pvalues):
        sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"  {name:<18} {coef:>10.4f} {se:>10.4f} {tstat:>10.4f} {pval:>10.4f} {sig}")
    print(f"\n  R² = {res_taylor.rsquared:.4f}")
    print(f"  Adj. R² = {res_taylor.rsquared_adj:.4f}")
    print(f"  N = {len(Y_taylor)}")

    # ─── STEP 3: Save Taylor residuals ────────────────────────────────────
    print("\n" + "─"*60)
    print("STEP 3: Taylor-Rule Residuals (Monetary Policy Shocks)")

    eps_hat = pd.Series(res_taylor.resid, index=data_matrix.index, name='taylor_shock')
    print(f"  eps_hat shape: {eps_hat.shape}")
    print(f"  eps_hat mean:  {eps_hat.mean():.6f}")
    print(f"  eps_hat std:   {eps_hat.std():.4f}")
    print(f"  eps_hat range: [{eps_hat.min():.4f}, {eps_hat.max():.4f}]")

    # Save to CSV
    shock_path = OUT_DIR / 'taylor_rule_shocks.csv'
    eps_hat.to_csv(shock_path, header=True)
    print(f"  Saved shocks to: {shock_path}")

    # ─── STEP 4a: Baseline VAR(1) and Cholesky rate residuals ────────────
    print("\n" + "─"*60)
    print("STEP 4a: Baseline VAR(1) Cholesky Rate Residuals")

    var1_model  = VAR(var_df)
    var1_result = var1_model.fit(1, trend='c')
    max_eig = float(max(abs(np.linalg.eigvals(var1_result.coefs[0]))))
    print(f"  VAR(1) max eigenvalue: {max_eig:.4f}")
    print(f"  VAR(1) nobs: {var1_result.nobs}")

    # VAR residuals: shape (nobs, K), order is [tot, gdp, cpi, fx, rate]
    var_resids = pd.DataFrame(
        var1_result.resid,
        columns=['tot', 'gdp', 'cpi', 'fx', 'rate'],
        index=var_df.index[1:]   # VAR(1) loses first obs
    )
    chol_rate_resid = var_resids['rate']

    print(f"  Cholesky rate residual std: {chol_rate_resid.std():.4f}")
    print(f"  Cholesky rate residual range: [{chol_rate_resid.min():.4f}, {chol_rate_resid.max():.4f}]")

    # ─── STEP 4b: First-stage F-stat ──────────────────────────────────────
    print("\n" + "─"*60)
    print("STEP 4b: First-Stage F-Statistic (Proxy Relevance)")

    # Align eps_hat with chol_rate_resid
    common_idx = chol_rate_resid.index.intersection(eps_hat.index)
    eps_aligned  = eps_hat.reindex(common_idx)
    chol_aligned = chol_rate_resid.reindex(common_idx)

    # Drop NaNs
    mask = ~(eps_aligned.isna() | chol_aligned.isna())
    eps_fs   = eps_aligned[mask].values
    chol_fs  = chol_aligned[mask].values
    n_fs     = mask.sum()
    print(f"  First-stage N: {n_fs}")

    X_fs = sm.add_constant(eps_fs)
    res_fs = sm.OLS(chol_fs, X_fs).fit(cov_type='HC3')

    # F-stat for the instrument (test that eps_hat coeff != 0)
    fstat_fs = res_fs.fvalue
    r2_fs    = res_fs.rsquared
    coef_fs  = res_fs.params[1]
    se_fs    = res_fs.bse[1]
    tstat_fs = res_fs.tvalues[1]

    print(f"  First-stage coefficient on eps_hat: {coef_fs:.4f} (SE={se_fs:.4f}, t={tstat_fs:.4f})")
    print(f"  First-stage F-stat: {fstat_fs:.4f}")
    print(f"  First-stage R²: {r2_fs:.4f}")

    if fstat_fs > 10:
        print(f"  >> F > 10: instrument is RELEVANT. Proceeding with LP-IV.")
        instrument_relevant = True
    else:
        print(f"  >> F <= 10: instrument may be WEAK. LP-IV results should be interpreted with caution.")
        instrument_relevant = True   # still run it but flag

    # ─── STEP 4c: Correlation ─────────────────────────────────────────────
    print("\n" + "─"*60)
    print("STEP 4c: Correlation between eps_hat and Cholesky Rate Shock")

    corr = np.corrcoef(eps_fs, chol_fs)[0, 1]
    print(f"  Correlation(eps_hat, Cholesky rate residual) = {corr:.4f}")

    # ─── STEP 4d: LP-IV (2SLS) ────────────────────────────────────────────
    print("\n" + "─"*60)
    print("STEP 4d: LP-IV with Taylor Residual as Instrument")
    print("  Specification: y_{t+h} - y_{t-1} = alpha_h + beta_h*(delta_rate [IV: eps_hat])")
    print("                  + gamma1*y_{t-1} + gamma2*y_{t-2}")
    print("                  + delta1*delta_rate_{t-1} + delta2*delta_rate_{t-2}")
    print("                  + lambda*covid_t + e_{t,h}")

    lp_gdp   = var_df['gdp'].values
    lp_rate  = var_df['rate'].values
    cov_arr  = covid_dummy.reindex(var_df.index).fillna(0).values
    N_lp     = len(lp_gdp)

    # Build eps_hat aligned to var_df index
    eps_lp = eps_hat.reindex(var_df.index)

    max_lag = 2
    H = 9
    beta_iv  = np.zeros(H)
    se_iv    = np.zeros(H)
    tstat_iv = np.zeros(H)

    for h in range(H):
        rows_dep  = []
        rows_X    = []
        rows_Z    = []   # instruments
        rows_Xend = []   # endogenous regressor

        for t in range(max_lag, N_lp - h):
            # dependent variable: cumulative GDP change
            dep = lp_gdp[t + h] - lp_gdp[t - 1]

            # exogenous controls (same as endogenous but excluding d_rate[t])
            gdp_lag1 = lp_gdp[t - 1]
            gdp_lag2 = lp_gdp[t - 2] if t >= 2 else 0.0
            rate_lag1 = lp_rate[t - 1]
            rate_lag2 = lp_rate[t - 2] if t >= 2 else 0.0
            covid_t  = cov_arr[t]

            # endogenous: d_rate[t]
            d_rate_t = lp_rate[t]

            # instrument: eps_hat[t]
            eps_t = eps_lp.iloc[t] if t < len(eps_lp) else np.nan

            if np.isnan(eps_t):
                continue

            # Exogenous regressors (no d_rate[t])
            exog_row = [1.0, gdp_lag1, gdp_lag2, rate_lag1, rate_lag2, covid_t]

            rows_dep.append(dep)
            rows_X.append(exog_row)
            rows_Xend.append(d_rate_t)
            rows_Z.append(eps_t)

        if len(rows_dep) < 10:
            print(f"  h={h}: insufficient obs ({len(rows_dep)}), skipping")
            beta_iv[h] = np.nan
            se_iv[h]   = np.nan
            continue

        Y_h    = np.array(rows_dep)
        X_exog = np.array(rows_X)      # (n, 6)
        D_h    = np.array(rows_Xend)   # endogenous regressor
        Z_h    = np.array(rows_Z)      # instrument

        n_h = len(Y_h)

        # Manual 2SLS:
        # First stage: project D on [exog, Z]
        X_fs_h = np.column_stack([X_exog, Z_h])   # (n, 7)
        res_first = sm.OLS(D_h, X_fs_h).fit()
        D_hat = res_first.fittedvalues             # predicted d_rate

        # Second stage: regress Y on [exog, D_hat] with HC3 SEs
        # But for valid 2SLS SEs we need to use the original D, not D_hat
        X_2sls_2nd = np.column_stack([X_exog, D_hat])  # (n, 7)
        X_2sls_orig = np.column_stack([X_exog, D_h])

        # 2SLS estimator: beta = (X_hat'X)^{-1} X_hat'Y
        # where X_hat = [exog, D_hat], X = [exog, D]
        try:
            XhTXh = X_2sls_2nd.T @ X_2sls_orig
            XhTY  = X_2sls_2nd.T @ Y_h
            beta_2sls = np.linalg.solve(XhTXh, XhTY)

            # Residuals using ORIGINAL D
            resid_2sls = Y_h - X_2sls_orig @ beta_2sls

            # HC3 variance: sandwich with X_hat
            # V = (X_hat'X)^{-1} (X_hat' diag(e^2/(1-h_ii)^2) X_hat) (X'X_hat)^{-1}
            # Simplified HC3: use hat values from X_hat
            H_diag = np.diag(X_2sls_2nd @ np.linalg.pinv(X_2sls_2nd.T @ X_2sls_2nd) @ X_2sls_2nd.T)
            H_diag = np.clip(H_diag, 0, 0.999)
            e_hc3  = resid_2sls / (1 - H_diag)

            meat = X_2sls_2nd.T @ np.diag(e_hc3**2) @ X_2sls_2nd
            bread_inv = np.linalg.pinv(XhTXh)
            V_2sls = bread_inv @ meat @ bread_inv.T

            beta_iv[h]  = beta_2sls[-1]   # coefficient on d_rate (last column)
            se_iv[h]    = np.sqrt(np.abs(V_2sls[-1, -1]))
            tstat_iv[h] = beta_2sls[-1] / se_iv[h] if se_iv[h] > 0 else np.nan

        except np.linalg.LinAlgError as e_err:
            print(f"  h={h}: linear algebra error: {e_err}")
            beta_iv[h] = np.nan
            se_iv[h]   = np.nan
            tstat_iv[h] = np.nan

    print(f"\n  LP-IV Results (HC3 SEs, 2SLS):")
    print(f"  {'h':>4} {'beta_h':>10} {'SE':>10} {'t-stat':>10} {'90% CI Lo':>12} {'90% CI Hi':>12}")
    print(f"  {'-'*62}")
    for h in range(H):
        ci_lo = beta_iv[h] - 1.645 * se_iv[h]
        ci_hi = beta_iv[h] + 1.645 * se_iv[h]
        sig   = ('*' if abs(tstat_iv[h]) > 1.645 else '') if not np.isnan(tstat_iv[h]) else ''
        print(f"  {h:>4} {beta_iv[h]:>10.4f} {se_iv[h]:>10.4f} {tstat_iv[h]:>10.4f} "
              f"{ci_lo:>12.4f} {ci_hi:>12.4f} {sig}")

    peak_h   = int(np.nanargmin(beta_iv))
    peak_val = float(np.nanmin(beta_iv))
    print(f"\n  LP-IV peak GDP response: {peak_val:.4f} pp at h={peak_h}")

    # ─── STEP 5: Figure ───────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("STEP 5: Saving Figure")

    h_arr = np.arange(H)
    ci_lo_plot = beta_iv - 1.645 * se_iv
    ci_hi_plot = beta_iv + 1.645 * se_iv

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    # Panel A: Taylor Rule Residuals over time
    ax0 = axes[0]
    ax0.bar(range(len(eps_hat)), eps_hat.values, color=C_DARK, alpha=0.7, width=0.8)
    ax0.axhline(0, color=C_LIGHT, lw=0.8)
    # Label x-ticks by year
    tick_idx = []
    tick_lbl = []
    for i, ts in enumerate(eps_hat.index):
        if ts.quarter == 1:
            tick_idx.append(i)
            tick_lbl.append(str(ts.year))
    ax0.set_xticks(tick_idx[::2])
    ax0.set_xticklabels(tick_lbl[::2], rotation=45, fontsize=8)
    ax0.set_title('Taylor-Rule Residuals\n(Monetary Policy Shocks)')
    ax0.set_ylabel('Shock (pp)')
    ax0.set_xlabel('Quarter')

    # Panel B: LP-IV impulse response
    ax1 = axes[1]
    ax1.fill_between(h_arr, ci_lo_plot, ci_hi_plot, alpha=0.3, color=C_BAND1,
                     label='90% CI (HC3)')
    ax1.plot(h_arr, beta_iv, color=C_DARK, lw=2.0, label='LP-IV')
    ax1.axhline(0, color=C_LIGHT, lw=0.8)
    ax1.set_title('LP-IV: GDP Response\nto 100bp Rate Shock')
    ax1.set_xlabel('Quarters after shock')
    ax1.set_ylabel('GDP cumulative response (pp)')
    ax1.legend(fontsize=8)

    # Panel C: Proxy relevance scatter
    ax2 = axes[2]
    ax2.scatter(eps_fs, chol_fs, color=C_DARK, alpha=0.6, s=18)
    # Fit line
    xr = np.linspace(eps_fs.min(), eps_fs.max(), 100)
    yr = res_fs.params[0] + res_fs.params[1] * xr
    ax2.plot(xr, yr, color='#cc0000', lw=1.5, label=f'Slope={coef_fs:.3f}')
    ax2.set_title(f'Proxy Relevance\nF={fstat_fs:.2f}, R²={r2_fs:.3f}, corr={corr:.3f}')
    ax2.set_xlabel('Taylor-Rule Residual (eps_hat)')
    ax2.set_ylabel('Cholesky Rate Shock (VAR residual)')
    ax2.legend(fontsize=8)

    fig.suptitle('Option 1: Taylor-Rule Residual as MP Shock — Proxy SVAR', fontsize=11, y=1.01)
    plt.tight_layout()

    out_path = OUT_DIR / 'option1_taylor_lp.pdf'
    fig.savefig(out_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure to: {out_path}")

    # ─── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)
    print(f"\n  Survey inflation expectations ('PN01270PM') found: {survey_found}")
    print(f"  Adaptive expectation used: rolling 4-quarter mean of lagged CPI")
    print(f"\n  Taylor Rule OLS:")
    for name, coef, se, tstat, pval in zip(
            col_names_taylor,
            res_taylor.params, res_taylor.bse,
            res_taylor.tvalues, res_taylor.pvalues):
        print(f"    {name:<18} coef={coef:+.4f}  SE={se:.4f}  t={tstat:+.4f}  p={pval:.4f}")
    print(f"    R² = {res_taylor.rsquared:.4f}  Adj.R² = {res_taylor.rsquared_adj:.4f}  N={len(Y_taylor)}")

    print(f"\n  Taylor shock stats:")
    print(f"    mean={eps_hat.mean():.6f}  std={eps_hat.std():.4f}  "
          f"min={eps_hat.min():.4f}  max={eps_hat.max():.4f}")

    print(f"\n  Proxy-SVAR Diagnostics:")
    print(f"    First-stage F-stat:        {fstat_fs:.4f}")
    print(f"    First-stage R²:            {r2_fs:.4f}")
    print(f"    Correlation(eps, Chol):    {corr:.4f}")
    print(f"    Instrument relevant (F>10): {fstat_fs > 10}")

    print(f"\n  LP-IV (h=0..8):")
    for h in range(H):
        print(f"    h={h}: beta={beta_iv[h]:+.4f}  SE={se_iv[h]:.4f}  t={tstat_iv[h]:+.4f}")
    print(f"\n  LP-IV Peak: {peak_val:.4f} pp at h={peak_h}")
    print(f"  Output figure: {out_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
