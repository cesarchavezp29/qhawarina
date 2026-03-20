#!/usr/bin/env python3
"""
AUDIT SCRIPT: BVAR, Taylor-rule IV, ToT IV for poverty equation.
All numerical results saved to text files. No rounding before saving.
"""
import sys, io, warnings, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from scipy import stats
from pathlib import Path

ROOT    = Path('D:/Nexus/nexus')
PANEL   = ROOT / 'data/processed/national/panel_national_monthly.parquet'
OUT_DIR = ROOT / 'paper/output/robustness'
AUDIT   = Path('/home/claude/audit')
AUDIT.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("AUDIT SCRIPT: BVAR + Taylor-rule + ToT IV")
print("=" * 70)

# ─── POVERTY DATA (same as all other scripts) ────────────────────────────────
POV_DATA = [
    (2005, 6.282, -2.7), (2006, 7.555, -5.0), (2007, 8.470, -6.3),
    (2008, 9.185, -5.0), (2009, 1.123, -3.4), (2010, 8.283, -4.5),
    (2011, 6.380, -3.5), (2012, 6.145, -2.3), (2013, 5.827, -1.7),
    (2014, 2.453, -0.4), (2015, 3.223, -1.7), (2016, 3.975, -1.1),
    (2017, 2.515,  0.0), (2018, 3.957, -1.6), (2019, 2.250, -0.6),
    (2022, 2.857,  1.5), (2023, -0.345, 1.5), (2024, 3.473, -2.1),
]
POV_YEARS = [r[0] for r in POV_DATA]
POV_GDP   = np.array([r[1] for r in POV_DATA])
POV_DPOV  = np.array([r[2] for r in POV_DATA])

# ─── DATA LOADING (identical to all P1-P6 scripts) ───────────────────────────
def load_var_data():
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
    monthly = frames[0].join(frames[1:], how='outer').sort_index()
    q = pd.DataFrame()
    q['rate_level'] = monthly['rate_raw'].resample('QE').mean()
    q['d_rate']     = q['rate_level'].diff()
    q['gdp']        = monthly['gdp_m'].resample('QE').sum()
    q['cpi']        = monthly['cpi_m'].resample('QE').sum()
    q['fx']         = monthly['fx_m'].resample('QE').mean().pct_change() * 100
    q['tot']        = monthly['tot_m'].resample('QE').mean().pct_change() * 100
    q = q.dropna()
    q = q.loc['2004-04-01':'2025-09-30']
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
    print(f"VAR data loaded: T={len(var_df)}")
    return var_df, q['rate_level'].copy(), covid_dummy, q

var_df, rate_level, covid_dummy, q_full = load_var_data()

# Fit baseline VAR(1)
var_model  = VAR(var_df)
var_result = var_model.fit(1, trend='c')
A1_ols     = var_result.coefs[0]          # (5,5)
Sigma_ols  = np.array(var_result.sigma_u)  # (5,5)
T          = var_result.nobs               # effective obs
K          = 5

print(f"Baseline VAR(1): T={T}, K={K}")
print(f"Max eigenvalue of A1: {max(abs(np.linalg.eigvals(A1_ols))):.4f}")

# Baseline Cholesky peak for comparison
P_ols = np.linalg.cholesky(Sigma_ols)
norm_ols = P_ols[4, 4]
irfs_baseline = []
Ah = np.eye(K)
for h in range(9):
    irfs_baseline.append((Ah @ P_ols[:, 4])[1] / norm_ols)
    Ah = Ah @ A1_ols
irfs_baseline = np.array(irfs_baseline)
freq_chol_peak = irfs_baseline.min()
freq_chol_h    = int(np.argmin(irfs_baseline))
print(f"Frequentist Cholesky peak: {freq_chol_peak:.6f} at h={freq_chol_h}")


# ============================================================
print("\n" + "=" * 70)
print("STEP 1: BVAR WITH MINNESOTA PRIOR")
print("=" * 70)
# ============================================================

# --- Prepare design matrices for Bayesian estimation ---
# VAR(1) with constant: Y = X @ B + E
# Y: (T, K),  X: (T, K+1) = [Y_{t-1}, 1]

Y = var_df.values[1:]          # (T-1, K)  response
X_lag = var_df.values[:-1]     # (T-1, K)  lagged regressors
ones  = np.ones((Y.shape[0], 1))
X     = np.hstack([X_lag, ones])  # (T-1, K+1)
T_eff = Y.shape[0]
print(f"Bayesian estimation: T_eff={T_eff}, K={K}")

# OLS posterior (just for reference)
B_ols_full = np.linalg.lstsq(X, Y, rcond=None)[0]  # (K+1, K)

# Minnesota prior specification
# Prior on vec(B): B ~ N(B_0, V_0)
# B_0: diagonal AR(1) with 0 on diagonals (stationary prior)
# V_0: Minnesota structure - diagonal for simplicity (independence assumption)

lambda1 = 0.2   # overall tightness
lambda2 = 0.5   # cross-variable shrinkage

# Prior mean: all coefficients = 0 (white noise / stationary prior)
B0 = np.zeros((K + 1, K))  # (K+1, K): [K lag coefficients; 1 constant]

# Prior variance for each coefficient b_{ij}:
# Own lag: (lambda1 / sigma_i)^2
# Cross lag: (lambda1 * lambda2 / sigma_j)^2
# Constant: large (diffuse)
sigma_sq = np.diag(Sigma_ols)  # residual variances from OLS

V0_diag = np.zeros(K * (K + 1))   # flattened diagonal of prior covariance
# Convention: B is (K+1, K) stacked as [lag coefficients (K rows); constant (1 row)]
# We build V0 as a block-diagonal-like structure
# Rows 0..K-1 = lag coefficients, row K = constant

# For equation i (column i of B), regressor j (row j of X, j<K = lagged):
#   own lag (j==i): var = (lambda1 * sigma_i)^2 / sigma_i = lambda1^2 * sigma_i
#   cross lag (j!=i): var = (lambda1 * lambda2)^2 * sigma_i / sigma_j
#   constant: var = 1e6 (diffuse)

# Build full prior variance matrix (block diagonal over equations)
# Using standard Minnesota: each equation independently
# For equation e, prior covariance over its (K+1) regressors:
prior_vars = []  # list of (K+1,) arrays, one per equation
for e in range(K):
    v_eq = np.zeros(K + 1)
    for j in range(K):
        if j == e:
            v_eq[j] = (lambda1 ** 2) * sigma_sq[e]   # own lag
        else:
            v_eq[j] = (lambda1 * lambda2) ** 2 * sigma_sq[e] / sigma_sq[j]  # cross
    v_eq[K] = 1e6   # diffuse constant
    prior_vars.append(v_eq)

print("Minnesota prior variances (equation 1 = GDP):")
print(f"  Own lag variance (GDP): {prior_vars[1][1]:.6f}")
print(f"  Cross lag variance (Rate->GDP): {prior_vars[1][4]:.6f}")

# Normal-Inverse-Wishart conjugate posterior
# Y = X @ B + E,  E ~ N(0, Sigma)
# Prior: vec(B|Sigma) ~ N(vec(B0), Sigma ⊗ V0_B) -- use Sigma-independent prior for simplicity
# We use equation-by-equation Normal-Wishart (independent across equations)

# For conjugate NIW posterior:
# Posterior of B|Sigma,Y: N(B_post, V_post ⊗ Sigma) where:
#   V_post^{-1} = V_0^{-1} + X'X
#   B_post = V_post @ (V_0^{-1} @ B0 + X'Y)
# Posterior of Sigma|Y: IW(S_post, nu_post) where:
#   nu_post = T_eff + K + 2
#   S_post = S0 + (Y - X @ B_post)' @ (Y - X @ B_post) + (B_post - B0)' @ V_0^{-1} @ (B_post - B0)

# Prior IW: S0 = diag(sigma_sq) (scale), nu0 = K+2 (minimally informative)
nu0 = K + 2
S0  = np.diag(sigma_sq)

# Compute equation-by-equation posterior for B (approximation: treat Sigma as known=OLS estimate)
# Then use that B_post for the joint NIW posterior

B_post_list = []
V_post_list = []
for e in range(K):
    V0_e_inv = np.diag(1.0 / prior_vars[e])
    V_post_e_inv = V0_e_inv + X.T @ X
    V_post_e = np.linalg.inv(V_post_e_inv)
    B_post_e = V_post_e @ (V0_e_inv @ B0[:, e] + X.T @ Y[:, e])
    B_post_list.append(B_post_e)
    V_post_list.append(V_post_e)

B_post = np.column_stack(B_post_list)   # (K+1, K)

# Posterior IW parameters
resid_post = Y - X @ B_post
S_post = S0 + resid_post.T @ resid_post
nu_post = nu0 + T_eff

print(f"Posterior nu: {nu_post}")
print(f"S_post diagonal (first 3): {np.diag(S_post)[:3]}")

# Draw from posterior
N_DRAWS = 5000
np.random.seed(42)

peak_draws = []
h_draws    = []
irf_draws  = []

from scipy.stats import invwishart, multivariate_normal

n_failed = 0
for draw in range(N_DRAWS):
    try:
        # Draw Sigma from IW posterior
        Sigma_draw = invwishart.rvs(df=nu_post, scale=S_post)

        # Draw B from Normal posterior given Sigma_draw
        # For each equation e: b_e | Sigma_draw ~ N(B_post[:,e], Sigma_draw[e,e] * V_post_e)
        A1_draw = np.zeros((K, K))
        for e in range(K):
            cov_b = Sigma_draw[e, e] * V_post_list[e]
            b_draw = multivariate_normal.rvs(mean=B_post_list[e], cov=cov_b)
            A1_draw[e, :] = b_draw[:K]  # take only lag coefficients (not constant)

        # Stability check
        eigs = abs(np.linalg.eigvals(A1_draw))
        if max(eigs) >= 1.0:
            n_failed += 1
            continue

        # Cholesky of drawn Sigma
        try:
            P_draw = np.linalg.cholesky(Sigma_draw)
        except np.linalg.LinAlgError:
            n_failed += 1
            continue

        norm_draw = P_draw[4, 4]
        if abs(norm_draw) < 1e-10:
            n_failed += 1
            continue

        # Compute IRF h=0..8
        irf_h = np.zeros(9)
        Ah_d  = np.eye(K)
        for h in range(9):
            irf_h[h] = (Ah_d @ P_draw[:, 4])[1] / norm_draw
            Ah_d = Ah_d @ A1_draw

        peak_val = irf_h.min()
        peak_h_  = int(np.argmin(irf_h))
        peak_draws.append(peak_val)
        h_draws.append(peak_h_)
        irf_draws.append(irf_h)

    except Exception as ex:
        n_failed += 1
        continue

peak_draws = np.array(peak_draws)
print(f"\nSuccessful draws: {len(peak_draws)} / {N_DRAWS} (failed: {n_failed})")

if len(peak_draws) < 100:
    print("FAILURE: Too few successful draws. Reporting failure.")
    with open(AUDIT / 'bvar_results.txt', 'w') as f:
        f.write("BVAR COMPUTATION FAILED: fewer than 100 stable posterior draws\n")
        f.write(f"Successful draws: {len(peak_draws)}\n")
        f.write(f"Frequentist Cholesky peak for comparison: {freq_chol_peak}\n")
    raise SystemExit("BVAR failed")

# Compute statistics
bvar_median = float(np.median(peak_draws))
bvar_mean   = float(np.mean(peak_draws))
bvar_68_lo  = float(np.percentile(peak_draws, 16))
bvar_68_hi  = float(np.percentile(peak_draws, 84))
bvar_90_lo  = float(np.percentile(peak_draws, 5))
bvar_90_hi  = float(np.percentile(peak_draws, 95))
bvar_95_lo  = float(np.percentile(peak_draws, 2.5))
bvar_95_hi  = float(np.percentile(peak_draws, 97.5))
bvar_median_h = int(np.median(h_draws))

print(f"\nBVAR posterior median peak: {bvar_median:.6f} at h={bvar_median_h}")
print(f"BVAR posterior mean peak:   {bvar_mean:.6f}")
print(f"BVAR 68% CI: [{bvar_68_lo:.6f}, {bvar_68_hi:.6f}]")
print(f"BVAR 90% CI: [{bvar_90_lo:.6f}, {bvar_90_hi:.6f}]")
print(f"BVAR 95% CI: [{bvar_95_lo:.6f}, {bvar_95_hi:.6f}]")

# Sensitivity: lambda1 = 0.1
print("\n--- Sensitivity: lambda1=0.1 ---")
lambda1_s = 0.1
B_post_s_list = []
V_post_s_list = []
for e in range(K):
    v0_s = np.zeros(K + 1)
    for j in range(K):
        if j == e:
            v0_s[j] = (lambda1_s ** 2) * sigma_sq[e]
        else:
            v0_s[j] = (lambda1_s * lambda2) ** 2 * sigma_sq[e] / sigma_sq[j]
    v0_s[K] = 1e6
    V0_e_inv_s = np.diag(1.0 / v0_s)
    V_post_e_s = np.linalg.inv(V0_e_inv_s + X.T @ X)
    B_post_e_s = V_post_e_s @ (V0_e_inv_s @ B0[:, e] + X.T @ Y[:, e])
    B_post_s_list.append(B_post_e_s)
    V_post_s_list.append(V_post_e_s)
B_post_s = np.column_stack(B_post_s_list)
resid_s = Y - X @ B_post_s
S_post_s = S0 + resid_s.T @ resid_s

peak_s01 = []
for draw in range(N_DRAWS):
    try:
        Sig_d = invwishart.rvs(df=nu_post, scale=S_post_s)
        A1_d  = np.zeros((K, K))
        for e in range(K):
            cov_b = Sig_d[e, e] * V_post_s_list[e]
            b_d   = multivariate_normal.rvs(mean=B_post_s_list[e], cov=cov_b)
            A1_d[e, :] = b_d[:K]
        if max(abs(np.linalg.eigvals(A1_d))) >= 1.0:
            continue
        P_d = np.linalg.cholesky(Sig_d + np.eye(K) * 1e-10)
        norm_d = P_d[4, 4]
        irf_h = np.zeros(9)
        Ah_d = np.eye(K)
        for h in range(9):
            irf_h[h] = (Ah_d @ P_d[:, 4])[1] / norm_d
            Ah_d = Ah_d @ A1_d
        peak_s01.append(irf_h.min())
    except:
        continue

bvar_s01_median = float(np.median(peak_s01)) if peak_s01 else float('nan')
bvar_s01_mean   = float(np.mean(peak_s01)) if peak_s01 else float('nan')
print(f"lambda1=0.1: median={bvar_s01_median:.6f}, mean={bvar_s01_mean:.6f}, draws={len(peak_s01)}")

# Sensitivity: lambda1 = 0.5
print("\n--- Sensitivity: lambda1=0.5 ---")
lambda1_s2 = 0.5
B_post_s2_list = []
V_post_s2_list = []
for e in range(K):
    v0_s2 = np.zeros(K + 1)
    for j in range(K):
        if j == e:
            v0_s2[j] = (lambda1_s2 ** 2) * sigma_sq[e]
        else:
            v0_s2[j] = (lambda1_s2 * lambda2) ** 2 * sigma_sq[e] / sigma_sq[j]
    v0_s2[K] = 1e6
    V0_e_inv_s2 = np.diag(1.0 / v0_s2)
    V_post_e_s2 = np.linalg.inv(V0_e_inv_s2 + X.T @ X)
    B_post_e_s2 = V_post_e_s2 @ (V0_e_inv_s2 @ B0[:, e] + X.T @ Y[:, e])
    B_post_s2_list.append(B_post_e_s2)
    V_post_s2_list.append(V_post_e_s2)
B_post_s2 = np.column_stack(B_post_s2_list)
resid_s2 = Y - X @ B_post_s2
S_post_s2 = S0 + resid_s2.T @ resid_s2

peak_s05 = []
for draw in range(N_DRAWS):
    try:
        Sig_d = invwishart.rvs(df=nu_post, scale=S_post_s2)
        A1_d  = np.zeros((K, K))
        for e in range(K):
            cov_b = Sig_d[e, e] * V_post_s2_list[e]
            b_d   = multivariate_normal.rvs(mean=B_post_s2_list[e], cov=cov_b)
            A1_d[e, :] = b_d[:K]
        if max(abs(np.linalg.eigvals(A1_d))) >= 1.0:
            continue
        P_d = np.linalg.cholesky(Sig_d + np.eye(K) * 1e-10)
        norm_d = P_d[4, 4]
        irf_h = np.zeros(9)
        Ah_d = np.eye(K)
        for h in range(9):
            irf_h[h] = (Ah_d @ P_d[:, 4])[1] / norm_d
            Ah_d = Ah_d @ A1_d
        peak_s05.append(irf_h.min())
    except:
        continue

bvar_s05_median = float(np.median(peak_s05)) if peak_s05 else float('nan')
bvar_s05_mean   = float(np.mean(peak_s05)) if peak_s05 else float('nan')
print(f"lambda1=0.5: median={bvar_s05_median:.6f}, mean={bvar_s05_mean:.6f}, draws={len(peak_s05)}")

# Save BVAR results
with open(AUDIT / 'bvar_results.txt', 'w') as f:
    f.write(f"BVAR posterior median peak GDP response: {bvar_median} at h={bvar_median_h}\n")
    f.write(f"BVAR posterior mean peak GDP response: {bvar_mean}\n")
    f.write(f"BVAR 68% credible interval: [{bvar_68_lo}, {bvar_68_hi}]\n")
    f.write(f"BVAR 90% credible interval: [{bvar_90_lo}, {bvar_90_hi}]\n")
    f.write(f"BVAR 95% credible interval: [{bvar_95_lo}, {bvar_95_hi}]\n")
    f.write(f"Number of posterior draws used: {len(peak_draws)} of {N_DRAWS} attempted\n")
    f.write(f"Prior tightness lambda1=0.2, lambda2=0.5\n")
    f.write(f"Sensitivity lambda1=0.1: median={bvar_s01_median}, mean={bvar_s01_mean}, n_draws={len(peak_s01)}\n")
    f.write(f"Sensitivity lambda1=0.5: median={bvar_s05_median}, mean={bvar_s05_mean}, n_draws={len(peak_s05)}\n")
    f.write(f"Frequentist Cholesky peak for comparison: {freq_chol_peak} at h={freq_chol_h}\n")
    f.write(f"Prior mean: all coefficients = 0 (white noise / stationary prior)\n")
    f.write(f"Prior IW: S0=diag(sigma_sq_OLS), nu0={nu0}\n")
    f.write(f"Posterior IW: nu_post={nu_post}\n")
    f.write(f"NOTE: equation-by-equation Minnesota prior, independent Normal draws given Sigma\n")

# Save distribution
pd.DataFrame({'bvar_peak_gdp': peak_draws}).to_csv(AUDIT / 'bvar_peak_distribution.csv', index=False)
print(f"\nBVAR results saved to {AUDIT / 'bvar_results.txt'}")
print(f"Distribution saved to {AUDIT / 'bvar_peak_distribution.csv'}")


# ============================================================
print("\n" + "=" * 70)
print("STEP 2: TAYLOR-RULE RESIDUAL")
print("=" * 70)
# ============================================================

# Build Taylor rule using quarterly data
# Δrate_t = α + β_cpi * cpi_t + β_gap * output_gap_t + β_lag_rate * rate_{t-1} + ρ * Δrate_{t-1} + ε_t
# NOTE: No inflation expectations data available. Using contemporaneous CPI growth.

# Get quarterly series (already FWL-adjusted)
d_rate     = q_full['d_rate'].copy()       # quarterly rate change (FWL)
cpi_q      = q_full['cpi'].copy()          # quarterly CPI growth (FWL)
rate_level = q_full['rate_level'].copy()   # level of rate (FWL)
gdp_q      = q_full['gdp'].copy()          # quarterly GDP growth (FWL)
tot_q      = q_full['tot'].copy()

# HP filter for output gap (lambda=1600 for quarterly)
from statsmodels.tsa.filters.hp_filter import hpfilter
gdp_cycle, gdp_trend = hpfilter(gdp_q, lamb=1600)
output_gap = gdp_cycle  # cyclical component

# Align series for Taylor rule regression
# Need: d_rate, cpi (contemp), output_gap (contemp), rate_{t-1} (lagged level), d_rate_{t-1}
tr_df = pd.DataFrame({
    'd_rate':       d_rate,
    'cpi':          cpi_q,
    'output_gap':   output_gap,
    'rate_lag1':    rate_level.shift(1),
    'd_rate_lag1':  d_rate.shift(1),
}).dropna()

print(f"Taylor rule sample: T={len(tr_df)}")

Y_tr  = tr_df['d_rate'].values
X_tr_raw = tr_df[['cpi', 'output_gap', 'rate_lag1', 'd_rate_lag1']].values
X_tr  = sm.add_constant(X_tr_raw)

tr_ols = sm.OLS(Y_tr, X_tr).fit()
print(f"\nTaylor rule OLS:")
print(f"  R²: {tr_ols.rsquared:.6f}")
print(f"  Coefficients: {tr_ols.params}")
print(f"  P-values:     {tr_ols.pvalues}")

taylor_resid = tr_ols.resid  # This is ε̂_t (Taylor shock)
print(f"  Taylor residual std: {taylor_resid.std():.6f}")
print(f"  Taylor residual N: {len(taylor_resid)}")

# Get Cholesky VAR residuals for the rate equation
# The rate equation residual is var_result.resid[:, 4] (rate is index 4)
var_resid_df    = pd.DataFrame(np.array(var_result.resid), index=var_df.index[1:], columns=var_df.columns)
var_resid_rate  = var_resid_df['rate'].values   # numpy array, length T-1
var_resid_dates = var_df.index[1:]              # dates for VAR residuals

# Align Taylor residuals with VAR residuals by date
tr_dates = tr_df.index
common_dates = tr_dates.intersection(var_resid_dates)
print(f"\nCommon dates for first-stage: N={len(common_dates)}")

if len(common_dates) < 20:
    print("WARNING: Too few common dates for first-stage regression")
    fs_F = float('nan')
    fs_corr = float('nan')
    fs_r2 = float('nan')
else:
    # Extract aligned series
    taylor_resid_aligned = pd.Series(np.array(taylor_resid), index=tr_df.index).loc[common_dates].values
    var_rate_aligned     = pd.Series(var_resid_rate, index=var_resid_dates).loc[common_dates].values

    # First-stage regression: u_rate = a + b * taylor_shock + eps
    X_fs = sm.add_constant(taylor_resid_aligned)
    fs_ols = sm.OLS(var_rate_aligned, X_fs).fit()

    fs_F    = float(fs_ols.fvalue)
    fs_r2   = float(fs_ols.rsquared)
    fs_corr = float(np.corrcoef(taylor_resid_aligned, var_rate_aligned)[0, 1])

    print(f"\nFirst-stage regression (u_rate ~ taylor_shock):")
    print(f"  F-statistic: {fs_F:.6f}")
    print(f"  R²: {fs_r2:.6f}")
    print(f"  Correlation: {fs_corr:.6f}")
    print(f"  N: {len(common_dates)}")

# Save Taylor results
with open(AUDIT / 'taylor_rule_results.txt', 'w') as f:
    f.write(f"Taylor rule R²: {tr_ols.rsquared}\n")
    params = tr_ols.params
    pvals  = tr_ols.pvalues
    f.write(f"Taylor rule coefficients: alpha={params[0]}, beta_cpi={params[1]}, beta_gap={params[2]}, beta_lag_rate={params[3]}, rho={params[4]}\n")
    f.write(f"Taylor rule p-values: alpha={pvals[0]}, beta_cpi={pvals[1]}, beta_gap={pvals[2]}, beta_lag_rate={pvals[3]}, rho={pvals[4]}\n")
    f.write(f"Taylor residual first-stage F-statistic: {fs_F}\n")
    f.write(f"Taylor residual correlation with Cholesky VAR rate shock: {fs_corr}\n")
    f.write(f"First-stage R²: {fs_r2}\n")
    f.write(f"Common dates for first-stage: N={len(common_dates)}\n")
    f.write(f"Inflation expectations data used: NO — used contemporaneous CPI growth\n")
    if not np.isnan(fs_F) and fs_F < 10:
        f.write(f"LP-IV not estimated: F={fs_F:.4f} < 10 (weak instrument threshold)\n")
    elif not np.isnan(fs_F) and fs_F >= 10:
        f.write(f"F >= 10: instrument potentially relevant. LP-IV could be estimated.\n")
    f.write(f"Taylor rule sample T: {len(tr_df)}\n")
    f.write(f"NOTE: FWL-adjusted data used (same as VAR baseline)\n")

print(f"\nTaylor results saved to {AUDIT / 'taylor_rule_results.txt'}")


# ============================================================
print("\n" + "=" * 70)
print("STEP 3: TERMS OF TRADE IV FOR POVERTY EQUATION")
print("=" * 70)
# ============================================================

# Annual ToT growth: aggregate quarterly ToT to annual
# ToT is quarterly pct change — annualize by averaging 4 quarters
monthly_raw = pd.read_parquet(PANEL)
monthly_raw['date'] = pd.to_datetime(monthly_raw['date'])
tot_monthly = monthly_raw[monthly_raw['series_id'] == 'PN38923BM'][['date', 'value_raw']].copy()
tot_monthly = tot_monthly.set_index('date').sort_index()['value_raw']

# Annual average of monthly ToT level, then YoY pct change
tot_annual_level = tot_monthly.resample('YE').mean()
tot_annual_yoy   = tot_annual_level.pct_change() * 100  # YoY % change

# Align with poverty years
pov_df = pd.DataFrame({
    'year':     POV_YEARS,
    'gdp_yoy':  POV_GDP,
    'd_poverty': POV_DPOV,
})
pov_df['date'] = pd.to_datetime([f"{y}-12-31" for y in POV_YEARS])
pov_df = pov_df.set_index('date')

# Map ToT annual to poverty years
tot_annual_yoy.index = tot_annual_yoy.index.year
pov_df['tot_annual'] = pov_df['year'].map(tot_annual_yoy)

# Drop rows with missing ToT
pov_complete = pov_df.dropna(subset=['tot_annual'])
N_iv = len(pov_complete)
print(f"\nToT IV sample: N={N_iv} observations")
print(f"Years: {list(pov_complete['year'])}")

y_pov  = pov_complete['d_poverty'].values
X_gdp  = pov_complete['gdp_yoy'].values
Z_tot  = pov_complete['tot_annual'].values

# a. OLS baseline
X_ols_c = sm.add_constant(X_gdp)
ols_res  = sm.OLS(y_pov, X_ols_c).fit()
ols_beta = float(ols_res.params[1])
ols_se   = float(ols_res.bse[1])
ols_t    = float(ols_res.tvalues[1])
ols_p    = float(ols_res.pvalues[1])
ols_r2   = float(ols_res.rsquared)
print(f"\nOLS: beta={ols_beta:.6f}, SE={ols_se:.6f}, t={ols_t:.6f}, p={ols_p:.6f}, R²={ols_r2:.6f}, N={N_iv}")

# b. First stage: GDP = gamma0 + gamma1*ToT + v
Z_tot_c   = sm.add_constant(Z_tot)
fs_tot    = sm.OLS(X_gdp, Z_tot_c).fit()
gamma1    = float(fs_tot.params[1])
gamma1_se = float(fs_tot.bse[1])
fs_F_tot  = float(fs_tot.fvalue)   # F-stat on gamma1=0
fs_r2_tot = float(fs_tot.rsquared)
print(f"First stage: gamma1={gamma1:.6f}, SE={gamma1_se:.6f}, F={fs_F_tot:.6f}, R²={fs_r2_tot:.6f}")

# c. Reduced form: DPoverty = alpha + delta*ToT + eps
rf_res  = sm.OLS(y_pov, Z_tot_c).fit()
delta   = float(rf_res.params[1])
delta_se= float(rf_res.bse[1])
delta_p = float(rf_res.pvalues[1])
delta_r2= float(rf_res.rsquared)
print(f"Reduced form: delta={delta:.6f}, SE={delta_se:.6f}, p={delta_p:.6f}, R²={delta_r2:.6f}")

# d. 2SLS manually
# beta_IV = (Z'Y) / (Z'X) where Z is instrument, X is endogenous, Y is outcome
# In matrix form: beta_IV = (Z'X)^{-1} Z'Y (with constant)
# Using projection: X_hat = Z*(Z'Z)^{-1}Z'X, then OLS of Y on X_hat
X_hat = Z_tot_c @ np.linalg.inv(Z_tot_c.T @ Z_tot_c) @ Z_tot_c.T @ X_ols_c
tsls_res = sm.OLS(y_pov, X_hat).fit()
# 2SLS SE: use correct sandwich formula
# var(beta_IV) = sigma^2_2sls * (X_hat'X_hat)^{-1} where sigma^2_2sls uses true residuals
beta_iv    = float(tsls_res.params[1])
resid_iv   = y_pov - X_ols_c @ np.array([tsls_res.params[0], beta_iv])
sigma2_iv  = np.sum(resid_iv**2) / (N_iv - 2)
XhXh_inv   = np.linalg.inv(X_hat.T @ X_hat)
var_beta_iv = sigma2_iv * XhXh_inv[1, 1]
se_iv       = float(np.sqrt(var_beta_iv))
wald_iv     = float((beta_iv / se_iv)**2)
print(f"2SLS: beta_IV={beta_iv:.6f}, SE={se_iv:.6f}, Wald={wald_iv:.6f}")

# e. Anderson-Rubin confidence set
# AR test statistic: for each hypothesized beta0, regress (y - X*beta0) on Z
# AR(beta0) = ((y-X*beta0) - Z_resid_mean) ... use F-test
# 95% CI: invert AR test at 5% level

beta_grid = np.linspace(-5, 5, 10001)
ar_pvals  = []
for b0 in beta_grid:
    e0 = y_pov - b0 * X_gdp          # residuals under H0: beta=b0 (absorb constant)
    # Regress e0 on [Z, constant], test joint significance of Z coefficient
    ar_ols = sm.OLS(e0, Z_tot_c).fit()
    # Use F-test on the Z coefficient (excluding constant)
    ar_F   = ar_ols.fvalue
    ar_p   = ar_ols.f_pvalue
    ar_pvals.append(float(ar_p))

ar_pvals  = np.array(ar_pvals)
ar_accept = beta_grid[ar_pvals >= 0.05]  # AR 95% CI = beta values not rejected

if len(ar_accept) == 0:
    ar_lo, ar_hi = float('nan'), float('nan')
    ar_note = "EMPTY SET — no beta value accepted at 5% level"
elif ar_accept[0] == beta_grid[0] or ar_accept[-1] == beta_grid[-1]:
    ar_lo = float(ar_accept[0])
    ar_hi = float(ar_accept[-1])
    ar_note = "POSSIBLY UNBOUNDED — grid endpoints hit"
else:
    ar_lo = float(ar_accept[0])
    ar_hi = float(ar_accept[-1])
    ar_note = "bounded"

print(f"Anderson-Rubin 95% CI: [{ar_lo:.4f}, {ar_hi:.4f}] ({ar_note})")

# f. Hausman test
hausman_stat = float((beta_iv - ols_beta)**2 / (se_iv**2 - ols_se**2)) if (se_iv**2 - ols_se**2) > 0 else float('nan')
hausman_p    = float(1 - stats.chi2.cdf(abs(hausman_stat), df=1)) if not np.isnan(hausman_stat) else float('nan')
print(f"Hausman: stat={hausman_stat:.6f}, p={hausman_p:.6f}")

# Diagnosis
if fs_F_tot < 10:
    diagnosis = f"WEAK INSTRUMENT: F={fs_F_tot:.4f} < 10 (Stock-Yogo threshold). 2SLS estimates are unreliable."
elif fs_F_tot < 16.38:
    diagnosis = f"POTENTIALLY WEAK: F={fs_F_tot:.4f} < 16.38 (10% maximal IV size distortion threshold). Interpret with caution."
else:
    diagnosis = f"STRONG INSTRUMENT: F={fs_F_tot:.4f} >= 16.38"

print(f"Diagnosis: {diagnosis}")

with open(AUDIT / 'tot_iv_results.txt', 'w') as f:
    f.write(f"OLS baseline: beta={ols_beta}, SE={ols_se}, t={ols_t}, p={ols_p}, R2={ols_r2}, N={N_iv}\n")
    f.write(f"First stage: gamma1={gamma1}, SE={gamma1_se}, F={fs_F_tot}, R2={fs_r2_tot}\n")
    f.write(f"Reduced form: delta={delta}, SE={delta_se}, p={delta_p}, R2={delta_r2}\n")
    f.write(f"2SLS: beta_IV={beta_iv}, SE={se_iv}\n")
    f.write(f"2SLS Wald statistic: {wald_iv}\n")
    f.write(f"Anderson-Rubin 95% CI: [{ar_lo}, {ar_hi}]\n")
    f.write(f"Anderson-Rubin note: {ar_note}\n")
    f.write(f"Hausman test: statistic={hausman_stat}, p={hausman_p}\n")
    f.write(f"Diagnosis: {diagnosis}\n")
    f.write(f"ToT annual series: YoY change in annual-average monthly level\n")
    f.write(f"Years in sample: {list(pov_complete['year'])}\n")
    f.write(f"Missing ToT years (excluded): {[y for y in POV_YEARS if y not in list(pov_complete['year'])]}\n")

print(f"\nToT IV results saved to {AUDIT / 'tot_iv_results.txt'}")


# ============================================================
print("\n" + "=" * 70)
print("STEP 4: VERIFICATION — PRINT ALL AUDIT FILES")
print("=" * 70)
# ============================================================

for fname in ['bvar_results.txt', 'taylor_rule_results.txt', 'tot_iv_results.txt']:
    fpath = AUDIT / fname
    print(f"\n{'='*50}")
    print(f"FILE: {fpath}")
    print('='*50)
    with open(fpath) as f:
        print(f.read())

# Copy to output dir
import shutil
out_audit = OUT_DIR / 'audit'
out_audit.mkdir(exist_ok=True)
for fname in ['bvar_results.txt', 'taylor_rule_results.txt', 'tot_iv_results.txt', 'bvar_peak_distribution.csv']:
    src = AUDIT / fname
    if src.exists():
        shutil.copy(src, out_audit / fname)
        print(f"Copied {fname} to {out_audit}")

print("\n" + "=" * 70)
print("AUDIT SCRIPT COMPLETE")
print("=" * 70)
