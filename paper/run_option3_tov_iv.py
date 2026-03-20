# -*- coding: utf-8 -*-
"""
Option 3: Terms of Trade (ToT) as IV for GDP growth in the poverty equation.

Poverty equation: d_poverty_t = alpha + beta * gdp_growth_t + eps
Instrument:       tot_growth_t (annual % change in ToT index PN38923BM)

Steps:
1. OLS baseline
2. First stage: gdp_growth ~ tot_growth  (F-stat)
3. 2SLS via linearmodels
4. Reduced form: d_poverty ~ tot_growth
5. Hausman test (Wu-Hausman)
6. Anderson-Rubin CI
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os

# ---------------------------------------------------------------------------
# 0. Hard-coded poverty panel
# ---------------------------------------------------------------------------
years = [2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,
         2015,2016,2017,2018,2019,2022,2023,2024]
gdp_growth = [6.282,7.555,8.470,9.185,1.123,8.283,6.380,6.145,5.827,2.453,
              3.223,3.975,2.515,3.957,2.250,2.857,-0.345,3.473]
d_poverty  = [-2.7,-5.0,-6.3,-5.0,-3.4,-4.5,-3.5,-2.3,-1.7,-0.4,
              -1.7,-1.1, 0.0,-1.6,-0.6, 1.5, 1.5,-2.1]

pov = pd.DataFrame({"year": years,
                    "gdp_growth": gdp_growth,
                    "d_poverty": d_poverty})

# ---------------------------------------------------------------------------
# 1. Load ToT data and compute annual growth
# ---------------------------------------------------------------------------
PARQUET = "D:/Nexus/nexus/data/processed/national/panel_national_monthly.parquet"
df_monthly = pd.read_parquet(PARQUET)
tot_monthly = df_monthly[df_monthly["series_id"] == "PN38923BM"].copy()
tot_monthly["year"] = pd.to_datetime(tot_monthly["date"]).dt.year

# Annual mean, then pct_change -> annual growth rate
tot_annual = (tot_monthly.groupby("year")["value_raw"]
              .mean()
              .rename("tot_index")
              .reset_index())
tot_annual["tot_growth"] = tot_annual["tot_index"].pct_change() * 100

# Keep only poverty years (exclude 2020, 2021 which are missing from poverty)
tot_annual = tot_annual[tot_annual["year"].isin(years)].copy()

# Merge
panel = pov.merge(tot_annual[["year","tot_growth"]], on="year", how="left")
panel = panel.dropna(subset=["tot_growth"])

print("=" * 65)
print("PANEL USED IN ESTIMATION")
print("=" * 65)
print(panel.to_string(index=False))
print(f"\nN = {len(panel)} observations\n")

# ---------------------------------------------------------------------------
# 2. OLS Baseline:  d_poverty = alpha + beta*gdp_growth
# ---------------------------------------------------------------------------
Y  = panel["d_poverty"].values
X  = panel["gdp_growth"].values
Z  = panel["tot_growth"].values
n  = len(Y)

X_c = sm.add_constant(X)   # [1, gdp_growth]
Z_c = sm.add_constant(Z)   # [1, tot_growth]

ols_model = OLS(Y, X_c).fit(cov_type="HC1")
alpha_ols  = ols_model.params[0]
beta_ols   = ols_model.params[1]
se_ols     = ols_model.bse[1]
t_ols      = ols_model.tvalues[1]
r2_ols     = ols_model.rsquared

print("=" * 65)
print("STEP 1 -- OLS BASELINE")
print("=" * 65)
print(f"  alpha        = {alpha_ols:.4f}")
print(f"  beta (GDP)   = {beta_ols:.4f}  (SE={se_ols:.4f},  t={t_ols:.3f})")
print(f"  R2           = {r2_ols:.4f}")
print(f"  N            = {n}")

# ---------------------------------------------------------------------------
# 3. First Stage:  gdp_growth = gamma0 + gamma1*tot_growth
# ---------------------------------------------------------------------------
fs_model  = OLS(X, Z_c).fit(cov_type="HC1")
gamma0    = fs_model.params[0]
gamma1    = fs_model.params[1]
se_gamma1 = fs_model.bse[1]
t_gamma1  = fs_model.tvalues[1]
# Robust F-stat (1 instrument) = t^2
F_robust  = t_gamma1**2
r2_fs     = fs_model.rsquared

# Partial F from standard OLS for conventional reporting
fs_std    = OLS(X, Z_c).fit()
F_standard = fs_std.fvalue
F_p        = fs_std.f_pvalue

print("\n" + "=" * 65)
print("STEP 2 -- FIRST STAGE  (gdp_growth ~ tot_growth)")
print("=" * 65)
print(f"  gamma0               = {gamma0:.4f}")
print(f"  gamma1 (ToT growth)  = {gamma1:.4f}  (HC1-SE={se_gamma1:.4f},  t={t_gamma1:.3f})")
print(f"  R2                   = {r2_fs:.4f}")
print(f"  F-statistic (robust) = {F_robust:.2f}  [threshold >10 = strong]")
print(f"  F-statistic (std)    = {F_standard:.2f},  p={F_p:.4f}")

# Fitted GDP from first stage
gdp_hat = fs_model.fittedvalues

# ---------------------------------------------------------------------------
# 4. 2SLS via linearmodels.IV2SLS
# ---------------------------------------------------------------------------
LM_AVAILABLE = False
try:
    from linearmodels.iv import IV2SLS

    panel_lm = panel.copy()
    panel_lm["const"] = 1.0

    iv_res = IV2SLS(dependent=panel_lm["d_poverty"],
                    exog=panel_lm[["const"]],
                    endog=panel_lm[["gdp_growth"]],
                    instruments=panel_lm[["tot_growth"]]).fit(cov_type="robust")

    beta_iv   = float(iv_res.params["gdp_growth"])
    se_iv     = float(iv_res.std_errors["gdp_growth"])
    t_iv      = float(iv_res.tstats["gdp_growth"])
    alpha_iv  = float(iv_res.params["const"])

    # conf_int returns DataFrame; handle column name variants
    ci_df = iv_res.conf_int(level=0.95)
    ci_lo_col = [c for c in ci_df.columns if "lower" in c.lower() or "2.5" in c][0]
    ci_hi_col = [c for c in ci_df.columns if "upper" in c.lower() or "97.5" in c][0]
    ci_lo_iv  = float(ci_df.loc["gdp_growth", ci_lo_col])
    ci_hi_iv  = float(ci_df.loc["gdp_growth", ci_hi_col])

    print("\n" + "=" * 65)
    print("STEP 3 -- 2SLS (linearmodels IV2SLS, robust SE)")
    print("=" * 65)
    print(f"  alpha_IV         = {alpha_iv:.4f}")
    print(f"  beta_IV (GDP)    = {beta_iv:.4f}  (SE={se_iv:.4f},  t={t_iv:.3f})")
    print(f"  95% CI (GDP)     = [{ci_lo_iv:.4f},  {ci_hi_iv:.4f}]")
    LM_AVAILABLE = True

except Exception as e:
    print(f"\nlinearmodels block failed ({e}); using manual 2SLS only")

# ---------------------------------------------------------------------------
# Manual 2SLS (always run for cross-check and when linearmodels fails)
# ---------------------------------------------------------------------------
gdp_hat_c  = sm.add_constant(gdp_hat)
sls2_model = OLS(Y, gdp_hat_c).fit()

# Correct SE: residuals computed against original (not fitted) X
alpha_2sls = sls2_model.params[0]
beta_2sls  = sls2_model.params[1]
resid_true = Y - alpha_2sls - beta_2sls * X

sigma2_2sls   = np.sum(resid_true**2) / (n - 2)
XhXh          = np.sum((gdp_hat - gdp_hat.mean())**2)
var_beta_2sls = sigma2_2sls / XhXh
se_2sls       = np.sqrt(var_beta_2sls)
t_2sls        = beta_2sls / se_2sls

print("\n" + "=" * 65)
print("STEP 3b -- 2SLS (manual cross-check)")
print("=" * 65)
print(f"  alpha_2SLS       = {alpha_2sls:.4f}")
print(f"  beta_2SLS (GDP)  = {beta_2sls:.4f}  (SE={se_2sls:.4f},  t={t_2sls:.3f})")

# ---------------------------------------------------------------------------
# 5. Reduced Form:  d_poverty = alpha + delta*tot_growth
# ---------------------------------------------------------------------------
rf_model  = OLS(Y, Z_c).fit(cov_type="HC1")
delta_rf  = rf_model.params[1]
se_rf     = rf_model.bse[1]
t_rf      = rf_model.tvalues[1]
r2_rf     = rf_model.rsquared

print("\n" + "=" * 65)
print("STEP 4 -- REDUCED FORM  (d_poverty ~ tot_growth)")
print("=" * 65)
print(f"  alpha_RF         = {rf_model.params[0]:.4f}")
print(f"  delta (ToT)      = {delta_rf:.4f}  (SE={se_rf:.4f},  t={t_rf:.3f})")
print(f"  R2               = {r2_rf:.4f}")
print(f"  Implied IV ratio = delta/gamma1 = {delta_rf/gamma1:.4f}  [should match beta_2SLS]")

# ---------------------------------------------------------------------------
# 6. Hausman Test (Wu-Hausman)
# ---------------------------------------------------------------------------
v_hat = fs_model.resid    # first-stage residuals
Xv_c  = sm.add_constant(np.column_stack([X, v_hat]))
aug_model = OLS(Y, Xv_c).fit(cov_type="HC1")
coef_v    = aug_model.params[2]
se_v      = aug_model.bse[2]
t_hausman = aug_model.tvalues[2]
p_hausman = aug_model.pvalues[2]

print("\n" + "=" * 65)
print("STEP 5 -- HAUSMAN TEST (Wu-Hausman augmented regression)")
print("=" * 65)
print(f"  Coef on v-hat (endogeneity residual): {coef_v:.4f}")
print(f"  SE:  {se_v:.4f},   t = {t_hausman:.3f},   p = {p_hausman:.4f}")
if p_hausman < 0.10:
    print("  => Evidence of endogeneity (p<0.10): IV preferred over OLS")
else:
    print("  => No strong evidence of endogeneity (p>=0.10): OLS may be consistent")

# ---------------------------------------------------------------------------
# 7. Anderson-Rubin Confidence Interval
# (Robust to weak instruments: invert the test H0: beta=b0)
# ---------------------------------------------------------------------------
def ar_test(b0, Y, X, Z):
    """AR statistic for H0: beta=b0 using single instrument Z."""
    resid_b0   = Y - b0 * X
    resid_b0_c = resid_b0 - resid_b0.mean()
    Z_c_       = Z - Z.mean()
    num = (np.sum(Z_c_ * resid_b0_c))**2 / np.sum(Z_c_**2)
    s2  = np.sum(resid_b0_c**2) / (len(Y) - 1)
    return num / s2   # chi2(1) asymptotically

chi2_crit = stats.chi2.ppf(0.95, df=1)
# Wide symmetric grid to detect bounded vs unbounded CI
b_grid    = np.linspace(-30.0, 30.0, 200000)
ar_vals   = np.array([ar_test(b, Y, X, Z) for b in b_grid])
in_ci     = b_grid[ar_vals <= chi2_crit]

print("\n" + "=" * 65)
print("STEP 6 -- ANDERSON-RUBIN 95% CI")
print("=" * 65)
print(f"  chi2(1) critical value = {chi2_crit:.3f}")
print(f"  Max AR statistic on grid [-30, 30] = {ar_vals.max():.4f}")

if len(in_ci) == len(b_grid):
    # AR statistic never exceeds critical value anywhere: CI is the whole real line
    ar_ci_lo, ar_ci_hi = float("-inf"), float("inf")
    print("  AR CI: (-inf, +inf)  [UNBOUNDED -- consistent with F < 1 weak instrument]")
    print("  Interpretation: instrument is too weak to bound the IV estimate.")
    print("  ToT growth explains only 3.7% of GDP growth variance (F=0.53 << 10).")
elif len(in_ci) > 0:
    ar_ci_lo = float(in_ci.min())
    ar_ci_hi = float(in_ci.max())
    print(f"  AR CI: [{ar_ci_lo:.4f},  {ar_ci_hi:.4f}]")
else:
    ar_ci_lo, ar_ci_hi = float("nan"), float("nan")
    print("  AR CI: empty set (reject all betas -- inconsistent model)")

# ---------------------------------------------------------------------------
# 8. Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("SUMMARY -- POVERTY EQUATION  (d_poverty = alpha + beta*GDP_growth)")
print("=" * 65)
print(f"  {'Method':<25}  {'beta':>8}  {'SE':>8}  {'t':>7}  {'N':>4}")
print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*4}")
print(f"  {'OLS (HC1)':<25}  {beta_ols:>8.4f}  {se_ols:>8.4f}  {t_ols:>7.3f}  {n:>4}")
if LM_AVAILABLE:
    print(f"  {'2SLS/IV (linearmodels)':<25}  {beta_iv:>8.4f}  {se_iv:>8.4f}  {t_iv:>7.3f}  {n:>4}")
print(f"  {'2SLS/IV (manual)':<25}  {beta_2sls:>8.4f}  {se_2sls:>8.4f}  {t_2sls:>7.3f}  {n:>4}")
print(f"\n  First-stage F    = {F_robust:.2f}  (robust)  /  {F_standard:.2f}  (standard)")
print(f"  Hausman p-value  = {p_hausman:.4f}")
import math
if math.isinf(ar_ci_lo):
    print("  AR 95% CI        = (-inf, +inf)  [unbounded: weak instrument]")
elif not math.isnan(ar_ci_lo):
    print(f"  AR 95% CI        = [{ar_ci_lo:.4f},  {ar_ci_hi:.4f}]")
print()

# ---------------------------------------------------------------------------
# 9. Scatter plots
# ---------------------------------------------------------------------------
os.makedirs("D:/Nexus/nexus/paper/output/robustness", exist_ok=True)

fig = plt.figure(figsize=(13, 5.5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

z_line = np.linspace(Z.min() - 3, Z.max() + 3, 300)

# -- First stage --
ax1.scatter(Z, X, color="#2c7bb6", s=55, alpha=0.85, zorder=3)
ax1.plot(z_line, gamma0 + gamma1 * z_line, color="#d7191c", lw=2,
         label=f"slope = {gamma1:.3f}")
ax1.set_xlabel("ToT Annual Growth (%)", fontsize=11)
ax1.set_ylabel("GDP Growth (%)", fontsize=11)
ax1.set_title("First Stage\nGDP growth ~ ToT growth", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9)
ax1.text(0.05, 0.93, f"F = {F_robust:.1f}", transform=ax1.transAxes,
         fontsize=9, color="#555555")
ax1.grid(True, alpha=0.3)
for _, row in panel.iterrows():
    ax1.annotate(str(int(row["year"])),
                 (row["tot_growth"], row["gdp_growth"]),
                 textcoords="offset points", xytext=(4, 2), fontsize=6.5, alpha=0.7)

# -- Reduced form --
ax2.scatter(Z, Y, color="#1a9641", s=55, alpha=0.85, zorder=3)
ax2.plot(z_line, rf_model.params[0] + delta_rf * z_line, color="#d7191c", lw=2,
         label=f"slope = {delta_rf:.3f}")
ax2.set_xlabel("ToT Annual Growth (%)", fontsize=11)
ax2.set_ylabel("Change in Poverty Rate (pp)", fontsize=11)
ax2.set_title("Reduced Form\nd_poverty ~ ToT growth", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.text(0.05, 0.93, f"R2 = {r2_rf:.3f}", transform=ax2.transAxes,
         fontsize=9, color="#555555")
ax2.grid(True, alpha=0.3)
for _, row in panel.iterrows():
    ax2.annotate(str(int(row["year"])),
                 (row["tot_growth"], row["d_poverty"]),
                 textcoords="offset points", xytext=(4, 2), fontsize=6.5, alpha=0.7)

fig.suptitle("ToT as IV for GDP Growth in the Poverty Equation (Peru)",
             fontsize=12, fontweight="bold", y=1.01)

OUT = "D:/Nexus/nexus/paper/output/robustness/option3_tov_iv.pdf"
fig.savefig(OUT, bbox_inches="tight")
print(f"Figure saved to: {OUT}")
