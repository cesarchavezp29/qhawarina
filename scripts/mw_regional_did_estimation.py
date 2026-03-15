"""
mw_regional_did_estimation.py
================================
Regional DiD following Dube, Lester & Reich (2010).

  Y_idt = alpha_d + gamma_t + beta*(Post_t x Treat_d) + X'theta + eps_idt

Treatment intensities:
  (A) kaitz_pre    = MW_930 / median_V4_formal_wage_dept_2021
  (B) share_at_risk = share formal workers with wage in [0.9*930, 1.1*1025] in 2021

Outcomes:
  formal_v4, informal, log_wage, below_mw, employed

Event study: beta_22 and beta_23, base = 2021.

Robustness:
  - Alternate treatment: share_at_risk
  - Alternate formality: ocupinf_inei (INEI's own variable)

SE: statsmodels cluster-robust (cov_type='cluster', groups=dept).
    If wildboottest available, also report WCB p-values.
Weights: fac500a.

Output: D:/Nexus/nexus/exports/data/mw_regional_did_results.json
"""

import json, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
PANEL    = ROOT / "data/processed/mw_regional_did_panel.parquet"
OUT_JSON = ROOT / "exports/data/mw_regional_did_results.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

# ── wildboottest availability ──────────────────────────────────────────────────
try:
    from wildboottest.wildboottest import wildboottest
    HAS_WBT = True
    print("wildboottest: available")
except ImportError:
    HAS_WBT = False
    print("wildboottest: not available — skipping WCB p-values")

# ── Load ───────────────────────────────────────────────────────────────────────
print("\nLoading panel …")
df = pd.read_parquet(PANEL)
df["dept"] = df["dept"].astype(str).str.zfill(2)
print(f"  Shape: {df.shape}  |  Years: {sorted(df.year.unique())}  |  Depts: {df.dept.nunique()}")

# ── Derived outcomes ───────────────────────────────────────────────────────────
df["formal_inei"] = df["ocupinf_inei"].copy()   # 1=formal, 0=informal (NaN for 2024)
df["age2"]        = df["p208a"] ** 2

# ── Dataset splits ─────────────────────────────────────────────────────────────
df_main  = df[df.year.isin([2021, 2023])].copy()
df_event = df.copy()
BASE_DEPT = "15"

# ── Core regression function ───────────────────────────────────────────────────
def run_reg(Y_col, df_sub, treat_col, controls, post_col, groups_col,
            wt_col="fac500a", note="", event=False):
    """
    Weighted OLS with dept+year FEs and cluster-robust SE.
    Returns dict with key estimates.
    """
    needed = [Y_col, treat_col, groups_col, wt_col, "year", "dept"] + controls
    sub = df_sub.dropna(subset=[c for c in needed if c in df_sub.columns]).copy()
    sub = sub[sub[Y_col].notna()].copy()
    N = len(sub)
    if N < 30:
        return None

    # Outcome + weights
    Y = sub[Y_col].values.astype(float)
    w = sub[wt_col].values.astype(float)
    groups = sub[groups_col].values

    # Year dummies
    years = sorted(sub["year"].unique())
    yr_dummies = pd.get_dummies(sub["year"], prefix="yr").astype(float)
    drop_yr = f"yr_{min(years)}"
    if drop_yr in yr_dummies.columns:
        yr_dummies.drop(columns=[drop_yr], inplace=True)

    # Dept dummies
    dept_dummies = pd.get_dummies(sub["dept"], prefix="d").astype(float)
    drop_dept = f"d_{BASE_DEPT}"
    if drop_dept in dept_dummies.columns:
        dept_dummies.drop(columns=[drop_dept], inplace=True)

    # Treatment interaction(s)
    if not event:
        sub["treat"] = sub[post_col].astype(float) * sub[treat_col].astype(float)
        treat_cols = [post_col, "treat"]
    else:
        sub["treat_2022"] = (sub["year"] == 2022).astype(float) * sub[treat_col].astype(float)
        sub["treat_2023"] = (sub["year"] == 2023).astype(float) * sub[treat_col].astype(float)
        treat_cols = ["treat_2022", "treat_2023"]

    ctrl_df = sub[controls].copy() if controls else pd.DataFrame(index=sub.index)

    # Assemble X
    parts = [pd.Series(1.0, index=sub.index, name="const")]
    parts += [sub[c].rename(c) for c in treat_cols]
    if not event:
        parts += [yr_dummies]
    parts += [ctrl_df, dept_dummies]
    X = pd.concat(parts, axis=1).astype(float)

    # Drop near-constant columns
    std = X.std(axis=0)
    keep_mask = (std > 1e-10) | (X.columns == "const")
    X = X.loc[:, keep_mask]

    # WLS with clustered SE
    model  = sm.WLS(Y, X, weights=w)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": groups})

    def extract(param):
        if param not in result.params.index:
            return {}
        return {
            "coef":  float(result.params[param]),
            "se":    float(result.bse[param]),
            "tstat": float(result.tvalues[param]),
            "pval":  float(result.pvalues[param]),
            "ci_lo": float(result.conf_int().loc[param, 0]),
            "ci_hi": float(result.conf_int().loc[param, 1]),
        }

    out = {"outcome": Y_col, "treat_var": treat_col, "note": note,
           "N": N, "N_depts": int(len(np.unique(groups))),
           "r2": float(result.rsquared)}

    if not event:
        r = extract("treat")
        out.update({
            "beta":  r.get("coef",  np.nan),
            "se":    r.get("se",    np.nan),
            "tstat": r.get("tstat", np.nan),
            "pval":  r.get("pval",  np.nan),
            "ci_lo": r.get("ci_lo", np.nan),
            "ci_hi": r.get("ci_hi", np.nan),
        })

        # Wild cluster bootstrap p-value
        if HAS_WBT:
            try:
                wbt = wildboottest(model.fit(), cluster=groups,
                                   param="treat", B=999, seed=42)
                out["wcb_pval"] = float(wbt["p_val"])
            except Exception:
                out["wcb_pval"] = None
    else:
        r22 = extract("treat_2022")
        r23 = extract("treat_2023")
        out.update({
            "beta_2022":  r22.get("coef",  np.nan),
            "se_2022":    r22.get("se",    np.nan),
            "pval_2022":  r22.get("pval",  np.nan),
            "ci_lo_2022": r22.get("ci_lo", np.nan),
            "ci_hi_2022": r22.get("ci_hi", np.nan),
            "beta_2023":  r23.get("coef",  np.nan),
            "se_2023":    r23.get("se",    np.nan),
            "pval_2023":  r23.get("pval",  np.nan),
            "ci_lo_2023": r23.get("ci_lo", np.nan),
            "ci_hi_2023": r23.get("ci_hi", np.nan),
        })

    return out

# ── Sample mask factory ────────────────────────────────────────────────────────
def masks(df_use):
    emp    = df_use["ocu500"] == 1
    formal = emp & (df_use["formal_v4"] == 1)
    wage   = formal & (df_use["wage_monthly"] > 0) & df_use["log_wage"].notna()
    full   = pd.Series(True, index=df_use.index)
    return {"employed": emp, "formal": formal, "wage": wage, "full": full}

CONTROLS = ["p208a", "age2", "p207"]

# ── Specification table ────────────────────────────────────────────────────────
# (outcome, sample_key, note)
SPECS = [
    ("formal_v4",   "employed", "LPM, employed"),
    ("informal",    "employed", "LPM, employed"),
    ("log_wage",    "wage",     "OLS, formal w/ wage>0"),
    ("below_mw",    "formal",   "LPM, formal-employed"),
    ("employed",    "full",     "LPM, full working-age"),
]

# ── MAIN DiD ──────────────────────────────────────────────────────────────────
def print_main_header():
    print(f"\n{'Outcome':<18} {'Treat':>12} {'N':>8}  {'Beta':>10}  {'SE':>8}  {'t':>7}  {'p':>7}  {'95% CI'}")
    print("-"*90)

def print_main_row(r):
    if r is None: return
    star = "***" if r["pval"]<0.01 else ("**" if r["pval"]<0.05 else ("*" if r["pval"]<0.10 else ""))
    wcb  = f"  wcb={r['wcb_pval']:.3f}" if r.get("wcb_pval") is not None else ""
    print(f"  {r['outcome']:<16} {r['treat_var']:>12} {r['N']:>8,}  {r['beta']:>+10.4f}  {r['se']:>8.4f}  "
          f"{r['tstat']:>7.2f}  {r['pval']:>7.4f}{star}  [{r['ci_lo']:>+8.4f}, {r['ci_hi']:>+7.4f}]{wcb}")

def run_main_block(df_use, treat_col, label):
    print(f"\n--- Treatment: {label} ---")
    print_main_header()
    results = []
    m = masks(df_use)
    for outcome, skey, note in SPECS:
        mask = m[skey]
        sub  = df_use[mask].copy()
        if outcome not in sub.columns or sub[outcome].notna().sum() < 30:
            continue
        r = run_reg(outcome, sub, treat_col, CONTROLS, "post", "dept",
                    note=f"{note}, treat={treat_col}")
        print_main_row(r)
        if r:
            results.append(r)
    return results

print("\n" + "="*90)
print("MAIN DiD — 2021 vs 2023   (Post=year==2023, base=2021, dept+year FE, cluster SE)")
print("="*90)

main_kaitz       = run_main_block(df_main, "kaitz_pre",    "kaitz_pre")
main_share       = run_main_block(df_main, "share_at_risk","share_at_risk")

# Robustness: formal_inei with kaitz_pre
print(f"\n--- Robustness: formal_inei (INEI ocupinf) with kaitz_pre ---")
print_main_header()
m_main = masks(df_main)
inei_mask = m_main["employed"] & df_main["formal_inei"].notna()
r_inei = run_reg("formal_inei", df_main[inei_mask], "kaitz_pre", CONTROLS, "post", "dept",
                 note="LPM INEI ocupinf, robustness")
print_main_row(r_inei)
robustness_results = [r_inei] if r_inei else []

# ── EVENT STUDY ───────────────────────────────────────────────────────────────
print("\n" + "="*90)
print("EVENT STUDY — base=2021, beta_2022 and beta_2023   (treat=kaitz_pre)")
print("="*90)
print(f"\n{'Outcome':<18} {'N':>8}  {'b_22':>10}  {'se_22':>8}  {'p_22':>7}  {'b_23':>10}  {'se_23':>8}  {'p_23':>7}")
print("-"*95)

def star(p):
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))

m_ev = masks(df_event)
event_results = []
for outcome, skey, note in SPECS:
    mask = m_ev[skey]
    sub  = df_event[mask].copy()
    if outcome not in sub.columns or sub[outcome].notna().sum() < 30:
        continue
    r = run_reg(outcome, sub, "kaitz_pre", CONTROLS, "post_2023", "dept",
                note=note, event=True)
    if r:
        event_results.append(r)
        print(f"  {r['outcome']:<16} {r['N']:>8,}  "
              f"{r['beta_2022']:>+10.4f}  {r['se_2022']:>8.4f}  {r['pval_2022']:>7.4f}{star(r['pval_2022'])}  "
              f"{r['beta_2023']:>+10.4f}  {r['se_2023']:>8.4f}  {r['pval_2023']:>7.4f}{star(r['pval_2023'])}")

# ── ECONOMIC INTERPRETATION ───────────────────────────────────────────────────
print("\n" + "="*90)
print("ECONOMIC INTERPRETATION")
print("="*90)

kaitz_q = df_main["kaitz_pre"].quantile([0.25, 0.50, 0.75])
iqr_kaitz = float(kaitz_q[0.75] - kaitz_q[0.25])
iqr_share = float(df_main["share_at_risk"].quantile(0.75) - df_main["share_at_risk"].quantile(0.25))

print(f"\n  Kaitz_pre    : p25={kaitz_q[0.25]:.3f}  p50={kaitz_q[0.50]:.3f}  p75={kaitz_q[0.75]:.3f}  IQR={iqr_kaitz:.3f}")
sar_q = df_main["share_at_risk"].quantile([0.25, 0.50, 0.75])
print(f"  share_at_risk: p25={sar_q[0.25]:.3f}  p50={sar_q[0.50]:.3f}  p75={sar_q[0.75]:.3f}  IQR={iqr_share:.3f}")
print()

print(f"  IQR effects (moving from p25 to p75 dept):")
for r in main_kaitz:
    if r and not np.isnan(r.get("beta", np.nan)):
        eff = r["beta"] * iqr_kaitz
        print(f"    {r['outcome']:<18}: beta={r['beta']:>+.4f}  IQR effect={eff:>+.4f}  p={r['pval']:.4f}"
              + ("***" if r["pval"]<0.01 else "**" if r["pval"]<0.05 else "*" if r["pval"]<0.10 else ""))

# ── SAVE JSON ─────────────────────────────────────────────────────────────────
def clean(x):
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    return x

output = {
    "metadata": {
        "spec": "Y_idt = alpha_d + gamma_t + beta*(Post*Treat) + age+age2+sex + eps",
        "post": "year==2023",
        "base": "year==2021",
        "se_type": "cluster-robust at dept level (25 clusters)",
        "weights": "fac500a",
        "formality": "V4 (p419a1, p510, p511a, p512a, p517)",
        "source": "ENAHO Panel 2020-2024 (srienaho-978)",
        "generated": "2026-03-15",
    },
    "main_kaitz":    [clean(r) for r in main_kaitz if r],
    "main_share":    [clean(r) for r in main_share if r],
    "robustness":    [clean(r) for r in robustness_results if r],
    "event_study":   [clean(r) for r in event_results if r],
    "kaitz_summary": {
        "p25": float(kaitz_q[0.25]),
        "p50": float(kaitz_q[0.50]),
        "p75": float(kaitz_q[0.75]),
        "iqr": iqr_kaitz,
        "min": float(df_main["kaitz_pre"].min()),
        "max": float(df_main["kaitz_pre"].max()),
    },
    "share_summary": {
        "p25": float(sar_q[0.25]),
        "p50": float(sar_q[0.50]),
        "p75": float(sar_q[0.75]),
        "iqr": iqr_share,
    },
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
print(f"\nWrote {OUT_JSON}")
print("DONE.")
