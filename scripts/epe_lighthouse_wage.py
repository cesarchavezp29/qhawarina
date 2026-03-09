"""
epe_lighthouse_wage.py
THE LIGHTHOUSE EFFECT: MW as a reference price for informal wages.

HYPOTHESIS: MW increase raises wages of informal workers earning near MW,
even though MW legally doesn't bind them. MW acts as a "farol" (lighthouse)
in informal bargaining — when the official anchor shifts, informal wages follow.

Literature:
  Maloney (2004): strong lighthouse in Brazil, Mexico, Colombia
  Lombardo et al. (2024): lighthouse confirmed for 6 LAC countries incl. Peru
  Khamis (2013): Argentina lighthouse > formal pass-through

DESIGN:
  For each event, match workers across pre/post waves (conglome+vivienda+hogar+codperso).
  Split by formality status in pre-period (Def B: p222_pre in {1,2,3}).

  INFORMAL stayers (informal pre + employed post):
    Treatment: wage_pre in [0.80*MW_old, MW_new)  — earning near old MW
    Control:   wage_pre in [MW_new,      1.5*MW_new) — above new MW
    Outcome:   d_log_wage = log(wage_post) - log(wage_pre)
    Model:     d_log_wage ~ treat  (OLS, HC1 SEs)

  FORMAL stayers (same bands, same outcome) — comparison benchmark.

  If β_lighthouse > 0: informal wages near-MW rose more than informal wages
  above-MW → lighthouse effect confirmed.

FORMALITY DEFINITION: Def B — p222 in {1,2,3} (validated: mean|gap|=2.7pp vs INEI)
VARIABLE CODES: verified from data audit (2003/2012/2022 .sav labels)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pathlib
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm
import statsmodels.formula.api as smf

OUT      = pathlib.Path("D:/Nexus/nexus/data/raw/epe/srienaho")
SAVE_DIR = pathlib.Path("D:/Nexus/nexus/data/results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MW_EVENTS = [
    ("2003_Oct", 410, 460,  92,  95),
    ("2006_Jan", 500, 530, 138, 142),
    ("2007_Aug", 530, 550, 179, 187),
    ("2008_Jan", 550, 600, 191, 196),
    ("2011_Aug", 600, 675, 264, 272),
    ("2012_Jun", 675, 750, 287, 299),
    ("2015_Dec", 750, 850, 485, 496),
    ("2016_May", 850, 930, 496, 514),
    ("2022_Apr", 930, 1025, 742, 766),
]

FORMAL_CODES = {1.0, 2.0, 3.0}   # EsSalud, Seguro Privado, Ambos (Def B)


def load_sav(code):
    d = OUT / str(code)
    savs = sorted(d.glob("*.sav")) + sorted(d.glob("*.SAV"))
    if not savs:
        return None
    df, _ = pyreadstat.read_sav(str(savs[0]))
    df.columns = [c.lower() for c in df.columns]
    return df


def make_pid(df):
    return (df["conglome"].astype(str).str.strip() + "_" +
            df["vivienda"].astype(str).str.strip() + "_" +
            df["hogar"].astype(str).str.strip() + "_" +
            df["codperso"].astype(str).str.strip())


def is_formal(s):
    return pd.to_numeric(s, errors="coerce").isin(FORMAL_CODES)


def ols_beta(y, treat):
    """OLS Δlog_wage ~ treat with HC1 SEs. Returns (beta, se, pval, n)."""
    t = treat.rename("treat").astype(float)
    X = sm.add_constant(t)
    try:
        res = sm.OLS(y, X).fit(cov_type="HC1")
        b   = res.params["treat"]
        se  = res.bse["treat"]
        p   = res.pvalues["treat"]
        return b, se, p, int(len(y))
    except Exception as e:
        print(f"      [OLS error: {e}]")
        return np.nan, np.nan, np.nan, 0


def run_event(event_name, mw_old, mw_new, code_pre, code_post):
    pre  = load_sav(code_pre)
    post = load_sav(code_post)
    if pre is None or post is None:
        print(f"\nSKIP {event_name}: file not found")
        return None

    pre["pid"]  = make_pid(pre)
    post["pid"] = make_pid(post)

    # ── Pre-wave: employed with positive income ─────────────────────────────
    pre_emp = pre[
        (pd.to_numeric(pre["ocu200"],   errors="coerce") == 1) &
        (pd.to_numeric(pre.get("ingprin", pd.Series(dtype=float)),
                       errors="coerce") > 0)
    ].copy()

    pre_emp["wage_pre"]    = pd.to_numeric(pre_emp["ingprin"], errors="coerce")
    pre_emp["formal_pre"]  = is_formal(pre_emp["p222"]).astype(int)
    pre_emp["log_wage_pre"] = np.log(pre_emp["wage_pre"])

    # ── Post-wave: employed with positive income ────────────────────────────
    post_sub = post[["pid", "ocu200", "ingprin"]].copy()
    post_sub["employed_post"] = (
        pd.to_numeric(post_sub["ocu200"], errors="coerce") == 1
    ).astype(int)
    post_sub["wage_post"] = pd.to_numeric(post_sub["ingprin"], errors="coerce")
    post_sub = post_sub.drop(columns=["ocu200", "ingprin"])

    merged = pre_emp.merge(post_sub, on="pid", how="inner")

    # ── Stayers: employed and positive wage in both waves ──────────────────
    stayers = merged[
        (merged["employed_post"] == 1) &
        (merged["wage_post"] > 0)
    ].copy()
    stayers["log_wage_post"] = np.log(stayers["wage_post"])
    stayers["d_log_wage"]    = stayers["log_wage_post"] - stayers["log_wage_pre"]

    # Drop extreme wage changes (likely misreports: |Δ| > 1.5 log-points)
    stayers = stayers[stayers["d_log_wage"].abs() <= 1.5].copy()

    # ── Band definitions ───────────────────────────────────────────────────
    # Treatment: workers earning near old MW (below new MW)
    # Control: workers earning above new MW but below 1.5*MW_new
    treat_lo = 0.80 * mw_old
    treat_hi = mw_new              # strict: < MW_new
    ctrl_lo  = mw_new              # at or above new MW
    ctrl_hi  = 1.50 * mw_new      # cap at 1.5x

    treat_m = stayers["wage_pre"].between(treat_lo, treat_hi - 0.01)
    ctrl_m  = stayers["wage_pre"].between(ctrl_lo,  ctrl_hi)

    # ── Run by formality group ─────────────────────────────────────────────
    results = {}
    print(f"\n{'─'*72}")
    print(f"  {event_name}  MW: S/{mw_old}->S/{mw_new}  (+{(mw_new/mw_old-1)*100:.1f}%)")

    for group, label, formal_val in [
        ("informal", "INFORMAL (lighthouse)", 0),
        ("formal",   "FORMAL  (benchmark)",   1),
    ]:
        fmask = (stayers["formal_pre"] == formal_val)
        sub   = stayers[fmask]
        t_sub = sub[treat_m[sub.index]]
        c_sub = sub[ctrl_m[sub.index]]

        n_t = len(t_sub)
        n_c = len(c_sub)
        print(f"\n  {label}:")
        print(f"    n_treat={n_t}  n_ctrl={n_c}")

        if n_t < 10 or n_c < 10:
            print("    SKIP (n < 10)")
            results[group] = (np.nan, np.nan, np.nan, 0)
            continue

        # Descriptive
        print(f"    Treat: mean_wage_pre=S/{t_sub['wage_pre'].mean():.0f}  "
              f"mean_Δlog={t_sub['d_log_wage'].mean():+.3f} ({np.exp(t_sub['d_log_wage'].mean())-1:+.1%})")
        print(f"    Ctrl:  mean_wage_pre=S/{c_sub['wage_pre'].mean():.0f}  "
              f"mean_Δlog={c_sub['d_log_wage'].mean():+.3f} ({np.exp(c_sub['d_log_wage'].mean())-1:+.1%})")

        # DiD regression
        df_did  = pd.concat([t_sub, c_sub])
        treat_v = treat_m[df_did.index].astype(int)
        b, se, p, n = ols_beta(df_did["d_log_wage"], treat_v)
        stars = ("***" if p < 0.01 else " **" if p < 0.05
                 else "  *" if p < 0.10 else "    ")
        print(f"    DiD β = {b:+.4f}  SE={se:.4f}  p={p:.3f}  {stars}")
        print(f"    → {np.exp(b)-1:+.1%} wage premium for treated vs ctrl")
        results[group] = (b, se, p, n)

    b_inf, se_inf, p_inf, n_inf = results.get("informal",   (np.nan,)*4)
    b_for, se_for, p_for, n_for = results.get("formal",     (np.nan,)*4)
    if not np.isnan(b_inf) and not np.isnan(b_for):
        diff = b_inf - b_for
        print(f"\n  LIGHTHOUSE vs FORMAL DIFFERENCE: {diff:+.4f} ({np.exp(diff)-1:+.1%})")
        if diff > 0:
            print("    Informal pass-through > Formal → STRONG lighthouse")
        elif diff > -0.02:
            print("    Informal pass-through ≈ Formal → lighthouse present")
        else:
            print("    Informal pass-through < Formal → weaker lighthouse")

    return {
        "event":   event_name,
        "mw_old":  mw_old, "mw_new": mw_new,
        "mw_pct":  round((mw_new / mw_old - 1) * 100, 1),
        # informal (lighthouse)
        "n_inf_treat": results.get("informal", (np.nan,)*4)[3],
        "b_inf":   b_inf, "se_inf":  se_inf, "p_inf":  p_inf,
        "pct_inf": round((np.exp(b_inf) - 1) * 100, 1) if not np.isnan(b_inf) else np.nan,
        # formal (benchmark)
        "n_for_treat": results.get("formal", (np.nan,)*4)[3],
        "b_for":   b_for, "se_for":  se_for, "p_for":  p_for,
        "pct_for": round((np.exp(b_for) - 1) * 100, 1) if not np.isnan(b_for) else np.nan,
        # difference
        "b_diff":  round(b_inf - b_for, 4) if not (np.isnan(b_inf) or np.isnan(b_for)) else np.nan,
    }


# ── RUN ─────────────────────────────────────────────────────────────────────
print("\n" + "#"*72)
print("  LIGHTHOUSE EFFECT — Informal Wage Pass-Through")
print("  MW as reference price in informal wage bargaining")
print("  Def B formality: p222 in {1,2,3}, validated mean|gap|=2.7pp vs INEI")
print("#"*72)

summary = []
for name, mw_old, mw_new, c_pre, c_post in MW_EVENTS:
    r = run_event(name, mw_old, mw_new, c_pre, c_post)
    if r:
        summary.append(r)


# ── SUMMARY TABLE ───────────────────────────────────────────────────────────
print(f"\n\n{'#'*72}")
print("  LIGHTHOUSE SUMMARY")
print(f"{'#'*72}")
print(f"  {'Event':<12} {'MW%':>5}  "
      f"{'β_INFORMAL':>11} {'%':>7} {'p':>6}  "
      f"{'β_FORMAL':>10} {'%':>7} {'p':>6}  "
      f"{'b_diff':>7}  note")
print("  " + "─"*80)


def sig(p):
    if np.isnan(p):
        return "   "
    return "***" if p < 0.01 else " **" if p < 0.05 else "  *" if p < 0.10 else "    "


for r in summary:
    note = ""
    if not np.isnan(r["b_inf"]) and not np.isnan(r["b_for"]):
        if r["b_diff"] > 0:
            note = "LH>formal"
        elif r["b_diff"] > -0.02:
            note = "LH~formal"
        else:
            note = "LH<formal"
    print(f"  {r['event']:<12} {r['mw_pct']:>4.1f}%  "
          f"{r['b_inf']:>+11.4f} {r['pct_inf']:>+6.1f}%{sig(r['p_inf'])} "
          f"{r['b_for']:>+10.4f} {r['pct_for']:>+6.1f}%{sig(r['p_for'])} "
          f"{r['b_diff']:>+7.4f}  {note}")

df_s = pd.DataFrame(summary)
print("  " + "─"*80)

# Pooled IVW — informal
mask_inf = df_s["se_inf"].notna() & (df_s["se_inf"] > 0)
if mask_inf.sum() > 1:
    w    = 1 / df_s.loc[mask_inf, "se_inf"]**2
    ivw  = np.average(df_s.loc[mask_inf, "b_inf"], weights=w)
    w2   = (w**2).sum() / w.sum()**2
    se_p = np.sqrt(w2) * (1 / w.mean())   # approximate pooled SE
    print(f"\n  Pooled IVW informal (lighthouse): β = {ivw:+.4f}  "
          f"({np.exp(ivw)-1:+.1%})")

# Pooled IVW — formal
mask_for = df_s["se_for"].notna() & (df_s["se_for"] > 0)
if mask_for.sum() > 1:
    w   = 1 / df_s.loc[mask_for, "se_for"]**2
    ivw = np.average(df_s.loc[mask_for, "b_for"], weights=w)
    print(f"  Pooled IVW formal  (benchmark):  β = {ivw:+.4f}  "
          f"({np.exp(ivw)-1:+.1%})")

# Save
df_s.to_csv(SAVE_DIR / "epe_lighthouse_wage_results.csv", index=False)
print(f"\n  Saved -> data/results/epe_lighthouse_wage_results.csv")

print(f"\n\n{'#'*72}")
print("  INTERPRETATION GUIDE")
print(f"{'#'*72}")
print("""
  β_INFORMAL (lighthouse): wage change for informal workers near-vs-above MW.
    Positive = informal workers earning near old MW got relative wage gains
    when MW was raised → MW reference price effect confirmed.

  β_FORMAL (benchmark): wage change for formal workers near-vs-above MW.
    This is the standard MW compliance effect for the formal sector.

  b_diff = β_informal - β_formal:
    > 0 → lighthouse exceeds formal pass-through (strong anchoring)
    ≈ 0 → similar pass-through (MW equally anchors both sectors)
    < 0 → formal sector compliance dominates (lighthouse weak in this event)

  Literature context:
    Maloney (2004): lighthouse dominates in high-informality LAC economies
    Lombardo et al. (2024): 6 LAC countries confirm lighthouse, Peru included
    Khamis (2013): Argentina informal β > formal β
    Expected Peru finding: moderate lighthouse (~0.05-0.10) given ~60% informality
""")
