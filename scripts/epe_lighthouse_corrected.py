"""
epe_lighthouse_corrected.py
Track 2 Lighthouse: MW effect on formality transitions.

FORMALITY DEFINITION: Def B — Formal = p222 in {1, 2, 3}
  Validated against INEI published Lima Metro rates: mean |gap| = 2.7pp
  All 9 event years within ±5pp of INEI official rate.

VARIABLE MAPPING (verified from data audit, 2003/2012/2022):
  p206: 2=Independiente, 3=Empleado, 4=Obrero
  p222: 1=EsSalud, 2=Seguro Privado, 3=Ambos = FORMAL
        4=Otro, 5=No afiliado, 6=SIS(2022 unlabeled) = INFORMAL
  ocu200: 1=Ocupado
  ingprin: principal income (monthly, soles)

PANEL MATCHING: conglome + vivienda + hogar + codperso (unique person ID)
TREATMENT BAND: ingprin_pre in [0.85*MW_old, MW_new)
CONTROL BAND:   ingprin_pre in [1.05*MW_new, 2.0*MW_old)

LIGHTHOUSE QUESTION: Among workers near the old MW threshold who kept
their job, did they become MORE formal after the MW increase?
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pathlib
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm

OUT = pathlib.Path("D:/Nexus/nexus/data/raw/epe/srienaho")
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

FORMAL_CODES = {1.0, 2.0, 3.0}   # EsSalud, Seguro Privado, Ambos


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


def is_formal(p222_series):
    return pd.to_numeric(p222_series, errors="coerce").isin(FORMAL_CODES)


def run_lighthouse(event_name, mw_old, mw_new, code_pre, code_post):
    pre  = load_sav(code_pre)
    post = load_sav(code_post)
    if pre is None or post is None:
        print(f"\nSKIP {event_name}: file not found")
        return None

    pre["pid"]  = make_pid(pre)
    post["pid"] = make_pid(post)

    # ── Pre-wave: employed workers with valid income ────────────────────────
    pre_emp = pre[
        (pd.to_numeric(pre["ocu200"], errors="coerce") == 1) &
        (pd.to_numeric(pre.get("ingprin", pd.Series(dtype=float)),
                       errors="coerce") > 0)
    ].copy()

    pre_emp["ingprin_num"] = pd.to_numeric(pre_emp["ingprin"], errors="coerce")
    pre_emp["formal_pre"]  = is_formal(pre_emp["p222"]).astype(int)
    pre_emp["occ"]         = pd.to_numeric(pre_emp["p206"], errors="coerce")

    # ── Treatment / control bands ──────────────────────────────────────────
    treat_lo = 0.85 * mw_old
    treat_hi = mw_new         # < MW_new (earners near old threshold)
    ctrl_lo  = 1.05 * mw_new  # above new MW (not directly affected)
    ctrl_hi  = 2.0  * mw_old  # cap to avoid very-high-earner noise

    treat_mask = pre_emp["ingprin_num"].between(treat_lo, treat_hi - 0.01)
    ctrl_mask  = pre_emp["ingprin_num"].between(ctrl_lo,  ctrl_hi)

    pre_band = pre_emp[treat_mask | ctrl_mask].copy()
    pre_band["treat"] = treat_mask[pre_band.index].astype(int)  # 1=treat, 0=ctrl

    # ── Panel merge with post-wave ─────────────────────────────────────────
    post_sub = post[["pid", "ocu200", "p222"]].copy()
    post_sub["employed_post"] = (
        pd.to_numeric(post_sub["ocu200"], errors="coerce") == 1
    ).astype(int)
    post_sub["formal_post"] = is_formal(post_sub["p222"]).astype(int)
    post_sub = post_sub.drop(columns=["ocu200", "p222"])

    merged = pre_band.merge(post_sub, on="pid", how="inner")

    n_treat = (merged["treat"] == 1).sum()
    n_ctrl  = (merged["treat"] == 0).sum()

    # ── Stayers: employed in both waves ────────────────────────────────────
    stayers = merged[merged["employed_post"] == 1].copy()
    n_st_t  = (stayers["treat"] == 1).sum()
    n_st_c  = (stayers["treat"] == 0).sum()

    print(f"\n{'─'*72}")
    print(f"  {event_name}  MW: S/{mw_old}->S/{mw_new}  (+{(mw_new/mw_old-1)*100:.1f}%)")
    print(f"  Band matched: treat={n_treat}, ctrl={n_ctrl}"
          f"  -> stayers: treat={n_st_t}, ctrl={n_st_c}")

    if n_st_t < 20 or n_st_c < 20:
        print("  SKIP: insufficient stayers (< 20)")
        return None

    # ── Raw DiD ────────────────────────────────────────────────────────────
    def frate(df, treat_val):
        s = df[df["treat"] == treat_val]
        return s["formal_pre"].mean(), s["formal_post"].mean(), len(s)

    fp_t, fq_t, nt = frate(stayers, 1)
    fp_c, fq_c, nc = frate(stayers, 0)
    delta_t = fq_t - fp_t
    delta_c = fq_c - fp_c
    did_raw = delta_t - delta_c

    print(f"\n  Formality rate (Def B: p222 in {{1,2,3}}):")
    print(f"    Treat (n={nt}): pre={fp_t:.1%}  post={fq_t:.1%}  delta={delta_t:+.1%}")
    print(f"    Ctrl  (n={nc}): pre={fp_c:.1%}  post={fq_c:.1%}  delta={delta_c:+.1%}")
    print(f"    Raw DiD:        {did_raw:+.1%}")

    # ── OLS DiD: formal_post ~ treat + formal_pre ──────────────────────────
    X = sm.add_constant(stayers[["treat", "formal_pre"]])
    y = stayers["formal_post"]
    try:
        res  = sm.OLS(y, X).fit(cov_type="HC1")
        beta = res.params["treat"]
        se   = res.bse["treat"]
        pval = res.pvalues["treat"]
        stars = ("***" if pval < 0.01 else "**" if pval < 0.05
                 else "*" if pval < 0.10 else "   ")
        print(f"  OLS DiD:  beta={beta:+.4f}  SE={se:.4f}  p={pval:.3f}  {stars}")
    except Exception as e:
        beta, se, pval = np.nan, np.nan, np.nan
        print(f"  OLS failed: {e}")

    # ── Breakdown by occupation ────────────────────────────────────────────
    print(f"\n  DiD by occupation (stayers):")
    for code, name in [(3, "empleado(3)"), (4, "obrero(4)"), (2, "independiente(2)")]:
        sub = stayers[stayers["occ"] == code]
        t   = sub[sub["treat"] == 1]
        c   = sub[sub["treat"] == 0]
        if len(t) < 5 or len(c) < 5:
            continue
        d = (t["formal_post"].mean() - t["formal_pre"].mean()) - \
            (c["formal_post"].mean() - c["formal_pre"].mean())
        print(f"    {name:<22}: n_treat={len(t):3d}  n_ctrl={len(c):3d}  DiD={d:+.1%}")

    # ── Transition matrix (treated stayers) ────────────────────────────────
    ts = stayers[stayers["treat"] == 1]
    if len(ts) > 0:
        i2f = ((ts["formal_pre"] == 0) & (ts["formal_post"] == 1)).sum()
        f2i = ((ts["formal_pre"] == 1) & (ts["formal_post"] == 0)).sum()
        ii  = ((ts["formal_pre"] == 0) & (ts["formal_post"] == 0)).sum()
        ff  = ((ts["formal_pre"] == 1) & (ts["formal_post"] == 1)).sum()
        print(f"\n  Transitions (treated stayers, n={len(ts)}):")
        print(f"    Informal -> Formal:  {i2f:3d} ({i2f/len(ts)*100:.1f}%)")
        print(f"    Formal   -> Informal:{f2i:3d} ({f2i/len(ts)*100:.1f}%)")
        print(f"    Stay Informal:       {ii:3d}  ({ii/len(ts)*100:.1f}%)")
        print(f"    Stay Formal:         {ff:3d}  ({ff/len(ts)*100:.1f}%)")

    return {
        "event": event_name, "mw_old": mw_old, "mw_new": mw_new,
        "mw_pct": round((mw_new / mw_old - 1) * 100, 1),
        "n_st_treat": int(n_st_t), "n_st_ctrl": int(n_st_c),
        "formal_pre_treat": round(fp_t, 4), "formal_post_treat": round(fq_t, 4),
        "formal_pre_ctrl":  round(fp_c, 4), "formal_post_ctrl":  round(fq_c, 4),
        "delta_treat": round(delta_t, 4), "delta_ctrl": round(delta_c, 4),
        "did_raw": round(did_raw, 4),
        "beta_ols": round(beta, 4) if not np.isnan(beta) else np.nan,
        "se_ols":   round(se, 4)   if not np.isnan(se)   else np.nan,
        "pval_ols": round(pval, 4) if not np.isnan(pval) else np.nan,
    }


# ── RUN ─────────────────────────────────────────────────────────────────────
print("\n" + "#"*72)
print("  LIGHTHOUSE (Track 2) — CORRECTED")
print("  MW Effect on Worker Formality Transitions")
print("  Formality: Def B (p222 in {1,2,3}), validated: mean|gap|=2.7pp vs INEI")
print("#"*72)

summary = []
for name, mw_old, mw_new, c_pre, c_post in MW_EVENTS:
    r = run_lighthouse(name, mw_old, mw_new, c_pre, c_post)
    if r:
        summary.append(r)


# ── SUMMARY TABLE ───────────────────────────────────────────────────────────
print(f"\n\n{'#'*72}")
print("  LIGHTHOUSE SUMMARY — Formality DiD (stayers only)")
print(f"{'#'*72}")
print(f"  {'Event':<12} {'N_tr':>5} {'N_ct':>5}  "
      f"{'delta_T':>8} {'delta_C':>8}  {'DiD_raw':>8} {'beta_ols':>9} {'p':>6}  sig")
print("  " + "─"*72)


def sig(p):
    if np.isnan(p):
        return "   "
    return "***" if p < 0.01 else " **" if p < 0.05 else "  *" if p < 0.10 else "   "


for r in summary:
    print(f"  {r['event']:<12} {r['n_st_treat']:>5} {r['n_st_ctrl']:>5}  "
          f"{r['delta_treat']:>+8.1%} {r['delta_ctrl']:>+8.1%}  "
          f"{r['did_raw']:>+8.1%} {r['beta_ols']:>+9.4f} {r['pval_ols']:>6.3f}  "
          f"{sig(r['pval_ols'])}")

df_s = pd.DataFrame(summary)
if len(df_s) > 0:
    print("  " + "─"*72)
    # Weighted mean (by n_st_treat)
    w = df_s["n_st_treat"].values
    wdid = np.average(df_s["did_raw"], weights=w)
    print(f"  Weighted mean DiD (by n_treat): {wdid:+.1%}")

    # IVW pooled beta
    mask = df_s["se_ols"].notna() & (df_s["se_ols"] > 0)
    if mask.sum() > 1:
        wts = 1 / df_s.loc[mask, "se_ols"]**2
        ivw = np.average(df_s.loc[mask, "beta_ols"], weights=wts)
        print(f"  Pooled IVW beta:                {ivw:+.4f}")

    df_s.to_csv(SAVE_DIR / "epe_lighthouse_corrected.csv", index=False)
    print(f"\n  Saved -> data/results/epe_lighthouse_corrected.csv")

print(f"\n\n{'#'*72}")
print("  INTERPRETATION NOTES")
print(f"{'#'*72}")
print("  Positive DiD = treated workers became MORE formal relative to control.")
print("  Negative DiD = treated workers became LESS formal (net informalization).")
print("  Panel matching captures workers who stay employed in BOTH waves.")
print("  Workers who lose jobs post-MW (captured by employment DiD / Lee bounds)")
print("  are NOT included here.")
print("  Formality def B (EsSalud + private + ambos) validated vs INEI: 2.7pp gap.")
print("")
print("  PREVIOUS LIGHTHOUSE WAS WRONG:")
print("    epe_panel_did_all.py used p213 for formality -- p213 = job search,")
print("    NOT health insurance. Those results are INVALID and discarded.")
print("  EMPLOYMENT DiD (Track 4) and LEE BOUNDS remain VALID (no formality var).")
