"""
mw_multi_event_regional_did.py
================================
Multi-event regional DiD for minimum wage effects using ENAHO cross-section.

Events:
  7:  Pre=2015 (MW=S/750), Post=2016/2017 (MW raised to S/850 Dec-2015,
      then S/930 May-2016). Kaitz_pre = 750 / dept_median_formal_2015.
  8:  Pre=2017 (MW=S/930), Post=2018/2019 (MW CONSTANT — parallel trend check).
      Kaitz_pre = 930 / dept_median_formal_2017.
  9:  Pre=2021 (MW=S/930), Post=2022/2023 (MW raised to S/1,025 May-2022).
      [Results read from existing JSON — already estimated with ENAHO Panel 978]

Pool events 7+8+9 via inverse-variance weighting (IVW) with Cochran Q.

Specification:
  Y_idt = alpha_d + gamma_t + beta(Post_t x Kaitz_d_pre) + Edad + Edad^2 + Sexo + eps

Outputs:
  exports/data/mw_canonical_results.json  (multi-event canonical)
  Copy to: D:/qhawarina/public/assets/data/mw_canonical_results.json
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import pathlib
import warnings
import numpy as np
import pandas as pd
import pyreadstat
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT   = pathlib.Path("D:/Nexus/nexus")
CS_DIR = ROOT / "data/raw/enaho/cross_section"    # cross-section modules
OUT    = ROOT / "exports/data"
OUT.mkdir(parents=True, exist_ok=True)

EXISTING_REGIONAL_JSON = ROOT / "exports/data/mw_regional_did_results.json"
OUT_JSON               = OUT / "mw_canonical_results.json"
OUT_QHAW               = pathlib.Path("D:/qhawarina/public/assets/data/mw_canonical_results.json")

# ── MW schedule ────────────────────────────────────────────────────────────────
MW_YEAR = {
    2015: 750,   # full year
    2016: 930,   # Dec-2015 → 850, May-2016 → 930 (annual survey captures mixture)
    2017: 930,
    2018: 930,
    2019: 930,
    2021: 930,
    2022: 1025,  # raised May 2022
    2023: 1025,
}

DEPT_NAMES = {
    "01": "Amazonas",     "02": "Ancash",       "03": "Apurimac",
    "04": "Arequipa",     "05": "Ayacucho",     "06": "Cajamarca",
    "07": "Callao",       "08": "Cusco",        "09": "Huancavelica",
    "10": "Huanuco",      "11": "Ica",          "12": "Junin",
    "13": "La Libertad",  "14": "Lambayeque",   "15": "Lima",
    "16": "Loreto",       "17": "Madre de Dios","18": "Moquegua",
    "19": "Pasco",        "20": "Piura",        "21": "Puno",
    "22": "San Martin",   "23": "Tacna",        "24": "Tumbes",
    "25": "Ucayali",
}

LOAD_COLS = [
    "conglome", "vivienda", "hogar", "codperso", "ubigeo",
    "ocu500", "ocupinf", "i524a1", "fac500a",
    "p207", "p208a",
]

# ── Helper: find module-05 .dta file ──────────────────────────────────────────
def find_m05(year: int) -> pathlib.Path | None:
    folder = CS_DIR / f"modulo_05_{year}"
    if not folder.exists():
        return None
    for pat in [f"enaho01a-{year}-500.dta", f"enaho01a-{year}_500.dta",
                f"Enaho01a-{year}-500.dta"]:
        p = folder / pat
        if p.exists():
            return p
    candidates = list(folder.glob("*.dta"))
    # Filter out lookup tables (small files)
    candidates = [f for f in candidates if f.stat().st_size > 5_000_000]
    return candidates[0] if candidates else None


# ── Load one ENAHO cross-section year ─────────────────────────────────────────
def load_year(year: int) -> pd.DataFrame | None:
    path = find_m05(year)
    if path is None:
        print(f"  [{year}] Module 05 NOT FOUND")
        return None
    print(f"  [{year}] Loading {path.name} ...", end=" ", flush=True)
    df, _ = pyreadstat.read_dta(str(path), apply_value_formats=False,
                                usecols=[c.upper() for c in LOAD_COLS] +
                                        [c.lower() for c in LOAD_COLS] +
                                        LOAD_COLS,
                                row_limit=0)
    df.columns = [c.lower() for c in df.columns]
    # Keep only columns we need (handle duplicates from case mismatch)
    df = df.loc[:, ~df.columns.duplicated()]
    for col in LOAD_COLS:
        if col not in df.columns:
            print(f"\n    MISSING col: {col}")

    df["year"] = year
    df["dept"] = df["ubigeo"].astype(str).str.zfill(6).str[:2]
    # Keep valid departments
    df = df[df["dept"].isin(DEPT_NAMES.keys())].copy()
    # Employment: ocu500 == 1 means has principal occupation
    df["employed"] = (df["ocu500"] == 1).astype(int)
    # Formality: ocupinf == 2 = formal (INEI official)
    df["formal"] = ((df["ocu500"] == 1) & (df["ocupinf"] == 2)).astype(int)
    # Wage: annual imputed income / 12
    df["wage_monthly"] = df["i524a1"] / 12
    df["wage_monthly"] = df["wage_monthly"].clip(lower=0)
    df["log_wage"] = np.where(df["wage_monthly"] > 0,
                              np.log(df["wage_monthly"]), np.nan)
    # Age
    df["edad"]    = pd.to_numeric(df["p208a"], errors="coerce")
    df["edad_sq"] = df["edad"] ** 2
    # Sex (1=male, 2=female → recode to 0/1)
    df["sexo"] = (df["p207"] == 2).astype(int)
    # Weight
    df["fac500a"] = pd.to_numeric(df["fac500a"], errors="coerce").fillna(1)

    # Restrict to working-age (18-70)
    df = df[(df["edad"] >= 18) & (df["edad"] <= 70)].copy()
    print(f"{len(df):,} obs ({df['formal'].sum():,} formal)")
    return df


# ── Compute department Kaitz in pre-year ──────────────────────────────────────
def compute_kaitz(df_pre: pd.DataFrame, mw_pre: float) -> pd.DataFrame:
    """Weighted median formal wage by dept in pre-year → Kaitz = MW/median."""
    formal = df_pre[(df_pre["employed"] == 1) & (df_pre["ocupinf"] == 2) &
                    (df_pre["wage_monthly"] > 0)].copy()
    rows = []
    for dept, grp in formal.groupby("dept"):
        wages = grp["wage_monthly"].values
        wts   = grp["fac500a"].values
        # Weighted median
        order = np.argsort(wages)
        wages_s, wts_s = wages[order], wts[order]
        cumw = np.cumsum(wts_s)
        med  = wages_s[np.searchsorted(cumw, cumw[-1] / 2)]
        rows.append({
            "dept": dept,
            "median_formal": med,
            "kaitz_pre": round(mw_pre / med, 4),
            "n_formal": len(grp),
        })
    kaitz_df = pd.DataFrame(rows)
    print(f"  Kaitz pre: n_depts={len(kaitz_df)} "
          f"range=[{kaitz_df.kaitz_pre.min():.3f}, {kaitz_df.kaitz_pre.max():.3f}] "
          f"median={kaitz_df.kaitz_pre.median():.3f}")
    return kaitz_df


# ── Run single-event DiD ──────────────────────────────────────────────────────
def run_event_did(event_id: int, label: str, mw_old: float,
                  pre_year: int, post_years: list[int],
                  dfs: dict[int, pd.DataFrame]) -> dict:
    print(f"\n{'='*60}")
    print(f"EVENT {event_id}: {label}")
    print(f"  Pre: {pre_year}  Post: {post_years}  MW_pre={mw_old}")

    df_pre = dfs.get(pre_year)
    if df_pre is None:
        print("  [SKIP] Missing pre-year data")
        return {}

    # Kaitz
    kaitz_df = compute_kaitz(df_pre, mw_old)

    # Stack years
    frames = [df_pre.copy()]
    for py in post_years:
        if dfs.get(py) is not None:
            frames.append(dfs[py].copy())
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.merge(kaitz_df[["dept", "kaitz_pre"]], on="dept", how="inner")

    # Post dummy (1 for post-treatment years)
    df_all["post"] = df_all["year"].isin(post_years).astype(int)
    df_all["post_kaitz"] = df_all["post"] * df_all["kaitz_pre"]

    results = {
        "id": event_id,
        "name": label,
        "mw_old": mw_old,
        "year_pre": pre_year,
        "year_post": post_years,
        "n_obs": len(df_all),
        "n_departments": int(df_all["dept"].nunique()),
        "kaitz_range": [round(float(kaitz_df.kaitz_pre.min()), 3),
                        round(float(kaitz_df.kaitz_pre.max()), 3)],
        "note": ("PARALLEL TREND CHECK — no MW change during this period"
                 if mw_old == MW_YEAR.get(post_years[0], mw_old) else ""),
    }

    def run_ols(outcome: str, sample_mask=None, label_out="") -> dict | None:
        df_tmp = df_all.copy()
        if sample_mask is not None:
            df_tmp = df_tmp[sample_mask(df_tmp)]
        df_tmp = df_tmp[["dept", "year", "post_kaitz", "kaitz_pre",
                          "edad", "edad_sq", "sexo", "fac500a", outcome]].dropna()
        if len(df_tmp) < 500 or df_tmp["dept"].nunique() < 5:
            return None
        try:
            formula = f"{outcome} ~ C(dept) + C(year) + post_kaitz + edad + edad_sq + sexo"
            m = smf.wls(formula, data=df_tmp, weights=df_tmp["fac500a"]).fit(
                cov_type="cluster",
                cov_kwds={"groups": df_tmp["dept"].astype(str)},
                disp=0
            )
            ci = m.conf_int().loc["post_kaitz"]
            return {
                "beta":    round(float(m.params["post_kaitz"]), 5),
                "se":      round(float(m.bse["post_kaitz"]),    5),
                "ci_low":  round(float(ci.iloc[0]),             5),
                "ci_high": round(float(ci.iloc[1]),             5),
                "p":       round(float(m.pvalues["post_kaitz"]), 4),
                "n":       len(df_tmp),
            }
        except Exception as e:
            print(f"    OLS error ({label_out}): {e}")
            return None

    def run_event_study(outcome: str, sample_mask=None) -> list | None:
        df_tmp = df_all.copy()
        if sample_mask is not None:
            df_tmp = df_tmp[sample_mask(df_tmp)]
        rows_es = []
        for py in post_years:
            df_tmp[f"y{py}_kaitz"] = (df_tmp["year"] == py).astype(int) * df_tmp["kaitz_pre"]
        yr_terms = " + ".join([f"y{py}_kaitz" for py in post_years])
        df_tmp = df_tmp[["dept", "year", "kaitz_pre", "edad", "edad_sq", "sexo",
                          "fac500a", outcome] + [f"y{py}_kaitz" for py in post_years]].dropna()
        if len(df_tmp) < 500:
            return None
        try:
            formula = f"{outcome} ~ C(dept) + C(year) + {yr_terms} + edad + edad_sq + sexo"
            m = smf.wls(formula, data=df_tmp, weights=df_tmp["fac500a"]).fit(
                cov_type="cluster",
                cov_kwds={"groups": df_tmp["dept"].astype(str)},
                disp=0
            )
            for py in post_years:
                key = f"y{py}_kaitz"
                if key in m.params.index:
                    ci = m.conf_int().loc[key]
                    rows_es.append({
                        "year": py,
                        "beta":   round(float(m.params[key]),     5),
                        "se":     round(float(m.bse[key]),        5),
                        "ci_low": round(float(ci.iloc[0]),        5),
                        "ci_high":round(float(ci.iloc[1]),        5),
                        "p":      round(float(m.pvalues[key]),    4),
                    })
            return rows_es
        except Exception as e:
            print(f"    Event study error: {e}")
            return None

    # ── Employment ─────────────────────────────────────────────────────────────
    print("  Running employment DiD...")
    r_emp = run_ols("employed")
    if r_emp:
        print(f"    beta={r_emp['beta']:+.4f}  se={r_emp['se']:.4f}  "
              f"p={r_emp['p']:.3f}  n={r_emp['n']:,}")
    results["employment"] = r_emp or {}

    # ── Formalization ──────────────────────────────────────────────────────────
    print("  Running formalization DiD (employed workers)...")
    r_frm = run_ols("formal", sample_mask=lambda d: d["employed"] == 1)
    if r_frm:
        print(f"    beta={r_frm['beta']:+.4f}  se={r_frm['se']:.4f}  "
              f"p={r_frm['p']:.3f}  n={r_frm['n']:,}")
    results["formalization"] = r_frm or {}

    # ── Log wages ──────────────────────────────────────────────────────────────
    print("  Running log-wage DiD (formal employed)...")
    r_wg = run_ols("log_wage",
                   sample_mask=lambda d: (d["formal"] == 1) & (d["log_wage"].notna()))
    if r_wg:
        print(f"    beta={r_wg['beta']:+.4f}  se={r_wg['se']:.4f}  "
              f"p={r_wg['p']:.3f}  n={r_wg['n']:,}")
    results["log_wage"] = r_wg or {}

    # ── Event study ────────────────────────────────────────────────────────────
    print("  Running event study...")
    es_emp = run_event_study("employed")
    es_frm = run_event_study("formal", sample_mask=lambda d: d["employed"] == 1)
    es_wg  = run_event_study("log_wage",
                             sample_mask=lambda d: (d["formal"]==1) & (d["log_wage"].notna()))
    results["event_study"] = {
        "employment":    es_emp or [],
        "formalization": es_frm or [],
        "log_wage":      es_wg  or [],
    }
    if es_emp:
        for row in es_emp:
            print(f"    emp  {row['year']}: beta={row['beta']:+.4f}  "
                  f"se={row['se']:.4f}  p={row['p']:.3f}")
    if es_frm:
        for row in es_frm:
            print(f"    frm  {row['year']}: beta={row['beta']:+.4f}  "
                  f"se={row['se']:.4f}  p={row['p']:.3f}")
    if es_wg:
        for row in es_wg:
            print(f"    wage {row['year']}: beta={row['beta']:+.4f}  "
                  f"se={row['se']:.4f}  p={row['p']:.3f}")

    return results


# ── IVW pooling ───────────────────────────────────────────────────────────────
def pool_ivw(event_results: list[dict], outcome: str) -> dict:
    betas, ses = [], []
    for er in event_results:
        r = er.get(outcome, {})
        if r and "beta" in r and "se" in r and r["se"] > 0:
            betas.append(r["beta"])
            ses.append(r["se"])
    if len(betas) < 2:
        return {}
    betas_a = np.array(betas)
    ses_a   = np.array(ses)
    ws      = 1.0 / ses_a**2
    beta_p  = float(np.sum(ws * betas_a) / np.sum(ws))
    se_p    = float(1.0 / np.sqrt(np.sum(ws)))
    z       = beta_p / se_p
    p_p     = float(2 * (1 - stats.norm.cdf(abs(z))))
    Q       = float(np.sum(ws * (betas_a - beta_p)**2))
    I2      = max(0.0, (Q - (len(betas)-1)) / Q * 100) if Q > 0 else 0.0
    ci_lo   = beta_p - 1.96 * se_p
    ci_hi   = beta_p + 1.96 * se_p
    return {
        "beta":     round(beta_p, 5),
        "se":       round(se_p, 5),
        "ci_low":   round(ci_lo, 5),
        "ci_high":  round(ci_hi, 5),
        "p":        round(p_p, 4),
        "Q":        round(Q, 3),
        "I2":       round(I2, 1),
        "n_events": len(betas),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("MULTI-EVENT MW DiD — REGIONAL IDENTIFICATION (DLR 2010)")
    print("="*60)

    # ── Load cross-section years needed for events 7 and 8 ───────────────────
    print("\n[1] Loading ENAHO cross-section data...")
    dfs = {}
    for yr in [2015, 2016, 2017, 2018, 2019]:
        df = load_year(yr)
        if df is not None:
            dfs[yr] = df

    # ── Event 7: Pre=2015, Post=2016/2017 ────────────────────────────────────
    ev7 = run_event_did(
        event_id=7,
        label="2015-2017 (S/750→S/930: Dec-2015 + May-2016 combined)",
        mw_old=750.0,
        pre_year=2015,
        post_years=[2016, 2017],
        dfs=dfs,
    )

    # ── Event 8: Pre=2017, Post=2018/2019 (NO MW change — trend check) ───────
    ev8 = run_event_did(
        event_id=8,
        label="2017-2019 (MW constant at S/930 — parallel trend check)",
        mw_old=930.0,
        pre_year=2017,
        post_years=[2018, 2019],
        dfs=dfs,
    )

    # ── Event 9: Read from existing JSON ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("EVENT 9: 2021-2023 (S/930→S/1,025) — reading existing results")
    ev9 = {}
    try:
        regional = json.load(open(EXISTING_REGIONAL_JSON))
        # Pull main kaitz results
        def pull_result(outcome_key: str) -> dict:
            r = next((x for x in regional.get("main_kaitz", [])
                      if x["outcome"] == outcome_key), None)
            if not r:
                return {}
            return {
                "beta":   r["beta"],  "se":     r["se"],
                "ci_low": r["ci_lo"], "ci_high": r["ci_hi"],
                "p":      r["pval"],
            }
        def pull_event_study(outcome_key: str) -> list:
            es = next((x for x in regional.get("event_study", [])
                       if x["outcome"] == outcome_key), None)
            if not es:
                return []
            rows = []
            for yr, suf in [(2022, "2022"), (2023, "2023")]:
                rows.append({
                    "year":    yr,
                    "beta":    es[f"beta_{suf}"],
                    "se":      es[f"se_{suf}"],
                    "ci_low":  es[f"ci_lo_{suf}"],
                    "ci_high": es[f"ci_hi_{suf}"],
                    "p":       es[f"pval_{suf}"],
                })
            return rows

        ev9 = {
            "id": 9, "name": "2021-2023 (S/930→S/1,025, May 2022)",
            "mw_old": 930.0, "year_pre": 2021, "year_post": [2022, 2023],
            "n_obs":         regional["main_kaitz"][0].get("N", 0) if regional.get("main_kaitz") else 0,
            "n_departments": regional["main_kaitz"][0].get("N_depts", 25) if regional.get("main_kaitz") else 25,
            "kaitz_range":   [round(regional["kaitz_summary"]["min"], 3),
                              round(regional["kaitz_summary"]["max"], 3)],
            "employment":    pull_result("employed"),
            "formalization": pull_result("formal_v4"),
            "log_wage":      pull_result("log_wage"),
            "event_study": {
                "employment":    pull_event_study("employed"),
                "formalization": pull_event_study("formal_v4"),
                "log_wage":      pull_event_study("log_wage"),
            },
        }
        print(f"  Employment:    beta={ev9['employment'].get('beta','?'):+}  "
              f"p={ev9['employment'].get('p','?')}")
        print(f"  Formalization: beta={ev9['formalization'].get('beta','?'):+}  "
              f"p={ev9['formalization'].get('p','?')}")
        print(f"  Log wage:      beta={ev9['log_wage'].get('beta','?'):+}  "
              f"p={ev9['log_wage'].get('p','?')}")
    except Exception as e:
        print(f"  ERROR reading event 9: {e}")

    # ── Pooled IVW (events 7+9 only — event 8 is trend check, excluded) ──────
    print(f"\n{'='*60}")
    print("POOLED IVW (Events 7 + 9 — real MW changes)")
    real_events = [x for x in [ev7, ev9] if x]
    pooled = {}
    for outcome in ["employment", "formalization", "log_wage"]:
        pooled[outcome] = pool_ivw(real_events, outcome)
        p = pooled[outcome]
        if p:
            print(f"  {outcome:15s}: beta={p['beta']:+.4f}  se={p['se']:.4f}  "
                  f"p={p['p']:.3f}  I2={p['I2']:.1f}%  Q={p['Q']:.2f}")

    # ── Print full comparison table ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("=== MULTI-EVENT MW DiD RESULTS ===")
    all_events_display = [("Event 7 (2015-2017, S/750→930)", ev7),
                          ("Event 8 (2017-2019, trend check)", ev8),
                          ("Event 9 (2021-2023, S/930→1025)", ev9)]
    for outcome in ["employment", "formalization", "log_wage"]:
        print(f"\n{outcome.upper()}:")
        for name, er in all_events_display:
            if not er:
                continue
            r = er.get(outcome, {})
            if r and "beta" in r:
                print(f"  {name}: "
                      f"beta={r['beta']:+.4f}  se={r['se']:.4f}  "
                      f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  "
                      f"p={r['p']:.3f}  n={r.get('n') or '?'}")
            else:
                print(f"  {name}: NO RESULT")
        p = pooled.get(outcome, {})
        if p:
            print(f"  POOLED (IVW, events 7+9): "
                  f"beta={p['beta']:+.4f}  se={p['se']:.4f}  "
                  f"[{p['ci_low']:+.4f}, {p['ci_high']:+.4f}]  "
                  f"p={p['p']:.3f}  I2={p['I2']:.1f}%  Q={p['Q']:.2f}")

    print(f"\nEVENT STUDY COEFFICIENTS:")
    for name, er in all_events_display:
        if not er:
            continue
        es = er.get("event_study", {})
        emp_es = es.get("employment", [])
        frm_es = es.get("formalization", [])
        print(f"\n  {name}:")
        for row in emp_es:
            print(f"    emp  {row['year']}: beta={row['beta']:+.5f}  "
                  f"[{row['ci_low']:+.5f}, {row['ci_high']:+.5f}]  p={row['p']:.3f}")
        for row in frm_es:
            print(f"    frm  {row['year']}: beta={row['beta']:+.5f}  "
                  f"[{row['ci_low']:+.5f}, {row['ci_high']:+.5f}]  p={row['p']:.3f}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_data = {
        "events":  [x for x in [ev7, ev8, ev9] if x],
        "pooled":  pooled,
        "metadata": {
            "specification": "Y_idt = alpha_d + gamma_t + beta(Post_t x Kaitz_d_pre) + Edad + Edad^2 + Sexo + eps",
            "formality_definition": "INEI oficial (ocupinf==2). Event 9 uses V4 (ENAHO Panel 978).",
            "weight_variable": "fac500a",
            "cluster_level": "departamento",
            "data_source_7": "ENAHO cross-section module 500, 2015-2017",
            "data_source_8": "ENAHO cross-section module 500, 2017-2019 (trend check)",
            "data_source_9": "ENAHO Panel 978 (2020-2024), module 1477",
            "n_events": 3,
            "last_updated": "2026-03-15",
            "note_event_8": "MW unchanged 2017-2019; serves as parallel trend test",
            "note_pooled": "IVW pool uses events 7 and 9 only (real MW changes). Event 8 excluded.",
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT_JSON}")

    # Copy to qhawarina
    try:
        import shutil
        shutil.copy(OUT_JSON, OUT_QHAW)
        print(f"Copied: {OUT_QHAW}")
    except Exception as e:
        print(f"Copy to qhawarina FAILED: {e}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
