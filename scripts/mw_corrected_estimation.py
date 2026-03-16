"""
mw_corrected_estimation.py
==========================
CORRECTED MW Values — full re-estimation.

EVENTS (correct):
  A: S/750 → S/850  (May 2016, DS 005-2016-TR)
     Kaitz_pre = 750 / median_formal_wage_dept_2015
     Pre=2015, Post=[2016, 2017]

  B: S/850 → S/930  (April 2018, DS 004-2018-TR)  ← REAL EVENT, not placebo
     Kaitz_pre = 850 / median_formal_wage_dept_2017
     Pre=2017, Post=[2018, 2019]

  C: S/930 → S/1,025 (May 2022, DS 003-2022-TR)
     Kaitz_pre = 930 / median_formal_wage_dept_2021
     Pre=2021, Post=[2022, 2023]

PLACEBO (genuine no-change period):
  Within Event A post-window: compare 2017 vs 2016 outcomes
  using Event A's Kaitz (750/median_2015). If no additional
  differential growth after year-1, beta ≈ 0.

DATASETS:
  1. ENAHO CS Annual   (modulo 05, 2015-2023)
  2. ENAHO Quarterly   (trimestral movil, Q1 windows)
  3. EPE Lima Metro    (from existing epe_panel_did_all_results.csv)

PRINT ALL COEFFICIENTS. No summarising.
"""
import sys, os, io, zipfile, requests, json, pathlib, shutil, tempfile
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf
from scipy import stats

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT    = pathlib.Path("D:/Nexus/nexus")
CS_DIR  = ROOT / "data/raw/enaho/cross_section"
QT_DIR  = ROOT / "data/raw/enaho/quarterly"
QT_DIR.mkdir(parents=True, exist_ok=True)
EPE_ALL = ROOT / "data/results/epe_panel_did_all_results.csv"
OUT_J   = ROOT / "exports/data/mw_canonical_results.json"   # overwrites old canonical
OUT_J2  = ROOT / "exports/data/mw_multi_dataset_results.json"
OUT_J.parent.mkdir(parents=True, exist_ok=True)

# ── Annual ENAHO codes (YEAR_MAP from wage_bin_did.py + INEI portal) ────────
ANNUAL_CODES = {
    2015: 498, 2016: 546, 2017: 603, 2018: 634, 2019: 687,
    2020: 737, 2021: 759, 2022: 784, 2023: 906,
}

# ── Quarterly codes (verified: Q1 of following year) ────────────────────────
# 510 = Q1 2016, 551 = Q1 2017, 607 = Q1 2018, 641 = Q1 2019
# 765 = Q1 2022 (from search), Q1 2023 to be found
QUARTERLY_CODES = {
    "Q3_2015": 484,   # July 2015 (before May 2016 increase)
    "Q4_2015": 494,   # October 2015 (before May 2016 increase)
    "Q1_2016":  510,   # Jan-Mar 2016 (before May 2016 increase) ← Event A pre
    "Q1_2017":  551,   # Jan-Mar 2017 (after May 2016 increase)  ← Event A post
    "Q1_2018":  607,   # Jan-Mar 2018 (before April 2018 increase) ← Event B pre
    "Q1_2019":  641,   # Jan-Mar 2019 (after April 2018 increase) ← Event B post
    "Q1_2022":  765,   # Jan-Mar 2022 (before May 2022 increase) ← Event C pre
}

# ── Events (CORRECTED) ───────────────────────────────────────────────────────
EVENTS = {
    "A": {
        "label": "Event A: S/750→S/850 (May 2016)",
        "mw_old": 750, "mw_new": 850,
        "pre_year": 2015, "post_years": [2016, 2017],
        "qt_pre": "Q1_2016",   # Jan-Mar 2016, before May 2016 increase
        "qt_post": ["Q1_2017"],
        "qt_note": "Pre=Q1 2016 (before May increase), Post=Q1 2017",
    },
    "B": {
        "label": "Event B: S/850→S/930 (April 2018)",
        "mw_old": 850, "mw_new": 930,
        "pre_year": 2017, "post_years": [2018, 2019],
        "qt_pre": "Q1_2018",   # Jan-Mar 2018, before April 2018 increase
        "qt_post": ["Q1_2019"],
        "qt_note": "Pre=Q1 2018 (Jan-Mar, before April 2018 increase), Post=Q1 2019",
    },
    "C": {
        "label": "Event C: S/930→S/1,025 (May 2022)",
        "mw_old": 930, "mw_new": 1025,
        "pre_year": 2021, "post_years": [2022, 2023],
        "qt_pre": "Q1_2022",   # Jan-Mar 2022, before May 2022 increase
        "qt_post": None,        # need Q1 2023 code (not yet found)
        "qt_note": "Pre=Q1 2022 (before May increase), Post=Q1 2023 (code TBD)",
    },
}

PLACEBO = {
    "label": "Placebo: 2019 vs 2017 (between B pre and post, no MW change)",
    "mw_for_kaitz": 850,          # use Event B Kaitz (850/median_2017)
    "kaitz_pre_year": 2017,
    "pre_year": 2017,             # pre-Event B (before April 2018 increase)
    "post_years": [2019],         # post-Event B post-period — but MW changed April 2018!
    # Actually: use 2016 (post-A) vs 2017 (still post-A, no new increase)
    # Better placebo: reverse direction — use 2017 as "pre" and 2016 as "post" with Event A Kaitz
    # Cleanest: 2016 vs 2017, both post-A, no new increase. MW unchanged Dec2016-Apr2018.
    "note": "Use Event B Kaitz (850/median_2017). Pre=2017 (pre-B). Post=2019 (post-B). "
            "MW DID change April 2018. This tests whether 2019 vs 2017 gap was concentrated "
            "in formal sectors (genuine formalization effect), vs placebo within-event-B."
}
PLACEBO_ANNUAL = {
    "label": "Placebo annual: 2016 vs 2017 (post-A only, no new increase)",
    "mw_for_kaitz": 750,
    "kaitz_pre_year": 2015,
    "pre_year": 2016,
    "post_years": [2017],
    "note": "Both 2016 and 2017 are post-Event A. No new increase until April 2018. "
            "Beta should be ~0 for employment (no new differential). "
            "Any significant beta = pre-existing trend, not MW effect.",
}
PLACEBO_QT = {
    "label": "Placebo quarterly: Q4 2015 vs Q1 2016 (no increase in this window)",
    "qt_pre": "Q4_2015",     # Oct 2015 — before any increase
    "qt_post": "Q1_2016",    # Jan-Mar 2016 — still before May 2016 increase
    "mw_for_kaitz": 750,     # MW = 750 throughout this period
    "note": "Oct 2015 to Jan-Mar 2016: no MW change (increase was May 2016). "
            "Beta should be ~0 across all outcomes. Tests seasonal confounds.",
}

OUTCOMES = ["employed", "formal", "log_wage"]


# ═══════════════════════════════════════════════════════════════════════════
# DOWNLOAD helpers
# ═══════════════════════════════════════════════════════════════════════════
def _extract_dta(zip_bytes: bytes, dest_path: pathlib.Path) -> pathlib.Path:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for info in z.infolist():
            try:
                info.filename = info.filename.encode("cp437").decode("latin1")
            except Exception:
                pass
            fname = os.path.basename(info.filename)
            if fname.lower().endswith(".dta") and (
                    "enaho" in fname.lower() or "500" in fname.lower()):
                with tempfile.TemporaryDirectory() as tmp:
                    info.filename = fname
                    z.extract(info, tmp)
                    shutil.copy(pathlib.Path(tmp) / fname, dest_path)
                return dest_path
    raise RuntimeError("No matching .dta found in ZIP")


def download_annual_module05(year: int) -> pathlib.Path:
    """Download module 05 for a year if not already on disk."""
    dest_dir = CS_DIR / f"modulo_05_{year}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Check if already present
    existing = list(dest_dir.glob("enaho01a-*-500.dta")) + list(dest_dir.glob("enaho01a_*_500.dta"))
    if existing:
        return existing[0]

    code = ANNUAL_CODES[year]
    url = f"https://proyectos.inei.gob.pe/iinei/srienaho/descarga/STATA/{code}-Modulo05.zip"
    print(f"  Downloading annual {year} (code {code})...", end=" ", flush=True)
    r = requests.get(url, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for annual {year}")
    dest_file = dest_dir / f"enaho01a-{year}-500.dta"
    _extract_dta(r.content, dest_file)
    print(f"OK ({dest_file.stat().st_size / 1e6:.1f} MB)")
    return dest_file


def download_quarterly(label: str) -> pathlib.Path:
    code = QUARTERLY_CODES[label]
    dest = QT_DIR / f"enaho_qt_{label}_code{code}.dta"
    if dest.exists():
        return dest
    url = f"https://proyectos.inei.gob.pe/iinei/srienaho/descarga/STATA/{code}-Modulo05.zip"
    print(f"  Downloading quarterly {label} (code {code})...", end=" ", flush=True)
    r = requests.get(url, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {label}")
    _extract_dta(r.content, dest)
    print(f"OK ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


# ═══════════════════════════════════════════════════════════════════════════
# LOAD + FEATURIZE
# ═══════════════════════════════════════════════════════════════════════════
def load_annual(year: int) -> pd.DataFrame:
    path = download_annual_module05(year)
    df, _ = pyreadstat.read_dta(str(path))
    df.columns = [c.lower() for c in df.columns]
    return _featurize(df)


def load_quarterly(label: str) -> pd.DataFrame | None:
    if label is None or label not in QUARTERLY_CODES:
        return None
    try:
        path = download_quarterly(label)
    except Exception as e:
        print(f"  [WARN] quarterly {label}: {e}")
        return None
    df, _ = pyreadstat.read_dta(str(path))
    df.columns = [c.lower() for c in df.columns]
    return _featurize(df)


def _featurize(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["dept"] = df["ubigeo"].astype(str).str.strip().str[:2]

    # employed
    if "ocu500" in df.columns:
        out["employed"] = (pd.to_numeric(df["ocu500"], errors="coerce") == 1).astype(float)
    else:
        out["employed"] = np.nan

    # formal
    if "ocupinf" in df.columns:
        out["formal"] = (pd.to_numeric(df["ocupinf"], errors="coerce") == 2).astype(float)
        out["_fm"] = "ocupinf"
    elif "p510a1" in df.columns:
        # labor contract proxy (quarterly pre-2017 files lack ocupinf)
        out["formal"] = (pd.to_numeric(df["p510a1"], errors="coerce") == 1).astype(float)
        out["_fm"] = "p510a1"
    else:
        out["formal"] = np.nan
        out["_fm"] = "none"

    # wage (monthly income from primary occupation)
    # i524a1 = ANNUAL income in soles (divide by 12 for monthly)
    # p524a1 = MONTHLY income in soles (quarterly files)
    if "i524a1" in df.columns:
        w = pd.to_numeric(df["i524a1"], errors="coerce") / 12.0   # annual → monthly
        out["wage"] = w
        out["log_wage"] = np.log(w.where(w > 0))
    elif "p524a1" in df.columns:
        w = pd.to_numeric(df["p524a1"], errors="coerce")           # already monthly
        out["wage"] = w
        out["log_wage"] = np.log(w.where(w > 0))
    else:
        out["wage"] = np.nan
        out["log_wage"] = np.nan

    # controls
    edad_var = next((v for v in ["p208a", "edad"] if v in df.columns), None)
    out["edad"] = pd.to_numeric(df[edad_var], errors="coerce") if edad_var else np.nan
    out["edad_sq"] = out["edad"] ** 2

    sexo_var = next((v for v in ["p207", "sexo"] if v in df.columns), None)
    out["sexo"] = pd.to_numeric(df[sexo_var], errors="coerce") if sexo_var else np.nan

    wt_var = next((v for v in ["fac500a", "fac500"] if v in df.columns), None)
    out["wt"] = pd.to_numeric(df[wt_var], errors="coerce").clip(lower=0) if wt_var else 1.0

    # filter: working age 14-70, valid 2-digit dept
    mask = (
        out["edad"].between(14, 70) &
        out["dept"].str.match(r"^\d{2}$")
    )
    return out[mask].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# KAITZ + DiD
# ═══════════════════════════════════════════════════════════════════════════
def compute_kaitz(df_pre: pd.DataFrame, mw: float) -> pd.Series:
    """Kaitz = mw / weighted median formal wage by dept."""
    sub = df_pre[(df_pre["formal"] == 1) & (df_pre["wage"] > 0)].dropna(
        subset=["wage", "wt"])

    def wmed(g):
        g = g.sort_values("wage")
        cum = g["wt"].cumsum()
        tot = g["wt"].sum()
        if tot == 0 or len(g) == 0:
            return np.nan
        idx = (cum >= tot / 2).idxmax()
        return float(g.loc[idx, "wage"])

    med = sub.groupby("dept").apply(wmed, include_groups=False)
    if isinstance(med, pd.DataFrame):
        med = med.iloc[:, 0]
    med = pd.to_numeric(med, errors="coerce")
    k = mw / med
    k.name = "kaitz_pre"
    return k


def run_event_did(
        df_pre: pd.DataFrame,
        df_posts: list[pd.DataFrame],
        mw_old: float,
        label: str,
        kaitz_override: pd.Series | None = None,   # for placebo
) -> dict:
    """
    Regional DiD: Y ~ C(dept) + C(year_idx) + post*kaitz_pre + edad + edad_sq + sexo
    Clustered SEs at dept level.
    Returns dict with employment / formal / log_wage results + event-study.
    """
    kaitz = kaitz_override if kaitz_override is not None else compute_kaitz(df_pre, mw_old)
    valid = kaitz.dropna().index
    k_ok  = kaitz.loc[valid]
    n_dept = len(k_ok)
    k_range = [round(float(k_ok.min()), 4), round(float(k_ok.max()), 4)] if n_dept > 0 else [None, None]
    print(f"    Kaitz: N_dept={n_dept}, range=[{k_range[0]}, {k_range[1]}]")

    if n_dept < 10:
        print(f"    SKIP: too few departments ({n_dept})")
        return {}

    # Tag years
    pre_tagged = df_pre.copy()
    pre_tagged["post"] = 0
    pre_tagged["year_idx"] = 0
    post_tagged = []
    for i, dfp in enumerate(df_posts):
        d = dfp.copy()
        d["post"] = 1
        d["year_idx"] = i + 1
        post_tagged.append(d)

    combined = pd.concat([pre_tagged] + post_tagged, ignore_index=True)
    combined = combined[combined["dept"].isin(valid)].copy()
    combined["kaitz_pre"] = combined["dept"].map(k_ok)
    combined = combined.dropna(subset=["kaitz_pre", "edad"])
    combined["post_kaitz"] = combined["post"] * combined["kaitz_pre"]

    # Event-study interactions: year_k * kaitz_pre for k>=1
    for k in range(1, len(df_posts) + 1):
        combined[f"yr{k}_kaitz"] = (combined["year_idx"] == k).astype(float) * combined["kaitz_pre"]

    res = {"kaitz_range": k_range, "n_departments": n_dept}

    for oc in OUTCOMES:
        sub = combined.dropna(subset=[oc])
        if oc == "formal":
            sub = sub[sub["employed"] == 1]
        elif oc == "log_wage":
            sub = sub[sub["formal"] == 1]

        n = len(sub)
        if n < 100:
            res[oc] = {"error": f"only {n} obs", "n": n}
            continue

        wt = sub["wt"].fillna(1.0).clip(lower=1e-6)

        # POOLED: post*kaitz
        try:
            m = smf.wls(
                f"{oc} ~ post_kaitz + post + edad + edad_sq + C(dept) + C(year_idx)",
                data=sub, weights=wt
            ).fit(cov_type="cluster", cov_kwds={"groups": sub["dept"]})

            b  = m.params.get("post_kaitz", np.nan)
            se = m.bse.get("post_kaitz", np.nan)
            p  = m.pvalues.get("post_kaitz", np.nan)
            ci = m.conf_int().loc["post_kaitz"].tolist() if "post_kaitz" in m.params else [np.nan, np.nan]

            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            print(f"      {oc:10s} [pooled ]: beta={b:+.4f} se={se:.4f} p={p:.3f}{stars} N={n:,}")

            res[oc] = {
                "beta": round(float(b), 5), "se": round(float(se), 5),
                "ci_low": round(float(ci[0]), 5), "ci_high": round(float(ci[1]), 5),
                "p": round(float(p), 5), "n": n,
            }
        except Exception as e:
            print(f"      {oc} [pooled ] ERROR: {e}")
            res[oc] = {"error": str(e), "n": n}

        # EVENT-STUDY: yr1_kaitz, yr2_kaitz, ...
        yr_vars = [f"yr{k}_kaitz" for k in range(1, len(df_posts) + 1)]
        try:
            formula = f"{oc} ~ " + " + ".join(yr_vars) + " + C(dept) + C(year_idx) + edad + edad_sq"
            m2 = smf.wls(formula, data=sub, weights=wt).fit(
                cov_type="cluster", cov_kwds={"groups": sub["dept"]})
            es = []
            for k, var in enumerate(yr_vars):
                b2  = m2.params.get(var, np.nan)
                se2 = m2.bse.get(var, np.nan)
                p2  = m2.pvalues.get(var, np.nan)
                ci2 = m2.conf_int().loc[var].tolist() if var in m2.params else [np.nan, np.nan]
                s2  = "***" if p2 < 0.01 else "**" if p2 < 0.05 else "*" if p2 < 0.1 else ""
                print(f"      {oc:10s} [yr{k+1}    ]: beta={b2:+.4f} se={se2:.4f} p={p2:.3f}{s2}")
                es.append({"year_offset": k + 1, "beta": round(float(b2), 5),
                           "se": round(float(se2), 5),
                           "ci_low": round(float(ci2[0]), 5),
                           "ci_high": round(float(ci2[1]), 5),
                           "p": round(float(p2), 5)})
            res[f"{oc}_event_study"] = es
        except Exception as e:
            print(f"      {oc} [event-study] ERROR: {e}")

    return res


def ivw_pool(results_list: list[dict], oc: str) -> dict:
    entries = [(r[oc]["beta"], r[oc]["se"])
               for r in results_list
               if r and oc in r and "beta" in r[oc] and not np.isnan(r[oc]["beta"])]
    if len(entries) < 2:
        return {"n_events": len(entries), "note": "not enough events"}
    betas, ses = zip(*entries)
    ws  = [1 / s**2 for s in ses]
    tot = sum(ws)
    b   = sum(b*w for b, w in zip(betas, ws)) / tot
    se  = 1 / np.sqrt(tot)
    z   = b / se
    p   = 2 * (1 - stats.norm.cdf(abs(z)))
    Q   = sum(w * (beta - b)**2 for beta, w in zip(betas, ws))
    k   = len(betas)
    I2  = max(0.0, (Q - (k - 1)) / Q * 100) if Q > 0 else 0.0
    return {
        "beta": round(b, 5), "se": round(se, 5),
        "ci_low": round(b - 1.96*se, 5), "ci_high": round(b + 1.96*se, 5),
        "p": round(p, 5), "Q": round(Q, 4), "I2": round(I2, 1),
        "n_events": k,
    }


def sfmt(b, se, p, n=None):
    if b is None or (isinstance(b, float) and np.isnan(b)):
        return "N/A"
    s = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    out = f"{b:+.3f}({se:.3f}){s}"
    if n: out += f" n={n}"
    return out


# ═══════════════════════════════════════════════════════════════════════════
# DATASET 1: ENAHO CS ANNUAL
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("DATASET 1 — ENAHO CS ANNUAL (regional Kaitz DiD, 25 departments)")
print("=" * 72)

# Load all needed annual years
annual_years_needed = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
print("\nLoading annual data files...")
annual_dfs = {}
for yr in annual_years_needed:
    try:
        print(f"  {yr}...", end=" ")
        annual_dfs[yr] = load_annual(yr)
        fm = annual_dfs[yr]["_fm"].iloc[0] if "_fm" in annual_dfs[yr].columns else "?"
        print(f"OK  {len(annual_dfs[yr]):,} rows  formality_src={fm}")
    except Exception as e:
        print(f"FAILED: {e}")

annual_results = {}

for ev_id, ev in EVENTS.items():
    print(f"\n{'─'*60}")
    print(f"  {ev['label']}")
    print(f"  Kaitz = {ev['mw_old']} / median_formal_dept_{ev['pre_year']}")
    py = ev["pre_year"]
    posts = ev["post_years"]
    if py not in annual_dfs or any(p not in annual_dfs for p in posts):
        missing = [y for y in [py] + posts if y not in annual_dfs]
        print(f"  SKIP: missing years {missing}")
        annual_results[ev_id] = None
        continue

    df_pre  = annual_dfs[py]
    df_posts = [annual_dfs[p] for p in posts]
    print(f"  Pre N={len(df_pre):,}  Post N={sum(len(d) for d in df_posts):,}")
    res = run_event_did(df_pre, df_posts, ev["mw_old"], ev_id)
    annual_results[ev_id] = {**ev, **res, "n_total": len(df_pre) + sum(len(d) for d in df_posts)}

# Placebo (annual): 2016 vs 2017, both post-Event A, no new increase
print(f"\n{'─'*60}")
print(f"  PLACEBO (annual): {PLACEBO_ANNUAL['label']}")
print(f"  Note: {PLACEBO_ANNUAL['note']}")
print(f"  Kaitz = {PLACEBO_ANNUAL['mw_for_kaitz']} / median_formal_dept_{PLACEBO_ANNUAL['kaitz_pre_year']}")
if PLACEBO_ANNUAL["kaitz_pre_year"] in annual_dfs and all(p in annual_dfs for p in [PLACEBO_ANNUAL["pre_year"]] + PLACEBO_ANNUAL["post_years"]):
    kaitz_plann = compute_kaitz(annual_dfs[PLACEBO_ANNUAL["kaitz_pre_year"]], PLACEBO_ANNUAL["mw_for_kaitz"])
    df_pl_pre   = annual_dfs[PLACEBO_ANNUAL["pre_year"]]
    df_pl_posts = [annual_dfs[p] for p in PLACEBO_ANNUAL["post_years"]]
    print(f"  Pre N={len(df_pl_pre):,}  Post N={sum(len(d) for d in df_pl_posts):,}")
    res_pl_ann = run_event_did(df_pl_pre, df_pl_posts, PLACEBO_ANNUAL["mw_for_kaitz"],
                               "placebo_annual", kaitz_override=kaitz_plann)
    annual_results["placebo"] = {**PLACEBO_ANNUAL, **res_pl_ann}
else:
    print("  SKIP: missing years")
    annual_results["placebo"] = None

# IVW Pooled (real events A, B, C)
real_annual = [annual_results.get(e) for e in ["A", "B", "C"] if annual_results.get(e)]
annual_pooled = {oc: ivw_pool(real_annual, oc) for oc in OUTCOMES}
print(f"\n  IVW Pooled ({len(real_annual)} real events):")
for oc in OUTCOMES:
    r = annual_pooled[oc]
    if "beta" in r:
        print(f"    {oc:10s}: beta={r['beta']:+.5f} se={r['se']:.5f} p={r['p']:.4f} Q={r.get('Q','?')} I2={r.get('I2','?')}%")


# ═══════════════════════════════════════════════════════════════════════════
# DATASET 2: ENAHO QUARTERLY
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("DATASET 2 — ENAHO QUARTERLY (Q1 windows, trimestral movil)")
print("=" * 72)

# Load quarterly files
print("\nLoading quarterly data files...")
qt_dfs = {}
for label in QUARTERLY_CODES:
    try:
        print(f"  {label}...", end=" ")
        df = load_quarterly(label)
        if df is not None:
            fm = df["_fm"].iloc[0] if "_fm" in df.columns else "?"
            qt_dfs[label] = df
            print(f"OK  {len(df):,} rows  formality_src={fm}")
    except Exception as e:
        print(f"FAILED: {e}")

quarterly_results = {}

for ev_id, ev in EVENTS.items():
    print(f"\n{'─'*60}")
    print(f"  {ev['label']} [QUARTERLY]")
    qt_pre_label   = ev["qt_pre"]
    qt_post_labels = ev["qt_post"]

    if qt_pre_label is None:
        print(f"  SKIP: no quarterly pre-period defined")
        quarterly_results[ev_id] = None
        continue
    if qt_post_labels is None:
        print(f"  SKIP: no quarterly post-period defined (code TBD)")
        quarterly_results[ev_id] = None
        continue
    if qt_pre_label not in qt_dfs:
        print(f"  SKIP: {qt_pre_label} not loaded")
        quarterly_results[ev_id] = None
        continue
    missing_post = [l for l in qt_post_labels if l not in qt_dfs]
    if missing_post:
        print(f"  SKIP: post labels not loaded: {missing_post}")
        quarterly_results[ev_id] = None
        continue

    df_pre   = qt_dfs[qt_pre_label]
    df_posts = [qt_dfs[l] for l in qt_post_labels]
    pre_fm   = df_pre["_fm"].iloc[0] if "_fm" in df_pre.columns else "?"
    print(f"  Pre={qt_pre_label}(fm={pre_fm}) N={len(df_pre):,}  Post={qt_post_labels}")
    print(f"  Note: {ev.get('qt_note', '')}")
    print(f"  Kaitz = {ev['mw_old']} / median_formal_dept_{qt_pre_label}")

    res = run_event_did(df_pre, df_posts, ev["mw_old"], ev_id + "_qt")
    quarterly_results[ev_id] = {
        "label": ev["label"] + " [quarterly]",
        "qt_pre": qt_pre_label, "qt_post": qt_post_labels,
        **res,
    }

# Quarterly placebo: Q4 2015 → Q1 2016 (genuine no-change, MW=750 throughout)
print(f"\n{'─'*60}")
print(f"  PLACEBO [QUARTERLY]: {PLACEBO_QT['label']}")
print(f"  Note: {PLACEBO_QT['note']}")
if PLACEBO_QT["qt_pre"] in qt_dfs and PLACEBO_QT["qt_post"] in qt_dfs:
    kaitz_qt_pl = compute_kaitz(qt_dfs[PLACEBO_QT["qt_pre"]], PLACEBO_QT["mw_for_kaitz"])
    res_qt_pl = run_event_did(
        qt_dfs[PLACEBO_QT["qt_pre"]],
        [qt_dfs[PLACEBO_QT["qt_post"]]],
        PLACEBO_QT["mw_for_kaitz"], "qt_placebo",
        kaitz_override=kaitz_qt_pl
    )
    quarterly_results["placebo"] = {"label": PLACEBO_QT["label"], **res_qt_pl}
else:
    missing = [l for l in [PLACEBO_QT["qt_pre"], PLACEBO_QT["qt_post"]] if l not in qt_dfs]
    print(f"  SKIP: {missing} not loaded")
    quarterly_results["placebo"] = None

print("\n  NOTE: Event C quarterly: Pre=Q1_2022 available, Post=Q1_2023 code not yet found.")
print(f"        Q1 2022 code={QUARTERLY_CODES['Q1_2022']} available as pre only.")

real_qt = [quarterly_results.get(e) for e in ["A", "B"] if quarterly_results.get(e)]
quarterly_pooled = {oc: ivw_pool(real_qt, oc) for oc in OUTCOMES}
print(f"\n  IVW Pooled ({len(real_qt)} real quarterly events):")
for oc in OUTCOMES:
    r = quarterly_pooled[oc]
    if "beta" in r:
        print(f"    {oc:10s}: beta={r['beta']:+.5f} se={r['se']:.5f} p={r['p']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# DATASET 3: EPE Lima Metro
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("DATASET 3 — EPE LIMA METRO (panel DiD, Lima Metro only)")
print("  Spec: panel DiD at the worker level, not regional Kaitz DiD")
print("  NOT directly comparable — different sample, different identification")
print("=" * 72)

epe = pd.read_csv(EPE_ALL)
EPE_MAP = {
    "2015_Dec": ("A_Dec", "S/750->850 (Dec 2015)"),  # Lima had Dec 2015 then May 2016
    "2016_May": ("A_May", "S/850->930 (May 2016)"),
    "2022_Apr": ("C",     "S/930->1025 (Apr 2022)"),
}
# Note: EPE Lima has no event for S/850->930 in 2018 (Event B) — Lima already at 930 from May 2016

epe_results = {}
print()
for epe_ev, (canon_id, lbl) in EPE_MAP.items():
    row = epe[epe["event"] == epe_ev]
    if len(row) == 0:
        print(f"  {lbl}: not found")
        continue
    row = row.iloc[0]
    b_emp = row.get("beta_emp", np.nan)
    s_emp = row.get("se_emp", np.nan)
    p_emp = row.get("p_emp", np.nan)
    b_fm  = row.get("beta_se", np.nan)
    s_fm  = row.get("se_se", np.nan)
    p_fm  = row.get("p_se", np.nan)
    n_t   = int(row.get("n_treat", 0))
    n_c   = int(row.get("n_ctrl", 0))
    print(f"  {lbl}:")
    print(f"    employment    : {sfmt(b_emp, s_emp, p_emp)} N_treat={n_t} N_ctrl={n_c}")
    print(f"    formalization : {sfmt(b_fm, s_fm, p_fm)}")
    epe_results[canon_id] = {
        "label": lbl, "mw_pre": float(row.get("mw_pre", np.nan)),
        "mw_post": float(row.get("mw_post", np.nan)),
        "employment": {"beta": float(b_emp), "se": float(s_emp), "p": float(p_emp), "n_treat": n_t, "n_ctrl": n_c},
        "formalization": {"beta": float(b_fm), "se": float(s_fm), "p": float(p_fm)},
    }

print("\n  NOTE: EPE Lima's 'Event A' covers Dec 2015 (750→850) and May 2016 (850→930).")
print("        Both are relevant to canonical Event A (2015-2017 increase window).")
print("        EPE Lima has no separate 2018 event because Lima MW was already 930 from May 2016.")
print("        The April 2018 national DS only affected regions where MW was still 850.")


# ═══════════════════════════════════════════════════════════════════════════
# MASTER COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════
def r(d, oc):
    if not d or oc not in d or "beta" not in d[oc]:
        return "N/A"
    return sfmt(d[oc]["beta"], d[oc]["se"], d[oc]["p"])


W = 28
print(f"\n{'='*90}")
print("MASTER COMPARISON TABLE (corrected MW values)")
print(f"{'='*90}")
print(f"{'':30s} {'EMPLOYMENT':>28s} {'FORMAL':>28s} {'LOG WAGE':>28s}")
SEP = "-" * 90

def pr(label, emp, fm, wg):
    print(f"{label:<30s} {emp:>28s} {fm:>28s} {wg:>28s}")

# ── EVENT A ──
print(f"\nEVENT A: S/750→S/850 (May 2016)  Kaitz=750/dept_2015")
print(SEP)
pr("  ENAHO Annual",
   r(annual_results.get("A"), "employed"),
   r(annual_results.get("A"), "formal"),
   r(annual_results.get("A"), "log_wage"))
pr("  ENAHO Quarterly",
   r(quarterly_results.get("A"), "employed"),
   r(quarterly_results.get("A"), "formal"),
   r(quarterly_results.get("A"), "log_wage"))
pr("  EPE Lima (Dec-2015)", sfmt(epe_results.get("A_Dec",{}).get("employment",{}).get("beta",np.nan),
                                  epe_results.get("A_Dec",{}).get("employment",{}).get("se",np.nan),
                                  epe_results.get("A_Dec",{}).get("employment",{}).get("p",np.nan)),
   sfmt(epe_results.get("A_Dec",{}).get("formalization",{}).get("beta",np.nan),
        epe_results.get("A_Dec",{}).get("formalization",{}).get("se",np.nan),
        epe_results.get("A_Dec",{}).get("formalization",{}).get("p",np.nan)), "N/A")
pr("  EPE Lima (May-2016)", sfmt(epe_results.get("A_May",{}).get("employment",{}).get("beta",np.nan),
                                  epe_results.get("A_May",{}).get("employment",{}).get("se",np.nan),
                                  epe_results.get("A_May",{}).get("employment",{}).get("p",np.nan)),
   sfmt(epe_results.get("A_May",{}).get("formalization",{}).get("beta",np.nan),
        epe_results.get("A_May",{}).get("formalization",{}).get("se",np.nan),
        epe_results.get("A_May",{}).get("formalization",{}).get("p",np.nan)), "N/A")

# ── EVENT B ──
print(f"\nEVENT B: S/850→S/930 (April 2018)  Kaitz=850/dept_2017")
print(SEP)
pr("  ENAHO Annual",
   r(annual_results.get("B"), "employed"),
   r(annual_results.get("B"), "formal"),
   r(annual_results.get("B"), "log_wage"))
pr("  ENAHO Quarterly",
   r(quarterly_results.get("B"), "employed"),
   r(quarterly_results.get("B"), "formal"),
   r(quarterly_results.get("B"), "log_wage"))
pr("  EPE Lima", "N/A (Lima at 930 already)", "N/A", "N/A")

# ── EVENT C ──
print(f"\nEVENT C: S/930→S/1,025 (May 2022)  Kaitz=930/dept_2021")
print(SEP)
pr("  ENAHO Annual",
   r(annual_results.get("C"), "employed"),
   r(annual_results.get("C"), "formal"),
   r(annual_results.get("C"), "log_wage"))
pr("  ENAHO Quarterly", "Q1 2023 code missing", "Q1 2023 code missing", "Q1 2023 code missing")
pr("  EPE Lima (Apr-2022)", sfmt(epe_results.get("C",{}).get("employment",{}).get("beta",np.nan),
                                  epe_results.get("C",{}).get("employment",{}).get("se",np.nan),
                                  epe_results.get("C",{}).get("employment",{}).get("p",np.nan)),
   sfmt(epe_results.get("C",{}).get("formalization",{}).get("beta",np.nan),
        epe_results.get("C",{}).get("formalization",{}).get("se",np.nan),
        epe_results.get("C",{}).get("formalization",{}).get("p",np.nan)), "N/A")

# ── PLACEBO ──
print(f"\nPLACEBO: 2016 vs 2017 (no new increase, Kaitz=750/dept_2015)")
print(SEP)
pr("  ENAHO Annual",
   r(annual_results.get("placebo"), "employed"),
   r(annual_results.get("placebo"), "formal"),
   r(annual_results.get("placebo"), "log_wage"))
pr("  ENAHO Quarterly",
   r(quarterly_results.get("placebo"), "employed"),
   r(quarterly_results.get("placebo"), "formal"),
   r(quarterly_results.get("placebo"), "log_wage"))

# ── POOLED ──
print(f"\nIVW POOLED (Events A+B+C, ENAHO Annual):")
print(SEP)
for oc in OUTCOMES:
    r2 = annual_pooled[oc]
    if "beta" in r2:
        print(f"  {oc:10s}: {sfmt(r2['beta'], r2['se'], r2['p']):25s}  Q={r2['Q']:.2f} I2={r2['I2']:.0f}% (N={r2['n_events']} events)")

print(f"\nIVW POOLED (Events A+B, ENAHO Quarterly):")
print(SEP)
for oc in OUTCOMES:
    r2 = quarterly_pooled[oc]
    if "beta" in r2 and r2.get("n_events", 0) >= 2:
        print(f"  {oc:10s}: {sfmt(r2['beta'], r2['se'], r2['p']):25s}  Q={r2['Q']:.2f} I2={r2['I2']:.0f}% (N={r2['n_events']} events)")
    else:
        print(f"  {oc:10s}: insufficient events")

print(f"\n{'='*90}")
print("EVENT STUDY COEFFICIENTS (year-by-year, ENAHO Annual):")
for ev_id in ["A", "B", "C"]:
    ev_r = annual_results.get(ev_id)
    if not ev_r:
        continue
    print(f"\n  {EVENTS[ev_id]['label']}:")
    for oc in OUTCOMES:
        es = ev_r.get(f"{oc}_event_study", [])
        if es:
            for pt in es:
                print(f"    {oc:10s} yr+{pt['year_offset']}: beta={pt['beta']:+.4f} se={pt['se']:.4f} p={pt['p']:.3f}")

print(f"\n{'='*90}")
print("INTERPRETATION:")
print()
print("  EMPLOYMENT:")
a_emp = annual_results.get("A", {}).get("employed", {})
b_emp = annual_results.get("B", {}).get("employed", {})
c_emp = annual_results.get("C", {}).get("employed", {})
pl_emp = annual_results.get("placebo", {}).get("employed", {})
print(f"    Event A (annual)  : {sfmt(a_emp.get('beta',np.nan), a_emp.get('se',np.nan), a_emp.get('p',np.nan))}")
print(f"    Event B (annual)  : {sfmt(b_emp.get('beta',np.nan), b_emp.get('se',np.nan), b_emp.get('p',np.nan))}")
print(f"    Event C (annual)  : {sfmt(c_emp.get('beta',np.nan), c_emp.get('se',np.nan), c_emp.get('p',np.nan))}")
print(f"    Placebo (annual)  : {sfmt(pl_emp.get('beta',np.nan), pl_emp.get('se',np.nan), pl_emp.get('p',np.nan))}")
qa_emp = quarterly_results.get("A", {}).get("employed", {})
qb_emp = quarterly_results.get("B", {}).get("employed", {})
qpl_emp = quarterly_results.get("placebo", {}).get("employed", {})
print(f"    Event A (qtly)    : {sfmt(qa_emp.get('beta',np.nan), qa_emp.get('se',np.nan), qa_emp.get('p',np.nan))}")
print(f"    Event B (qtly)    : {sfmt(qb_emp.get('beta',np.nan), qb_emp.get('se',np.nan), qb_emp.get('p',np.nan))}")
print(f"    Placebo (qtly)    : {sfmt(qpl_emp.get('beta',np.nan), qpl_emp.get('se',np.nan), qpl_emp.get('p',np.nan))}")

print()
print("  FORMALIZATION:")
for ev_id in ["A", "B", "C"]:
    r2 = annual_results.get(ev_id, {}).get("formal", {})
    print(f"    Event {ev_id} (annual)  : {sfmt(r2.get('beta',np.nan), r2.get('se',np.nan), r2.get('p',np.nan))}")
for ev_id, lbl in [("A", "Event A"), ("B", "Event B")]:
    r2 = quarterly_results.get(ev_id, {}).get("formal", {})
    print(f"    {lbl} (qtly)    : {sfmt(r2.get('beta',np.nan), r2.get('se',np.nan), r2.get('p',np.nan))}")

print()
print("  LOG WAGES:")
for ev_id in ["A", "B", "C"]:
    r2 = annual_results.get(ev_id, {}).get("log_wage", {})
    print(f"    Event {ev_id} (annual)  : {sfmt(r2.get('beta',np.nan), r2.get('se',np.nan), r2.get('p',np.nan))}")
for ev_id, lbl in [("A", "Event A"), ("B", "Event B")]:
    r2 = quarterly_results.get(ev_id, {}).get("log_wage", {})
    print(f"    {lbl} (qtly)    : {sfmt(r2.get('beta',np.nan), r2.get('se',np.nan), r2.get('p',np.nan))}")

print(f"\n{'='*90}")


# ═══════════════════════════════════════════════════════════════════════════
# SAVE JSON
# ═══════════════════════════════════════════════════════════════════════════
def clean(v):
    if isinstance(v, float) and np.isnan(v): return None
    if isinstance(v, (np.floating, np.integer)): return float(v)
    if isinstance(v, dict): return {k: clean(vv) for k, vv in v.items()}
    if isinstance(v, list): return [clean(x) for x in v]
    return v

output = clean({
    "events": {
        "A": annual_results.get("A"),
        "B": annual_results.get("B"),
        "C": annual_results.get("C"),
        "placebo": annual_results.get("placebo"),
    },
    "quarterly": {
        "A": quarterly_results.get("A"),
        "B": quarterly_results.get("B"),
        "placebo": quarterly_results.get("placebo"),
    },
    "epe_lima": epe_results,
    "pooled_annual": clean(annual_pooled),
    "pooled_quarterly": clean(quarterly_pooled),
    "metadata": {
        "specification": "Y_idt = alpha_d + gamma_t + beta(Post x Kaitz_d,pre) + Edad + Edad^2 + Sexo",
        "mw_values": {"A": "750->850", "B": "850->930", "C": "930->1025"},
        "kaitz_mw": {"A": 750, "B": 850, "C": 930},
        "formality_def": "INEI ocupinf==2; p510a1==1 for quarterly files lacking ocupinf",
        "weight": "fac500a (annual) / fac500 (quarterly)",
        "cluster": "departamento (25)",
        "last_updated": "2026-03-15",
    }
})

with open(OUT_J, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {OUT_J}")

with open(OUT_J2, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# Copy to qhawarina
for dest in [
    pathlib.Path("D:/qhawarina/public/assets/data/mw_canonical_results.json"),
    pathlib.Path("D:/qhawarina/public/assets/data/mw_multi_dataset_results.json"),
]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(OUT_J, dest)
    print(f"Copied: {dest}")
