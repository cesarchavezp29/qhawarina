"""Validation analyses for IRP/IRE paper.

Runs:
  (a) FX correlation analysis (EMBIG not available via BCRP API)
  (b) Formal event study with t-tests
  (c) Jordà local projections at weekly frequency
  (d) Google Trends correlation (attempt)
  (e) EPU neighboring countries (attempt)

All output is printed in LaTeX-ready format.
"""

import warnings
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats
import httpx

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_PATH = DATA_DIR / "processed/daily_instability/daily_index.parquet"

# ── Events ───────────────────────────────────────────────────────────────────
EVENTS = [
    ("2025-03-18", "Cerron Desaparece"),
    ("2025-04-23", "Impunidad Boluarte"),
    ("2025-06-16", "Vizcarra Prison"),
    ("2025-07-17", "Fragmentacion Izquierda"),
    ("2025-08-19", "Colapso Ministerial"),
    ("2025-10-10", "Paro Transportistas"),
    ("2025-10-22", "Vacancia Boluarte"),
    ("2025-11-13", "Castillo Condenado"),
    ("2026-01-21", "Censura Jeri"),
    ("2026-02-04", "Paro II"),
    ("2026-03-12", "Crisis Camisea"),
    ("2026-03-18", "Voto Confianza"),
]

BCRP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Referer": "https://estadisticas.bcrp.gob.pe/",
}
BCRP_BASE = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_index() -> pd.DataFrame:
    """Load IRP/IRE daily index."""
    df = pd.read_parquet(INDEX_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "political_index", "economic_index"]].copy()


def parse_bcrp_date(s: str) -> pd.Timestamp:
    """Parse BCRP daily date like '02.Ene.25' → 2025-01-02."""
    month_map = {
        "Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Ago": 8, "Set": 9, "Sep": 9, "Oct": 10, "Nov": 11, "Dic": 12,
    }
    parts = s.split(".")
    day = int(parts[0])
    month = month_map[parts[1]]
    year = 2000 + int(parts[2])
    return pd.Timestamp(year, month, day)


def fetch_daily_fx() -> pd.DataFrame:
    """Fetch daily PEN/USD interbank sale rate from BCRP API."""
    url = f"{BCRP_BASE}/PD04638PD/json/20250101/20260318/esp"
    try:
        r = httpx.get(url, timeout=20, headers=BCRP_HEADERS)
        r.raise_for_status()
        data = r.json()
        periods = data.get("periods", [])
        rows = []
        for p in periods:
            val = p["values"][0]
            if val and val != "n.d.":
                rows.append({"date": parse_bcrp_date(p["name"]), "fx": float(val)})
        df = pd.DataFrame(rows)
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  WARNING: Could not fetch FX data: {e}")
        return pd.DataFrame(columns=["date", "fx"])


# ─────────────────────────────────────────────────────────────────────────────
# (a) FX correlation analysis (EMBIG unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def run_fx_correlation(idx: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("SECTION (a): Financial Market Correlation Analysis")
    print("=" * 70)
    print("Note: EMBIG daily data is blocked (HTTP 403) on the BCRP API.")
    print("Using PEN/USD daily interbank rate (PD04638PD) as benchmark.")
    print()

    fx = fetch_daily_fx()
    if fx.empty:
        print("  ERROR: FX data unavailable. Skipping correlation analysis.")
        return

    print(f"  FX data: {len(fx)} trading days, "
          f"{fx['date'].min().date()} to {fx['date'].max().date()}")

    # Merge on date
    merged = pd.merge(idx, fx, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)
    print(f"  Merged observations: {len(merged)}")

    # Daily FX change (first difference)
    merged["dfx"] = merged["fx"].diff()

    # Weekly aggregate (Monday week)
    merged["week"] = merged["date"].dt.to_period("W")
    weekly = merged.groupby("week").agg(
        irp=("political_index", "mean"),
        ire=("economic_index", "mean"),
        dfx=("dfx", "sum"),
        fx=("fx", "last"),
    ).dropna()
    weekly["dfx_weekly"] = weekly["fx"].diff()

    # Monthly aggregate
    merged["month"] = merged["date"].dt.to_period("M")
    monthly = merged.groupby("month").agg(
        irp=("political_index", "mean"),
        ire=("economic_index", "mean"),
        fx=("fx", "last"),
    ).dropna()
    monthly["dfx"] = monthly["fx"].diff()

    print()
    print("LaTeX table: Correlation of IRP/IRE with PEN/USD Change")
    print()

    rows = []
    for label, sub, irp_col, ire_col, outcome_col in [
        ("Daily",   merged.dropna(subset=["dfx"]),  "political_index", "economic_index", "dfx"),
        ("Weekly",  weekly.dropna(subset=["dfx_weekly"]), "irp", "ire", "dfx_weekly"),
        ("Monthly", monthly.dropna(subset=["dfx"]), "irp", "ire", "dfx"),
    ]:
        n = len(sub)
        r_irp, p_irp = stats.pearsonr(sub[irp_col], sub[outcome_col])
        r_ire, p_ire = stats.pearsonr(sub[ire_col], sub[outcome_col])
        rows.append((label, r_irp, p_irp, r_ire, p_ire, n))
        print(f"  {label}: N={n}, r(IRP)={r_irp:.3f} (p={p_irp:.3f}), "
              f"r(IRE)={r_ire:.3f} (p={p_ire:.3f})")

    print()
    print(r"\begin{table}[h!]")
    print(r"\centering")
    print(r"\caption{Correlation of IRP/IRE with PEN/USD Daily Change}")
    print(r"\label{tab:fx}")
    print(r"\begin{tabular}{lrrrrrr}")
    print(r"\toprule")
    print(r"Frequency & $r(\text{IRP})$ & $p$-value & $r(\text{IRE})$ & $p$-value & $N$ \\")
    print(r"\midrule")
    for label, r_irp, p_irp, r_ire, p_ire, n in rows:
        print(f"{label:<8} & ${r_irp:+.3f}$ & {p_irp:.3f} & ${r_ire:+.3f}$ & {p_ire:.3f} & {n:3d} \\\\")
    print(r"\bottomrule")
    print(r"\multicolumn{6}{l}{\footnotesize FX data from BCRP (TC Interbancario Venta).} \\")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ─────────────────────────────────────────────────────────────────────────────
# (b) Formal event study
# ─────────────────────────────────────────────────────────────────────────────

def run_event_study(idx: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("SECTION (b): Formal Event Study")
    print("=" * 70)

    event_dates = [pd.Timestamp(d) for d, _ in EVENTS]

    # Mark all days within ±5 of any event
    all_dates = idx["date"].values
    in_any_window = np.zeros(len(idx), dtype=bool)
    for ed in event_dates:
        window = (idx["date"] >= ed - timedelta(days=5)) & \
                 (idx["date"] <= ed + timedelta(days=5))
        in_any_window |= window.values

    non_event_irp = idx.loc[~in_any_window, "political_index"]
    non_event_ire = idx.loc[~in_any_window, "economic_index"]
    mu_irp = non_event_irp.mean()
    mu_ire = non_event_ire.mean()
    n_non = len(non_event_irp)
    std_irp = non_event_irp.std()
    std_ire = non_event_ire.std()

    print(f"  Non-event days: {n_non}")
    print(f"  Non-event mean IRP: {mu_irp:.2f} (std={std_irp:.2f})")
    print(f"  Non-event mean IRE: {mu_ire:.2f} (std={std_ire:.2f})")
    print()

    results = []
    for (event_date_str, label) in EVENTS:
        ed = pd.Timestamp(event_date_str)

        def window_avg(days):
            mask = (idx["date"] >= ed - timedelta(days=days)) & \
                   (idx["date"] <= ed + timedelta(days=days))
            sub = idx.loc[mask]
            return sub["political_index"].mean(), sub["economic_index"].mean(), len(sub)

        irp_1, ire_1, n1 = window_avg(1)
        irp_3, ire_3, n3 = window_avg(3)
        irp_5, ire_5, n5 = window_avg(5)

        # Abnormal IRP = window_3 - non_event_mean
        ab_irp = irp_3 - mu_irp

        # T-test: one-sided, event-window vs non-event
        mask5 = (idx["date"] >= ed - timedelta(days=5)) & \
                (idx["date"] <= ed + timedelta(days=5))
        event_irp_vals = idx.loc[mask5, "political_index"].values
        t_stat, p_two = stats.ttest_ind(event_irp_vals, non_event_irp.values,
                                        alternative="two-sided", equal_var=False)
        p_one = p_two / 2 if t_stat > 0 else 1.0

        results.append({
            "label": label,
            "date": event_date_str,
            "irp_1": irp_1,
            "irp_3": irp_3,
            "ire_3": ire_3,
            "irp_5": irp_5,
            "ab_irp": ab_irp,
            "t_stat": t_stat,
            "p_one": p_one,
            "n_window": n5,
        })

        sig = "*" if p_one < 0.10 else ("**" if p_one < 0.05 else "")
        print(f"  {label[:22]:<22}: IRP±3={irp_3:.1f}, IRE±3={ire_3:.1f}, "
              f"Abnormal={ab_irp:+.1f}, t={t_stat:.2f}, p(one-sided)={p_one:.3f}{sig}")

    # Summary
    sig_count = sum(1 for r in results if r["p_one"] < 0.10)
    print(f"\n  {sig_count}/{len(results)} events significant at 10% (one-sided)")

    print()
    print(r"\begin{table}[h!]")
    print(r"\centering")
    print(r"\caption{Event Study: IRP Abnormal Readings Around Known Episodes}")
    print(r"\label{tab:event_study}")
    print(r"\small")
    print(r"\begin{tabular}{lrrrrrr}")
    print(r"\toprule")
    print(r"Event & Date & IRP\,$\pm$1d & IRP\,$\pm$3d & IRE\,$\pm$3d & Abnormal IRP & $p$-value \\")
    print(r"\midrule")
    for r in results:
        p_str = f"{r['p_one']:.3f}"
        if r["p_one"] < 0.05:
            p_str += "$^{**}$"
        elif r["p_one"] < 0.10:
            p_str += "$^{*}$"
        ab_str = f"{r['ab_irp']:+.1f}"
        print(f"{r['label'][:22]:<22} & {r['date']} & "
              f"{r['irp_1']:.1f} & {r['irp_3']:.1f} & {r['ire_3']:.1f} & "
              f"{ab_str} & {p_str} \\\\")
    print(r"\midrule")
    print(f"Non-event baseline & --- & --- & {mu_irp:.1f} & {mu_ire:.1f} & 0.0 & --- \\\\")
    print(r"\bottomrule")
    print(r"\multicolumn{7}{l}{\footnotesize Abnormal IRP = window $\pm$3d mean minus non-event mean.} \\")
    print(r"\multicolumn{7}{l}{\footnotesize $p$-value: one-sided Welch $t$-test, event window $\pm$5d vs.\ all non-event days.} \\")
    print(r"\multicolumn{7}{l}{\footnotesize $^{*}p<0.10$, $^{**}p<0.05$. Non-event days = days outside any $\pm$5d window.} \\")
    print(r"\end{tabular}")
    print(r"\end{table}")

    return {
        "results": results,
        "mu_irp": mu_irp,
        "mu_ire": mu_ire,
        "sig_count": sig_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# (c) Jordà local projections
# ─────────────────────────────────────────────────────────────────────────────

def run_local_projections(idx: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("SECTION (c): Jordà Local Projections (Weekly Frequency)")
    print("=" * 70)

    try:
        import statsmodels.api as sm
    except ImportError:
        print("  ERROR: statsmodels not available. Skipping local projections.")
        return

    # Fetch FX for outcome
    fx = fetch_daily_fx()
    if fx.empty:
        print("  WARNING: FX unavailable. Skipping local projections.")
        return

    # Merge index and FX
    merged = pd.merge(idx, fx, on="date", how="inner").sort_values("date")
    merged["dfx"] = merged["fx"].diff()

    # Aggregate to weekly
    merged["week"] = merged["date"].dt.to_period("W")
    weekly = merged.groupby("week").agg(
        irp=("political_index", "mean"),
        ire=("economic_index", "mean"),
        dfx_sum=("dfx", "sum"),
        fx_last=("fx", "last"),
    ).dropna()
    weekly = weekly.sort_index()
    weekly["dfx_weekly"] = weekly["fx_last"].diff()
    weekly = weekly.dropna()

    n_weeks = len(weekly)
    print(f"  Weekly observations: {n_weeks}")

    if n_weeks < 20:
        print("  WARNING: Too few weeks for reliable local projections. Skipping.")
        return

    horizons = [1, 2, 3, 4]
    print()
    print(f"  Local projections: Y_{{t+h}} = a + b*IRP_t + c*IRE_t + d*Y_{{t-1}} + e_t")
    print(f"  Outcome Y = weekly PEN/USD change (sum of daily changes)")
    print()

    irp_coefs = []
    ire_coefs = []

    for h in horizons:
        y = weekly["dfx_weekly"].shift(-h)
        X = pd.DataFrame({
            "const": 1.0,
            "irp": weekly["irp"],
            "ire": weekly["ire"],
            "y_lag1": weekly["dfx_weekly"].shift(1),
        })
        valid = (~y.isna()) & (~X.isna().any(axis=1))
        y_v = y[valid]
        X_v = X[valid]

        if len(y_v) < 10:
            print(f"  h={h}: insufficient observations ({len(y_v)}). Skipping.")
            irp_coefs.append(None)
            ire_coefs.append(None)
            continue

        try:
            model = sm.OLS(y_v, X_v).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": 4},
            )
            b_irp = model.params["irp"]
            se_irp = model.bse["irp"]
            b_ire = model.params["ire"]
            se_ire = model.bse["ire"]
            p_irp = model.pvalues["irp"]
            p_ire = model.pvalues["ire"]
            irp_coefs.append((b_irp, se_irp, p_irp, len(y_v)))
            ire_coefs.append((b_ire, se_ire, p_ire, len(y_v)))
            ci90_lo = b_irp - 1.645 * se_irp
            ci90_hi = b_irp + 1.645 * se_irp
            print(f"  h={h}: IRP coef={b_irp:.4f} (SE={se_irp:.4f}, p={p_irp:.3f}), "
                  f"90% CI=[{ci90_lo:.4f}, {ci90_hi:.4f}], N={len(y_v)}")
        except Exception as e:
            print(f"  h={h}: regression error: {e}")
            irp_coefs.append(None)
            ire_coefs.append(None)

    # Print LaTeX table
    all_ok = all(c is not None for c in irp_coefs)
    if all_ok:
        print()
        print(r"\begin{table}[h!]")
        print(r"\centering")
        print(r"\caption{Jord\`{a} Local Projections: IRP/IRE Effect on Weekly PEN/USD Change}")
        print(r"\label{tab:lp}")
        print(r"\small")
        print(r"\begin{tabular}{lrrrrrr}")
        print(r"\toprule")
        print(r"Horizon $h$ & IRP coef. & SE & $p$-value & IRE coef. & SE & $N$ \\")
        print(r"\midrule")
        for h, irp_c, ire_c in zip(horizons, irp_coefs, ire_coefs):
            if irp_c is None:
                continue
            b, se, p, n = irp_c
            b2, se2, p2, _ = ire_c
            sig = "$^{**}$" if p < 0.05 else ("$^{*}$" if p < 0.10 else "")
            print(f"{h} week{'s' if h>1 else ' '} ahead & "
                  f"{b:.4f}{sig} & {se:.4f} & {p:.3f} & "
                  f"{b2:.4f} & {se2:.4f} & {n} \\\\")
        print(r"\bottomrule")
        print(r"\multicolumn{7}{l}{\footnotesize OLS with Newey-West SEs (4 lags). "
              r"Control: lagged outcome. $^{*}p<0.10$, $^{**}p<0.05$.} \\")
        print(r"\end{tabular}")
        print(r"\end{table}")
    else:
        print("  Some horizons failed. Partial results shown above.")


# ─────────────────────────────────────────────────────────────────────────────
# (d) Google Trends
# ─────────────────────────────────────────────────────────────────────────────

def run_google_trends(idx: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("SECTION (d): Google Trends (attempt)")
    print("=" * 70)

    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="es-PE", tz=-300)
        pytrends.build_payload(
            ["crisis política perú", "vacancia"],
            timeframe="2025-01-01 2026-03-18",
            geo="PE",
        )
        gt_df = pytrends.interest_over_time()
        if gt_df.empty:
            print("  Google Trends returned empty DataFrame.")
            return

        gt_df = gt_df.reset_index()
        gt_df = gt_df.rename(columns={"date": "week"})
        gt_df["week"] = pd.to_datetime(gt_df["week"])

        # Aggregate IRP to weekly to match
        idx_copy = idx.copy()
        idx_copy["week"] = idx_copy["date"].dt.to_period("W").dt.start_time
        weekly_irp = idx_copy.groupby("week")["political_index"].mean().reset_index()

        merged = pd.merge(weekly_irp, gt_df, on="week", how="inner")
        if "crisis política perú" in merged.columns:
            r, p = stats.pearsonr(merged["political_index"], merged["crisis política perú"])
            print(f"  r(IRP, 'crisis política perú') = {r:.3f} (p={p:.3f}, N={len(merged)})")
        if "vacancia" in merged.columns:
            r2, p2 = stats.pearsonr(merged["political_index"], merged["vacancia"])
            print(f"  r(IRP, 'vacancia') = {r2:.3f} (p={p2:.3f}, N={len(merged)})")

    except ImportError:
        print("  pytrends not installed. Skipping Google Trends analysis.")
    except Exception as e:
        print(f"  Google Trends failed: {e}")
        print("  Skipping (likely rate limit or network issue).")


# ─────────────────────────────────────────────────────────────────────────────
# (e) EPU neighboring countries
# ─────────────────────────────────────────────────────────────────────────────

def run_epu_comparison() -> None:
    print("\n" + "=" * 70)
    print("SECTION (e): EPU Neighboring Countries (attempt)")
    print("=" * 70)

    search_paths = [
        DATA_DIR / "raw",
        DATA_DIR / "external",
    ]
    patterns = ["*epu*", "*EPU*", "*colombia*epu*", "*chile*epu*"]

    found = []
    for base in search_paths:
        if base.exists():
            for pat in patterns:
                found.extend(base.glob(pat))
                found.extend(base.glob(f"**/{pat}"))

    if found:
        print(f"  Found EPU files: {[str(f) for f in found]}")
    else:
        print("  No EPU data found in data/raw/ or data/external/.")
        print("  Skipping EPU comparison. See policyuncertainty.com for Colombia/Chile EPU.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("IRP/IRE VALIDATION ANALYSES")
    print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load index
    idx = load_index()
    print(f"\nIndex loaded: {len(idx)} days, "
          f"{idx['date'].min().date()} to {idx['date'].max().date()}")
    print(f"IRP: mean={idx['political_index'].mean():.2f}, "
          f"std={idx['political_index'].std():.2f}")
    print(f"IRE: mean={idx['economic_index'].mean():.2f}, "
          f"std={idx['economic_index'].std():.2f}")

    # Run analyses
    run_fx_correlation(idx)
    event_results = run_event_study(idx)
    run_local_projections(idx)
    run_google_trends(idx)
    run_epu_comparison()

    print("\n" + "=" * 70)
    print("KEY NUMBERS FOR PAPER")
    print("=" * 70)
    if event_results:
        print(f"Non-event mean IRP: {event_results['mu_irp']:.2f}")
        print(f"Non-event mean IRE: {event_results['mu_ire']:.2f}")
        print(f"Events significant at 10%: {event_results['sig_count']}/12")
        for r in event_results["results"]:
            sig = "**" if r["p_one"] < 0.05 else ("*" if r["p_one"] < 0.10 else "")
            print(f"  {r['label'][:22]:<22}: Abnormal IRP={r['ab_irp']:+.1f}, p={r['p_one']:.3f}{sig}")

    print("\nDone.")


if __name__ == "__main__":
    main()
