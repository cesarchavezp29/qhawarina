"""Compute all statistics needed for the IRP/IRE working paper.

Run from the nexus root:
    python paper/compute_paper_stats.py

Outputs all Table 1-6 values plus summary statistics to stdout.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 72)
print("LOADING DATA")
print("=" * 72)

articles = pd.read_parquet(PROJECT_ROOT / "data/raw/rss/articles_classified.parquet")
print(f"articles_classified.parquet: {len(articles):,} rows x {len(articles.columns)} cols")
print(f"  Columns: {articles.columns.tolist()}")

df = pd.read_parquet(PROJECT_ROOT / "data/processed/daily_instability/daily_index.parquet")
print(f"daily_index.parquet: {len(df):,} rows x {len(df.columns)} cols")

with open(PROJECT_ROOT / "exports/data/political_index_daily.json", encoding="utf-8") as f:
    pol_json = json.load(f)
print(f"political_index_daily.json: loaded, keys={list(pol_json.keys())}")

# ──────────────────────────────────────────────────────────────────────────────
# Pre-processing
# ──────────────────────────────────────────────────────────────────────────────

# Parse dates in articles
articles["published"] = pd.to_datetime(articles["published"], utc=True)
articles["date"] = articles["published"].dt.tz_convert("America/Lima").dt.normalize().dt.tz_localize(None)
articles["date_only"] = articles["date"].dt.date

# Parse daily index
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"\nDate range (daily index): {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Total days: {len(df)}")
print(f"Total articles: {len(articles):,}")

# ──────────────────────────────────────────────────────────────────────────────
# TABLE 1 — Source-level statistics
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("TABLE 1 — Source-Level Statistics")
print("=" * 72)

# Get date range for mean daily articles
n_days_total = (articles["date"].max() - articles["date"].min()).days + 1

source_stats = []
for src, grp in articles.groupby("source"):
    n_art = len(grp)
    pol_scores = grp["political_score"].dropna()
    eco_scores = grp["economic_score"].dropna()
    pol_nonzero = pol_scores[pol_scores > 0]
    eco_nonzero = eco_scores[eco_scores > 0]

    # Daily counts per source
    daily_counts = grp.groupby("date_only").size()
    n_days_src = len(daily_counts)
    mean_daily = daily_counts.mean()

    pct_pol_20 = (pol_scores >= 20).mean() * 100
    pct_eco_20 = (eco_scores >= 20).mean() * 100

    source_stats.append({
        "Source": src,
        "N_articles": n_art,
        "N_days_active": n_days_src,
        "Mean_daily_articles": round(mean_daily, 1),
        "Mean_pol_score_nonzero": round(pol_nonzero.mean(), 1) if len(pol_nonzero) > 0 else 0.0,
        "Mean_eco_score_nonzero": round(eco_nonzero.mean(), 1) if len(eco_nonzero) > 0 else 0.0,
        "Pct_pol_gt20": round(pct_pol_20, 1),
        "Pct_eco_gt20": round(pct_eco_20, 1),
    })

tab1 = pd.DataFrame(source_stats).sort_values("N_articles", ascending=False)
print(tab1.to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────────
# TABLE 2 — Score distribution histogram
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("TABLE 2 — Score Distribution (decile bins)")
print("=" * 72)

pol_scores_all = articles["political_score"].dropna()
eco_scores_all = articles["economic_score"].dropna()

bins = [0, 1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
bin_labels = ["0", "1-10", "11-20", "21-30", "31-40", "41-50",
              "51-60", "61-70", "71-80", "81-90", "91-100"]

pol_counts = pd.cut(pol_scores_all, bins=bins, labels=bin_labels, right=False).value_counts().sort_index()
eco_counts = pd.cut(eco_scores_all, bins=bins, labels=bin_labels, right=False).value_counts().sort_index()

tab2 = pd.DataFrame({
    "Score_bin": bin_labels,
    "N_political": [pol_counts.get(b, 0) for b in bin_labels],
    "Pct_political": [round(pol_counts.get(b, 0) / len(pol_scores_all) * 100, 1) for b in bin_labels],
    "N_economic": [eco_counts.get(b, 0) for b in bin_labels],
    "Pct_economic": [round(eco_counts.get(b, 0) / len(eco_scores_all) * 100, 1) for b in bin_labels],
})
print(tab2.to_string(index=False))

print(f"\n  Total political scores: {len(pol_scores_all):,}")
print(f"  % political articles (pol > 20): {(pol_scores_all > 20).mean()*100:.1f}%")
print(f"  Total economic scores: {len(eco_scores_all):,}")
print(f"  % economic articles (eco > 20): {(eco_scores_all > 20).mean()*100:.1f}%")

# ──────────────────────────────────────────────────────────────────────────────
# TABLE 3 — Parameter sensitivity (week of March 14-18 2026)
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("TABLE 3 — Parameter Sensitivity (March 14-18, 2026)")
print("=" * 72)

# Load articles for re-computation
arts_recompute = articles.copy()
arts_recompute["political_score"] = pd.to_numeric(arts_recompute["political_score"], errors="coerce").fillna(0.0)
arts_recompute["economic_score"] = pd.to_numeric(arts_recompute["economic_score"], errors="coerce").fillna(0.0)

WEEK_DATES = pd.date_range("2026-03-14", "2026-03-18", freq="D")

def compute_irp_ire(arts, date_range, baseline_window, breadth_threshold, beta_pol, beta_eco):
    """Recompute IRP/IRE for a given date range with specified parameters."""
    # Full series needed for rolling baseline
    all_dates = pd.date_range(arts["date"].min(), arts["date"].max(), freq="D")

    rows = []
    for day in all_dates:
        day_df = arts[arts["date"] == day]
        n_total = len(day_df)
        if n_total == 0:
            rows.append({"date": day, "raw_pol": 0.0, "raw_eco": 0.0,
                         "n_pol_above": 0, "n_eco_above": 0, "n_total": 0})
            continue
        pol = day_df["political_score"].fillna(0.0)
        eco = day_df["economic_score"].fillna(0.0)
        rows.append({
            "date": day,
            "raw_pol": float(pol.sum()),
            "raw_eco": float(eco.sum()),
            "n_pol_above": int((pol >= breadth_threshold).sum()),
            "n_eco_above": int((eco >= breadth_threshold).sum()),
            "n_total": n_total,
        })

    result = pd.DataFrame(rows)
    result["baseline_pol"] = result["raw_pol"].rolling(window=baseline_window, min_periods=1).mean()
    result["baseline_eco"] = result["raw_eco"].rolling(window=baseline_window, min_periods=1).mean()
    result["intensity_pol"] = np.where(result["baseline_pol"] > 0,
                                       result["raw_pol"] / result["baseline_pol"] * 100.0, 0.0)
    result["intensity_eco"] = np.where(result["baseline_eco"] > 0,
                                       result["raw_eco"] / result["baseline_eco"] * 100.0, 0.0)
    n_tot = result["n_total"].astype(float)
    result["breadth_pol"] = np.where(n_tot > 0, result["n_pol_above"] / n_tot, 0.0)
    result["breadth_eco"] = np.where(n_tot > 0, result["n_eco_above"] / n_tot, 0.0)
    result["irp"] = result["intensity_pol"] * (result["breadth_pol"] ** beta_pol)
    result["ire"] = result["intensity_eco"] * (result["breadth_eco"] ** beta_eco)
    return result.set_index("date")

# Sensitivity grid
tab3_rows = []
for beta_p in [0.3, 0.5, 0.7]:
    for beta_e in [0.2, 0.3, 0.5]:
        for window in [60, 90, 120]:
            sens = compute_irp_ire(arts_recompute, WEEK_DATES,
                                   baseline_window=window, breadth_threshold=20,
                                   beta_pol=beta_p, beta_eco=beta_e)
            week = sens.loc[sens.index.isin(WEEK_DATES)]
            for date in WEEK_DATES:
                if date in week.index:
                    row = week.loc[date]
                    tab3_rows.append({
                        "beta_pol": beta_p,
                        "beta_eco": beta_e,
                        "window": window,
                        "date": date.strftime("%Y-%m-%d"),
                        "IRP": round(float(row["irp"]), 1),
                        "IRE": round(float(row["ire"]), 1),
                    })

tab3 = pd.DataFrame(tab3_rows)
# Pivot to show March 14-18 averages per param combination
tab3_agg = tab3.groupby(["beta_pol", "beta_eco", "window"])[["IRP", "IRE"]].mean().round(1).reset_index()
tab3_agg.columns = ["beta_pol", "beta_eco", "window_days", "IRP_avg", "IRE_avg"]
print(tab3_agg.to_string(index=False))

# Also show daily detail for baseline case
print("\n  Baseline (beta_pol=0.5, beta_eco=0.3, window=90) — daily detail:")
base_sens = compute_irp_ire(arts_recompute, WEEK_DATES,
                            baseline_window=90, breadth_threshold=20,
                            beta_pol=0.5, beta_eco=0.3)
for date in WEEK_DATES:
    if date in base_sens.index:
        row = base_sens.loc[date]
        print(f"    {date.date()}  IRP={row['irp']:.1f}  IRE={row['ire']:.1f}  "
              f"breadth_pol={row['breadth_pol']:.3f}  breadth_eco={row['breadth_eco']:.3f}")

# ──────────────────────────────────────────────────────────────────────────────
# TABLE 4 — Event detection
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("TABLE 4 — Event Detection (known events vs index spikes)")
print("=" * 72)

events_list = [
    ("Cerrón Desaparece (prófugo)", "2025-03-15"),
    ("Impunidad Boluarte", "2025-04-20"),
    ("Vizcarra Prisión Preventiva", "2025-06-15"),
    ("Fragmentación Izquierda Peruana", "2025-07-20"),
    ("Colapso Ministerial Boluarte", "2025-08-20"),
    ("Paro Transportistas", "2025-10-10"),
    ("Vacancia Boluarte", "2025-10-25"),
    ("Castillo Condenado", "2025-11-15"),
    ("Censura Jeri", "2026-01-20"),
    ("Paro Transportistas (segunda ola)", "2026-02-05"),
    ("Crisis Energética (Camisea + Petroperú)", "2026-03-10"),
    ("Voto de confianza gabinete Miralles", "2026-03-15"),
]

tab4_rows = []
df_indexed = df.set_index("date")

for event_name, event_date_str in events_list:
    event_date = pd.Timestamp(event_date_str)
    window_start = event_date - pd.Timedelta(days=3)
    window_end = event_date + pd.Timedelta(days=3)
    window_df = df_indexed.loc[
        (df_indexed.index >= window_start) & (df_indexed.index <= window_end)
    ]
    if window_df.empty:
        tab4_rows.append({
            "Event": event_name, "Date": "—", "IRP_raw": "—", "IRP_smooth": "—",
            "IRE_raw": "—", "IRE_smooth": "—", "N_articles": "—",
            "breadth_pol": "—", "breadth_eco": "—",
        })
        continue

    # Date with highest IRP within window
    peak_idx = window_df["political_index"].idxmax()
    peak = window_df.loc[peak_idx]

    tab4_rows.append({
        "Event": event_name,
        "Date": peak_idx.strftime("%Y-%m-%d"),
        "IRP_raw": round(float(peak["political_index"]), 1),
        "IRP_smooth": round(float(peak.get("political_smooth", peak.get("irp_smooth", float("nan")))), 1),
        "IRE_raw": round(float(peak["economic_index"]), 1),
        "IRE_smooth": round(float(peak.get("economic_smooth", peak.get("ire_smooth", float("nan")))), 1),
        "N_articles": int(peak["n_articles_total"]),
        "breadth_pol": round(float(peak["breadth_pol"]), 3),
        "breadth_eco": round(float(peak["breadth_eco"]), 3),
    })

tab4 = pd.DataFrame(tab4_rows)
print(tab4.to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────────
# TABLE 5 — FX correlation
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("TABLE 5 — FX Correlation (IRP/IRE vs PEN/USD)")
print("=" * 72)

# Build FX series from JSON
fx_series = pol_json.get("daily_fx_series", [])
fx_df = pd.DataFrame(fx_series)
fx_df["date"] = pd.to_datetime(fx_df["date"])
fx_df = fx_df.set_index("date").sort_index()
fx_df["fx_change"] = fx_df["fx"].pct_change() * 100  # daily % change

# Merge with daily index
merged = df.set_index("date").join(fx_df, how="inner")
merged = merged.dropna(subset=["fx_change", "political_index", "economic_index"])

print(f"\n  Merged daily series: {len(merged)} days with both IRP and FX data")

# Daily correlation
r_irp_daily, p_irp_daily = scipy_stats.pearsonr(
    merged["political_index"], merged["fx_change"]
) if len(merged) > 5 else (float("nan"), float("nan"))

r_ire_daily, p_ire_daily = scipy_stats.pearsonr(
    merged["economic_index"], merged["fx_change"]
) if len(merged) > 5 else (float("nan"), float("nan"))

# Weekly averages
merged["year_week"] = merged.index.to_period("W")
weekly = merged.groupby("year_week").agg({
    "political_index": "mean",
    "economic_index": "mean",
    "fx_change": "sum",  # weekly cumulative FX change
}).dropna()

r_irp_weekly, p_irp_weekly = scipy_stats.pearsonr(
    weekly["political_index"], weekly["fx_change"]
) if len(weekly) > 5 else (float("nan"), float("nan"))

r_ire_weekly, p_ire_weekly = scipy_stats.pearsonr(
    weekly["economic_index"], weekly["fx_change"]
) if len(weekly) > 5 else (float("nan"), float("nan"))

# Monthly averages
merged["year_month"] = merged.index.to_period("M")
monthly = merged.groupby("year_month").agg({
    "political_index": "mean",
    "economic_index": "mean",
    "fx_change": "sum",
}).dropna()

r_irp_monthly, p_irp_monthly = scipy_stats.pearsonr(
    monthly["political_index"], monthly["fx_change"]
) if len(monthly) > 5 else (float("nan"), float("nan"))

r_ire_monthly, p_ire_monthly = scipy_stats.pearsonr(
    monthly["economic_index"], monthly["fx_change"]
) if len(monthly) > 5 else (float("nan"), float("nan"))

tab5 = pd.DataFrame([
    {"Frequency": "Daily",   "IRP_r": round(r_irp_daily, 3),   "IRP_p": round(p_irp_daily, 3),
     "IRE_r": round(r_ire_daily, 3),   "IRE_p": round(p_ire_daily, 3),
     "N_obs": len(merged)},
    {"Frequency": "Weekly",  "IRP_r": round(r_irp_weekly, 3),  "IRP_p": round(p_irp_weekly, 3),
     "IRE_r": round(r_ire_weekly, 3),  "IRE_p": round(p_ire_weekly, 3),
     "N_obs": len(weekly)},
    {"Frequency": "Monthly", "IRP_r": round(r_irp_monthly, 3), "IRP_p": round(p_irp_monthly, 3),
     "IRE_r": round(r_ire_monthly, 3), "IRE_p": round(p_ire_monthly, 3),
     "N_obs": len(monthly)},
])
print(tab5.to_string(index=False))

print(f"\n  Note: FX = PEN/USD. Higher FX = more soles per dollar = PEN depreciation.")
print(f"  Monthly IRP data points: {len(monthly)}")
print(f"  Monthly IRE data points: {len(monthly)}")

# ──────────────────────────────────────────────────────────────────────────────
# TABLE 6 — Summary statistics of IRP and IRE
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("TABLE 6 — Summary Statistics of Daily IRP and IRE")
print("=" * 72)

# Use only days with articles (exclude empty days)
df_valid = df[df["n_articles_total"] > 0].copy()
irp = df_valid["political_index"]
ire = df_valid["economic_index"]
irp_s = df_valid["political_smooth"]
ire_s = df_valid["economic_smooth"]

def summary_stats(series, name):
    s = series.dropna()
    ac1 = float(s.autocorr(lag=1)) if len(s) > 1 else float("nan")
    ac7 = float(s.autocorr(lag=7)) if len(s) > 7 else float("nan")
    ac30 = float(s.autocorr(lag=30)) if len(s) > 30 else float("nan")
    return {
        "Series": name,
        "Mean": round(s.mean(), 2),
        "Median": round(s.median(), 2),
        "Std": round(s.std(), 2),
        "Min": round(s.min(), 2),
        "Max": round(s.max(), 2),
        "Skewness": round(float(scipy_stats.skew(s)), 3),
        "Kurtosis": round(float(scipy_stats.kurtosis(s)), 3),
        "AC_lag1": round(ac1, 3),
        "AC_lag7": round(ac7, 3),
        "AC_lag30": round(ac30, 3),
        "N": len(s),
    }

tab6 = pd.DataFrame([
    summary_stats(irp, "IRP (raw)"),
    summary_stats(irp_s, "IRP (smoothed)"),
    summary_stats(ire, "IRE (raw)"),
    summary_stats(ire_s, "IRE (smoothed)"),
])
print(tab6.to_string(index=False))

print(f"\n  Date range of valid days: {df_valid['date'].min().date()} to {df_valid['date'].max().date()}")
print(f"  Total valid days (n_articles > 0): {len(df_valid)}")
print(f"  Total days in sample (incl. empty): {len(df)}")

# ──────────────────────────────────────────────────────────────────────────────
# ADDITIONAL STATISTICS
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("ADDITIONAL STATISTICS")
print("=" * 72)

# Date range
print(f"\n  Full date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"  Total days in sample: {len(df)}")
print(f"  Total articles processed: {len(articles):,}")

pol_pct = (pol_scores_all > 20).mean() * 100
eco_pct = (eco_scores_all > 20).mean() * 100
print(f"  % articles with political_score > 20: {pol_pct:.1f}%")
print(f"  % articles with economic_score  > 20: {eco_pct:.1f}%")

# IRP/IRE daily correlation
r_irp_ire, p_irp_ire = scipy_stats.pearsonr(
    df_valid["political_index"].dropna(),
    df_valid["economic_index"].dropna()
)
print(f"\n  IRP vs IRE daily correlation: r={r_irp_ire:.3f} (p={p_irp_ire:.3f})")

# Monthly IRP/IRE correlation
df_valid_m = df_valid.set_index("date")
df_valid_m["month"] = df_valid_m.index.to_period("M")
monthly_irp_ire = df_valid_m.groupby("month")[["political_index","economic_index"]].mean().dropna()
r_monthly_irp_ire, p_monthly_irp_ire = scipy_stats.pearsonr(
    monthly_irp_ire["political_index"], monthly_irp_ire["economic_index"]
) if len(monthly_irp_ire) > 5 else (float("nan"), float("nan"))
print(f"  IRP vs IRE monthly correlation: r={r_monthly_irp_ire:.3f} (p={p_monthly_irp_ire:.3f})")

# High IRP episodes (consecutive days above 150)
THRESHOLD_HIGH = 150
irp_series = df_valid.set_index("date")["political_index"]
above = (irp_series >= THRESHOLD_HIGH).astype(int)
episode_lengths = []
current_len = 0
for val in above.values:
    if val:
        current_len += 1
    elif current_len > 0:
        episode_lengths.append(current_len)
        current_len = 0
if current_len > 0:
    episode_lengths.append(current_len)

n_episodes = len(episode_lengths)
avg_episode = np.mean(episode_lengths) if episode_lengths else 0.0
days_above_150 = int(above.sum())
print(f"\n  Days with IRP >= 150: {days_above_150}")
print(f"  Distinct high-IRP episodes: {n_episodes}")
print(f"  Average episode duration (consecutive days >= 150): {avg_episode:.1f} days")
print(f"  Episode lengths: {sorted(episode_lengths, reverse=True)}")

# IRE high episodes
ire_series = df_valid.set_index("date")["economic_index"]
above_ire = (ire_series >= THRESHOLD_HIGH).astype(int)
episode_lengths_ire = []
current_len = 0
for val in above_ire.values:
    if val:
        current_len += 1
    elif current_len > 0:
        episode_lengths_ire.append(current_len)
        current_len = 0
if current_len > 0:
    episode_lengths_ire.append(current_len)
days_above_150_ire = int(above_ire.sum())
avg_episode_ire = np.mean(episode_lengths_ire) if episode_lengths_ire else 0.0
print(f"\n  Days with IRE >= 150: {days_above_150_ire}")
print(f"  Distinct high-IRE episodes: {len(episode_lengths_ire)}")
print(f"  Average IRE episode duration: {avg_episode_ire:.1f} days")

# Weekday analysis
df_valid_wd = df_valid.copy()
df_valid_wd["weekday"] = df_valid_wd["date"].dt.day_name()
weekday_stats = df_valid_wd.groupby("weekday")[["political_index","economic_index"]].mean()
print("\n  IRP/IRE by day of week:")
print(weekday_stats.round(1).to_string())

# Source concentration
src_counts = articles.groupby("source").size().sort_values(ascending=False)
top3 = src_counts.head(3)
print(f"\n  Top 3 sources by article count:")
for src, cnt in top3.items():
    print(f"    {src}: {cnt:,} ({cnt/len(articles)*100:.1f}%)")

# Peak values
print(f"\n  IRP all-time max: {irp.max():.1f} on {df_valid.loc[irp.idxmax(), 'date'].date()}")
print(f"  IRE all-time max: {ire.max():.1f} on {df_valid.loc[ire.idxmax(), 'date'].date()}")
print(f"  IRP all-time min (days w articles): {irp.min():.1f} on {df_valid.loc[irp.idxmin(), 'date'].date()}")
print(f"  IRE all-time min (days w articles): {ire.min():.1f} on {df_valid.loc[ire.idxmin(), 'date'].date()}")

# Monthly averages for the paper
print("\n  Monthly IRP/IRE averages:")
monthly_irp_ire_tab = df_valid_m.groupby("month")[["political_index","economic_index","n_articles_total"]].mean()
print(monthly_irp_ire_tab.round(1).to_string())

# FX descriptive statistics
fx_df_reset = fx_df.reset_index()
print(f"\n  FX series: {len(fx_df_reset)} daily observations")
print(f"  FX range: {fx_df['fx'].min():.4f} – {fx_df['fx'].max():.4f} PEN/USD")
print(f"  FX mean: {fx_df['fx'].mean():.4f}")
print(f"  FX std: {fx_df['fx'].std():.4f}")

print("\n" + "=" * 72)
print("DONE — all statistics computed")
print("=" * 72)
