"""
Google Trends validation for IRP/IRE paper.
Downloads weekly Trends data for Peru search terms and correlates with IRP/IRE.
"""
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
from scipy import stats
import time

pytrends = TrendReq(hl='es-PE', tz=-300, timeout=(10, 25))

# Download in two batches (pytrends max 5 terms at once, but 2 is safer)
time.sleep(1)
pytrends.build_payload(['paro nacional', 'vacancia'], timeframe='2025-01-01 2026-03-18', geo='PE')
df1 = pytrends.interest_over_time().drop(columns='isPartial', errors='ignore')

time.sleep(2)
pytrends.build_payload(['crisis Peru', 'camisea'], timeframe='2025-01-01 2026-03-18', geo='PE')
df2 = pytrends.interest_over_time().drop(columns='isPartial', errors='ignore')

time.sleep(2)
pytrends.build_payload(['inestabilidad politica', 'congreso Peru'], timeframe='2025-01-01 2026-03-18', geo='PE')
df3 = pytrends.interest_over_time().drop(columns='isPartial', errors='ignore')

trends = df1.join(df2, how='outer').join(df3, how='outer')
trends.index = pd.to_datetime(trends.index).tz_localize(None)
trends.index.name = 'week_start'
trends = trends[trends.index >= '2025-01-01']

print("Trends shape:", trends.shape)
print(trends.describe())
print("\nTop weeks for 'paro nacional':")
print(trends.nlargest(5, 'paro nacional')[['paro nacional', 'vacancia', 'crisis Peru']])

# Load IRP/IRE and aggregate to weekly
irp_df = pd.read_parquet('D:/nexus/nexus/data/processed/daily_instability/daily_index.parquet')
irp_df['date'] = pd.to_datetime(irp_df['date'])
irp_df = irp_df[irp_df['date'] >= '2025-01-01']
# Google Trends uses Sunday-based weeks; align IRP/IRE to same convention
# dayofweek: Mon=0 ... Sun=6; offset to prior Sunday = (dayofweek + 1) % 7
irp_df['week_start'] = irp_df['date'] - pd.to_timedelta((irp_df['date'].dt.dayofweek + 1) % 7, unit='D')
weekly = irp_df.groupby('week_start').agg(
    IRP=('political_index', 'mean'),
    IRE=('economic_index', 'mean')
).reset_index()
weekly['week_start'] = pd.to_datetime(weekly['week_start'])
weekly = weekly.set_index('week_start')

# Merge
merged = trends.join(weekly, how='inner')
print(f"\nMerged shape: {merged.shape}")
print(merged.tail(5))

# Correlations
results = []
for term in trends.columns:
    sub = merged[[term, 'IRP', 'IRE']].dropna()
    if len(sub) < 10:
        continue
    r_irp, p_irp = stats.pearsonr(sub[term], sub['IRP'])
    r_ire, p_ire = stats.pearsonr(sub[term], sub['IRE'])
    results.append({
        'Term': term,
        'r_IRP': r_irp, 'p_IRP': p_irp,
        'r_IRE': r_ire, 'p_IRE': p_ire,
        'N': len(sub)
    })

results_df = pd.DataFrame(results).sort_values('r_IRP', ascending=False)
print("\n=== CORRELATION RESULTS ===")
print(results_df.to_string(index=False))

# Generate LaTeX table output
print("\n=== LaTeX TABLE ===")
print("\\begin{tabular}{lrrrrrr}")
print("\\toprule")
print("Search term (Google Trends, Peru) & $r(\\text{IRP})$ & $p$-value & $r(\\text{IRE})$ & $p$-value & $N$ \\\\")
print("\\midrule")
for _, row in results_df.iterrows():
    term = row['Term'].replace('_', ' ').title()
    print(f"{term} & ${row['r_IRP']:.3f}$ & {row['p_IRP']:.3f} & ${row['r_IRE']:.3f}$ & {row['p_IRE']:.3f} & {int(row['N'])} \\\\")
print("\\bottomrule")
print("\\end{tabular}")

# Save data
merged.to_csv('D:/nexus/nexus/paper/google_trends_irp.csv')
print("\nSaved to paper/google_trends_irp.csv")
