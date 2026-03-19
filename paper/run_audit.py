"""
False Positive Audit — Section 5.7
Sample 100 articles per political_score quintile, re-score with Claude claude-haiku-4-5-20251001,
compute agreement and false-positive rates.
"""

import pandas as pd
import numpy as np
import anthropic
import json, time, re
from pathlib import Path

PARQUET = Path("D:/Nexus/nexus/data/raw/rss/articles_classified.parquet")
OUT     = Path("D:/Nexus/nexus/paper/audit_results.json")
N_PER_QUINTILE = 100
SEED = 42

df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["political_score","title","summary"])
df = df[df["title"].str.len() > 10]

# Use fixed score buckets instead of qcut (many zeros break quantile edges)
# Q1: 0, Q2: 1-15, Q3: 16-35, Q4: 36-60, Q5: 61-100
def assign_quintile(s):
    if s == 0:   return "Q1"
    elif s <= 15: return "Q2"
    elif s <= 35: return "Q3"
    elif s <= 60: return "Q4"
    else:         return "Q5"

df["quintile"] = df["political_score"].apply(assign_quintile)

sample_frames = []
for q in ["Q1","Q2","Q3","Q4","Q5"]:
    qdf = df[df["quintile"] == q]
    n = min(N_PER_QUINTILE, len(qdf))
    sample_frames.append(qdf.sample(n, random_state=SEED))

sample = pd.concat(sample_frames).reset_index(drop=True)
print(f"Total sample: {len(sample)} articles across 5 quintiles")
print(sample.groupby("quintile")["political_score"].describe()[["min","max","mean","count"]])

client = anthropic.Anthropic()

SYSTEM = """You are an expert at scoring Peruvian news articles for political and economic risk content.
Given an article title and summary, output ONLY a JSON object with:
- "political_score": integer 0-100 (0=no political content, 100=major political crisis/event)
- "economic_score": integer 0-100 (0=no economic content, 100=major economic shock)
- "category": one of ["no_risk","campaign_electoral","political_governance","economic_domestic","economic_foreign","social_conflict","crime_security","opinion_column","other"]
- "rationale": 1 sentence explanation

Scoring guide:
0-10: No relevant content or purely informational
11-30: Minor political/economic mentions
31-60: Moderate political/economic events (campaign news, policy debates, routine economic data)
61-80: Significant events (minister resignation, large protests, major economic shift)
81-100: Crisis-level events (constitutional crisis, market crash, state of emergency)"""

results = []
errors = 0

for i, row in sample.iterrows():
    text = f"Title: {row['title']}\nSummary: {str(row['summary'])[:500]}"
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=SYSTEM,
            messages=[{"role":"user","content": text}]
        )
        raw = msg.content[0].text.strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
        else:
            parsed = json.loads(raw)
        results.append({
            "idx": int(i),
            "quintile": str(row["quintile"]),
            "original_pol": float(row["political_score"]),
            "original_econ": float(row["economic_score"]),
            "audit_pol": int(parsed.get("political_score", -1)),
            "audit_econ": int(parsed.get("economic_score", -1)),
            "category": parsed.get("category","unknown"),
            "rationale": parsed.get("rationale",""),
            "title": row["title"][:120],
            "source": row["source"],
        })
        if len(results) % 50 == 0:
            print(f"  {len(results)}/{len(sample)} done...")
        time.sleep(0.05)
    except Exception as e:
        errors += 1
        print(f"  Error on row {i}: {e}")
        results.append({
            "idx": int(i), "quintile": str(row["quintile"]),
            "original_pol": float(row["political_score"]), "original_econ": float(row["economic_score"]),
            "audit_pol": -1, "audit_econ": -1, "category":"error","rationale":"API error",
            "title":row["title"][:120],"source":row["source"]
        })

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nDone. {len(results)} articles scored, {errors} errors. Saved to {OUT}")

rdf = pd.DataFrame([r for r in results if r["audit_pol"] >= 0])
rdf["delta"] = rdf["audit_pol"] - rdf["original_pol"]
rdf["agree"] = rdf["delta"].abs() <= 20

print("\n=== AGREEMENT BY QUINTILE (within +/-20 pts) ===")
print(rdf.groupby("quintile")[["agree","delta"]].agg({"agree":"mean","delta":"mean"}).round(3))
print("\n=== CATEGORY DISTRIBUTION ===")
print(rdf["category"].value_counts())
print("\n=== OVERALL AGREEMENT ===", rdf["agree"].mean().round(3))
print("=== MEAN DELTA (audit - original) ===", rdf["delta"].mean().round(2))

# False positive rate: original >= 40, audit < 20
fp = rdf[(rdf["original_pol"] >= 40) & (rdf["audit_pol"] < 20)]
print(f"\n=== FALSE POSITIVES (orig>=40, audit<20): {len(fp)} of {len(rdf[rdf['original_pol']>=40])} high-score articles ===")
if len(fp):
    print(fp[["quintile","original_pol","audit_pol","category","title"]].to_string())
