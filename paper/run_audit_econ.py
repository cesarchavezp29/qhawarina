"""Economic-dimension false positive audit — Section 5.8"""
import pandas as pd, numpy as np, anthropic, json, time, re
from pathlib import Path

PARQUET = Path("D:/Nexus/nexus/data/raw/rss/articles_classified.parquet")
OUT     = Path("D:/Nexus/nexus/paper/audit_results_econ.json")
N_PER_QUINTILE = 100
SEED = 42

df = pd.read_parquet(PARQUET)
df = df[df["economic_score"] > 0].dropna(subset=["economic_score","title","summary"])
df = df[df["title"].str.len() > 10]

def assign_quintile(s):
    if s <= 10:  return "Q1"
    elif s <= 15: return "Q2"
    elif s <= 20: return "Q3"
    elif s <= 35: return "Q4"
    else:         return "Q5"

df["quintile"] = df["economic_score"].apply(assign_quintile)

sample_frames = []
for q in ["Q1","Q2","Q3","Q4","Q5"]:
    qdf = df[df["quintile"] == q]
    n = min(N_PER_QUINTILE, len(qdf))
    sample_frames.append(qdf.sample(n, random_state=SEED))

sample = pd.concat(sample_frames).reset_index(drop=True)
print(f"Total sample: {len(sample)} articles")
print(sample.groupby("quintile")["economic_score"].describe()[["min","max","mean","count"]])

client = anthropic.Anthropic()

SYSTEM = """You are an expert at scoring Peruvian news articles for economic risk content.
Given an article title and summary, output ONLY a JSON object with:
- "economic_score": integer 0-100 (0=no economic risk content, 100=major economic shock/crisis)
- "political_score": integer 0-100
- "category": one of ["no_economic_risk","routine_economic_data","sectoral_news","policy_fiscal","commodity_price","foreign_economic","labor_employment","financial_markets","opinion_economic","other"]
- "rationale": 1 sentence

Economic scoring guide:
0-10: No economic risk or routine statistics with no policy implications
11-25: Minor economic mentions, sectoral news, commodity prices without Peru impact
26-50: Moderate economic events (policy changes, sector-level shocks, significant price moves)
51-75: Significant economic events (major policy shift, large market move, sector crisis)
76-100: Crisis-level (market crash, bank failure, sovereign default risk, major fiscal emergency)"""

results = []
errors = 0
for i, row in sample.iterrows():
    text = f"Title: {row['title']}\nSummary: {str(row['summary'])[:500]}"
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=300, system=SYSTEM,
            messages=[{"role":"user","content": text}]
        )
        raw = msg.content[0].text.strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        parsed = json.loads(m.group() if m else raw)
        results.append({
            "idx": int(i), "quintile": str(row["quintile"]),
            "original_pol": float(row["political_score"]),
            "original_econ": float(row["economic_score"]),
            "audit_pol": int(parsed.get("political_score", -1)),
            "audit_econ": int(parsed.get("economic_score", -1)),
            "category": parsed.get("category","unknown"),
            "rationale": parsed.get("rationale",""),
            "title": row["title"][:120], "source": row["source"],
        })
        if len(results) % 50 == 0:
            print(f"  {len(results)}/{len(sample)} done...")
        time.sleep(0.05)
    except Exception as e:
        errors += 1
        results.append({"idx":int(i),"quintile":str(row["quintile"]),"original_pol":float(row["political_score"]),
            "original_econ":float(row["economic_score"]),"audit_pol":-1,"audit_econ":-1,
            "category":"error","rationale":"API error","title":row["title"][:120],"source":row["source"]})

with open(OUT,"w",encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nDone. {len(results)} articles, {errors} errors.")

rdf = pd.DataFrame([r for r in results if r["audit_econ"] >= 0])
rdf["delta"] = rdf["audit_econ"] - rdf["original_econ"]
rdf["agree"] = rdf["delta"].abs() <= 20
rdf["agree15"] = rdf["delta"].abs() <= 15

print("\n=== AGREEMENT BY QUINTILE ===")
print(rdf.groupby("quintile")[["agree","agree15","delta"]].agg({"agree":"mean","agree15":"mean","delta":"mean"}).round(3))
print("\n=== CATEGORY DISTRIBUTION ===")
print(rdf["category"].value_counts())
print("\n=== OVERALL AGREEMENT (±20) ===", rdf["agree"].mean().round(3))
print("=== MEAN DELTA ===", rdf["delta"].mean().round(2))

# False positives: original>=25, audit<10
fp = rdf[(rdf["original_econ"] >= 25) & (rdf["audit_econ"] < 10)]
hi = rdf[rdf["original_econ"] >= 25]
print(f"\n=== FALSE POSITIVES (orig>=25, audit<10): {len(fp)} of {len(hi)} ({100*len(fp)/max(len(hi),1):.1f}%) ===")
if len(fp): print(fp[["quintile","original_econ","audit_econ","category","title"]].to_string())

# Overscoring
over = rdf[rdf["delta"] < -20]
print(f"\n=== OVERSCORING (audit < original - 20): {len(over)} ({100*len(over)/len(rdf):.1f}%) ===")
print(over.groupby("category")["delta"].agg(["count","mean"]).round(2))
