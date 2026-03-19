# Qhawarina Daily Pipeline — Complete Reference

**Last updated:** 2026-03-18
**Maintainer:** Qhawarina.pe
**Stack:** Python 3.11 · Windows Task Scheduler · Plotly · Claude Haiku (AI-GPR)

---

## Table of Contents

1. [What This Pipeline Does](#1-what-this-pipeline-does)
2. [Architecture Overview](#2-architecture-overview)
3. [Daily Schedule & Timing](#3-daily-schedule--timing)
4. [Step-by-Step Pipeline](#4-step-by-step-pipeline)
5. [All Data Sources](#5-all-data-sources)
6. [Key Files & Directories](#6-key-files--directories)
7. [Index Definitions (IRP, IRE, IPC, etc.)](#7-index-definitions-irp-ire-ipc-etc)
8. [How to Check If the Pipeline Ran Today](#8-how-to-check-if-the-pipeline-ran-today)
9. [How to Run Manually](#9-how-to-run-manually)
10. [Monthly Steps (runs on the 15th)](#10-monthly-steps-runs-on-the-15th)
11. [Known Bugs & Gotchas](#11-known-bugs--gotchas)
12. [Troubleshooting](#12-troubleshooting)
13. [Output Files → Website Mapping](#13-output-files--website-mapping)

---

## 1. What This Pipeline Does

Qhawarina is a real-time economic intelligence platform for Peru. Every day, the pipeline:

- **Scrapes** ~42,000 supermarket product prices (Plaza Vea, Metro, Wong)
- **Builds** a political and economic instability index from Peruvian news RSS feeds, classified by Claude Haiku
- **Exports** all nowcast data (GDP, inflation, poverty, political risk) as JSON/CSV files
- **Syncs** those files to the website's `/public/assets/data/` directory

The pipeline does **not** retrain models daily. Models are trained separately (see `scripts/train_*.py`). The pipeline only runs inference on the latest data.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WINDOWS TASK SCHEDULER                       │
│                    Triggers at 21:00 Lima (CT)                      │
│                         = 02:00 UTC next day                        │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
                  run_daily_pipeline.py
                            │
          ┌─────────────────┼──────────────────┐
          │                 │                  │
          ▼                 ▼                  ▼
  STEP 1: Scrape     STEP 2: Political   STEP 3: Export
  Supermarkets       Index (RSS+Haiku)   Web Data
          │                 │                  │
          ▼                 ▼                  ▼
  supermarket/       daily_instability/  exports/
  YYYY-MM-DD.parq    daily_index.parq    *.json / *.csv
                                               │
                                               ▼
                                        STEP 4: Sync
                                        to D:/qhawarina/
                                        public/assets/data/
                                               │
                                    (optional) ▼
                                        STEP 5: Git Push
                                        (--push flag only)
```

---

## 3. Daily Schedule & Timing

| Time (Lima / PET) | Time (UTC) | Event |
|---|---|---|
| 21:00 | 02:00 +1d | Pipeline starts |
| 21:00–21:20 | 02:00–02:20 | Supermarket scrape (3 stores, ~20 min) |
| 21:20–21:28 | 02:20–02:28 | Political index RSS fetch + Haiku classification |
| 21:28–21:31 | 02:28–02:31 | Export web data (all JSON/CSV generation) |
| 21:31 | 02:31 | Sync to website directory |
| — | — | Git push (disabled by default, use `--push`) |

**Total runtime:** ~30 minutes

**Note on timezone:** Lima is UTC-5 (PET, no DST). The pipeline runs at 21:00 Lima = 02:00 UTC next calendar day. This means log files dated "2026-03-18" contain timestamps from 21:00–21:31 Lima, which appear as 02:00–02:31 UTC on 2026-03-19. RSS articles fetched near midnight Lima can be assigned to the *next* calendar day — see Known Bugs.

---

## 4. Step-by-Step Pipeline

### Step 1 — Supermarket Scrape

**Script:** inline in `run_daily_pipeline.py` → `src/ingestion/supermarket.py`

Scrapes all products from:
- **Plaza Vea** (~15,000 products)
- **Metro** (~12,000 products)
- **Wong** (~15,000 products)

Saves daily snapshot parquet: `data/raw/supermarket/YYYY-MM-DD.parquet`

Fields: `store`, `product_name`, `price`, `unit`, `category`, `sku`, `scraped_at`

Success indicator in log:
```
Saved 42654 products for 2026-03-18
  metro       12447
  plazavea    15282
  wong        14925
```

---

### Step 2 — Political Instability Index

**Script:** `scripts/build_daily_index.py`

1. Fetches RSS feeds from ~13 Peruvian news sources (configured in `config/rss_feeds.yaml`)
2. Deduplicates articles (URL hash + title similarity)
3. Classifies each article with **Claude Haiku** (AI-GPR methodology)
   - Returns: `is_political` (bool), `is_economic` (bool), `severity_score` (0–100)
   - **Fallback:** if `ANTHROPIC_API_KEY` is not set → keyword classifier (lower quality)
4. Builds IRP and IRE using the Intensity × Breadth^β formula (Iacoviello & Tong 2026)
5. Saves to: `data/processed/daily_instability/daily_index.parquet`

**CRITICAL:** The parquet at `data/processed/daily_instability/daily_index.parquet` is the **single source of truth** for IRP/IRE. The old `data/processed/daily/daily_index.parquet` is a legacy file stuck at 2026-03-13 — ignore it.

---

### Step 3 — Export Web Data

**Script:** `scripts/export_web_data.py`

Reads all model outputs and produces JSON/CSV for the website:

| Output file | Content |
|---|---|
| `exports/political_risk.json` | IRP + IRE daily series + today's value |
| `exports/gdp_nowcast.json` | Quarterly GDP nowcast |
| `exports/inflation_nowcast.json` | Monthly CPI nowcast |
| `exports/poverty_nowcast.json` | Annual poverty by department |
| `exports/supermarket_prices.json` | Daily price index |
| `exports/poverty_districts_full.csv` | 1,891 districts with poverty proxy |

The `[AI-GPR] IRP today = X  IRE today = Y` line printed during this step reports the **last complete day's** IRP and IRE (filtered to rows with ≥100 articles to avoid partial-day UTC bleed — see Known Bugs).

---

### Step 4 — Sync to Web

**Script:** `scripts/sync_web_data.py`

Copies all files from `exports/` to `D:/qhawarina/public/assets/data/`.

---

### Step 5 — Git Push (optional)

Run with `--push` flag. Commits and pushes changed files in the qhawarina repo.

```bash
python scripts/run_daily_pipeline.py --push
```

---

## 5. All Data Sources

| Source | What | Frequency | Script |
|---|---|---|---|
| Plaza Vea / Metro / Wong | Product prices | Daily | `src/ingestion/supermarket.py` |
| BCRP API | GDP, credit, electricity, employment, inflation | Monthly | `scripts/update_bcrp.py` |
| RSS feeds (13 sources) | News articles | Daily | `build_daily_index.py` |
| Claude Haiku (Anthropic) | Article classification | Daily | `build_daily_index.py` |
| INEI ENAHO | Official poverty rates | Annual (May pub.) | Manual import |
| NOAA-VIIRS (NTL) | Nighttime lights satellite | Monthly | `scripts/update_ntl_monthly.py` |
| MEF/SUNAT | Tax revenue, fiscal data | Monthly | `scripts/update_bcrp.py` |

---

## 6. Key Files & Directories

```
D:/nexus/nexus/
│
├── PIPELINE.md                         ← YOU ARE HERE
├── scripts/
│   ├── run_daily_pipeline.py           ← MAIN ENTRY POINT (run this)
│   ├── build_daily_index.py            ← Political/economic index from RSS
│   ├── export_web_data.py              ← Generates all JSON/CSV for website
│   ├── sync_web_data.py                ← Copies exports/ → qhawarina/public/
│   ├── generate_poverty_nowcast.py     ← Monthly poverty model (runs on 15th)
│   ├── build_poverty_nowcast.py        ← Builds vintage panel for poverty model
│   ├── update_bcrp.py                  ← Downloads BCRP indicator data
│   └── README.md                       ← Scripts reference
│
├── data/
│   ├── raw/
│   │   ├── supermarket/                ← Daily parquets YYYY-MM-DD.parquet
│   │   ├── rss/                        ← articles_cache.parquet (all articles)
│   │   │                                  articles_classified.parquet (labeled)
│   │   └── satellite/                  ← NTL monthly rasters
│   └── processed/
│       ├── daily_instability/
│       │   └── daily_index.parquet     ← IRP/IRE SOURCE OF TRUTH (updated daily)
│       ├── daily/
│       │   └── daily_index.parquet     ← LEGACY — stuck at 2026-03-13, IGNORE
│       ├── national/                   ← Monthly national panel
│       └── regional/                   ← Monthly departmental panel (25 depts)
│
├── results/                            ← Model outputs (nowcasts, backtests)
│   ├── backtest_gdp.parquet
│   ├── backtest_inflation.parquet
│   ├── backtest_poverty.parquet
│   └── district_poverty_nowcast.parquet
│
├── exports/                            ← Final web-ready JSON/CSV (copied to website)
│
├── logs/
│   ├── daily_pipeline_YYYY-MM-DD.log   ← One per day — check this first
│   ├── daily_index.log                 ← RSS classification log (verbose)
│   ├── update_bcrp.log                 ← BCRP download log
│   └── README.md                       ← Log reading guide
│
└── config/
    ├── settings.py                     ← All paths and constants
    └── rss_feeds.yaml                  ← RSS sources list
```

---

## 7. Index Definitions (IRP, IRE, IPC, etc.)

### IRP — Índice de Riesgo Político
Political instability index. Built daily from Peruvian news using the AI-GPR methodology (Iacoviello & Tong 2026):

```
IRP = Intensity_pol × Breadth_pol^β
```

- **Intensity:** average severity score of political articles (0–100 per article from Haiku)
- **Breadth:** share of RSS sources covering political instability that day
- **β = 0.5** (breadth exponent for political)
- **Scale:** 0–300+ (no fixed ceiling). Typical range: 40–150. Values >180 = high alert.
- **Yesterday (2026-03-17):** 197.2 — elevated
- **Today (2026-03-18):** 226.2 — highest recorded day

### IRE — Índice de Riesgo Económico
Economic instability index. Same formula as IRP but for economic articles:

- **β = 0.3** (lower because economic coverage is more concentrated)
- **Scale:** 0–400+. Typical range: 80–200. Values >250 = high alert.
- **Today (2026-03-18):** 163.0

### IPC (Supermarket) — Índice de Precios al Consumidor (Alta Frecuencia)
Daily price index built from supermarket scrape. Base = first observation.

```
IPC_daily = geometric_mean(price_t / price_base) × 100
```

Covers ~42,000 SKUs across 3 chains. Lima-biased. Excludes services (~50% of official CPI).

### IPD — Índice de Precios Diarios
Same as IPC above. Used interchangeably in the website.

### Poverty Nowcast
Departmental poverty rate for 2025, predicted using Gradient Boosting Regressor (GBR) trained on ENAHO 2004–2024 panel data. Updated monthly on the 15th. See `scripts/build_poverty_nowcast.py`.

---

## 8. How to Check If the Pipeline Ran Today

```bash
# 1. Check the log exists and has PIPELINE COMPLETE
cat D:/nexus/nexus/logs/daily_pipeline_2026-03-18.log

# 2. Quick status check
tail -15 D:/nexus/nexus/logs/daily_pipeline_$(date +%Y-%m-%d).log

# 3. Check the IRP/IRE values for today (from the source parquet)
python -c "
import pandas as pd
df = pd.read_parquet('D:/nexus/nexus/data/processed/daily_instability/daily_index.parquet')
print(df[['date','political_index','economic_index','n_articles_total']].tail(5).to_string())
"
```

A healthy log ends with:
```
PIPELINE COMPLETE — HH:MM:SS
  scrape       OK
  political    OK
  export       OK
  sync         OK
```

---

## 9. How to Run Manually

```bash
cd D:/nexus/nexus

# Full pipeline (no git push)
python scripts/run_daily_pipeline.py

# Full pipeline + git push to deploy
python scripts/run_daily_pipeline.py --push

# Skip scrape (prices already collected today)
python scripts/run_daily_pipeline.py --skip-scrape

# Skip political index (RSS already classified today)
python scripts/run_daily_pipeline.py --skip-political

# Just re-export and sync (fastest, no data collection)
python scripts/run_daily_pipeline.py --skip-scrape --skip-political

# Check what today's IRP/IRE would be without running the full pipeline
python scripts/export_web_data.py
```

---

## 10. Monthly Steps (runs on the 15th)

On the 15th of each month, the pipeline also runs:

1. **Regional BCRP download** — fetches latest departmental indicators (credit, electricity, employment, etc.)
2. **Regional panel rebuild** — rebuilds `panel_departmental_monthly.parquet`
3. **Poverty vintage panel** — rebuilds model input features via `build_poverty_nowcast.py`
4. **Generate poverty nowcast** — runs GBR model → `poverty_nowcast.json` → copies to website

Log indicators:
```
Monthly poverty: skipping (today is the 18th, runs on the 15th)   ← normal
MONTHLY POVERTY NOWCAST — 2026-03-15                               ← runs on 15th
```

To force-run the poverty nowcast on a non-15th day:
```bash
python scripts/generate_poverty_nowcast.py
```

---

## 11. Known Bugs & Gotchas

### BUG 1 — Partial-day UTC bleed in IRP/IRE log line (FIXED 2026-03-18)

**Symptom:** Log shows `IRP today = 12.5  IRE today = 9.4` but real IRP is ~200+.

**Root cause:** Pipeline runs at 21:00 Lima = 02:00 UTC next day. By export time (21:30 Lima), the RSS fetcher has already bucketed ~50 articles into "tomorrow" (e.g. 2026-03-19). The parquet gets a partial row for tomorrow with very few articles and artificially low IRP/IRE. Old code used `.iloc[-1]` which picked this partial row.

**Fix:** `export_web_data.py` now filters to rows with `n_articles_total >= 100` before computing today's value. The log now also prints the date and article count:
```
[AI-GPR] IRP today = 226.2  IRE today = 163.0  (date: 2026-03-18, articles: 932)
```

### BUG 2 — Two daily_index.parquet files (stale legacy file)

**Symptom:** If you read `data/processed/daily/daily_index.parquet`, you get data only up to 2026-03-13. This file is a legacy artifact.

**The correct parquet is:** `data/processed/daily_instability/daily_index.parquet` (updated daily, currently goes to 2026-03-19).

**Fix:** Do not use `data/processed/daily/daily_index.parquet`. It should be deleted or archived.

### BUG 3 — ANTHROPIC_API_KEY not set → keyword fallback

**Symptom:** Log shows `WARNING: ANTHROPIC_API_KEY not set — using keyword classifier fallback`. IRP/IRE values are lower quality.

**Fix:** Ensure `.env` file in `D:/nexus/nexus/` contains:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### GOTCHA — Git push is OFF by default

The pipeline does NOT push to GitHub automatically. Run with `--push` to deploy. This is intentional to allow review before publishing.

### GOTCHA — Poverty nowcast only on 15th

The poverty model runs on the 15th. On all other days, the log line `Monthly poverty: skipping (today is the Nth, runs on the 15th)` is normal and expected.

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| No log file for today | Task Scheduler didn't run | Check Windows Task Scheduler, ensure task is enabled |
| `scrape FAILED` | Supermarket site changed layout or blocked | Check `src/ingestion/supermarket.py`, test manually |
| `political FAILED` | RSS timeout or Haiku API error | Check `logs/daily_index.log` for details |
| `export FAILED` | Missing parquet or schema change | Check `logs/` for traceback, run `export_web_data.py` manually |
| IRP/IRE show 0 | `ANTHROPIC_API_KEY` missing | Add key to `.env` |
| Website not updated | Sync ran but no git push | Run with `--push` or manually copy exports/ |
| Poverty data stale | Monthly step failed on 15th | Run `python scripts/generate_poverty_nowcast.py` manually |
| `daily/daily_index.parquet` stale at 2026-03-13 | Legacy file, not updated | Use `daily_instability/daily_index.parquet` instead |

---

## 13. Output Files → Website Mapping

| Export file | Website page |
|---|---|
| `political_risk.json` | `/estadisticas/riesgo-politico` |
| `supermarket_prices.json` | `/estadisticas/inflacion/precios-alta-frecuencia` |
| `gdp_nowcast.json` | `/estadisticas/pbi` |
| `inflation_nowcast.json` | `/estadisticas/inflacion` |
| `poverty_nowcast.json` | `/estadisticas/pobreza` + `/estadisticas/pobreza/mapas` |
| `poverty_districts_full.csv` | `/estadisticas/pobreza/distritos` |
| `poverty_quarterly.json` | `/estadisticas/pobreza/trimestral` |
| `poverty_monthly.json` | `/estadisticas/pobreza/trimestral` |

All files live at: `D:/qhawarina/public/assets/data/`
