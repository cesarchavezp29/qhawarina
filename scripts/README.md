# Scripts Reference — Qhawarina Pipeline

> For the full pipeline overview, see [`../PIPELINE.md`](../PIPELINE.md)

---

## Entry Points (run these)

| Script | When to run | What it does |
|---|---|---|
| `run_daily_pipeline.py` | Daily (Task Scheduler) | Full daily pipeline: scrape → index → export → sync |
| `generate_poverty_nowcast.py` | Monthly (or on demand) | Runs poverty GBR model, exports JSON |
| `update_bcrp.py` | On demand | Downloads fresh BCRP indicator data |
| `export_web_data.py` | On demand | Re-exports all JSON/CSV to exports/ without re-collecting data |
| `sync_web_data.py` | On demand | Copies exports/ to D:/qhawarina/public/assets/data/ |

### Daily pipeline flags

```bash
python scripts/run_daily_pipeline.py                    # standard run (no push)
python scripts/run_daily_pipeline.py --push             # run + git commit + push
python scripts/run_daily_pipeline.py --skip-scrape      # skip supermarket scrape
python scripts/run_daily_pipeline.py --skip-political   # skip RSS/Haiku step
```

---

## Data Collection Scripts

| Script | Data collected | Output |
|---|---|---|
| `scrape_supermarket_prices.py` | Prices from Vea/Metro/Wong | `data/raw/supermarket/YYYY-MM-DD.parquet` |
| `update_bcrp.py` | Monthly BCRP national series | `data/raw/bcrp/*.parquet` |
| `download_regional_bcrp.py` | Monthly BCRP by department | `data/raw/bcrp/regional/*.parquet` |
| `update_ntl_monthly.py` | NOAA-VIIRS nighttime lights | `data/raw/satellite/` |
| `backfill_gdelt.py` | Historical GDELT news events | `data/raw/gdelt/` |

---

## Processing Scripts

| Script | What it builds | Output |
|---|---|---|
| `build_daily_index.py` | IRP + IRE daily instability indices | `data/processed/daily_instability/daily_index.parquet` |
| `build_panel.py` | National monthly feature panel | `data/processed/national/panel_national_monthly.parquet` |
| `build_regional_panel.py` | Departmental monthly feature panel | `data/processed/regional/panel_departmental_monthly.parquet` |
| `build_price_index.py` | Supermarket daily price index | `data/processed/prices/price_index_daily.parquet` |
| `build_poverty_nowcast.py` | Poverty model inputs (vintage panel) | `data/processed/regional/vintage_panel.parquet` |
| `build_financial_stress.py` | Financial stress composite | `data/processed/national/financial_stress.parquet` |
| `build_political_index.py` | [LEGACY] Old political index builder | — |

---

## Model / Nowcast Scripts

| Script | Model | Output |
|---|---|---|
| `generate_nowcast.py` | GDP + Inflation (GBR) | `results/backtest_gdp.parquet`, `results/backtest_inflation.parquet` |
| `generate_poverty_nowcast.py` | Poverty (GBR panel) | `results/district_poverty_nowcast.parquet`, `exports/poverty_nowcast.json` |
| `generate_poverty_quarterly.py` | Poverty quarterly (Chow-Lin) | `exports/poverty_quarterly.json` |
| `generate_poverty_monthly.py` | Poverty monthly (rolling) | `exports/poverty_monthly.json` |
| `generate_inflation_regional_nowcast.py` | Inflation by department | `exports/inflation_regional.json` |
| `run_inflation_backtest.py` | Inflation backtest (walk-forward) | `results/backtest_inflation.parquet` |
| `run_gdp_backtest.py` | GDP backtest | `results/backtest_gdp.parquet` |
| `run_poverty_backtest.py` | Poverty backtest (2012–2024) | `results/backtest_poverty.parquet` |

---

## Export & Sync Scripts

| Script | What it does |
|---|---|
| `export_web_data.py` | Reads all parquets, generates JSON/CSV in `exports/` |
| `sync_web_data.py` | Copies `exports/` → `D:/qhawarina/public/assets/data/` |
| `generate_csv_exports.py` | Generates downloadable CSV files for users |
| `generate_maps.py` | Generates static map PNGs (not used in live site) |

---

## Diagnostic Scripts

```bash
# Validate all pipeline outputs are present and recent
python scripts/validate_pipeline.py

# Check what the current nowcast values are
python scripts/test_forecasts.py

# Check political index data (article counts, IRP/IRE by day)
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/daily_instability/daily_index.parquet')
print(df[['date','political_index','economic_index','n_articles_total']].tail(10).to_string())
"

# Check supermarket scrape coverage
python -c "
import pandas as pd, glob
files = sorted(glob.glob('data/raw/supermarket/2026-*.parquet'))
print(f'Last {len(files[-5:])} scrape files:')
for f in files[-5:]:
    df = pd.read_parquet(f)
    print(f'  {f[-18:]}: {len(df):,} products')
"
```

---

## Script Dependencies (execution order)

```
BCRP download          NTL download
      │                      │
      ▼                      ▼
build_panel.py      build_regional_panel.py
      │                      │
      ├──────────────────────┤
      │                      │
      ▼                      ▼
generate_nowcast.py   build_poverty_nowcast.py
(GDP + inflation)            │
      │                      ▼
      │           generate_poverty_nowcast.py
      │
      ▼
export_web_data.py  ←── build_daily_index.py
      │                      │
      │              (RSS + Haiku daily)
      ▼
sync_web_data.py
      │
      ▼
D:/qhawarina/public/assets/data/
```

---

## Important Constants (config/settings.py)

| Constant | Value | Used for |
|---|---|---|
| `PROCESSED_DAILY_DIR` | `data/processed/daily_instability/` | IRP/IRE parquet location |
| `RESULTS_DIR` | `data/results/` | Model output parquets |
| `TARGETS_DIR` | `data/targets/` | Official INEI/BCRP targets |
| `EXPORTS_DIR` | `exports/` | Web-ready JSON/CSV |
| `RAW_RSS_DIR` | `data/raw/rss/` | RSS articles cache |

---

## CRITICAL: Two daily_index.parquet Files (Legacy Issue)

```
data/processed/daily/daily_index.parquet            ← LEGACY. Stuck at 2026-03-13. IGNORE.
data/processed/daily_instability/daily_index.parquet ← CORRECT. Updated daily. USE THIS.
```

`PROCESSED_DAILY_DIR` in `config/settings.py` points to `daily_instability/` (correct).
The legacy file at `daily/` should be deleted to avoid confusion.
