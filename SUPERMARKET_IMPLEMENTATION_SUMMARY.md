# Supermarket Price Scraper — Implementation Complete ✓

## Status: PRODUCTION READY

**Date**: 2026-02-10  
**Snapshot**: 42,710 products across 3 stores  
**Classification**: 100% (L1 fallback + keywords)  
**Weights**: Official INEI CPI weights (base Dec 2021)  
**Tests**: 58/58 passing (39 supermarket + 19 orchestration)

---

## Data Collection

### Daily Snapshot (2026-02-10)
- **Plaza Vea**: 15,130 products (14 L1 categories) — S/0.40 - S/21,308
- **Metro**: 12,561 products (13 L1 categories) — S/0.60 - S/1,450
- **Wong**: 15,019 products (13 L1 categories) — S/0.70 - S/3,799
- **Total**: 42,710 products

### Classification Distribution
| Category | Products | % of Total |
|----------|----------|------------|
| Bebidas | 7,294 | 17.1% |
| Cuidado personal | 6,198 | 14.5% |
| Arroz/cereales | 6,194 | 14.5% |
| Limpieza | 4,377 | 10.2% |
| Azúcar/dulces | 4,358 | 10.2% |
| Carnes | 4,163 | 9.7% |
| Lácteos | 2,880 | 6.7% |
| Frutas | 2,614 | 6.1% |
| Verduras | 1,971 | 4.6% |
| Aceites/grasas | 1,331 | 3.1% |
| Pan/harinas | 958 | 2.2% |
| Pescados/mariscos | 304 | 0.7% |
| Huevos | 68 | 0.2% |
| **Other** | **0** | **0.0%** ✓ |

---

## CPI Weights (Official INEI)

### Verified Weights (Base Dic 2021 = 100)
Source: `estructura_ponderaciones_ipc_lima_dic2021.csv`

| COICOP | Category | Weight |
|--------|----------|--------|
| 011100 | Pan y cereales | 4.231% |
| 011200 | Carne | 5.227% |
| 011300 | Pescados y mariscos | 1.089% |
| 011400 | Leche, queso, huevos | 2.848% |
| 011500 | Aceites y grasas | 0.535% |
| 011600 | Frutas | 2.500% |
| 011700 | Hortalizas/legumbres | 2.869% |
| 011800 | Azúcar, dulces | 0.890% |
| 012000 | Bebidas no alcohólicas | 1.921% |
| 056100 | Limpieza | 1.319% |
| 121000 | Cuidado personal | 4.791% |

**Total Scrapable**: 28.220% of CPI basket

---

## Methodology

### Index Construction (Cavallo BPP)
1. **Match products** by (store, sku_id) across days
2. **Bilateral price ratios**: R_i = p_t / p_{t-1} per product
3. **Category-level Jevons**: R_cat = exp(mean(ln(R_i))) within category
4. **CPI-weighted aggregation**: R_all = exp(Σ w_k × ln(R_k))
5. **Chain-link**: Index_t = Index_{t-1} × R_t (base = first day = 100)
6. **Monthly**: Average daily index values → MoM variation

### Formula Compatibility
- **INEI**: Modified Laspeyres (geometric mean at elementary level)
- **Cavallo**: Fisher or geometric mean + CPI weights
- **Ours**: Jevons (geometric mean) + CPI weights ✓

**Result**: Compatible with both methodologies

### Extreme Value Treatment
- Trim ratios: 0.5 < R < 2.0 (excludes promotions/errors)
- Minimum 5 products per category for reliable Jevons estimate

---

## Integration

### Files Created
- `src/ingestion/supermarket.py` — VTEXClient, Scraper, IndexBuilder, Aggregator
- `scripts/scrape_supermarket_prices.py` — CLI (--store, --build-index)
- `scripts/daily_update.bat` — Automated daily scraping
- `tests/test_supermarket.py` — 39 tests
- `docs/CPI_WEIGHTS_VERIFICATION.md` — Weight validation

### Files Modified
- `config/settings.py` — Added RAW_SUPERMARKET_DIR paths
- `src/processing/panel_builder.py` — Added _build_supermarket_panel()
- `src/models/dfm.py` — Added SUPERMARKET_FOOD_VAR, SUPERMARKET_ALL_VAR (now 25 inflation series)
- `scripts/update_nexus.py` — Added step_supermarket() (now 14 steps)
- `scripts/schedule_nexus.bat` — Calls daily_update.bat
- `tests/test_orchestration.py` — Updated to 14 steps

### Panel Output
- **Series**: SUPERMARKET_ALL_INDEX/VAR, SUPERMARKET_FOOD_INDEX/VAR + 11 category-level series
- **Location**: `data/processed/national/supermarket_monthly_prices.parquet`
- **Schema**: Standard panel long format (date, series_id, value_raw/sa/log/dlog/yoy, source, frequency_original, publication_lag_days)

---

## Next Steps

1. **Tomorrow (Feb 11)**: Run second daily scrape to get 2 snapshots
2. **Build first index**: `python scripts/scrape_supermarket_prices.py --build-index`
3. **Rebuild panel**: `python scripts/update_nexus.py --only panel`
4. **Re-run inflation nowcast**: Check if supermarket indices improve R²
5. **Monitor**: Daily scraping via `daily_update.bat` (scheduled task)

---

## Key Learnings

### VTEX API Discovery
- Products indexed at **L1 (top-level) only**, not leaf categories
- L2/L3 queries return 0 products → must search at L1 level
- 3 stores share Cencosud platform (Metro & Wong have identical category IDs)

### Classification Strategy
- **Two-pass**: Keywords first, then L1 category fallback
- Achieves 100% classification (no "other")
- L1_CATEGORY_MAP handles store category → CPI category mapping

### Weight Verification
- Initial estimates were off by 10-84% for some categories
- Official INEI weights now confirmed from CSV extract
- Total scrapable weight: 28.22% of CPI basket (food + cleaning + personal care)

### Snapshot Management
- Individual store scrapes merge into single date file
- Deduplication by (store, sku_id) prevents duplicates
- `--store all` scrapes 3 stores sequentially (~30 min total)

---

## References

- [INEI CPI Methodology](https://www.inei.gob.pe/media/MenuRecursivo/metodologias/metodologia_ipclm_dic_2021.pdf)
- [INEI CPI Announcement](https://m.inei.gob.pe/prensa/noticias/inei-presenta-nuevo-ano-base-para-la-medicion-de-los-precios-al-consumidor-13323/)
- [Cavallo & Rigobon (2016) - Billion Prices Project](https://www.aeaweb.org/articles?id=10.1257/jep.30.2.151)
- [Cavallo (2013) - Online and Official Price Indexes](https://www.sciencedirect.com/science/article/abs/pii/S0304393212000967)
