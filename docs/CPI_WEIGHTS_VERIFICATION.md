# CPI Weights Verification - Peru IPC Base Dec 2021=100

## Confirmed Division-Level Weights (Source: INEI)

From [INEI official announcement](https://m.inei.gob.pe/prensa/noticias/inei-presenta-nuevo-ano-base-para-la-medicion-de-los-precios-al-consumidor-13323/):

| Division | Category | Weight (%) |
|----------|----------|------------|
| 01 | Alimentos y Bebidas no Alcohólicas | 22.96 |
| 02 | Bebidas Alcohólicas, Tabaco y Estupefacientes | 1.61 |
| 03 | Prendas de Vestir y Calzado | 4.19 |
| 04 | Alojamiento, Agua, Electricidad, Gas y Otros Combustibles | 10.55 |
| 05 | Muebles, Artículos para el Hogar y Conservación de Vivienda | 5.10 |
| 06 | Salud | 3.48 |
| 07 | Transporte | 12.39 |
| 08 | Comunicaciones | 4.77 |
| 09 | Recreación y Cultura | 3.95 |
| 10 | Enseñanza | 8.60 |
| 11 | Restaurantes y Hoteles | 15.88 |
| 12 | Bienes y Servicios Diversos | 6.45 |

**Total**: 99.93% (rounding differences)

## Sub-Group Weights (Partial - Web Search Findings)

From search results, we found:
- **Pan y cereales**: 4.438% (within Division 01)
- **Carnes**: 5.410% (within Division 01)
- **Leche, queso y huevos**: 2.892% (within Division 01)
- **Aceites y grasas**: 0.538% (within Division 01)

Source: Multiple INEI technical reports (2022-2023)

## Our Current Implementation

In `src/ingestion/supermarket.py` we use:

```python
CPI_WEIGHTS_RAW = {
    "arroz_cereales": 4.91,     # Pan y cereales
    "aceites_grasas": 0.99,     # Aceites y grasas
    "azucar_dulces": 0.67,      # Azúcar, mermeladas, miel, chocolate
    "lacteos": 2.69,            # Leche, queso, huevos (dairy portion)
    "carnes": 4.18,             # Carnes y preparados de carne
    "pescados_mariscos": 1.18,  # Pescado y mariscos
    "pan_harinas": 4.06,        # Otros productos alimenticios n.e.p.
    "frutas": 2.13,             # Frutas
    "verduras": 3.55,           # Legumbres y hortalizas
    "huevos": 0.80,             # Huevos (split from dairy)
    "bebidas": 2.45,            # Bebidas no alcohólicas
    "limpieza": 1.20,           # Limpieza del hogar (within vivienda)
    "cuidado_personal": 5.73,   # Cuidados personales
}
```

**Total**: 34.44%

## Discrepancies

1. **Pan y cereales**: 4.91 (ours) vs 4.438 (INEI) — **+0.47pp** (10% overweight)
2. **Aceites y grasas**: 0.99 (ours) vs 0.538 (INEI) — **+0.45pp** (84% overweight!)
3. **Carnes**: 4.18 (ours) vs 5.410 (INEI) — **-1.23pp** (23% underweight)
4. **Lácteos + huevos**: 2.69+0.80=3.49 (ours) vs 2.892 (INEI) — **+0.60pp** (21% overweight)

## OFFICIAL INEI WEIGHTS (CONFIRMED ✓)

From `estructura_ponderaciones_ipc_lima_dic2021.csv`:

| COICOP Code | Group | Weight (%) |
|-------------|-------|------------|
| 011100 | Pan y cereales | 4.231 |
| 011200 | Carne | 5.227 |
| 011300 | Pescados y mariscos | 1.089 |
| 011400 | Leche, queso y huevos | 2.848 |
| 011500 | Aceites y grasas | 0.535 |
| 011600 | Frutas | 2.500 |
| 011700 | Hortalizas, legumbres, papas y tubérculos | 2.869 |
| 011800 | Azúcar, mermelada, miel, chocolate y dulces | 0.890 |
| 011900 | Productos alimenticios n.e.p. | 0.856 |
| 012000 | Bebidas no alcohólicas | 1.921 |
| 056100 | Bienes para el hogar no duraderos (cleaning) | 1.319 |
| 121000 | Cuidado personal | 4.791 |

**Total Food (Division 01)**: 22.967%
**Total Scrapable**: 27.077%

## Index Methodology Comparison

### INEI Official Methodology
From [INEI Methodology Guide](https://www.inei.gob.pe/media/MenuRecursivo/metodologias/metodologia_ipclm_dic_2021.pdf):

- **Formula**: Modified Laspeyres (fixed weights from base period)
- **Elementary level**: Weighted geometric mean of elementary price indices
- **Aggregation**: CPI basket weights (from ENAPREF 2019-2020)
- **Base period**: December 2021 = 100

### Cavallo BPP Methodology
From [Cavallo & Rigobon (2016)](https://www.aeaweb.org/articles?id=10.1257/jep.30.2.151):

- **Formula**: Fisher index when category weights available, geometric mean otherwise
- **Elementary level**: Jevons (geometric mean of price ratios) for matched products
- **Aggregation**: Official CPI expenditure weights
- **Base**: Chain-linked daily index

### Our Implementation
In `src/ingestion/supermarket.py`:

- **Formula**: Jevons (geometric mean) at category level + CPI-weighted aggregation
- **Elementary level**: Bilateral price ratios (p_t / p_{t-1}) per matched product
- **Aggregation**: INEI CPI weights (weighted geometric mean)
- **Base**: Chain-linked from first day = 100

**Verdict**: Our approach is **compatible** with both methodologies. We use Jevons (geometric mean) like Cavallo's elementary aggregation, and CPI weights like both INEI and Cavallo. The only difference is we don't use Fisher index (geometric mean of Laspeyres & Paasche) — we use pure geometric mean which is simpler and appropriate for daily high-frequency data.

## Action Items

1. ✅ **COMPLETED**: Updated weights to match INEI official data
2. ✅ **COMPLETED**: Found complete INEI sub-group weights table
3. ✅ **COMPLETED**: Verified methodology compatibility with INEI and Cavallo

## Sources

- [INEI CPI Base 2021 Announcement](https://m.inei.gob.pe/prensa/noticias/inei-presenta-nuevo-ano-base-para-la-medicion-de-los-precios-al-consumidor-13323/)
- [Cavallo & Rigobon (2016) - Billion Prices Project](https://www.aeaweb.org/articles?id=10.1257/jep.30.2.151)
- BCRP IPC series (PN01297PM-PN01312PM for components)
