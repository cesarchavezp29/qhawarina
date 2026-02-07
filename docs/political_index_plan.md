# Plan de Implementacion: Indice de Inestabilidad Politica (v2)

## Cambios vs v1

1. **Google Trends eliminado** — reemplazado por **Estabilidad de Gabinete** (`days_since_last_cabinet_change`, inverso)
2. **Dos outputs**: `political_index_monthly.parquet` + `political_index_weekly.parquet` (sin diario)
3. **Ground truth se mantiene ordinal 1-3** — para validacion, NLP (1-5) se agrupa en 3 bins
4. **Wikipedia API (MediaWiki)** en vez de scraping HTML directo

---

## 1. Arquitectura General

```
data/raw/political/
  peru_political_events.xlsx          -- GROUND TRUTH (70 eventos curados)
  cabinet_timeline.parquet            -- cronologia completa de premiers (PCM)
  scraped_events_raw.parquet          -- eventos crudos de Wikipedia API
  events.parquet                      -- eventos procesados, clasificados, linkeados

src/ingestion/political.py            -- Wikipedia API + fuentes
src/nlp/classifier.py                 -- wrapper XLM-RoBERTa zero-shot
src/nlp/validator.py                  -- validacion con Claude API
src/processing/political_index.py     -- agregacion del indice compuesto
src/processing/cabinet_stability.py   -- calculo days_since_last_cabinet_change
src/analysis/event_study.py           -- event studies financieros

config/political_sources.yaml         -- paginas Wikipedia y configuracion

data/processed/political_instability/
  events.parquet                      -- eventos finales con scores
  cabinet_stability.parquet           -- serie mensual/semanal de estabilidad
  political_index_weekly.parquet      -- indice semanal (eventos + financiero + gabinete)
  political_index_monthly.parquet     -- indice compuesto completo (4 componentes)
```

---

## 2. Paginas Wikipedia a Consultar via MediaWiki API

### 2.1 Metodo: Wikipedia API (no scraping HTML)

Usar la **MediaWiki Action API** (`https://en.wikipedia.org/w/api.php` y `https://es.wikipedia.org/w/api.php`):

```python
import requests

def get_wikipedia_text(title: str, lang: str = "es") -> str:
    """Obtener texto plano de una pagina Wikipedia via API."""
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,      # Texto plano, no HTML
        "exsectionformat": "plain",
        "format": "json",
    }
    resp = requests.get(url, params=params)
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("extract", "")

# Ejemplo:
text = get_wikipedia_text("Baguazo", lang="es")
# Retorna texto plano con secciones separadas por \n\n
```

**Ventajas vs scraping:**
- No necesita BeautifulSoup ni lxml
- Rate limit generoso (200 req/s para bots sin User-Agent, ilimitado con User-Agent)
- Texto ya limpio (sin HTML tags, tablas, refs)
- Estructura estable (API oficial, no depende de layout HTML)
- Secciones accesibles individualmente via `exsectionformat`

### 2.2 Paginas del sheet `Sources` (17 paginas, ingles)

| # | Titulo Wikipedia (en) | Contenido esperado |
|---|----------------------|-------------------|
| 1 | `Alberto_Fujimori` | Vladivideos, huida, juicio |
| 2 | `Valentín_Paniagua` | Transicion 2000-2001 |
| 3 | `Alejandro_Toledo` | Gobierno 2001-2006, Ecoteva |
| 4 | `Alan_García` | Segundo gobierno, Petroaudios, suicidio |
| 5 | `Ollanta_Humala` | Gobierno 2011-2016, Odebrecht |
| 6 | `Pedro_Pablo_Kuczynski` | Gobierno, crisis, renuncia |
| 7 | `Martín_Vizcarra` | Disolucion, vacancia |
| 8 | `Francisco_Sagasti` | Transicion 2020-2021 |
| 9 | `Pedro_Castillo` | Autogolpe, arresto |
| 10 | `Dina_Boluarte` | Masacres, Rolexgate, vacancia |
| 11 | `2019_Peruvian_constitutional_crisis` | Disolucion del Congreso |
| 12 | `Peruvian_political_crisis_(2016–present)` | Crisis 2016-presente |
| 13 | `2022_Peruvian_self-coup_attempt` | Autogolpe de Castillo |
| 14 | `Pardon_of_Alberto_Fujimori` | Indulto, revocacion |
| 15 | `Arrest_and_trial_of_Alberto_Fujimori` | Juicio y condena |
| 16 | `Impeachment_of_Dina_Boluarte` | Vacancia 2025 |
| 17 | `Presidency_of_Dina_Boluarte` | Gobierno Boluarte |

### 2.3 Paginas adicionales para gaps 2001-2016 (18 paginas, espanol)

| # | Titulo Wikipedia (es) | Contenido esperado |
|---|----------------------|-------------------|
| 18 | `Presidente_del_Consejo_de_Ministros_del_Perú` | **Lista completa ~40+ premiers** — FUENTE PRINCIPAL para cabinet_timeline |
| 19 | `Arequipazo` | Protesta masiva jun 2002 |
| 20 | `Andahuaylazo` | Insurreccion ene 2005 |
| 21 | `Baguazo` | Masacre jun 2009, 33 muertos |
| 22 | `Conflicto_de_Conga` | Anti-minera 2011-2012 |
| 23 | `Conflicto_de_Tía_María` | Anti-minera 2015 |
| 24 | `Gobierno_de_Alejandro_Toledo` | Eventos gobierno Toledo |
| 25 | `Segundo_gobierno_de_Alan_García` | Eventos Garcia II |
| 26 | `Gobierno_de_Ollanta_Humala` | Eventos Humala |
| 27 | `Gobierno_de_Pedro_Pablo_Kuczynski` | Eventos PPK |
| 28 | `Gobierno_de_Martín_Vizcarra` | Eventos Vizcarra |
| 29 | `Gobierno_de_Pedro_Castillo` | Eventos Castillo |
| 30 | `Gobierno_de_Dina_Boluarte` | Eventos Boluarte |
| 31 | `Petroaudios` | Escandalo grabaciones Garcia |
| 32 | `Paro_agrario_en_el_Perú_de_2008` | Paro agrario |
| 33 | `Protestas_en_Perú_de_2022-2023` | Protestas post-Castillo |
| 34 | `Crisis_política_en_el_Perú_de_2017-presente` | Crisis completa |
| 35 | `Gabinetes_ministeriales_del_Perú` | Cambios de gabinete |

### 2.4 Fuentes no-Wikipedia

| # | Fuente | URL | Contenido |
|---|--------|-----|-----------|
| 36 | AS/COA | `https://www.as-coa.org/articles/perus-presidential-crisis-timeline` | Timeline crisis |
| 37 | Baker Institute | `https://www.bakerinstitute.org/research/political-crisis-peru` | Analisis academico |

---

## 3. Estructura Propuesta de `events.parquet`

```python
columns = {
    # Identificacion
    "event_id": int,                    # ID interno (autoincrement)
    "ground_truth_event_id": int | NaN, # Linkeo al dataset manual (1-70) o NaN

    # Temporal
    "date": datetime,                   # Fecha del evento
    "year": int,
    "month": int,

    # Contenido
    "event_description": str,           # Descripcion del evento
    "event_type": str,                  # Tipo (29 categorias del dataset)
    "president_affected": str,

    # Fuente
    "source": str,                      # "wikipedia_api", "congreso"
    "source_page": str,                 # Titulo de pagina Wikipedia
    "source_text_snippet": str,         # Extracto de texto fuente (max 500 chars)

    # Clasificacion NLP (escala 1-5)
    "severity_nlp": int,                # Score 1-5 de XLM-RoBERTa
    "severity_nlp_confidence": float,   # Confianza del modelo (0-1)
    "severity_nlp_label": str,          # Label asignado

    # NLP binned para comparacion con GT (escala 1-3)
    "severity_nlp_bin3": int,           # NLP 1-5 agrupado en 3 bins: {1-2}->1, {3}->2, {4-5}->3

    # Validacion Claude (solo para ~200 eventos)
    "severity_claude": int | NaN,       # Score 1-5 de Claude API
    "severity_claude_bin3": int | NaN,  # Claude binned a 1-3
    "severity_claude_reasoning": str,   # Razonamiento

    # Ground truth (solo para eventos matcheados) — ORDINAL 1-3, SIN RESCALAR
    "severity_gt": int | NaN,           # Severity 1-3 del dataset manual (tal cual)

    # Matching
    "match_method": str | NaN,          # "exact_date", "fuzzy_date", "text_similarity"
    "match_confidence": float | NaN,    # Score de matching (0-1)

    # Contexto
    "anticipated": int | NaN,           # 0=sorpresa, 1=esperado (del ground truth)
    "market_hours": int | NaN,
    "constitutional_basis": str | NaN,

    # Indice
    "severity_final": int,              # Score final 1-5 (combina NLP + Claude)
    "severity_final_bin3": int,         # Final binned 1-3
    "weight_in_index": float,           # Peso en el indice compuesto
}
```

**Binning NLP (1-5) → 3 bins para validacion vs GT:**

| NLP Score | Bin | GT Equivalente | Descripcion |
|:---------:|:---:|:--------------:|-------------|
| 1-2 | 1 (Low) | 1 | Rutina, tension menor |
| 3 | 2 (Medium) | 2 | Conflicto institucional |
| 4-5 | 3 (High) | 3 | Crisis, quiebre |

**Total esperado:** ~250-400 eventos

---

## 4. Ejemplo del Prompt de Clasificacion XLM-RoBERTa

```python
from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli",
    device=-1,  # CPU
)

candidate_labels = [
    "rutina politica",           # Score 1
    "tension politica moderada", # Score 2
    "conflicto institucional",   # Score 3
    "crisis constitucional",     # Score 4
    "quiebre institucional",     # Score 5
]

label_to_score = {
    "rutina politica": 1,
    "tension politica moderada": 2,
    "conflicto institucional": 3,
    "crisis constitucional": 4,
    "quiebre institucional": 5,
}

def classify_event(text: str) -> dict:
    result = classifier(text, candidate_labels, multi_label=False)
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    nlp_score = label_to_score[top_label]
    return {
        "severity_nlp": nlp_score,
        "severity_nlp_confidence": round(top_score, 4),
        "severity_nlp_label": top_label,
        "severity_nlp_bin3": 1 if nlp_score <= 2 else (2 if nlp_score == 3 else 3),
    }

# Ejemplo:
# "El Congreso voto la vacancia del presidente Vizcarra" → score 4-5, bin3=3
# "PPK fue inaugurado como presidente" → score 1, bin3=1
# "Se presento una mocion de vacancia" → score 2-3, bin3=1-2
```

**Notas tecnicas:**
- `joeddav/xlm-roberta-large-xnli` soporta espanol nativo
- Zero-shot: no necesita fine-tuning
- ~0.5s/evento CPU, ~3 min para ~400 eventos

---

## 5. Plan de Validacion con Claude API

### 5.1 Seleccion de Muestra

~200 eventos estratificados:
- **Todos** los 70 del ground truth
- ~50 cambios de premier
- ~50 protestas/conflictos
- ~30 aleatorios

### 5.2 Prompt Estructurado

```python
import anthropic

client = anthropic.Anthropic()

RUBRIC_PROMPT = """Eres un analista politico experto en Peru. Clasifica la severidad del
siguiente evento politico peruano en una escala de 1 a 5:

ESCALA:
1 = Rutina politica: cambio ministerial rutinario, inauguracion protocolar
2 = Tension politica moderada: amenaza de censura, interpelacion, protesta menor
3 = Conflicto institucional: voto de confianza, censura aprobada, protesta significativa
4 = Crisis constitucional: vacancia iniciada, disolucion amenazada, estado de emergencia
5 = Quiebre institucional: presidente destituido/renunciado, Congreso disuelto, autogolpe

EVENTO:
Fecha: {date}
Descripcion: {description}
Presidente en ejercicio: {president}

Responde SOLO con un JSON:
{{"score": <1-5>, "reasoning": "<explicacion en 1-2 oraciones>"}}"""

def validate_with_claude(event: dict) -> dict:
    prompt = RUBRIC_PROMPT.format(
        date=event["date"],
        description=event["event_description"],
        president=event["president_affected"],
    )
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    import json
    result = json.loads(response.content[0].text)
    score = result["score"]
    return {
        "severity_claude": score,
        "severity_claude_bin3": 1 if score <= 2 else (2 if score == 3 else 3),
        "severity_claude_reasoning": result["reasoning"],
    }
```

### 5.3 Metricas de Validacion (todo en escala 1-3)

Toda la validacion se hace comparando **bin3** vs **GT ordinal 1-3**:

1. **NLP bin3 vs GT:** Confusion matrix 3x3, accuracy >= 80%, Cohen's kappa >= 0.6
2. **Claude bin3 vs GT:** Mismas metricas
3. **NLP bin3 vs Claude bin3:** Inter-rater agreement, kappa >= 0.6
4. **Errores graves:** GT=3 clasificado como bin3=1, o GT=1 clasificado como bin3=3 → objetivo: 0
5. **Correlacion:** Spearman rho (ordinal) entre NLP bin3 y GT

### 5.4 Costo Estimado

- 200 eventos x ~400 tokens = ~80K tokens total
- Claude Sonnet: **~$1-3 total**

---

## 6. Plan de Matching: Eventos Scrapeados vs Ground Truth

### 6.1 Algoritmo de Matching

```python
from difflib import SequenceMatcher

def match_to_ground_truth(scraped_event: dict, ground_truth: pd.DataFrame,
                          date_tolerance_days: int = 3,
                          text_threshold: float = 0.4) -> dict | None:
    scraped_date = scraped_event["date"]
    scraped_text = scraped_event["event_description"].lower()

    candidates = []
    for _, gt_row in ground_truth.iterrows():
        date_diff = abs((scraped_date - gt_row["date"]).days)
        if date_diff > date_tolerance_days:
            continue

        gt_text = gt_row["event_description"].lower()
        text_sim = SequenceMatcher(None, scraped_text, gt_text).ratio()

        same_president = (
            scraped_event.get("president_affected", "").lower() ==
            str(gt_row.get("president_affected", "")).lower()
        )

        score = text_sim * 0.6 + (1 - date_diff / date_tolerance_days) * 0.3
        if same_president:
            score += 0.1

        if text_sim >= text_threshold or (date_diff == 0 and same_president):
            candidates.append({
                "ground_truth_event_id": gt_row["event_id"],
                "match_confidence": round(score, 3),
                "match_method": "exact_date" if date_diff == 0 else "fuzzy_date",
            })

    if not candidates:
        return None
    return max(candidates, key=lambda c: c["match_confidence"])
```

### 6.2 Criterios de Match

| Condicion | Accion |
|-----------|--------|
| Misma fecha + mismo presidente | Match automatico (confidence >= 0.9) |
| Fecha +/-1 dia + text_sim >= 0.5 | Match probable (>= 0.7) |
| Fecha +/-3 dias + text_sim >= 0.4 | Match posible (revision si < 0.6) |

**Target recall:** >= 85% (>= 60 de 70 eventos)

---

## 7. Eventos Faltantes a Buscar (Gaps 2001-2016)

### 7.1 Cambios de Premier (~40 desde 2001)

Fuente principal: pagina #18 (`Presidente_del_Consejo_de_Ministros_del_Perú`) via Wikipedia API.

**Estos premiers tambien alimentan el componente de Estabilidad de Gabinete (seccion 9).**

**Toledo (2001-2006) — 5 premiers:**
- Roberto Danino (jul 2001 - jul 2002)
- Luis Solari (jul 2002 - jun 2003)
- Beatriz Merino (jun 2003 - dic 2003)
- Carlos Ferrero (dic 2003 - ago 2005)
- Pedro Pablo Kuczynski (ago 2005 - jul 2006)

**Garcia II (2006-2011) — 5 premiers:**
- Jorge del Castillo (jul 2006 - oct 2008)
- Yehude Simon (oct 2008 - jul 2009)
- Javier Velasquez Quesquen (jul 2009 - sep 2010)
- Jose Antonio Chang (sep 2010 - mar 2011)
- Rosario Fernandez (mar 2011 - jul 2011)

**Humala (2011-2016) — 7 premiers:**
- Salomon Lerner Ghitis (jul 2011 - dic 2011)
- Oscar Valdes (dic 2011 - jul 2012)
- Juan Jimenez Mayor (jul 2012 - oct 2013)
- Cesar Villanueva (oct 2013 - feb 2014)
- Rene Cornejo (feb 2014 - jul 2014)
- Ana Jara (jul 2014 - abr 2015)
- Pedro Cateriano (abr 2015 - jul 2016)

**PPK + Vizcarra + Castillo + Boluarte (2016-2025) — ~15+ premiers**

### 7.2 Protestas y Conflictos Mayores

| Fecha | Evento | GT Severity |
|-------|--------|:-----------:|
| Jun 2002 | **Arequipazo** — protesta contra privatizacion, 2 muertos | 3 |
| Ene 2005 | **Andahuaylazo** — insurreccion Antauro Humala, 6 muertos | 3 |
| Jun 2009 | **Baguazo** — enfrentamiento Bagua, 33 muertos | 3 |
| Nov 2011 | **Conflicto Conga** — anti-minera, estado de emergencia | 3 |
| May 2015 | **Conflicto Tia Maria** — anti-minera, 3 muertos | 3 |
| Feb 2008 | **Paro agrario nacional** — 72h, bloqueo carreteras | 2 |
| Jul 2007 | **Paro de maestros** — SUTEP, huelga indefinida | 2 |

### 7.3 Otros Eventos Faltantes

| Fecha | Evento | GT Severity |
|-------|--------|:-----------:|
| Oct 2008 | **Petroaudios** — grabaciones corrupcion | 2-3 |
| Jun 2013 | **Caso Ecoteva** — Toledo/Odebrecht $35M | 2 |
| Mar 2018 | **Kenjivideos** — compra de votos | 2-3 |

---

## 8. Pipeline Completo de Procesamiento

```
PASO 1: WIKIPEDIA API
  35 paginas via MediaWiki API (extracts) --> texto plano con eventos

PASO 2: EXTRACCION DE EVENTOS
  Regex + heuristicas para extraer:
  - Fecha del evento
  - Descripcion (1-2 oraciones)
  - Tipo de evento (29 categorias)
  - Presidente afectado

PASO 3: EXTRACCION CABINET TIMELINE
  De pagina #18 (PCM): extraer lista completa de premiers con fechas
  Guardar en cabinet_timeline.parquet: premier, start_date, end_date, president
  Calcular days_since_last_change para cada fecha

PASO 4: DEDUPLICACION
  Agrupar eventos por fecha +/- 1 dia + similitud texto > 0.7
  Mantener version mas completa

PASO 5: MATCHING vs GROUND TRUTH
  Algoritmo de la seccion 6
  Reportar recall: objetivo >= 85% (>= 60 de 70)

PASO 6: CLASIFICACION XLM-ROBERTA
  Clasificar todos los ~300-400 eventos (escala 1-5)
  Calcular bin3 para cada evento

PASO 7: VALIDACION CLAUDE API (~200 eventos)
  Clasificar, calcular bin3
  Comparar NLP bin3 vs Claude bin3 vs GT (1-3)

PASO 8: VALIDACION vs GROUND TRUTH
  Comparar severity_nlp_bin3 vs severity_gt (ambos 1-3)
  Reportar: accuracy, kappa, confusion matrix, errores graves

PASO 9: GUARDAR events.parquet + cabinet_timeline.parquet
```

---

## 9. Componente de Estabilidad de Gabinete (REEMPLAZA Google Trends)

### 9.1 Logica

La **duracion del gabinete actual** es un proxy directo de estabilidad politica. Un premier que lleva 400 dias indica estabilidad; uno que lleva 15 dias indica crisis reciente.

```python
def compute_cabinet_stability(cabinet_timeline: pd.DataFrame,
                               date_range: pd.DatetimeIndex) -> pd.Series:
    """Calcula days_since_last_cabinet_change para cada fecha.

    Returns inverted z-score: mayor valor = mayor INESTABILIDAD.
    (Pocos dias desde cambio = alta inestabilidad)
    """
    # Para cada fecha, encontrar el cambio de gabinete mas reciente
    days_since = pd.Series(index=date_range, dtype=float)
    for date in date_range:
        # Ultimo cambio <= date
        past_changes = cabinet_timeline[cabinet_timeline["start_date"] <= date]
        if past_changes.empty:
            days_since[date] = np.nan
        else:
            last_change = past_changes["start_date"].max()
            days_since[date] = (date - last_change).days

    # Invertir: pocos dias = alta inestabilidad
    # z-score negado: days_since bajo → z-score alto (inestable)
    stability_zscore = -zscore_rolling(days_since, window=60)
    return stability_zscore
```

### 9.2 Cabinet Timeline Output

`cabinet_timeline.parquet`:
```
premier_name, start_date, end_date, president, duration_days
```

~40+ filas (2001-2025). Fuente: Wikipedia API pagina #18.

### 9.3 Resample a semanal/mensual

- **Semanal:** `days_since_last_change` al viernes de cada semana
- **Mensual:** promedio del mes (o valor a fin de mes)

---

## 10. Componente Financiero (BCRP)

Series a descargar (buscar codigos diarios y/o mensuales):
- **EMBI Peru** (riesgo pais)
- **Tipo de cambio PEN/USD** (PN01246PM es mensual; buscar diario)
- **Tasa interbancaria**
- **Reservas internacionales**

Calculos (agregados a mensual para el indice):
- Volatilidad FX mensual: std de retornos diarios dentro del mes
- EMBI z-score mensual: `(embi_m - rolling_mean_60) / rolling_std_60`

Para la **version semanal** del indice:
- Mismos calculos pero agregados a semana (viernes)

Event studies para los 15 high-impact events:
- Ventana [-7, +7] dias
- Abnormal EMBI spread change
- Test anticipated=0 vs anticipated=1

---

## 11. Agregacion del Indice Compuesto

### 11.1 Estandarizacion

Cada componente: z-score sobre ventana rolling de 5 anios (60 meses):
```
z_i,t = (x_i,t - mean_60(x_i)) / std_60(x_i)
```

### 11.2 Ponderacion

| Componente | Peso | Frecuencia | Fuente |
|------------|:----:|:----------:|--------|
| Institucional (eventos NLP) | 35% | Mensual | Wikipedia API + NLP |
| Financiero (EMBI + FX vol) | 25% | Mensual (de diario) | BCRP |
| Estabilidad de gabinete | 20% | Mensual (de PCM timeline) | Wikipedia API |
| Confianza empresarial | 20% | Mensual | PD12912AM (ya en panel) |

### 11.3 Output

**`political_index_monthly.parquet`** (indice principal):
```
date, events_zscore, financial_zscore, cabinet_zscore, confidence_zscore, composite_index
```

**`political_index_weekly.parquet`** (actualizacion rapida):
```
date (viernes), events_zscore, financial_zscore, cabinet_zscore, weekly_index
```
- Nota: confianza empresarial (PD12912AM) es mensual, asi que el indice semanal usa 3 componentes con pesos renormalizados: eventos 44%, financiero 31%, gabinete 25%.

---

## 12. Tests Automatizados

```python
# test_political.py

# 1. Ground truth recall
def test_scraper_recall():
    """El scraper encuentra >= 85% de los 70 eventos del GT."""
    assert matched_count >= 60

# 2. Severity accuracy (bin3 vs GT, ambos 1-3)
def test_severity_accuracy_bin3():
    """NLP bin3 matches GT ordinal 1-3 para >= 80% de eventos matcheados."""
    matched = events[events["severity_gt"].notna()]
    correct = (matched["severity_nlp_bin3"] == matched["severity_gt"]).sum()
    assert correct / len(matched) >= 0.80

# 3. No errores graves (en escala 1-3)
def test_no_severe_errors():
    """GT=3 nunca clasificado bin3=1, GT=1 nunca clasificado bin3=3."""
    matched = events[events["severity_gt"].notna()]
    severe = (
        ((matched["severity_gt"] == 3) & (matched["severity_nlp_bin3"] == 1)) |
        ((matched["severity_gt"] == 1) & (matched["severity_nlp_bin3"] == 3))
    ).sum()
    assert severe == 0

# 4. High impact spikes
def test_high_impact_spikes():
    """Los 15 high-impact events producen z-score > 1.5 en el indice mensual."""
    for event in high_impact_events:
        month = event["date"].to_period("M")
        zscore = monthly_index.loc[month, "composite_index"]
        assert zscore > 1.5, f"Event {event['event_id']} z-score = {zscore}"

# 5. Event study signs
def test_event_study_embi():
    """EMBI spread sube en ventana alrededor de crisis."""
    for event in high_impact_events:
        embi_change = compute_embi_change(event, window=7)
        assert embi_change > 0

# 6. Anticipated vs surprise
def test_anticipated_effect():
    """Eventos sorpresa producen mayor reaccion financiera."""
    surprise = [e for e in high_impact if e["anticipated"] == 0]
    expected = [e for e in high_impact if e["anticipated"] == 1]
    mean_surprise = np.mean([abs(e["embi_change"]) for e in surprise])
    mean_expected = np.mean([abs(e["embi_change"]) for e in expected])
    assert mean_surprise > mean_expected

# 7. Cabinet stability
def test_cabinet_timeline_complete():
    """Cabinet timeline tiene >= 35 premiers desde 2001."""
    assert len(cabinet_timeline) >= 35

def test_cabinet_stability_inversely_related():
    """Months with recent cabinet changes have higher instability score."""
    recent = monthly_index[monthly_index["cabinet_zscore"] > 1.0]
    # These months should overlap with known crisis periods
    assert len(recent) >= 10
```

---

## 13. Dependencias Nuevas

```toml
# pyproject.toml additions
[project.optional-dependencies]
political = [
    "transformers>=4.40",
    "torch>=2.0",
    "anthropic>=0.25",
    "openpyxl>=3.1",       # para leer peru_political_events.xlsx
]
```

**Eliminado vs v1:** `pytrends`, `beautifulsoup4`, `lxml` (ya no necesarios).

**Nota:** `requests` ya esta instalado (via httpx). `openpyxl` puede ya estar como dep de pandas.

---

## 14. Estimacion de Trabajo

| Componente | Complejidad |
|------------|-------------|
| Wikipedia API (35 paginas) | Baja (API oficial, texto plano) |
| Extraccion de eventos + deduplicacion | Media-Alta |
| Cabinet timeline extraction | Baja (tabla estructurada) |
| Cabinet stability computation | Baja (~50 lineas) |
| Matching vs ground truth | Media |
| XLM-RoBERTa clasificacion + binning | Baja (wrapper) |
| Claude API validacion | Baja (~$1-3) |
| BCRP series financieras | Baja (reutilizar BCRPClient) |
| Agregacion del indice (monthly + weekly) | Media |
| Event studies | Media |
| Tests | Media |
| **Total** | **~700-1000 lineas de codigo nuevo** |

---

## 15. Riesgos y Mitigaciones

| Riesgo | Mitigacion |
|--------|-----------|
| Wikipedia API rate limit | Generoso (200 req/s), 35 paginas = trivial |
| Extraccion de fechas de texto plano | Regex + patrones de fecha espanol/ingles |
| PCM page structure changes | API extract es texto plano, no HTML — mas estable |
| XLM-RoBERTa accuracy insuficiente | Claude API como segundo clasificador + ensemble |
| EMBI Peru diario no disponible en BCRP | Usar mensual como fallback |
| torch muy pesado | Usar `torch-cpu` o `onnxruntime` como backend |
| Pocos eventos 2001-2016 | Complementar con AS/COA, Baker Institute |
