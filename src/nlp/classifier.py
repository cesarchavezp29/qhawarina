"""Claude API classifier for political event severity.

Uses Claude Haiku for direct severity classification on the ordinal
1-3 scale matching ground truth. Batched prompts for efficiency.
"""

import json
import logging
import time

import pandas as pd

logger = logging.getLogger("nexus.nlp.classifier")

# Binning: 1-5 → ordinal 1-3
SCORE_TO_BIN3 = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}

# Labels (kept for reference/compatibility)
CANDIDATE_LABELS = [
    "rutina politica",            # Score 1
    "tension politica moderada",  # Score 2
    "conflicto institucional",    # Score 3
    "crisis constitucional",      # Score 4
    "quiebre institucional",      # Score 5
]

LABEL_TO_SCORE = {
    "rutina politica": 1,
    "tension politica moderada": 2,
    "conflicto institucional": 3,
    "crisis constitucional": 4,
    "quiebre institucional": 5,
}

SEVERITY_PROMPT = """You are an expert political analyst specializing in Peru. Classify the severity of each political event on a 1-3 ordinal scale:

SCALE:
1 = LOW: Routine cabinet change, protocol inauguration, minor political tension, interpellation, minor protest
2 = MEDIUM: Vote of confidence, approved censure, significant protest, corruption scandal, arrest of former official, judicial ruling on political case, state of emergency
3 = HIGH: President removed/resigned, Congress dissolved, self-coup, massacre/mass killings, president arrested, major constitutional crisis

IMPORTANT: Consider the broader context of each event's month. An event that might seem routine in isolation (e.g., cabinet reshuffle, investigation opened) should be rated HIGHER if it occurs during a month with massacres, mass protests, or constitutional crises.
{context_block}
EVENTS TO CLASSIFY:
{events_block}

Respond with ONLY a JSON array. Each element: {{"id": <event_number>, "score": <1-3>, "label": "<one of: low/medium/high>"}}
No explanation, no markdown, just the JSON array."""

SINGLE_EVENT_PROMPT = """You are an expert political analyst specializing in Peru. Classify the severity of this political event on a 1-3 ordinal scale:

SCALE:
1 = LOW: Routine cabinet change, protocol inauguration, minor political tension
2 = MEDIUM: Vote of confidence, censure, significant protest, corruption scandal, arrest/trial of politician, state of emergency
3 = HIGH: President removed/resigned, Congress dissolved, self-coup, massacre, major constitutional crisis

EVENT:
Date: {date}
Description: {description}
President: {president}

Respond with ONLY a JSON object: {{"score": <1-3>, "label": "<low/medium/high>"}}"""

# Module-level client (lazy)
_client = None


def _get_client():
    """Lazy-load the Anthropic client."""
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
        logger.info("Anthropic client initialized")
    return _client


def classify_event(
    text: str,
    date: str = "",
    president: str = "",
    client=None,
    model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """Classify a single event with Claude API.

    Returns dict with: severity_claude (1-5 mapped), severity_claude_confidence,
    severity_claude_label, severity_claude_bin3 (1-3).
    """
    if client is None:
        client = _get_client()

    prompt = SINGLE_EVENT_PROMPT.format(
        date=date,
        description=text,
        president=president,
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        score3 = max(1, min(3, int(result["score"])))
        # Map bin3 back to 1-5 scale for column compatibility
        bin3_to_5 = {1: 1, 2: 3, 3: 5}
        score5 = bin3_to_5[score3]
        label = result.get("label", "")

        return {
            "severity_claude": score5,
            "severity_claude_confidence": 1.0,
            "severity_claude_label": label,
            "severity_claude_bin3": score3,
        }
    except Exception as e:
        logger.warning("Classification failed: %s", e)
        return {
            "severity_claude": 3,
            "severity_claude_confidence": 0.0,
            "severity_claude_label": f"ERROR: {e}",
            "severity_claude_bin3": 2,
        }


def _build_month_contexts(
    all_events: list[dict],
    min_events: int = 3,
    top_n: int = 3,
) -> dict[str, str]:
    """Build context summaries for months with many events.

    Groups events by YYYY-MM, and for months with >= min_events,
    picks the top_n longest descriptions as representative context.
    Returns dict mapping YYYY-MM -> context string.
    """
    from collections import defaultdict
    by_month: dict[str, list[dict]] = defaultdict(list)
    for ev in all_events:
        ym = ev["date"][:7]  # YYYY-MM
        by_month[ym].append(ev)

    contexts = {}
    for ym, month_events in by_month.items():
        if len(month_events) < min_events:
            continue
        # Pick top_n longest descriptions as most informative
        sorted_evs = sorted(month_events, key=lambda e: len(e["description"]), reverse=True)
        snippets = [e["description"][:80].strip() for e in sorted_evs[:top_n]]
        contexts[ym] = (
            f"{ym}: {len(month_events)} events this month, including: "
            + "; ".join(snippets)
        )
    return contexts


def _classify_batch_chunk(
    events: list[dict],
    client,
    model: str = "claude-haiku-4-5-20251001",
    month_contexts: dict[str, str] | None = None,
) -> list[dict]:
    """Classify a batch of events in a single API call.

    events: list of dicts with id, date, description, president.
    month_contexts: optional dict mapping YYYY-MM -> context string.
    Returns list of dicts with id, score (1-3), label.
    """
    # Build context block from relevant months
    context_block = ""
    if month_contexts:
        relevant_months = set()
        for e in events:
            ym = e["date"][:7]
            if ym in month_contexts:
                relevant_months.add(ym)
        if relevant_months:
            ctx_lines = [month_contexts[ym] for ym in sorted(relevant_months)]
            context_block = "\nMONTH CONTEXT (for reference):\n" + "\n".join(ctx_lines) + "\n"

    # Build events block
    lines = []
    for e in events:
        lines.append(f"Event {e['id']}: [{e['date']}] ({e['president']}) {e['description'][:300]}")
    events_block = "\n".join(lines)

    prompt = SEVERITY_PROMPT.format(events_block=events_block, context_block=context_block)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=len(events) * 50 + 100,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown fencing if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        results = json.loads(raw)
        return results
    except Exception as e:
        logger.warning("Batch classification failed: %s", e)
        # Return defaults
        return [{"id": ev["id"], "score": 2, "label": "medium"} for ev in events]


def classify_events_batch(
    events_df: pd.DataFrame,
    text_column: str = "event_description",
    batch_size: int = 20,
    client=None,
    model: str = "claude-haiku-4-5-20251001",
    delay: float = 0.3,
) -> pd.DataFrame:
    """Classify all events in a DataFrame using Claude API.

    Sends events in batches of batch_size per API call for efficiency.
    Adds columns: severity_claude, severity_claude_confidence, severity_claude_label,
    severity_claude_bin3.
    """
    if client is None:
        client = _get_client()

    df = events_df.copy()
    n = len(df)

    logger.info("Classifying %d events with Claude API (batch_size=%d)...", n, batch_size)

    # Prepare event dicts
    all_events = []
    for i, (idx, row) in enumerate(df.iterrows()):
        date_str = row["date"]
        if hasattr(date_str, "strftime"):
            date_str = date_str.strftime("%Y-%m-%d")
        all_events.append({
            "id": i + 1,
            "idx": idx,
            "date": str(date_str),
            "description": str(row[text_column]),
            "president": str(row.get("president_affected", "")),
        })

    # Initialize result columns
    df["severity_claude"] = 3
    df["severity_claude_confidence"] = 0.0
    df["severity_claude_label"] = ""
    df["severity_claude_bin3"] = 2

    # Build month contexts for enriched classification
    month_contexts = _build_month_contexts(all_events)
    if month_contexts:
        logger.info("  Month contexts built for %d months with 3+ events", len(month_contexts))

    # Process in batches
    bin3_to_5 = {1: 1, 2: 3, 3: 5}
    processed = 0

    for batch_start in range(0, n, batch_size):
        batch = all_events[batch_start:batch_start + batch_size]
        results = _classify_batch_chunk(
            batch, client=client, model=model, month_contexts=month_contexts,
        )

        # Map results back by id
        result_map = {}
        for r in results:
            rid = int(r["id"])
            result_map[rid] = r

        for ev in batch:
            eid = ev["id"]
            idx = ev["idx"]
            r = result_map.get(eid, {"score": 2, "label": "medium"})
            score3 = max(1, min(3, int(r["score"])))
            label = r.get("label", "")

            df.loc[idx, "severity_claude_bin3"] = score3
            df.loc[idx, "severity_claude"] = bin3_to_5[score3]
            df.loc[idx, "severity_claude_confidence"] = 1.0
            df.loc[idx, "severity_claude_label"] = label

        processed += len(batch)
        if processed % 100 == 0 or processed == n:
            logger.info("  Classified %d/%d events", processed, n)

        time.sleep(delay)

    # Summary
    dist3 = df["severity_claude_bin3"].value_counts().sort_index()
    logger.info("Bin3 distribution: %s", dict(dist3))
    dist5 = df["severity_claude"].value_counts().sort_index()
    logger.info("Score distribution (mapped 1-5): %s", dict(dist5))

    return df


ARTICLE_CLASSIFICATION_PROMPT = """Eres un analista de riesgo para Peru. Clasifica cada articulo segun su impacto en la ESTABILIDAD politica o economica del pais.

REGLAS CRITICAS:
- Solo clasifica como "political" o "economic" si el evento AMENAZA la estabilidad institucional o economica de Peru.
- Noticias internacionales son "irrelevant" EXCEPTO si afectan directamente a Peru (aranceles a exportaciones peruanas, precio del cobre, decisiones del FMI sobre Peru).
- Noticias empresariales rutinarias (ventas, proyectos, nombramientos, multas regulatorias) = "irrelevant".
- Reportes meteorologicos, deportes, cultura, farándula, tecnología = "irrelevant".
- Tipo de cambio diario, variaciones bursatiles normales (<2%) = "irrelevant".

CATEGORIA: "political", "economic", "both", "irrelevant"
- political: vacancia, censura, corrupcion gubernamental, crisis de gobernabilidad, protestas masivas, conflictos sociales graves
- economic: crisis financiera, colapso sectorial, huelgas que paralizan sectores, suspension de operaciones mineras, arbitrajes grandes contra el Estado, caidas bursatiles severas (>3%), devaluacion fuerte
- both: eventos con impacto politico Y economico simultaneo
- irrelevant: todo lo demas (noticias rutinarias, internacionales sin impacto Peru, empresariales, deportes, etc.)

SEVERIDAD (solo para political/economic/both):
1 = BAJO: conflicto menor localizado, queja sectorial, critica aislada, indicador ligeramente negativo
2 = MEDIO: protesta que escala, investigacion fiscal activa, suspension de operaciones, caida significativa de indicadores, arbitraje internacional
3 = ALTO: vacancia/censura presidencial, colapso de mercado, emergencia nacional, crisis institucional grave, conflicto social con muertos

EJEMPLOS de "irrelevant":
- "Tipo de cambio hoy" → irrelevant
- "V&V cerro 2025 con 20% de avance en ventas" → irrelevant
- "Indecopi multa a Santillana por errores en libro" → irrelevant
- "JPMorgan opina sobre liquidacion del mercado" → irrelevant (opinion internacional)
- "Foro de Davos reune a lideres" → irrelevant (internacional)
- "Estuardo Ortiz: 2026 puede ser nuestro año record" → irrelevant (empresarial)
- "Proinversión: se adjudicaron 69 proyectos APP" → irrelevant (estadistica rutinaria)

EJEMPLOS de clasificacion correcta:
- "6 mociones de censura contra el presidente" → political, severity 3
- "Nexa suspende operacion de Atacocha por bloqueo" → economic, severity 2
- "Aenza anuncia arbitraje contra el Estado por US$67M" → economic, severity 2
- "Aranceles de Trump afectan exportaciones peruanas" → economic, severity 2
- "Fiscalia investiga reuniones del presidente" → political, severity 3

SE ESTRICTO. En caso de duda, clasifica como "irrelevant". Prefiere severidad baja.

ARTICULOS:
{articles_block}

Responde SOLO con un JSON array:
[{{"id": 1, "category": "political", "severity": 2}}, ...]"""

SEVERITY_LABEL_MAP = {1: "low", 2: "medium", 3: "high"}


def _classify_articles_chunk(
    articles: list[dict],
    client,
    model: str = "claude-haiku-4-5-20251001",
) -> list[dict]:
    """Classify a batch of articles in a single API call.

    articles: list of dicts with id, date, title, summary.
    Returns list of dicts with id, category, severity.
    """
    lines = []
    for a in articles:
        lines.append(f"[{a['id']}] Fecha: {a['date']} | {a['title']}. {a['summary'][:200]}")
    articles_block = "\n".join(lines)

    prompt = ARTICLE_CLASSIFICATION_PROMPT.format(articles_block=articles_block)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=len(articles) * 60 + 100,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        results = json.loads(raw)
        return results
    except Exception as e:
        logger.warning("Article batch classification failed: %s", e)
        return [{"id": a["id"], "category": "irrelevant", "severity": 1} for a in articles]


def classify_articles_batch(
    articles_df: pd.DataFrame,
    batch_size: int = 20,
    client=None,
    model: str = "claude-haiku-4-5-20251001",
    delay: float = 0.3,
) -> pd.DataFrame:
    """Classify news articles for political/economic instability.

    Adds columns: article_category, article_severity, article_severity_label.

    Parameters
    ----------
    articles_df : DataFrame with columns: title, summary, published
    batch_size : number of articles per API call
    client : optional Anthropic client
    model : Claude model to use
    delay : seconds between API calls

    Returns
    -------
    DataFrame with classification columns added.
    """
    if client is None:
        client = _get_client()

    df = articles_df.copy()
    n = len(df)

    logger.info("Classifying %d articles with Claude API (batch_size=%d)...", n, batch_size)

    # Prepare article dicts
    all_articles = []
    for i, (idx, row) in enumerate(df.iterrows()):
        date_str = ""
        if pd.notna(row.get("published")):
            pub = row["published"]
            if hasattr(pub, "strftime"):
                date_str = pub.strftime("%Y-%m-%d")
            else:
                date_str = str(pub)[:10]

        all_articles.append({
            "id": i + 1,
            "idx": idx,
            "date": date_str,
            "title": str(row.get("title", "")),
            "summary": str(row.get("summary", "")),
        })

    # Initialize result columns
    df["article_category"] = "irrelevant"
    df["article_severity"] = 0
    df["article_severity_label"] = ""

    processed = 0
    for batch_start in range(0, n, batch_size):
        batch = all_articles[batch_start:batch_start + batch_size]
        results = _classify_articles_chunk(batch, client=client, model=model)

        # Map results back
        result_map = {}
        for r in results:
            rid = int(r["id"])
            result_map[rid] = r

        for a in batch:
            aid = a["id"]
            idx = a["idx"]
            r = result_map.get(aid, {"category": "irrelevant", "severity": 1})

            category = r.get("category", "irrelevant")
            if category not in ("political", "economic", "both", "irrelevant"):
                category = "irrelevant"

            raw_sev = r.get("severity")
            severity = max(0, min(3, int(raw_sev))) if raw_sev is not None else 1
            if category == "irrelevant":
                severity = 0

            df.loc[idx, "article_category"] = category
            df.loc[idx, "article_severity"] = severity
            df.loc[idx, "article_severity_label"] = SEVERITY_LABEL_MAP.get(severity, "")

        processed += len(batch)
        if processed % 100 == 0 or processed == n:
            logger.info("  Classified %d/%d articles", processed, n)

        if batch_start + batch_size < n:
            time.sleep(delay)

    # Summary
    cat_dist = df["article_category"].value_counts()
    logger.info("Category distribution: %s", dict(cat_dist))
    sev_dist = df.loc[df["article_category"] != "irrelevant", "article_severity"].value_counts().sort_index()
    logger.info("Severity distribution (non-irrelevant): %s", dict(sev_dist))

    return df


_POLITICAL_SYSTEM_PROMPT = """Eres un clasificador de riesgo político para Perú. Tu tarea es evaluar si un artículo de prensa señala una amenaza a la estabilidad política e institucional DOMÉSTICA del Perú.

DEFINICIÓN: Riesgo político = amenazas al orden institucional democrático peruano, la continuidad del gobierno, la estabilidad social, y el funcionamiento normal de los poderes del Estado EN PERÚ. NO incluye política extranjera a menos que impacte directamente a Perú.

Asigna un puntaje de 0 a 100 (usa CUALQUIER valor entero, no solo múltiplos de 10 o 20). PUNTOS DE REFERENCIA:

  0: Sin relevancia. Noticias internacionales sin impacto Perú, deportes, entretenimiento, operaciones rutinarias de empresas.
 20: Tensión menor. Fricción política de bajo nivel, declaraciones ordinarias de partidos, desacuerdos sin escalada. Ej: congresista critica al gobierno sin acción concreta.
 35: Inestabilidad baja-moderada. Huelgas de transporte o sectoriales, TC resuelve caso con implicancias electorales, candidato con antecedentes penales en campaña, protesta local sin violencia.
 50: Inestabilidad moderada. Disputas activas entre poderes del Estado, bancadas condicionando votos de confianza, investigaciones fiscales a figuras en ejercicio, protestas que escalan.
 65: Crisis significativa. Amenaza de vacancia o censura en proceso, movilización social extendida, confrontación seria Ejecutivo-Legislativo, ministro renunciado bajo presión política.
 80: Crisis severa. Amenaza inminente de caída del Ejecutivo/Legislativo, violencia política masiva, quiebre institucional.
100: Emergencia. Ruptura constitucional activa, colapso de gobernanza.

REGLA CLAVE — DIMENSIÓN DOMINANTE: Si un artículo trata sobre una ACCIÓN POLÍTICA (partidos condicionando al gobierno, bancadas amenazando censura, negociaciones de votos) usando una crisis económica como CONTEXTO o HERRAMIENTA, el puntaje político debe reflejar la acción política. Ejemplo: "Fuerza Popular usa la crisis energética para condicionar el voto de confianza" → pol≥55, aunque el contexto sea económico.

EJEMPLOS CALIBRADOS (úsalos como ancla):
- "Congreso debate moción de vacancia contra presidente" → pol=75
- "Bancada FP condiciona voto de confianza al gabinete" → pol=55
- "Transportistas inician paro nacional por alza de combustibles" → pol=35
- "TC admite habeas corpus de candidato con condena penal" → pol=40
- "Ministro de Economía renuncia por presión del Congreso" → pol=60
- "Congresista presenta denuncia constitucional contra premier" → pol=45
- "Candidato presidencial sin respaldo en región Arequipa" → pol=20
- "Aprobación del presidente cae a 18% según encuesta" → pol=25
- "Senace aprueba EIA de proyecto minero Trapiche" → pol=0
- "Hudbay presenta ITS para optimizar mina Constancia" → pol=0
- "Falabella abre nueva tienda en Trujillo" → pol=0
- "Copa Libertadores: Cristal clasifica a fase de grupos" → pol=0
- "Federico Salazar y Katia Condos anuncian separación" → pol=0
- "Sismo de magnitud 4.1 sacude Cañete" → pol=0

PALABRAS CLAVE QUE SEÑALAN RIESGO POLÍTICO (pol≥30 si el contexto es peruano):
vacancia, censura, interpelación, moción, voto de confianza, gabinete, premier, presidente del Consejo, bancada, partido político, elecciones, campaña electoral, JNE, ONPE, JEE, debate presidencial, inscripción de candidatos, plancha presidencial, alianza política, fragmentación política, Ejecutivo vs. Legislativo, crisis de gobernabilidad, golpe de Estado, protestas masivas, estado de emergencia, toque de queda, bloqueo de carretera por protesta política, Cerrón, Congreso, Poder Judicial (cuando involucra figuras políticas en casos de corrupción sistémica), huelga de transportistas, paro nacional, paro regional.

PALABRAS CLAVE QUE NO SON RIESGO POLÍTICO — pol=0 SALVO contexto político explícito:
PBI, inflación, tipo de cambio, dólar, BCRP, MEF, tasa de interés, exportaciones, importaciones, déficit fiscal, bonos soberanos, producción minera, créditos hipotecarios, empleo, canasta básica, inversión extranjera, bolsa de valores, Fed, bancos centrales extranjeros, criptomonedas, EIA (estudio de impacto ambiental), ITS (informe técnico sustentatorio), Senace, ProInversión (adjudicaciones rutinarias), expansión de empresa privada, apertura de tienda, resultado financiero de empresa.

REGLAS ADICIONALES:
- Aprobaciones técnicas de proyectos mineros/energéticos (EIA, ITS, Senace, Osinergmin regulatorio) = pol=0. Son decisiones administrativas, no políticas.
- Política extranjera sin impacto Perú = 0 (ej: "elecciones Colombia", "Trump habla con Putin") → 0.
- Deportes = 0 SIEMPRE (resultados, fichajes, transferencias, ligas europeas, Copa Libertadores, béisbol) → 0.
- Farándula/celebridades = 0 SIEMPRE. Conductores de TV, actores, influencers NO son actores políticos.
- Privatización/gestión de empresas estatales (Petroperú, EsSalud): si es decisión TÉCNICA o ECONÓMICA → pol=0-15. Solo sube pol si hay confrontación POLÍTICA explícita (moción de censura al ministro por Petroperú) → pol≥40.
- Gastronomía/turismo/cultura premiados = 0. El reconocimiento culinario NO es riesgo político.
- Crimen común (femicidio, robo, secuestro individual sin actores estatales) = pol=0-10.
- Tipo de cambio/dólar: SIEMPRE pol=0. Es indicador económico. EXCEPCIÓN ÚNICA: artículo dice explícitamente que una crisis POLÍTICA PERUANA hundió el sol.
- SENAMHI, pronósticos meteorológicos, sismos = 0.
- Noticias puramente económicas = pol=0 salvo vínculo EXPLÍCITO con acción política peruana.
- Usa cualquier entero: 5, 17, 33, 52, 68, 74, 91. Sé consistente."""

_ECONOMIC_SYSTEM_PROMPT = """Eres un clasificador de riesgo económico para Perú. Tu tarea es evaluar si un artículo de prensa señala una amenaza a la estabilidad económica del Perú, específicamente riesgos que afecten a Perú de manera DIRECTA y CONCRETA.

DEFINICIÓN: Riesgo económico = amenazas a la estabilidad macroeconómica, fiscal, financiera, o productiva DE PERÚ. Debe ser un riesgo que afecte a Perú MÁS que al promedio de economías emergentes.

Asigna un puntaje de 0 a 100 (usa CUALQUIER valor entero). PUNTOS DE REFERENCIA:

  0: Sin relevancia económica para Perú. Internacional genérico, deportes, farándula, clima rutinario, acciones legislativas/políticas sin impacto económico directo.
 20: Riesgo menor. Presiones sectoriales modestas, ajustes regulatorios sin urgencia, variaciones normales de precios.
 40: Estrés moderado. Deterioro en indicadores peruanos clave, disrupciones en sectores importantes, leyes con costo fiscal significativo (>S/500M).
 60: Vulnerabilidad significativa. Disrupciones mayores en motores económicos peruanos (minería, gas, exportaciones), inflación por encima de tendencia histórica, precio del petróleo >$90 impactando costos peruanos.
 80: Crisis severa. Colapso sectorial, crisis fiscal aguda, paralización de suministro energético nacional, estrés financiero sistémico.
100: Emergencia. Default soberano, quiebre bancario, pérdida total de confianza del mercado peruano.

EVENTOS DE ALTO IMPACTO ECONÓMICO PARA PERÚ — estos tipos de artículos deben recibir eco≥60:
- Ruptura o interrupción del gasoducto Camisea/TGP (suministro nacional de gas natural) → eco=70-85. Camisea abastece ~95% del gas natural peruano; su interrupción es una emergencia energética nacional.
- Precio del petróleo supera $90-100/barril por guerra, embargo o cierre de estrecho → eco=60-75. Perú importa derivados y el alza eleva inflación y costos de transporte directamente.
- Agencia Internacional de Energía (AIE/IEA) reporta caída de oferta global de millones de barriles/día → eco=65-75. Impacta precios de combustibles peruanos de forma inmediata.
- Datos oficiales de inflación peruana mostrando máximos de varios años (>4%) → eco=65-75.
- Precio de combustibles en Perú sube >20% → eco=60-70.
- Paralización de Las Bambas, Antamina u otra gran mina peruana → eco=65-80.
- Moody's, S&P o Fitch rebajan calificación soberana o perspectiva de Perú → eco=65-75.
- BBVA/BCP/Scotiabank recortan proyección de PBI de Perú en ≥0.5 puntos por shocks concretos → eco=45-60.
- Leyes aprobadas con costo fiscal directo >S/1,000 millones al año → eco=40-55.
- Exportaciones peruanas caen >10% por aranceles externos o cierre de mercados → eco=55-70.

EJEMPLOS CALIBRADOS (úsalos como ancla):
- "TGP reporta ruptura en gasoducto Camisea; suministro de gas interrumpido" → eco=80
- "Precio del petróleo supera $100 por guerra en Medio Oriente" → eco=68
- "IEA: cierre del estrecho de Ormuz reduce oferta mundial en 8 millones de barriles/día" → eco=72
- "Inflación en Perú alcanza 4.2%, máximo en 4 años, según INEI" → eco=68
- "Precio del balón de gas supera S/100 en Lima" → eco=55
- "BBVA ajusta proyección de PBI de Perú de 3.1% a 2.9% por crisis energética" → eco=48
- "Las Bambas suspende operaciones por bloqueo de comunidades" → eco=72
- "Moody's coloca deuda peruana en perspectiva negativa" → eco=65
- "Congreso aprueba CTS y gratificaciones para trabajadores CAS: costo S/2,800 millones/año" → eco=45
- "Aranceles de Trump afectan exportaciones peruanas de joyería y textiles" → eco=52
- "BCRP mantiene tasa de referencia en 4.25%" → eco=30
- "Tipo de cambio hoy: sol cierra a S/3.45" → eco=0
- "Bancadas condicionan voto de confianza usando crisis del gas como argumento" → eco=25 (el gas es contexto, la maniobra es política)
- "Delia Espinoza inhabilitada por el Congreso" → eco=0
- "Senace aprueba EIA de proyecto minero Trapiche" → eco=20 (aprobación regulatoria rutinaria, no crisis)
- "Reunión del ministro del Interior sobre seguridad ciudadana" → eco=0
- "Colegios regresan a clases presenciales" → eco=0
- "Camión de bomberos casi choca con auto particular" → eco=0
- "Atlético de Madrid ganó 5-2 a Tottenham" → eco=0
- "Federico Salazar y Katia Condos anuncian separación" → eco=0
- "Sismo de magnitud 4.1 sacude Cañete" → eco=0

REGLA CLAVE — DIMENSIÓN DOMINANTE: Si un artículo describe una MANIOBRA POLÍTICA que usa una crisis económica como CONTEXTO, eco=15-35 (el contexto), pol=50-70 (la acción). NO inflés eco porque el artículo menciona una crisis económica de fondo.

CONTENIDO POLÍTICO PURO — eco=0 OBLIGATORIO:
- Candidatos, encuestas de intención de voto, debates electorales, ONPE, JNE, inscripción de candidatos, plancha presidencial → eco=0 SIEMPRE.
- Voto de confianza, censura, interpelación, moción de vacancia → eco=0.
- Alianzas partidarias, cambios de bancada, fragmentación política → eco=0.
- Cobertura judicial de figuras políticas (Cerrón, Fujimori, magistrados, fiscales) sin impacto macroeconómico directo → eco=0.
- Inhabilitaciones, sanciones disciplinarias del Congreso → eco=0.
- "Confianza" o "respaldo" en sentido político (apoyo parlamentario, respaldo regional a candidato) → eco=0. Solo eco>0 si habla de confianza del INVERSOR, MERCADO o CALIFICADORA con datos.

PROHIBIDO — RAZONAMIENTO EN CADENA (eco=0 aunque la cadena parezca válida):
NO: "reunión de seguridad → orden público → economía mejora → eco>0" → eco=0.
NO: "escuelas regresan a clases → padres trabajan → productividad → eco>0" → eco=0.
NO: "accidente de tránsito → caos vial → pérdida económica → eco>0" → eco=0.
NO: "arresto de criminal → seguridad → inversión → eco>0" → eco=0.
NO: "incertidumbre política → confianza inversora → eco>0" → eco=0.
NO: "candidato sin respaldo → inestabilidad → eco>0" → eco=0.
eco>0 SOLO si el artículo describe impacto económico DIRECTO y OBSERVABLE con datos concretos o hechos físicos (paralización, corte de suministro, cifra de PBI, dato de inflación, rebaja de calificación).

REGLAS ADICIONALES:
- Deportes = 0 SIEMPRE (resultados, fichajes, Copa Libertadores, béisbol, NBA, NFL) → 0.
- Farándula y celebridades = 0.
- Sismos/temblores = 0 salvo que destruyan infraestructura productiva con cifras concretas.
- Política extranjera sin impacto económico directo en Perú = 0.
- Gastronomía/premios culturales = eco=0-10 máximo.
- Bitcoin/crypto = 0-15 máximo. Perú no es economía cripto-dependiente.
- Crimen individual = eco=0 salvo que interrumpa sectores productivos con datos.
- Tipo de cambio/dólar cotización diaria = eco=0-5. EXCEPCIÓN: movimiento ≥1% en un día con causa concreta analizada → eco=35-45.
- Fed/bancos centrales extranjeros = 0-20, SOLO si analiza impacto concreto en Perú/BCRP.
- Bolsas extranjeras caídas leves = 0. EXCEPCIÓN: caídas >3% que impacten activos peruanos con datos.
- Ajustes regulatorios rutinarios (SBS eleva límites de billetera digital, Sunafil recuerda RMV, Osinergmin fija tarifa) = eco=10-20 máximo.
- Expansión de empresa privada (tienda nueva, centro de distribución, nuevo modelo de moto) = eco=10-25. NO es riesgo económico sistémico.
- Usa cualquier entero: 5, 17, 33, 52, 68, 74, 91. Sé consistente."""

_DUAL_USER_TEMPLATE = """Evalúa los siguientes {n} artículos. Para CADA artículo, responde con un JSON en una línea separada. EXACTAMENTE {n} líneas, una por artículo, en orden.

{articles_block}

Responde EXACTAMENTE {n} líneas JSON, una por artículo, en orden:
{{"score": <0-100>}}
{{"score": <0-100>}}
..."""


def _score_batch(articles: list[dict], system_prompt: str, client, model: str) -> list[int | None]:
    """Single Haiku call for one dimension (political or economic).

    Returns list of integer scores (0-100) or None for parse failures.
    Raises RuntimeError if the API call itself fails (no fallback).
    """
    n = len(articles)
    lines = []
    for a in articles:
        lines.append(
            f"Artículo {a['id']}:\nTítulo: {a['title']}\nFuente: {a['source']}\n{a['summary'][:300]}"
        )
    articles_block = "\n\n".join(lines)
    user_msg = _DUAL_USER_TEMPLATE.format(n=n, articles_block=articles_block)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=n * 20 + 50,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()
    except Exception as e:
        raise RuntimeError(f"Haiku API call failed: {e}") from e

    scores: list[int | None] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            scores.append(max(0, min(100, int(obj["score"]))))
        except Exception:
            scores.append(None)

    # Pad/trim to exactly n
    while len(scores) < n:
        scores.append(None)
    return scores[:n]


def classify_articles_dual(
    articles_df: pd.DataFrame,
    batch_size: int = 20,
    client=None,
    model: str = "claude-haiku-4-5-20251001",
    delay: float = 0.3,
) -> pd.DataFrame:
    """Dual-prompt AI-GPR classifier (Iacoviello & Tong 2026).

    Each article receives TWO independent scores:
    - political_score (0-100): threat to Peru's political stability
    - economic_score (0-100): threat to Peru's economic stability

    Makes 2 API calls per batch (one per dimension). Temperature = 0.
    Raises RuntimeError on API failure — NO keyword fallback ever.

    Parameters
    ----------
    articles_df : DataFrame with columns: title, summary/description, published, source
    batch_size : articles per API call (20 recommended)
    client : optional pre-initialized Anthropic client
    model : Haiku model ID
    delay : seconds between batch pairs

    Returns
    -------
    DataFrame with political_score and economic_score columns added.
    """
    import os
    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Cannot classify without Haiku. "
                "NO keyword fallback — set the API key and retry."
            )
        client = _get_client()

    df = articles_df.copy()
    n = len(df)

    logger.info("Dual-classifying %d articles (2 calls per batch of %d)...", n, batch_size)

    # Prepare article dicts
    all_articles = []
    for i, (idx, row) in enumerate(df.iterrows()):
        pub = row.get("published", "")
        date_str = pub.strftime("%Y-%m-%d") if hasattr(pub, "strftime") else str(pub)[:10]
        summary = str(row.get("summary") or row.get("description") or "")
        all_articles.append({
            "id": i + 1,
            "idx": idx,
            "date": date_str,
            "title": str(row.get("title", "")),
            "summary": summary,
            "source": str(row.get("source", "")),
        })

    # Initialize columns as None
    df["political_score"] = None
    df["economic_score"] = None

    # ── Deterministic pre-filter ──────────────────────────────────────────────
    # Mark obvious zero-risk articles BEFORE the API call to save cost.
    # IMPORTANT: Articles are kept in the DataFrame (score=0, not dropped) so
    # they still contribute to N_it in the SWP denominator (AI-GPR methodology).
    # Only apply patterns where false-positive risk is near zero.

    titles_lower = df["title"].fillna("").str.lower()

    _pre_sports = (
        r"champions league|premier league|la liga|serie a|bundesliga|ligue 1|"
        r"real madrid|fc barcelona|atlético de madrid|manchester|liverpool|chelsea|"
        r"bayern|juventus|psg|inter miami|mbappé|haaland|"
        r"nba|nfl|mlb|nhl|grand slam|wimbledon|roland garros|"
        r"copa libertadores|copa del mundo|mundial de f[uú]tbol|clásico mundial de béisbol|"
        r"sporting cristal|alianza lima|universitario de deportes|fbc melgar|"
        r"fichaje|transferencia.*jugador|jugador.*transferencia|"
        r"partidos de hoy.*copa|copa.*partidos de hoy|horarios.*copa|copa.*horarios"
    )
    _pre_farandula = (
        r"cómo se conocieron|historia de amor|boda y|anuncio de ruptura|"
        r"confirma relaci[oó]n|ruptura sentimental|separaci[oó]n de pareja|"
        r"vida amorosa|conductora?.*amor|amor.*conductora?"
    )
    _pre_sismo = r"\b(sismo|sismos|temblor|temblores|terremoto|terremotos|movimiento s[ií]smico)\b"
    _pre_weather = (
        r"\bSENAMHI\b|pronóstico.*lluvia|pronóstico.*temperatura|"
        r"temperatura.*m[aá]xima.*m[ií]nima|lluvias para hoy|clima de hoy"
    )
    _pre_fx_routine = r"(precio del d[oó]lar|d[oó]lar hoy|divisa cierra|tipo de cambio hoy|sol se cotiza)"
    _pre_fx_large   = r"(puntos b[aá]sicos|pierde|gana|dispara|desploma|hunde|sube.*%|baja.*%)"
    _pre_gastro = r"\breceta\b|\brecetas\b|cómo preparar.*ingredientes|gastronomía.*premio|mejor restaurante del mundo"

    pre_mask = (
        titles_lower.str.contains(_pre_sports,    regex=True, na=False) |
        titles_lower.str.contains(_pre_farandula, regex=True, na=False) |
        titles_lower.str.contains(_pre_sismo,     regex=True, na=False) |
        titles_lower.str.contains(_pre_weather,   regex=True, na=False) |
        titles_lower.str.contains(_pre_gastro,    regex=True, na=False) |
        (
            titles_lower.str.contains(_pre_fx_routine, regex=True, na=False) &
            ~titles_lower.str.contains(_pre_fx_large,  regex=True, na=False)
        )
    )

    n_pre = int(pre_mask.sum())
    if n_pre > 0:
        df.loc[pre_mask, "political_score"] = 0
        df.loc[pre_mask, "economic_score"]  = 0
        logger.info(
            "Pre-filter: %d/%d articles set to 0 before API (sports=%d, farándula=%d, "
            "sismo=%d, weather=%d, gastro=%d, fx_routine=%d) — kept in df for N_it denominator",
            n_pre, n,
            titles_lower[pre_mask].str.contains(_pre_sports,    regex=True, na=False).sum(),
            titles_lower[pre_mask].str.contains(_pre_farandula, regex=True, na=False).sum(),
            titles_lower[pre_mask].str.contains(_pre_sismo,     regex=True, na=False).sum(),
            titles_lower[pre_mask].str.contains(_pre_weather,   regex=True, na=False).sum(),
            titles_lower[pre_mask].str.contains(_pre_gastro,    regex=True, na=False).sum(),
            (titles_lower[pre_mask].str.contains(_pre_fx_routine, regex=True, na=False) &
             ~titles_lower[pre_mask].str.contains(_pre_fx_large,  regex=True, na=False)).sum(),
        )

    # Only send non-pre-filtered articles to Haiku
    to_classify = [a for a in all_articles if not pre_mask.loc[a["idx"]]]
    n_api = len(to_classify)
    logger.info("Sending %d/%d articles to Haiku (%d pre-filtered)", n_api, n, n_pre)
    # ─────────────────────────────────────────────────────────────────────────

    processed = 0
    for batch_start in range(0, n_api, batch_size):
        batch = to_classify[batch_start : batch_start + batch_size]

        # Political call
        pol_scores = _score_batch(batch, _POLITICAL_SYSTEM_PROMPT, client, model)
        # Economic call
        eco_scores = _score_batch(batch, _ECONOMIC_SYSTEM_PROMPT, client, model)

        for i, a in enumerate(batch):
            idx = a["idx"]
            df.at[idx, "political_score"] = pol_scores[i]
            df.at[idx, "economic_score"] = eco_scores[i]

        processed += len(batch)
        if processed % 200 == 0 or processed == n_api:
            logger.info("  Dual-classified %d/%d articles (sent to API)", processed, n_api)

        if batch_start + batch_size < n_api:
            time.sleep(delay)

    # ── Deterministic post-filter ─────────────────────────────────────────────
    # Catch systematic errors the LLM makes regardless of prompt improvements.
    # These are keyword-based overrides for categories that should ALWAYS be 0.
    # Rule: only override when we are CERTAIN. When in doubt, leave the LLM score.

    titles = df["title"].fillna("").str.lower()

    # 1. Farándula / celebrity gossip → both=0
    _farandula = (
        r"cómo se conocieron|historia de amor|boda y|anuncio de ruptura|"
        r"confirma relaci[oó]n|romance|ruptura sentimental|separaci[oó]n de pareja|"
        r"novio|novia|coraz[oó]n roto|ex pareja|vida amorosa|"
        r"conductora?.*amor|amor.*conductora?|esposo.*actor|actriz.*esposo"
    )
    mask_farandula = titles.str.contains(_farandula, regex=True, na=False)
    df.loc[mask_farandula, "political_score"] = 0
    df.loc[mask_farandula, "economic_score"] = 0

    # 2. Sports (European leagues, international competitions, Peruvian clubs) → both=0
    _sports = (
        r"champions league|premier league|la liga|serie a|bundesliga|ligue 1|"
        r"real madrid|fc barcelona|atlético de madrid|manchester|liverpool|chelsea|"
        r"bayern|juventus|psg|inter miami|mbappé|haaland|messi|ronaldo|"
        r"nba|nfl|mlb|nhl|grand slam|wimbledon|roland garros|"
        r"copa libertadores|copa del mundo|mundial de f[uú]tbol|clásico mundial de béisbol|"
        r"sporting cristal|alianza lima|universitario de deportes|fbc melgar|"
        r"copa conmebol|fase de grupos.*copa|copa.*fase de grupos|"
        r"fichaje|transferencia.*jugador|jugador.*transferencia|"
        r"partidos de hoy.*copa|copa.*partidos de hoy|horarios.*copa|copa.*horarios"
    )
    mask_sports = titles.str.contains(_sports, regex=True, na=False)
    df.loc[mask_sports, "political_score"] = 0
    df.loc[mask_sports, "economic_score"] = 0

    # 3. Daily FX close (routine quote) → pol=0, eco≤5
    # Only suppress routine closes; large moves (puntos básicos, %) keep their eco score.
    _fx_routine = r"(precio del d[oó]lar|d[oó]lar hoy|divisa cierra|tipo de cambio hoy|sol se cotiza)"
    _fx_large_move = r"(puntos b[aá]sicos|pierde|gana|dispara|desploma|hunde|sube.*%|baja.*%)"
    mask_fx_routine = (
        titles.str.contains(_fx_routine, regex=True, na=False) &
        ~titles.str.contains(_fx_large_move, regex=True, na=False)
    )
    df.loc[mask_fx_routine, "economic_score"] = df.loc[mask_fx_routine, "economic_score"].clip(upper=5)
    df.loc[mask_fx_routine, "political_score"] = 0

    # 4. Earthquakes / seismic activity → both=0
    # Seismic events are natural, not political/economic instability signals.
    _sismo = r"\b(sismo|sismos|temblor|temblores|terremoto|terremotos|movimiento s[ií]smico)\b"
    mask_sismo = titles.str.contains(_sismo, regex=True, na=False)
    df.loc[mask_sismo, "political_score"] = 0
    df.loc[mask_sismo, "economic_score"] = 0

    # 5. Routine weather / SENAMHI → both=0
    # Only suppress pure forecasts; El Niño emergency declarations are NOT suppressed here.
    _weather_routine = (
        r"\bSENAMHI\b|pronóstico.*lluvia|pronóstico.*temperatura|"
        r"temperatura.*m[aá]xima.*m[ií]nima|lluvias para hoy|"
        r"clima de hoy|tiempo.*hoy.*Lima|"
        r"\bneblina\b|\bgarúa\b|llovizna.*Lima|Lima.*llovizna|"
        r"Lima la gris|amaneció con frío|temperatura bajó"
    )
    mask_weather = titles.str.contains(_weather_routine, regex=True, na=False)
    df.loc[mask_weather, "political_score"] = 0
    df.loc[mask_weather, "economic_score"] = 0

    # 6. Congressional sanctions / inhabilitaciones against political/judicial figures → eco=0
    # These are disciplinary legislative actions, not economic events.
    _inhabilitacion = (
        r"(inhabilita|inhabilitaci[oó]n|reconsideraci[oó]n de votaci[oó]n|"
        r"congreso rechaza|congreso aprueba inhabilit)"
    )
    mask_inhab = titles.str.contains(_inhabilitacion, regex=True, na=False)
    df.loc[mask_inhab, "economic_score"] = 0

    # 7. Pure political actions (voto de confianza, bancada, censura, moción) → eco=0
    # These are legislative/governance actions; economic impact is indirect (chain-reasoning trap).
    _pol_action = (
        r"\bvoto de confianza\b|"
        r"\bcensura\b.*\bgabinete\b|\bgabinete\b.*\bcensura\b|"
        r"\bmoción de vacancia\b|"
        r"\bbancada\b.*(apoya|rechaza|condiciona|exige|pide|voto)|"
        r"(apoya|rechaza|condiciona|exige|pide|voto).*\bbancada\b"
    )
    mask_pol_action = titles.str.contains(_pol_action, regex=True, na=False)
    df.loc[mask_pol_action, "economic_score"] = 0

    # 8. Electoral content (candidates, polls, JNE logistics) → eco=0
    _electoral = (
        r"(candidato|candidata).*(\bsin respaldo\b|encuesta|sondeo|debate.*versus|versus.*debate)|"
        r"\bjne\b.*candidat|\bje[ee]\b.*candidat|candidat.*\bjne\b|"
        r"inscripci[oó]n de candidat|plancha presidencial|intenci[oó]n de voto"
    )
    mask_electoral = titles.str.contains(_electoral, regex=True, na=False)
    df.loc[mask_electoral, "economic_score"] = 0

    # 9. Daily market brief column ("Mercados e indicadores") → pol=0
    # This is a routine financial data column, not a political instability signal.
    mask_mercados = titles.str.contains(r"mercados e indicadores", regex=False, na=False)
    df.loc[mask_mercados, "political_score"] = 0

    # 10. Recipes / gastronomy → both=0
    _gastronomia = (
        r"\breceta\b|\brecetas\b|cómo preparar|ingredientes.*cocina|"
        r"gastronomía.*premio|premio.*gastronomía|mejor restaurante|"
        r"cocina nikkei|cocina peruana.*reconoc"
    )
    mask_gastro = titles.str.contains(_gastronomia, regex=True, na=False)
    df.loc[mask_gastro, "political_score"] = 0
    df.loc[mask_gastro, "economic_score"] = 0

    # Exception guard: if crisis/emergency language is present, do NOT apply
    # lifestyle post-filters (rules 11-15). Real economic shocks can mention
    # holidays, weather, or travel in their headlines.
    _crisis_guard = (
        r"crisis|emergencia|desastre|destruy|colapso|pérdida.*millones|"
        r"inunda|sequía|helada.*producci[oó]n|ni[ñn]o costero|fen[oó]meno.*ni[ñn]o|"
        r"bloque[oó]|paro|huelga|protesta|desabastecimiento|escasez"
    )
    mask_crisis_exception = titles.str.contains(_crisis_guard, regex=True, na=False)

    # 11. Horoscopes / astrology → both=0
    _horoscope = (
        r"hor[oó]scopo|signo del zod[ií]aco|predicciones.*signo|"
        r"\btarot\b|carta astral|ascendente.*signo|signo.*ascendente"
    )
    mask_horoscope = titles.str.contains(_horoscope, regex=True, na=False) & ~mask_crisis_exception
    df.loc[mask_horoscope, "political_score"] = 0
    df.loc[mask_horoscope, "economic_score"] = 0

    # 12. Lottery / raffle results → both=0
    _lottery = r"la tinka|loter[ií]a.*resultado|sorteo.*ganador|resultado.*sorteo|n[uú]meros ganadores"
    mask_lottery = titles.str.contains(_lottery, regex=True, na=False) & ~mask_crisis_exception
    df.loc[mask_lottery, "political_score"] = 0
    df.loc[mask_lottery, "economic_score"] = 0

    # 13. Reality TV / entertainment shows → both=0
    _reality = (
        r"\bgran hermano\b|esto es guerra|al fondo hay sitio|la voz per[uú]|"
        r"\bfarándula\b|reality.*per[uú]|per[uú].*reality|"
        r"programa.*televis[i].*hoy|telenovela"
    )
    mask_reality = titles.str.contains(_reality, regex=True, na=False) & ~mask_crisis_exception
    df.loc[mask_reality, "political_score"] = 0
    df.loc[mask_reality, "economic_score"] = 0

    # 14. Holiday travel tips / personal savings listicles → both=0
    # These are lifestyle articles, not macro economic signals.
    _lifestyle_tips = (
        r"consejos para viajar|tips para (ahorrar|viajar)|c[oó]mo ahorrar en|"
        r"feriado largo.*(d[oó]nde ir|qu[eé] hacer|visitar)|"
        r"qu[eé] hacer en (semana santa|fiestas patrias|a[nñ]o nuevo|navidad)|"
        r"destinos para (semana santa|fiestas patrias|vacaciones)|"
        r"lugares para visitar.*feriado|feriado.*lugares para visitar"
    )
    mask_lifestyle = titles.str.contains(_lifestyle_tips, regex=True, na=False) & ~mask_crisis_exception
    df.loc[mask_lifestyle, "political_score"] = 0
    df.loc[mask_lifestyle, "economic_score"] = 0

    # 15. Personal finance advice (cap eco at 20, not zero — some marginal relevance)
    _personal_finance = (
        r"finanzas personales|c[oó]mo invertir.*bolsa|mejor cuenta de ahorro|"
        r"tarjeta de cr[eé]dito.*beneficio|AFP.*retiro.*pasos|"
        r"c[oó]mo retirar.*AFP|pasos para retirar"
    )
    mask_personal_fin = (
        titles.str.contains(_personal_finance, regex=True, na=False) &
        ~mask_crisis_exception &
        (df["economic_score"].fillna(0) > 20)
    )
    df.loc[mask_personal_fin, "economic_score"] = 20

    masks = [mask_farandula, mask_sports, mask_fx_routine, mask_sismo, mask_weather,
             mask_inhab, mask_pol_action, mask_electoral, mask_mercados, mask_gastro,
             mask_horoscope, mask_lottery, mask_reality, mask_lifestyle, mask_personal_fin]
    labels = ["farándula", "sports", "fx_routine", "sismo", "weather",
              "inhabilitacion", "pol_action", "electoral", "mercados", "gastro",
              "horoscope", "lottery", "reality", "lifestyle_tips", "personal_finance"]
    n_filtered = masks[0].copy()
    for m in masks[1:]:
        n_filtered = n_filtered | m
    if n_filtered.sum() > 0:
        counts = ", ".join(f"{lbl}={m.sum()}" for lbl, m in zip(labels, masks) if m.sum() > 0)
        logger.info("Post-filter applied to %d articles (%s)", n_filtered.sum(), counts)
    # ─────────────────────────────────────────────────────────────────────────

    # Summary
    pol_nonzero = (df["political_score"].fillna(0) > 0).sum()
    eco_nonzero = (df["economic_score"].fillna(0) > 0).sum()
    logger.info(
        "Done. Political nonzero: %d (%.1f%%)  Economic nonzero: %d (%.1f%%)",
        pol_nonzero, pol_nonzero / n * 100,
        eco_nonzero, eco_nonzero / n * 100,
    )

    return df


def compute_monthly_event_score(
    events_df: pd.DataFrame,
    score_column: str = "severity_claude",
) -> pd.Series:
    """Aggregate event severity to monthly frequency.

    Uses max severity in each month as the monthly score.
    Months without events get score 0.
    """
    df = events_df.copy()
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month")[score_column].max()
    return monthly


def compute_weekly_event_score(
    events_df: pd.DataFrame,
    score_column: str = "severity_claude",
) -> pd.Series:
    """Aggregate event severity to weekly frequency (Friday).

    Uses max severity in each week.
    """
    df = events_df.copy()
    df = df.set_index("date")
    weekly = df[score_column].resample("W-FRI").max()
    weekly = weekly.fillna(0)
    return weekly
