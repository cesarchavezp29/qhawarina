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


_POLITICAL_SYSTEM_PROMPT = """Clasificarás un artículo de prensa peruana. Evalúa ÚNICAMENTE el riesgo de inestabilidad política institucional en el Perú basándote en lo que el artículo afirma o implica directamente.

Riesgo político se define como: la amenaza, materialización o escalamiento de eventos adversos que afectan la estabilidad institucional del Perú, incluyendo: crisis de gobernabilidad, vacancias presidenciales, censuras ministeriales, rupturas entre poderes del Estado, protestas masivas que amenazan la continuidad del gobierno, crisis constitucionales, y conflictos sociales que escalan a nivel nacional.

Asigna un puntaje entero de 0 a 100:

0:      Sin relevancia política. Deportes, farándula, clima, recetas, tecnología, noticias internacionales sin impacto en Perú, obituarios, efemérides.
1-20:   Tensión menor. Declaraciones políticas rutinarias, sesiones legislativas ordinarias, nombramientos menores.
21-40:  Inestabilidad baja-moderada. Investigaciones parlamentarias, huelgas sectoriales, protestas locales, disputas partidarias.
41-60:  Inestabilidad significativa. Mociones de interpelación, cambios de gabinete bajo presión, protestas regionales que se expanden, conflictos entre poderes del Estado.
61-80:  Crisis mayor. Intentos de vacancia con firmas suficientes, censuras ministeriales aprobadas, protestas masivas nacionales, estados de emergencia.
81-100: Emergencia. Vacancia consumada, renuncia presidencial, ruptura constitucional, violencia política masiva, golpe de estado.

PUNTAJE 0 OBLIGATORIO para:
- Noticias internacionales sin impacto directo en la política peruana
- Eventos económicos puros (sin maniobra política)
- Deportes, farándula, entretenimiento, gastronomía
- Artículos sobre política de otros países
- Reportes climáticos o de desastres naturales (salvo que generen crisis política)

EJEMPLOS DE PUNTAJE 0 POLÍTICO:
- "Ruptura del gasoducto Camisea" → pol=0 (evento económico/industrial, no político)
- "BCR mantiene tasa de referencia" → pol=0 (política monetaria técnica)
- "Precio del dólar hoy" → pol=0
- "Trump anuncia aranceles" → pol=0 (política exterior extranjera — salvo que artículo describa crisis política interna peruana por ello)

REGLA: Si un artículo usa una crisis económica como CONTEXTO para una maniobra política (ej: "oposición condiciona voto de confianza por crisis energética"), el puntaje POLÍTICO debe ser alto. Lo que importa es la ACCIÓN política, no el tema económico."""

_ECONOMIC_SYSTEM_PROMPT = """Clasificarás un artículo de prensa peruana. Evalúa ÚNICAMENTE el riesgo de disrupción económica en el Perú basándote en lo que el artículo afirma o implica directamente.

Riesgo económico se define como: la amenaza, materialización o escalamiento de eventos adversos que DISRUMPEN la actividad económica normal del Perú, incluyendo: crisis financieras, colapso de sectores productivos, interrupciones en cadenas de suministro (energía, gas, minería), shocks de precios que afectan a la población, riesgo fiscal grave, paralizaciones económicas por conflictos, y amenazas comerciales externas (aranceles, sanciones).

Asigna un puntaje entero de 0 a 100:

0:      Sin relevancia económica O noticia económica sin elemento de RIESGO/DISRUPCIÓN. Deportes, farándula, clima sin impacto productivo, noticias positivas (crecimiento, inversión nueva, récords de exportación).
1-20:   Riesgo menor. Presiones sectoriales modestas, volatilidad cambiaria dentro de rangos normales, advertencias de analistas sin impacto inmediato.
21-40:  Estrés moderado. Deterioro de indicadores clave, conflictos laborales en sectores medianos, alzas de precios que afectan canasta básica.
41-60:  Vulnerabilidad significativa. Disrupciones en minería, gas o exportaciones principales, bloqueos de carreteras que afectan suministro, reformas que amenazan estabilidad fiscal.
61-80:  Crisis severa. Colapso sectorial, paralización energética nacional, fuga de capitales, downgrade crediticio.
81-100: Emergencia. Default soberano, colapso del sistema financiero, hiperinflación, paralización económica generalizada.

PUNTAJE 0 OBLIGATORIO para:
- Noticias económicas POSITIVAS (crecimiento, inversión, inauguraciones, premios empresariales, récords)
- Reportes RUTINARIOS sin shock (dólar hoy, tipo de cambio cierra en X, cotización de metales sin contexto de crisis)
- Tendencias ESTRUCTURALES crónicas (desempleo juvenil alto, informalidad laboral, brecha salarial — son problemas de largo plazo, NO disrupción aguda)
- Especulación sobre tecnología/futuro (IA reemplazará empleos, criptomonedas, tendencias globales)
- Consumo y estilo de vida (perfumería crece, turismo aumenta, nuevo restaurante, moda)
- Convocatorias laborales, ofertas de empleo, becas
- Maniobras POLÍTICAS que usan temas económicos como pretexto (ej: "Congreso bloquea reforma de pensiones" = político, no económico — el riesgo es la inestabilidad institucional)

EJEMPLOS DE PUNTAJE 0 (memoriza estos patrones — estos artículos deben recibir eco=0):
- "Roberto Chiabra será candidato presidencial" → eco=0 (candidatura política, no disrupción económica)
- "Eduardo Arana: Gobierno hará prevalecer autoridad" → eco=0 (declaración política)
- "Congreso aprueba cambios a ley de inversiones" → eco=0 (legislación, no disrupción — salvo que el artículo indique que causa fuga de capitales o rechazo de mercados)
- "MTC modifica reglamento portuario" → eco=0 (regulación rutinaria)
- "Sunafil: qué tipo de denuncias laborales se pueden realizar" → eco=0 (información al consumidor)
- "Desempleo en EEUU sube a 4.2%" → eco=0 (dato extranjero sin impacto directo en Perú)
- "Wall Street cae 2%" → eco=0 (mercado extranjero — salvo que artículo conecte con efecto en BVL o economía peruana con datos)
- "CAF firma acuerdos de cooperación" → eco=0 (noticia positiva/institucional)
- "Juliana Oxenford critica gestión ministerial" → eco=0 (opinión televisiva)
- "Indecopi multa a universidad" → eco=0 (regulación puntual, no disrupción sistémica)
- "SBS: cómo acceder a tu pensión" → eco=0 (información al consumidor)
- "Mujeres ocupan 4 de cada 10 puestos altos" → eco=0 (estadística social, no riesgo económico)

EJEMPLOS DE PUNTAJE ALTO (para contraste — solo estos tipos merecen eco≥40):
- "Ruptura del gasoducto Camisea paraliza suministro de gas al 64% del país" → eco=80 (disrupción energética nacional)
- "Trump anuncia arancel de 50% al cobre peruano" → eco=70 (amenaza directa a exportaciones)
- "Bloqueo de carreteras impide suministro de oxígeno a hospitales del sur" → eco=60 (disrupción de cadena de suministro con impacto en vidas)
- "BCR: expectativas empresariales caen a zona pesimista" → eco=45 (deterioro de indicadores macro)
- "Petroperú reporta pérdidas de S/2,000 millones" → eco=55 (crisis fiscal en empresa estatal)

REGLA: Riesgo económico = DISRUPCIÓN o AMENAZA ACTIVA a la actividad económica. NO es lo mismo que "noticia sobre economía". Si no hay disrupción, amenaza o shock, el puntaje es 0."""

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


def post_filter_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic post-filters to existing scored articles.

    Same rules as the inline post-filter in classify_articles_dual, but
    callable standalone so we can apply to an already-classified parquet
    without re-running the full Haiku classification.
    """
    import re as _re
    df = df.copy()
    titles = df["title"].fillna("").str.lower()

    _crisis_guard = (
        r"crisis|emergencia|desastre|destruy|colapso|pérdida.*millones|"
        r"inunda|sequía|helada.*producci[oó]n|ni[ñn]o costero|fen[oó]meno.*ni[ñn]o|"
        r"bloque[oó]|paro|huelga|protesta|desabastecimiento|escasez"
    )
    mask_crisis = titles.str.contains(_crisis_guard, regex=True, na=False)

    rules_eco_zero = [
        # Foreign stock markets
        (r"wall street|nasdaq|dow jones|s&p 500|bolsa de nueva york|"
         r"mercados? (de|en) (eeuu|estados unidos|asia|europa|china|jap[oó]n)|"
         r"bolsas (europeas|asi[aá]ticas|chinas)|nikkei|ftse|dax\b|cac 40", True),
        # Foreign unemployment / economic data
        (r"desempleo (en |de )(eeuu|estados unidos|china|europa|jap[oó]n|reino unido|alemania)|"
         r"inflaci[oó]n (en |de )(eeuu|estados unidos|china|europa)|"
         r"pib (de|en) (china|eeuu|estados unidos|europa|alemania)", True),
        # Daily market summary columns
        (r"mercados e indicadores|indicadores econ[oó]micos hoy|cierre de mercados|"
         r"resumen (de |)mercados|apertura de (la |)bolsa hoy", True),
        # Corporate earnings / positive growth
        (r"eleva (sus )?ingresos|reporta ganancias|incrementa (sus )?ventas|"
         r"registra crecimiento|aumenta (su )?facturaci[oó]n|utilidades crecen|"
         r"crece \d+% en (sus )?ventas|record de ventas|r[eé]cord de (ingresos|ganancias)", True),
        # Sports club business
        (r"sport boys.*(crec|ingres|venta|centenari|expand|presupuest)|"
         r"(alianza lima|universitario|cristal|melgar|ayacucho fc).*(crec|ingres|venta|presupuest|patrocin)", True),
        # Political candidacies (no crisis exception)
        (r"ser[aá] candidat[oa]|lanza (su )?candidatura|postula (a|como)|"
         r"inscribe (su )?candidatura|plancha presidencial|f[oó]rmula presidencial", False),
        # Consumer info / routine regulatory
        (r"sbs (publica|aprueba|establece).*(proyecto|norma|reglamento)|"
         r"indecopi (multa|sanciona|resuelve).{0,40}(universidad|colegio|empresa|proveedor)|"
         r"sunafil.*(denunci|orient|capacit|inform)|"
         r"c[oó]mo (acceder|obtener|solicitar|retirar).*(pensi[oó]n|afp|cts|gratificaci[oó]n|bono)|"
         r"premio.*(reconoc|otorg|entreg|galardon)|ranking de (empresas|universidades|pa[ií]ses)|"
         r"mujeres ocupan|brecha (salarial|de g[eé]nero)|paridad de g[eé]nero|equidad laboral", True),
    ]

    n_zeroed = 0
    for pattern, use_crisis_exception in rules_eco_zero:
        mask = titles.str.contains(pattern, regex=True, na=False)
        if use_crisis_exception:
            mask = mask & ~mask_crisis
        changed = mask & (df["economic_score"].fillna(0) > 0)
        n_zeroed += int(changed.sum())
        df.loc[mask, "economic_score"] = 0

    # FX routine cap at 10
    _fx_routine = r"(precio del d[oó]lar|d[oó]lar hoy|divisa cierra|tipo de cambio hoy|sol se cotiza)"
    _fx_large_move = r"(puntos b[aá]sicos|pierde|gana|dispara|desploma|hunde|sube.*%|baja.*%)"
    mask_fx = (
        titles.str.contains(_fx_routine, regex=True, na=False) &
        ~titles.str.contains(_fx_large_move, regex=True, na=False)
    )
    df.loc[mask_fx, "economic_score"] = df.loc[mask_fx, "economic_score"].clip(upper=10)

    logger.info("post_filter_scores: zeroed eco on %d articles", n_zeroed)
    return df


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

    # 16. Foreign stock markets (Wall Street, Nasdaq, Dow, Asian/European exchanges) → eco=0
    _foreign_markets = (
        r"wall street|nasdaq|dow jones|s&p 500|bolsa de nueva york|"
        r"mercados? (de|en) (eeuu|estados unidos|asia|europa|china|jap[oó]n)|"
        r"bolsas (europeas|asi[aá]ticas|chinas)|nikkei|ftse|dax\b|cac 40"
    )
    mask_foreign_markets = titles.str.contains(_foreign_markets, regex=True, na=False) & ~mask_crisis_exception
    df.loc[mask_foreign_markets, "economic_score"] = 0

    # 17. Foreign unemployment / economic data (not Peru-specific) → eco=0
    _foreign_econ_data = (
        r"desempleo (en |de )(eeuu|estados unidos|china|europa|jap[oó]n|reino unido|alemania)|"
        r"inflaci[oó]n (en |de )(eeuu|estados unidos|china|europa)|"
        r"pib (de|en) (china|eeuu|estados unidos|europa|alemania)"
    )
    mask_foreign_econ = titles.str.contains(_foreign_econ_data, regex=True, na=False) & ~mask_crisis_exception
    df.loc[mask_foreign_econ, "economic_score"] = 0

    # 18. Daily market summary columns → eco=0
    _market_summary = (
        r"mercados e indicadores|indicadores econ[oó]micos hoy|cierre de mercados|"
        r"resumen (de |)mercados|apertura de (la |)bolsa hoy"
    )
    mask_market_summary = titles.str.contains(_market_summary, regex=True, na=False) & ~mask_crisis_exception
    df.loc[mask_market_summary, "economic_score"] = 0

    # 19. Corporate earnings / positive growth → eco=0
    _corp_earnings = (
        r"eleva (sus )?ingresos|reporta ganancias|incrementa (sus )?ventas|"
        r"registra crecimiento|aumenta (su )?facturaci[oó]n|utilidades crecen|"
        r"crece \d+% en (sus )?ventas|record de ventas|r[eé]cord de (ingresos|ganancias)"
    )
    mask_corp_earnings = titles.str.contains(_corp_earnings, regex=True, na=False) & ~mask_crisis_exception
    df.loc[mask_corp_earnings, "economic_score"] = 0

    # 20. Sports club business / anniversary / expansion (not disruption) → eco=0
    _sports_biz = (
        r"sport boys.*(crec|ingres|venta|centenari|expand|presupuest)|"
        r"(alianza lima|universitario|cristal|melgar|ayacucho fc).*(crec|ingres|venta|presupuest|patrocin)"
    )
    mask_sports_biz = titles.str.contains(_sports_biz, regex=True, na=False) & ~mask_crisis_exception
    df.loc[mask_sports_biz, "economic_score"] = 0

    # 21. Political candidacies (not economic) → eco=0
    _candidacy = (
        r"ser[aá] candidat[oa]|lanza (su )?candidatura|postula (a|como)|"
        r"inscribe (su )?candidatura|plancha presidencial|f[oó]rmula presidencial"
    )
    mask_candidacy = titles.str.contains(_candidacy, regex=True, na=False)
    df.loc[mask_candidacy, "economic_score"] = 0

    # 22. Consumer info / regulatory routine → eco=0
    _consumer_info = (
        r"sbs (publica|aprueba|establece).*(proyecto|norma|reglamento)|"
        r"indecopi (multa|sanciona|resuelve).{0,40}(universidad|colegio|empresa|proveedor)|"
        r"sunafil.*(denunci|orient|capacit|inform)|"
        r"c[oó]mo (acceder|obtener|solicitar|retirar).*(pensi[oó]n|afp|cts|gratificaci[oó]n|bono)|"
        r"premio.*(reconoc|otorg|entreg|galardon)|ranking de (empresas|universidades|pa[ií]ses)|"
        r"mujeres ocupan|brecha (salarial|de g[eé]nero)|paridad de g[eé]nero|equidad laboral"
    )
    mask_consumer_info = titles.str.contains(_consumer_info, regex=True, na=False) & ~mask_crisis_exception
    df.loc[mask_consumer_info, "economic_score"] = 0

    masks = [mask_farandula, mask_sports, mask_fx_routine, mask_sismo, mask_weather,
             mask_inhab, mask_pol_action, mask_electoral, mask_mercados, mask_gastro,
             mask_horoscope, mask_lottery, mask_reality, mask_lifestyle, mask_personal_fin,
             mask_foreign_markets, mask_foreign_econ, mask_market_summary,
             mask_corp_earnings, mask_sports_biz, mask_candidacy, mask_consumer_info]
    labels = ["farándula", "sports", "fx_routine", "sismo", "weather",
              "inhabilitacion", "pol_action", "electoral", "mercados", "gastro",
              "horoscope", "lottery", "reality", "lifestyle_tips", "personal_finance",
              "foreign_markets", "foreign_econ", "market_summary",
              "corp_earnings", "sports_biz", "candidacy", "consumer_info"]
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
