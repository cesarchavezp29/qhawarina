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


_POLITICAL_SYSTEM_PROMPT = """Eres un clasificador de riesgo político para Perú. Tu tarea es evaluar si un artículo de prensa señala una amenaza a la estabilidad política e institucional doméstica del Perú.

DEFINICIÓN: Riesgo político abarca amenazas al orden institucional democrático, la continuidad del gobierno, la estabilidad social, y el funcionamiento normal de los poderes del Estado en Perú.

Asigna un puntaje de 0 a 100 (usa CUALQUIER valor entero, no solo múltiplos de 10 o 20). Los siguientes son PUNTOS DE REFERENCIA:

  0: Sin relevancia para la estabilidad política de Perú. Noticias internacionales, deportes, entretenimiento, cultura, operaciones gubernamentales rutinarias, legislación ordinaria sin controversia.
 20: Tensión menor. Fricción política de bajo nivel que no amenaza la continuidad institucional. Declaraciones políticas ordinarias, desacuerdos menores entre actores políticos.
 40: Inestabilidad moderada. Disputas escalando entre poderes del Estado, malestar social creciente en varias regiones, desafíos de gobernabilidad que podrían intensificarse.
 60: Crisis significativa. Confrontación activa entre actores institucionales, movilización social extendida, amenazas serias a la continuidad gubernamental o legislativa.
 80: Crisis severa. Amenaza inminente a la continuidad del Ejecutivo o Legislativo, malestar civil generalizado, quiebre de normas institucionales, violencia política.
100: Emergencia de régimen. Ruptura constitucional activa, violencia estatal contra civiles, colapso de la gobernanza democrática.

REGLAS:
- Evalúa SOLO eventos que afecten el orden político DOMÉSTICO de Perú.
- Política internacional (elecciones EEUU, conflictos globales) = 0, SALVO que desencadenen directamente una crisis política en Perú.
- Usa cualquier entero: 5, 17, 33, 52, 68, 74, 91 — el que mejor refleje el contenido.
- Evalúa basándote en lo que el artículo DICE o implica fuertemente, no en conocimiento externo.
- Obituarios, aniversarios, reseñas de libros/películas = 0.
- Sé consistente."""

_ECONOMIC_SYSTEM_PROMPT = """Eres un clasificador de riesgo económico para Perú. Tu tarea es evaluar si un artículo de prensa señala una amenaza a la estabilidad económica del Perú, específicamente riesgos que afecten al Perú de manera DESPROPORCIONADA respecto a otras economías.

DEFINICIÓN: Riesgo económico abarca amenazas a la estabilidad macroeconómica, fiscal, financiera, y productiva del Perú.

Asigna un puntaje de 0 a 100 (usa CUALQUIER valor entero, no solo múltiplos de 10 o 20). Los siguientes son PUNTOS DE REFERENCIA:

  0: Sin relevancia para la estabilidad económica de Perú. Noticias económicas internacionales generales sin impacto diferenciado en Perú, negocios rutinarios, deportes, entretenimiento.
 20: Preocupación económica menor. Ajustes de política rutinarios, dificultades sectoriales modestas, presiones fiscales manejables.
 40: Estrés económico moderado. Deterioro significativo en indicadores clave peruanos, vulnerabilidades notables en sectores de los que Perú depende desproporcionadamente.
 60: Vulnerabilidad económica significativa. Deterioro agudo que amenaza la estabilidad macroeconómica peruana, disrupciones mayores en los motores económicos primarios.
 80: Crisis económica severa. Estrés financiero sistémico en Perú, colapso de fuentes principales de ingreso nacional, intervenciones de emergencia requeridas.
100: Emergencia económica. Riesgo de default soberano, quiebre del sistema bancario, pérdida total de confianza del mercado.

PRUEBA CLAVE para eventos internacionales: "¿Este evento amenaza la economía de Perú MÁS que al promedio de economías emergentes?" Si no → 0. Si sí → puntaje según severidad.

REGLAS:
- Eventos económicos internacionales que afectan a todos los países por igual = 0.
- Usa cualquier entero: 5, 17, 33, 52, 68, 74, 91.
- Evalúa basándote en lo que el artículo DICE o implica fuertemente.
- Sé consistente."""

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

    processed = 0
    for batch_start in range(0, n, batch_size):
        batch = all_articles[batch_start : batch_start + batch_size]

        # Political call
        pol_scores = _score_batch(batch, _POLITICAL_SYSTEM_PROMPT, client, model)
        # Economic call
        eco_scores = _score_batch(batch, _ECONOMIC_SYSTEM_PROMPT, client, model)

        for i, a in enumerate(batch):
            idx = a["idx"]
            df.at[idx, "political_score"] = pol_scores[i]
            df.at[idx, "economic_score"] = eco_scores[i]

        processed += len(batch)
        if processed % 200 == 0 or processed == n:
            logger.info("  Dual-classified %d/%d articles", processed, n)

        if batch_start + batch_size < n:
            time.sleep(delay)

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
