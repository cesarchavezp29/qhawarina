"""Claude API classifier for political event severity.

Uses Claude Haiku for direct severity classification on the ordinal
1-3 scale matching ground truth. Batched prompts for efficiency.
"""

import json
import logging
import time

import pandas as pd

logger = logging.getLogger("nexus.nlp.classifier")

# ── Comprehensive sports pattern (used in pre-filter, inline post-filter, standalone post-filter) ──
_SPORTS_PATTERN = (
    # ── International club competitions ───────────────────────────────────────
    r"champions league|europa league|conference league|"
    r"copa libertadores|copa sudamericana|recopa sudamericana|"
    r"copa del mundo de clubes|mundial de clubes|"
    r"copa del mundo|mundial de f[uú]tbol|eurocopa|copa am[eé]rica|"
    r"clásico mundial de b[eé]isbol|world series|superbowl|super bowl|"
    # ── League names ──────────────────────────────────────────────────────────
    r"premier league|la liga\b|serie a\b|bundesliga|ligue 1|eredivisie|"
    r"mls\b|liga mx\b|brasileir[aã]o|liga argentina|primera divisi[oó]n argentina|"
    r"liga 1\b|liga1\b|liga 2\b|torneo apertura|torneo clausura|"   # Peru's Liga 1/2
    # ── European club names ───────────────────────────────────────────────────
    r"real madrid|fc barcelona|atlético de madrid|sevilla fc|real sociedad|"
    r"manchester (city|united)|liverpool fc|chelsea fc|arsenal fc|tottenham|"
    r"newcastle united|aston villa|"
    r"bayern munich|borussia dortmund|rb leipzig|bayer leverkusen|"
    r"juventus|inter de mil[aá]n|ac milan|napoli|as roma|fiorentina|lazio\b|"
    r"psg|paris saint-germain|olympique (de )?marsella|lyon\b|"
    r"ajax\b|psv eindhoven|benfica\b|porto\b|sporting\b|"
    # ── Latin American clubs ──────────────────────────────────────────────────
    r"boca juniors|river plate|racing club|independiente|san lorenzo|"
    r"flamengo|palmeiras|fluminense|corinthians|s[aã]o paulo|atletico mineiro|"
    r"nacional\b.*uruguay|pe[ñn]arol|colo colo|universidad de chile|"
    r"am[eé]rica\b.*m[eé]xico|chivas\b|cruz azul\b|pumas unam|tigres uanl|"
    r"inter miami|"
    # ── Peruvian Liga 1 clubs (all) ───────────────────────────────────────────
    r"sporting cristal|alianza lima|universitario de deportes|"
    r"fbc melgar|atlético grau|sport huancayo|cusco fc|"
    r"cienciano\b|ayacucho fc|adt de tarma|"
    r"carlos mannucci|uni[oó]n comercio|sport boys\b|"
    r"comerciantes unidos|univ[eé]rsitario\b.*f[uú]tbol|los chankas|"
    r"deportivo binacional|pirata fc|academia cantolao|"
    # ── Athletes (global stars) ───────────────────────────────────────────────
    r"messi|ronaldo|mbapp[eé]|haaland|neymar|vinicius|rodri\b|bellingham|"
    r"lebron james|steph curry|giannis|luka don[cč]i[cč]|shai gilgeous|"
    r"lionel messi|cristiano ronaldo|"
    # ── North American leagues ────────────────────────────────────────────────
    r"\bnba\b|\bnfl\b|\bmlb\b|\bnhl\b|\bmls\b|\bwnba\b|"
    # ── Tennis ───────────────────────────────────────────────────────────────
    r"grand slam|wimbledon|roland garros|us open.*tenis|abierto de australia|"
    r"djokovic|alcaraz|sinner\b|swiatek|"
    # ── F1 / motorsport ──────────────────────────────────────────────────────
    r"f[oó]rmula 1|formula one|gran premio de|gp de|max verstappen|lewis hamilton|"
    r"ferrari f1|mercedes f1|red bull racing|"
    # ── Olympics / multi-sport ────────────────────────────────────────────────
    r"juegos ol[ií]mpicos|olimpiadas|juegos panamericanos|"
    r"atletismo.*mundial|mundial.*atletismo|campeonato mundial.*nataci[oó]n|"
    # ── Generic sports match vocabulary (high-precision) ─────────────────────
    r"fichaje|transferencia.*jugador|jugador.*transferencia|"
    r"partidos de hoy.*copa|copa.*partidos de hoy|horarios.*liga|"
    r"tabla de posiciones|tabla acumulada|"
    r"copa conmebol|fase de grupos.*copa|copa.*fase de grupos|"
    r"jornada \d+.*liga|liga.*jornada \d+"
)

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


def _normalize_title(s: str) -> str:
    """Normalize a title for keyword matching.

    Strips accents, removes mojibake replacement chars (\ufffd),
    lowercases, and converts to plain ASCII.  This makes patterns
    portable regardless of whether the title was stored with correct
    UTF-8, Latin-1, or as corrupted 'replacement-character' sequences.
    """
    import unicodedata as _ud
    s = str(s).lower()
    s = s.replace("\ufffd", "").replace("\u0000", "")
    # Normalize composed forms → decomposed, then drop combining marks
    s = _ud.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s


def post_filter_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic post-filters to existing scored articles.

    Same rules as the inline post-filter in classify_articles_dual, but
    callable standalone so we can apply to an already-classified parquet
    without re-running the full Haiku classification.
    """
    import re as _re
    df = df.copy()
    # titles: raw lower — used by all legacy rules (which already have [oó] etc.)
    titles = df["title"].fillna("").str.lower()
    # nt: accent-stripped ASCII lower — used by all new rules; encoding-safe
    nt = df["title"].fillna("").apply(_normalize_title)

    _crisis_guard = (
        r"crisis|emergencia|desastre|destruy|colapso|perdida.*millones|"
        r"inunda|sequia|helada.*produccion|nino costero|fenomeno.*nino|"
        r"bloqueo|paro|huelga|protesta|desabastecimiento|escasez|"
        r"huaico|desborde|damnificados|alud|lluvias.*deja|lluvias.*afect"
    )
    mask_crisis = nt.str.contains(_crisis_guard, regex=True, na=False)

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
        # Health info / medical awareness (not systemic health crisis)
        (r"glaucoma|miopía|astigmatismo|chequeo.*médico|chequeos.*médicos|"
         r"c[oó]mo proteger.*(pacientes|salud)|c[oó]mo prevenir.*(c[aá]ncer|diabetes|hipertens)|"
         r"covid.{0,20}(c[oó]mo proteger|prevenci[oó]n|vacunaci[oó]n rutinaria)|"
         r"vacunaci[oó]n.{0,20}(calendario|rutinaria|anual)|"
         r"enfermedades.{0,20}m[aá]s comunes|s[ií]ntomas de|remedios para|"
         r"oncol[oó]gico.*chequeo|chequeo.*oncol[oó]gico", True),
        # Foreign economic agreements / IMF loans to non-Peru countries
        (r"fmi.{0,30}(acuerdo|pr[eé]stamo|programa).{0,30}(argentina|ecuador|colombia|m[eé]xico|bolivia|brasil)|"
         r"banco mundial.{0,30}(pr[eé]stamo|acuerdo).{0,30}(argentina|ecuador|colombia|m[eé]xico)|"
         r"(argentina|brasil|colombia).{0,30}(pr[eé]stamo|acuerdo).{0,30}fmi|"
         r"deuda (de|externa).{0,20}(argentina|brasil|colombia|m[eé]xico|bolivia)", True),
        # Product recalls / vehicle safety notices → not economic instability
        (r"indecopi alerta.{0,30}(revisi[oó]n|retiro|recall)|"
         r"llamados? a revisi[oó]n.{0,30}veh[ií]culo|"
         r"retiro del mercado.{0,30}(producto|veh[ií]culo|alimento)|"
         r"recall.{0,20}veh[ií]culo|"
         r"(ford|toyota|chevrolet|kia|hyundai|volkswagen).{0,30}(revisi[oó]n|retiro|defecto)", True),
        # Economist opinion columns (cap separately, not zero — handled below)
        # Gender diversity / talent features (not systemic economic risk)
        (r"talento femenino|ambiciones del talento|liderazgo femenino.*empresa|"
         r"mujeres en (cargos|puestos|roles).*(directivos|ejecutivos|gerenciales)|"
         r"radiograf[ií]a.*(g[eé]nero|mujer)|"
         r"chef.{0,30}(impulsa|promueve|cocina).{0,30}(comedores|popular)|"
         r"gastronomía.{0,30}(premio|reconoc|mejor.*mundo)|"
         r"mejor restaurante.{0,30}(per[uú]|latinoam[eé]rica|mundo)", True),
    ]

    # Economist opinion columns → eco capped at 20 (informed market signal, not crisis)
    _opinion_col = (
        r"elmer cuba.*:|: elmer cuba|"
        r"pablo lavado.*:|carlos adri[aá]nz[eé]n.*:|jorge gonz[aá]lez izquierdo.*:|"
        r"la (ha |han )?sacado barata|opina sobre la econom[ií]a|"
        r"economista.{0,20}(opina|analiza|explica|advierte sobre)"
    )
    mask_opinion = nt.str.contains(_opinion_col, regex=True, na=False)
    df.loc[mask_opinion & (df["economic_score"].fillna(0) > 20), "economic_score"] = 20

    # Sports matches / clubs / competitions → zero BOTH scores
    mask_sports_pf = nt.str.contains(_SPORTS_PATTERN, regex=True, na=False)
    n_sports_pol = int((mask_sports_pf & (df["political_score"].fillna(0) > 0)).sum())
    n_sports_eco = int((mask_sports_pf & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_sports_pf, "political_score"] = 0
    df.loc[mask_sports_pf, "economic_score"] = 0
    if n_sports_pol + n_sports_eco > 0:
        logger.info("post_filter_scores: sports zeroed pol=%d eco=%d articles", n_sports_pol, n_sports_eco)

    # Celebrity / tabloid / entertainment → zero BOTH scores (no crisis exception)
    _celeb = (
        r"maju mantilla|magaly medina|far[aá]ndula|reconciliaci[oó]n.*salcedo|"
        r"se agarran a golpes|pelea en combi|"
        r"supuesta reconciliaci[oó]n|infidelidad|ampay\b|esc[aá]ndalo sentimental|"
        r"romance entre|separaci[oó]n de|confirma su embarazo|boda de|"
        r"chollywood|showbiz|telenovela|reality (de amor|sentimental)|"
        # TV shows / entertainment personalities
        r"ale fuller|mayra go[ñn]i|flavia laos|ma[ñn]ana me caso|esto es guerra|"
        r"al fondo hay sitio|gran hermano|la voz per[uú]|yo soy\b|el gran chef|"
        r"natti natasha|bad bunny|taylor swift|reggaet[oó]n|concierto de|gira musical|"
        # TV premieres / show announcements
        r"primer adelanto de|estreno de.*serie|nueva temporada de|reaparecen con|"
        # Street crime / accidents (not systemic disruption)
        r"asaltan a|roban a|accidente de tr[aá]nsito|choque vehicular|atropell[oa]\b|"
        # Religious / opinion surveys
        r"preferencias religiosas|creencias religiosas|fe de los j[oó]venes|"
        r"religi[oó]n.*encuesta|encuesta.*religi[oó]n"
    )
    mask_celeb = nt.str.contains(_celeb, regex=True, na=False)
    n_celeb_eco = int((mask_celeb & (df["economic_score"].fillna(0) > 0)).sum())
    n_celeb_pol = int((mask_celeb & (df["political_score"].fillna(0) > 0)).sum())
    df.loc[mask_celeb, "economic_score"] = 0
    df.loc[mask_celeb, "political_score"] = 0
    if n_celeb_eco + n_celeb_pol > 0:
        logger.info("post_filter_scores: celeb/tabloid zeroed eco=%d pol=%d articles", n_celeb_eco, n_celeb_pol)

    # Foreign leader / ex-president health & personal news → BOTH scores 0
    # (their medical status has no bearing on Peru's economy or politics)
    _foreign_leader_health = (
        r"(expresidente|ex[ -]presidente|presidente).{0,40}"
        r"(brasile[ñn]o|argentino|colombiano|chileno|venezolano|boliviano|ecuatoriano|"
        r"paraguayo|uruguayo|estadounidense|franc[eé]s|alem[aá]n|espa[ñn]ol|"
        r"italiano|ingl[eé]s|chino|ruso|canadiense|japon[eé]s|coreano|turco|israelí).{0,80}"
        r"(estable|disfunci[oó]n|hospitalizado|internado|cirug[ií]a|recupera|"
        r"salud|diagn[oó]stico|fallece|muere|cancer|tumor|infarto|accidente cerebral)|"
        # Specific foreign ex-presidents by name + health context
        r"(bolsonaro|lula|milei|petro|maduro|boric|noboa|arce|lacalle).{0,60}"
        r"(estable|disfunci[oó]n|hospitalizado|internado|cirug[ií]a|recupera|salud|"
        r"diagn[oó]stico|fallece|muere|cancer|tumor|infarto)"
    )
    mask_fl_health = nt.str.contains(_foreign_leader_health, regex=True, na=False) & ~mask_crisis
    n_flh_eco = int((mask_fl_health & (df["economic_score"].fillna(0) > 0)).sum())
    n_flh_pol = int((mask_fl_health & (df["political_score"].fillna(0) > 0)).sum())
    df.loc[mask_fl_health, "economic_score"] = 0
    df.loc[mask_fl_health, "political_score"] = 0
    if n_flh_eco + n_flh_pol > 0:
        logger.info("post_filter_scores: foreign leader health zeroed eco=%d pol=%d articles", n_flh_eco, n_flh_pol)

    # Arts / cultural events with no political or economic relevance → BOTH scores 0
    _cultural = (
        r"fae lima|festival de artes esc[eé]nicas|festival artes esc[eé]nicas|"
        r"escena y memoria|cierre del fae|apertura del fae|"
        r"festival de (teatro|danza|jazz|cine documental|literatura) |"
        r"festival internacional de (teatro|danza|jazz|literatura)|"
        r"(temporada|estreno).{0,20}(ballet|[oó]pera).{0,30}(gran teatro|centro cultural)|"
        r"gran teatro nacional.{0,30}(programa|temporada|estreno)|"
        r"(inauguraci[oó]n|apertura|clausura).{0,30}exposici[oó]n.{0,30}(arte|pintura|escultura)|"
        r"bienal de arte|galería de arte.{0,20}inaugur"
    )
    mask_cultural = nt.str.contains(_cultural, regex=True, na=False)
    n_cult_eco = int((mask_cultural & (df["economic_score"].fillna(0) > 0)).sum())
    n_cult_pol = int((mask_cultural & (df["political_score"].fillna(0) > 0)).sum())
    df.loc[mask_cultural, "economic_score"] = 0
    df.loc[mask_cultural, "political_score"] = 0
    if n_cult_eco + n_cult_pol > 0:
        logger.info("post_filter_scores: cultural/arts zeroed eco=%d pol=%d articles", n_cult_eco, n_cult_pol)

    # Deaths of foreign intellectuals / cultural figures → BOTH scores 0
    # (crisis exception: if title also contains economic/political crisis language, keep)
    _intell_death = (
        r"(muere|fallece|muerte de).{0,120}"
        r"(fil[oó]sofos?|fil[oó]sofas?|"
        r"escritores?|escritoras?|"
        r"novelistas?|"
        r"poetas?|poetisas?|"
        r"dram[aá]turgos?|dram[aá]turgas?|"
        r"compositores?|compositoras?|"
        r"m[uú]sicos?|m[uú]sicas?|"
        r"pensadores?|pensadoras?|"
        r"te[oó]logos?|te[oó]logas?|"
        r"soci[oó]logos?|soci[oó]logas?|"
        r"antrop[oó]logos?|antrop[oó]logas?|"
        r"arque[oó]logos?|arque[oó]logas?|"
        r"historiadores?|historiadoras?|"
        r"ensayistas?|"
        r"fil[oó]logos?|fil[oó]logas?|"
        r"lingü[ií]stas?|"
        r"psic[oó]logos?|psic[oó]logas?|"
        r"pintores?|pintoras?|"
        r"escultores?|escultoras?|"
        r"artistas? pl[aá]sticos?|"
        r"actores?|actrices?|"
        r"directores? de cine|directoras? de cine|"
        r"bailarines?|bailarinas?|"
        r"coreógrafos?|coreógrafas?|"
        r"cantantes?|"
        r"guionistas?|"
        r"cineastas?)"
    )
    mask_intell = nt.str.contains(_intell_death, regex=True, na=False) & ~mask_crisis
    n_intell_eco = int((mask_intell & (df["economic_score"].fillna(0) > 0)).sum())
    n_intell_pol = int((mask_intell & (df["political_score"].fillna(0) > 0)).sum())
    df.loc[mask_intell, "economic_score"] = 0
    df.loc[mask_intell, "political_score"] = 0
    if n_intell_eco + n_intell_pol > 0:
        logger.info("post_filter_scores: intellectual/cultural deaths zeroed eco=%d pol=%d articles", n_intell_eco, n_intell_pol)

    # Rule 24: Foreign weather / local news in non-Peru locations → BOTH scores 0
    _foreign_weather_news = (
        r"(alerta|tormenta|hurac[aá]n|nieve|lluvia|calor|fr[ií]o|nevada|inundaci[oó]n)"
        r".{0,40}(florida|texas|california|nueva york|miami|new york|chicago|"
        r"m[eé]xico d\.?f\.?|buenos aires|bogot[aá]|santiago de chile|caracas|madrid|barcelona)"
        r"|"
        r"(florida|texas|california|nueva york|miami|chicago).{0,40}"
        r"(alerta|tormenta|hurac[aá]n|sismo|terremoto|incendio forestal)"
    )
    mask_foreign_weather = nt.str.contains(_foreign_weather_news, regex=True, na=False)
    n_fw_pol = int((mask_foreign_weather & (df["political_score"].fillna(0) > 0)).sum())
    n_fw_eco = int((mask_foreign_weather & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_foreign_weather, "political_score"] = 0
    df.loc[mask_foreign_weather, "economic_score"] = 0
    if n_fw_pol + n_fw_eco > 0:
        logger.info("post_filter_scores: foreign weather/news zeroed pol=%d eco=%d articles", n_fw_pol, n_fw_eco)

    # ── NEW FILTERS (systematic false-positive review, March 2026) ──────────────

    # Marketing / promotional content → eco=0 (crisis exception keeps price-shock alerts)
    _marketing = (
        r"hasta \d+% de (descuento|off\b)|oferta (especial|del d[ií]a|exclusiva)|"
        r"descuentos? de hasta|liquidaci[oó]n (de|total|por)|remate de (stock|inventario)|"
        r"campa[ñn]a.{0,20}publicitaria|lanzamiento.{0,20}(marca|producto|nuevo modelo)|"
        r"nuevo modelo.{0,20}(veh[ií]culo|auto|smartphone|celular|tablet)|"
        r"(promoci[oó]n|promo)\b.{0,20}(banco|financiera|tarjeta cr[eé]dito)|"
        r"tarjeta.{0,20}(puntos|millas|cashback|beneficios exclusivos)|"
        r"sin inter[eé]s.{0,20}cuotas|cuotas? sin inter[eé]s|"
        r"gratis (por|durante|al).{0,20}(mes|meses|a[ñn]o)|"
        r"abre (sus puertas|nuevo local|nueva tienda).{0,40}(lima|per[uú]|mall|centro comercial)"
    )
    mask_marketing = nt.str.contains(_marketing, regex=True, na=False) & ~mask_crisis
    n_mkt_eco = int((mask_marketing & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_marketing, "economic_score"] = 0
    if n_mkt_eco > 0:
        logger.info("post_filter_scores: marketing/promo zeroed eco=%d articles", n_mkt_eco)

    # Elections logistics / administrative procedures → eco=0 only
    # (ONPE/RENIEC/JNE logistics have political signal but zero economic-instability signal)
    _elections_admin = (
        r"(renovar?|actualizar?|tramitar?|sacar|obtener)\b.{0,30}(dni\b|documento.{0,10}identidad)|"
        r"dni (electr[oó]nico|biom[eé]trico|digital|f[ií]sico).{0,40}(tramit|expide|obtener|renovar)|"
        r"padr[oó]n electoral.{0,40}(habilitad|registrad|actualiz|consult)|"
        r"(local|mesa|c[eé]dula)\b.{0,20}(de votaci[oó]n|electoral)|"
        r"kit electoral|material electoral|lonjas? electorales?|"
        r"onpe\b.{0,50}(distribuye|entrega|instala|habilita|capacita|cronograma|sorteo)|"
        r"jne\b.{0,50}(inscripci[oó]n|registra|padr[oó]n|cronograma|admite|rechaza).{0,30}(candidat|partido)|"
        r"reniec\b.{0,50}(expide|emite|tramita|actualiza|moderniza|atiende)|"
        r"sorteo.{0,20}(miembros? de mesa|mesas? de sufragio)"
    )
    mask_elections_admin = nt.str.contains(_elections_admin, regex=True, na=False) & ~mask_crisis
    n_elec_eco = int((mask_elections_admin & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_elections_admin, "economic_score"] = 0
    if n_elec_eco > 0:
        logger.info("post_filter_scores: elections admin zeroed eco=%d articles", n_elec_eco)

    # Individual petty crime / police blotter → BOTH scores 0
    # ONLY zeroes individual incidents (mugging, petty theft, single arrest).
    # Does NOT touch: homicide rates, crime statistics, organized crime, extortion,
    # insecurity reports — those ARE economic signals (1.25×+ normal IRE).
    _crime_blotter = (
        r"asaltan? a.{0,30}(pasajero|combi\b|bus\b|taxi\b|ciclista|pe[aá]ton|transeúnte|"
        r"repartidor|delivery|mototaxi)\b|"
        r"roban? a.{0,30}(pasajero|turista|ciclista|pe[aá]ton|adulto mayor|ancian)\b|"
        r"(fue detenido|fue capturado|cayó) por (robo|hurto|asalto) (a|en|de)\b"
    )
    mask_crime_blot = nt.str.contains(_crime_blotter, regex=True, na=False) & ~mask_crisis
    n_cb_eco = int((mask_crime_blot & (df["economic_score"].fillna(0) > 0)).sum())
    n_cb_pol = int((mask_crime_blot & (df["political_score"].fillna(0) > 0)).sum())
    df.loc[mask_crime_blot, "economic_score"] = 0
    df.loc[mask_crime_blot, "political_score"] = 0
    if n_cb_eco + n_cb_pol > 0:
        logger.info("post_filter_scores: petty crime blotter zeroed eco=%d pol=%d articles", n_cb_eco, n_cb_pol)

    # Political inversion correction:
    # Classifier systematically assigns eco=60-70 pol=0 to political-legal articles.
    # These should be eco=0 AND pol≥60 (above-normal political risk).
    _pol_inversion = (
        # Prosecutor / judicial complaints by or against political figures
        r"(presenta|interpone|formulan?)\b.{0,20}denuncia (penal|constitucional)\b|"
        r"denuncia (penal|constitucional|formal).{0,10}contra.{0,60}"
        r"(fiscal|juez|ministro|congresist|premier|presidente|gobernador|funcionario)|"
        # Judicial/disciplinary sanctions on officials
        r"(fiscal[ií]a|poder judicial|jnj|tribunal constitucional).{0,50}"
        r"(suspendi[oó]|destituy[oó]|sancion[oó]|inhabilit[oó]).{0,40}"
        r"(fiscal|juez|magistrado|funcionario|ministro)|"
        r"(suspendido|destituido|inhabilitado|separado del cargo).{0,30}"
        r"(fiscal[ií]a|poder judicial|jnj|congreso|contralor[ií]a)\b|"
        # Named political figures in confrontation/accusation context
        r"(otárola|boluarte|vizcarra|humala|fujimori|keiko|pedro castillo|villanueva|"
        r"nicanor|domingo p[eé]rez|p[eé]rez crusat).{0,80}"
        r"(denuncia|acusa|demanda|querella|amparo|recusa|cuestiona|enfrenta|"
        r"no le tiene miedo|llegó al penal|fue trasladado al penal|cumple condena|"
        r"fue sentenciado|condenado a.{0,20}años)|"
        # Ex-president / senior official prison sentences → high political signal
        r"(expresidente|ex.presidente|ollanta humala|keiko fujimori|pedro castillo|"
        r"alberto fujimori|alejandro toledo|nadine heredia).{0,60}"
        r"(se ir[aá] preso|llegó al penal|trasladado al penal|ingresa al penal|"
        r"cumple condena|sentenciado a|condenado a|recibió.{0,10}años|"
        r"15 años|20 años|25 años).{0,30}"
    )
    mask_pol_inv = nt.str.contains(_pol_inversion, regex=True, na=False)
    # Zero eco (these are political, not economic)
    n_pi_eco = int((mask_pol_inv & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_pol_inv, "economic_score"] = 0
    # Boost pol to minimum 65 where it was under-scored (inverted by classifier)
    under_pol = mask_pol_inv & (df["political_score"].fillna(0) < 65)
    n_pi_pol = int(under_pol.sum())
    df.loc[under_pol, "political_score"] = 65
    if n_pi_eco + n_pi_pol > 0:
        logger.info(
            "post_filter_scores: political inversion corrected — eco zeroed=%d, pol boosted to 65 on %d articles",
            n_pi_eco, n_pi_pol,
        )

    # Foreign domestic politics unrelated to Peru → BOTH scores 0
    # (foreign elections, internal legislative debates — NOT tariffs/trade, which keep crisis exception)
    _foreign_domestic = (
        # Foreign election coverage
        r"elecciones (en |de )(colombia|argentina|brasil|m[eé]xico|chile|venezuela|"
        r"bolivia|ecuador|estados unidos|eeuu|reino unido|espa[ñn]a|francia|alemania|"
        r"canada|australia|india)\b|"
        # US Congress / Casa Blanca domestic items
        r"senado (de|en) (estados unidos|eeuu).{0,60}(aprueba|rechaza|debate|vota)\b|"
        r"congreso (de|en) (estados unidos|eeuu).{0,60}(aprueba|rechaza|debate|vota)\b|"
        # EU / UK parliament domestic legislation
        r"parlamento (europeo|brit[aá]nico|franc[eé]s|alem[aá]n|espa[ñn]ol).{0,60}"
        r"(aprueba|rechaza|debate|vota).{0,60}(migraci[oó]n|pensiones|sanidad|aborto|armas)|"
        # Specific foreign leaders' routine domestic announcements
        r"(bolsonaro|lula|milei|petro|boric|maduro|noboa|arce|lacalle|trump|biden|harris|"
        r"macron|scholz|starmer|meloni|modi|xi jinping|putin).{0,80}"
        r"(inaugur|posesion|posesi[oó]n|gana las elecciones|pierde las elecciones|"
        r"renuncia al cargo|es reelegido|asume el cargo)"
    )
    mask_foreign_dom = nt.str.contains(_foreign_domestic, regex=True, na=False) & ~mask_crisis
    n_fd_eco = int((mask_foreign_dom & (df["economic_score"].fillna(0) > 0)).sum())
    n_fd_pol = int((mask_foreign_dom & (df["political_score"].fillna(0) > 0)).sum())
    df.loc[mask_foreign_dom, "economic_score"] = 0
    df.loc[mask_foreign_dom, "political_score"] = 0
    if n_fd_eco + n_fd_pol > 0:
        logger.info("post_filter_scores: foreign domestic politics zeroed eco=%d pol=%d articles", n_fd_eco, n_fd_pol)

    # Entertainment / awards / pop-culture beyond _celeb → BOTH scores 0
    _entertainment_extra = (
        r"(concierto|gira|show|tour)\b.{0,30}(en lima|en per[uú]|llega a per[uú]|se presenta en lima)|"
        r"(nueva canci[oó]n|nuevo [aá]lbum|nuevo single|videoclip)\b.{0,30}de\b.{0,40}"
        r"(cantante|artista|banda|grupo musical)|"
        r"(cantante|artista|banda)\b.{0,30}(lanza|presenta|estrena)\b.{0,30}(canci[oó]n|disco|[aá]lbum)|"
        r"taquilla\b.{0,30}(r[eé]cord|supera los|bate los)|"
        r"(oscar|emmy|grammy|bafta|goya)\b.{0,30}(ganadores?|nominados?|ceremonia|gala)\b|"
        r"festival\b.{0,30}(de cannes|de venecia|de berlinal[eé]|sundance)\b|"
        r"pel[ií]cula.{0,30}(recauda|supera).{0,20}(mill[oó]n|millones).{0,20}(d[oó]lares|taquilla)"
    )
    mask_ent = nt.str.contains(_entertainment_extra, regex=True, na=False) & ~mask_crisis
    n_ent_eco = int((mask_ent & (df["economic_score"].fillna(0) > 0)).sum())
    n_ent_pol = int((mask_ent & (df["political_score"].fillna(0) > 0)).sum())
    df.loc[mask_ent, "economic_score"] = 0
    df.loc[mask_ent, "political_score"] = 0
    if n_ent_eco + n_ent_pol > 0:
        logger.info("post_filter_scores: entertainment/awards zeroed eco=%d pol=%d articles", n_ent_eco, n_ent_pol)

    # Peruvian weather / local disaster without economic disruption signal → eco=0
    # Crisis exception keeps flood/drought production losses, road blockages, etc.
    _peru_weather_routine = (
        r"(lluvia|llovizna|granizo|neblina|fr[ií]o intenso|ola de calor|"
        r"sismo.{0,10}(magnitud [1-3]\.|no se reportan da[ñn]os|sin v[ií]ctimas|leve)|"
        r"temblor.{0,10}(magnitud [1-3]\.|no se reportan da[ñn]os|sin v[ií]ctimas|leve))"
        r".{0,50}(igp|senamhi|pronostic|previsti|espera(n|ndo)|advierte|alerta amarilla)"
    )
    mask_peru_weather = nt.str.contains(_peru_weather_routine, regex=True, na=False) & ~mask_crisis
    n_pw_eco = int((mask_peru_weather & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_peru_weather, "economic_score"] = 0
    if n_pw_eco > 0:
        logger.info("post_filter_scores: routine weather zeroed eco=%d articles", n_pw_eco)

    # ── SCORE CORRECTIONS (boost where classifier under-scored) ─────────────────

    # Elections voter guide → eco=0, pol=max(pol,60)
    # Classifier gives eco=85 pol=0 to ballot/voting guides (should be purely political)
    _elections_guide = (
        r"c[oó]mo (es|ser[aá]|votar|marcar|llenar|evitar errores).{0,30}"
        r"(c[eé]dula|voto|boleta|sufragio)\b|"
        r"c[eé]dula (de votaci[oó]n|electoral).{0,40}(c[oó]mo|evitar|errores|marcar|votar)|"
        r"gu[ií]a (para votar|del elector|electoral).{0,40}(2025|2026)|"
        r"elecciones.{0,30}(c[oó]mo votar|paso a paso|gu[ií]a|evitar errores)"
    )
    mask_eg = nt.str.contains(_elections_guide, regex=True, na=False)
    df.loc[mask_eg, "economic_score"] = 0
    under_pol_eg = mask_eg & (df["political_score"].fillna(0) < 60)
    df.loc[under_pol_eg, "political_score"] = 60
    n_eg = int(mask_eg.sum())
    if n_eg > 0:
        logger.info("post_filter_scores: elections guide corrected eco=0 pol>=60 on %d articles", n_eg)

    # BCRP routine communications → eco=0 (new banknote security, coin designs, reserve stats)
    _bcrp_routine = (
        r"(bcrp|banco central).{0,60}(nuevo.{0,10}billete|hilo de seguridad|"
        r"dise[ñn]o.{0,10}(billete|moneda)|emisi[oó]n de (billete|moneda)|"
        r"nueva.{0,10}(serie|emisi[oó]n))|"
        r"(nuevo hilo de seguridad|nuevo dise[ñn]o|nueva serie).{0,60}(billete|moneda).{0,60}"
        r"(bcrp|banco central|precisiones)"
    )
    mask_bcrp_r = nt.str.contains(_bcrp_routine, regex=True, na=False) & ~mask_crisis
    n_bcrp = int((mask_bcrp_r & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_bcrp_r, "economic_score"] = 0
    if n_bcrp > 0:
        logger.info("post_filter_scores: BCRP routine zeroed eco=%d articles", n_bcrp)

    # Commercial chain openings / restaurant expansions → eco=0
    # (business growth news ≠ economic instability signal)
    _commercial_opening = (
        r"(prepara|planea|anuncia).{0,30}(nueva(s)? apertura|nuevo local|nueva tienda|"
        r"nueva(s)? tienda|expansi[oó]n.{0,20}(lima|per[uú]))|"
        r"(hamburguesa|pizzer[ií]a|fast food|cadena de.{0,20}restaurante|"
        r"cadena.{0,20}(gym|gimnasio|fitness)).{0,60}"
        r"(apertura|abre|inaugura|llega a (lima|per[uú]))"
    )
    mask_comm = nt.str.contains(_commercial_opening, regex=True, na=False) & ~mask_crisis
    n_comm = int((mask_comm & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_comm, "economic_score"] = 0
    if n_comm > 0:
        logger.info("post_filter_scores: commercial opening zeroed eco=%d articles", n_comm)

    # ── POLITICAL CALIBRATION ───────────────────────────────────────────────────

    # Presidential vacancy → pol ≥ 90, eco ≥ 40  (just below a coup)
    _pres_vacancy = (
        r"(admite|aprueba|debate|presenta[nm]|se alistan?|interpone[nm]|ingresa[nm]?)\b"
        r".{0,30}moci[oó]n.{0,10}de vacancia\b|"
        r"moci[oó]n.{0,10}de vacancia.{0,40}(admitida|aprobada|debatida|contra|presentada|ingresada)\b|"
        r"(8|ocho) presidentes.{0,20}(10|diez) a[ñn]os\b|"
        r"crisis pol[ií]tica.{0,30}(8|ocho) presidentes\b"
    )
    mask_pv = nt.str.contains(_pres_vacancy, regex=True, na=False)
    df.loc[mask_pv & (df["political_score"].fillna(0) < 90), "political_score"] = 90
    df.loc[mask_pv & (df["economic_score"].fillna(0) < 40), "economic_score"] = 40
    n_pv = int(mask_pv.sum())
    if n_pv > 0:
        logger.info("post_filter_scores: presidential vacancy pol>=90 eco>=40 on %d articles", n_pv)

    # Presidential censura / removal → pol ≥ 85, eco ≥ 40
    _pres_removal = (
        r"(es|fue|ser[aá]) censurado.{0,40}(congreso|pleno)\b|"
        r"(congreso|pleno).{0,50}(censura|censuró|cens[uó]).{0,40}(presidente|premier)\b|"
        r"(nuevo|siguiente) presidente.{0,40}(elegir[aá]|ser[aá] elegido|reemplazar[aá]).{0,30}per[uú]\b|"
        r"gobierno de.{0,30}terminó.{0,30}(cens[uó]|congreso|pleno)\b|"
        r"(caída|fin del gobierno).{0,30}(cens[uó]|vacancia|congreso)\b"
    )
    mask_pr = nt.str.contains(_pres_removal, regex=True, na=False)
    df.loc[mask_pr & (df["political_score"].fillna(0) < 85), "political_score"] = 85
    df.loc[mask_pr & (df["economic_score"].fillna(0) < 40), "economic_score"] = 40
    n_pr = int(mask_pr.sum())
    if n_pr > 0:
        logger.info("post_filter_scores: presidential removal pol>=85 eco>=40 on %d articles", n_pr)

    # Premier removal / censura → pol ≥ 65
    _premier_removal = (
        r"(congreso|pleno).{0,40}(censura|censuró|cens[uó]).{0,40}(premier|presidente del consejo)\b|"
        r"(premier|presidente del consejo).{0,40}(censurado|removido|destituido)\b|"
        r"nuevo premier.{0,30}(nombrad|designad|elegid|asumirá)\b"
    )
    mask_pmr = nt.str.contains(_premier_removal, regex=True, na=False)
    df.loc[mask_pmr & (df["political_score"].fillna(0) < 65), "political_score"] = 65
    n_pmr = int(mask_pmr.sum())
    if n_pmr > 0:
        logger.info("post_filter_scores: premier removal pol>=65 on %d articles", n_pmr)

    # Minister removal / censura / interpelación → pol ≥ 55
    _minister_removal = (
        r"(congreso|pleno).{0,40}(censura|censuró|cens[uó]).{0,40}ministro\b|"
        r"ministro.{0,40}(censurado|removido|destituido).{0,30}(congreso|gobierno|presidencia)\b|"
        r"interpelaci[oó]n.{0,30}(ministro|premier).{0,30}(aprobada|debate|hoy|este\s+\w+es)\b"
    )
    mask_mr = nt.str.contains(_minister_removal, regex=True, na=False)
    df.loc[mask_mr & (df["political_score"].fillna(0) < 55), "political_score"] = 55
    n_mr = int(mask_mr.sum())
    if n_mr > 0:
        logger.info("post_filter_scores: minister removal pol>=55 on %d articles", n_mr)

    # ── ECONOMIC CALIBRATION ────────────────────────────────────────────────────

    # Camisea / gas crisis → eco ≥ 80  (just below El Niño / recession)
    _gas_crisis = (
        r"(camisea|gnv\b|gas natural vehicular|gas licuado).{0,60}"
        r"(corte|restricci[oó]n|escasez|emergencia|crisis|colapso|desabastecimiento|"
        r"incendio|deflagraci[oó]n|explosi[oó]n|ruptura|falla)\b|"
        r"(corte|restricci[oó]n|emergencia|crisis).{0,40}(camisea|gnv\b|gas natural vehicular|"
        r"gas licuado|ducto|gasoducto)\b"
    )
    mask_gc = nt.str.contains(_gas_crisis, regex=True, na=False)
    df.loc[mask_gc & (df["economic_score"].fillna(0) < 80), "economic_score"] = 80
    n_gc = int(mask_gc.sum())
    if n_gc > 0:
        logger.info("post_filter_scores: gas crisis eco>=80 on %d articles", n_gc)

    # Petroperú crisis / bankruptcy / bailout → eco ≥ 60
    _petroperu = (
        r"petroper[uú].{0,60}"
        r"(quiebra|concurso de acreedores|rescate|bail.?out|crisis|colapso|cierre|"
        r"pérdida|deuda|insolvencia|intervenida|liquidaci[oó]n|subsidio|aporte del estado)\b"
    )
    mask_pp = nt.str.contains(_petroperu, regex=True, na=False)
    df.loc[mask_pp & (df["economic_score"].fillna(0) < 60), "economic_score"] = 60
    n_pp = int(mask_pp.sum())
    if n_pp > 0:
        logger.info("post_filter_scores: Petroperú crisis eco>=60 on %d articles", n_pp)

    # Congress proposing fiscal/economic expansion policies → eco ≥ 60
    _fiscal_expansion = (
        r"(congreso|pleno).{0,60}"
        r"(aprueba|aprobó|debate|propone|impulsa).{0,40}"
        r"(aumento del sueldo m[ií]nimo|retiro de afp|retiro de cts|"
        r"bonificaci[oó]n|gratificaci[oó]n doble|deuda agrar[a-z]+|"
        r"condonaci[oó]n.{0,20}deuda|shock.{0,20}fiscal|gasto p[uú]blico)\b"
    )
    mask_fe = nt.str.contains(_fiscal_expansion, regex=True, na=False)
    df.loc[mask_fe & (df["economic_score"].fillna(0) < 60), "economic_score"] = 60
    n_fe = int(mask_fe.sum())
    if n_fe > 0:
        logger.info("post_filter_scores: fiscal expansion eco>=60 on %d articles", n_fe)

    # Criminality statistics (Sinadef, INEI, PNP monthly peaks) → eco ≥ 55
    _crime_stats = (
        r"sinadef.{0,40}(homicidio|feminicidio|muerte violenta|asesinato)\b|"
        r"(registr[oó]|reporta).{0,20}(homicidios|feminicidios|muertes violentas)"
        r".{0,20}(por d[ií]a|al mes|en.*mes|en.*semana)\b|"
        r"tasa de (homicidio|criminalidad|violencia).{0,30}(per[uú]|lima|nacional)\b|"
        r"[ií]ndice de (criminalidad|violencia|inseguridad).{0,30}(registr|aument|subi[oó]|alcanz)\b|"
        r"(extorsi[oó]n|sicariato).{0,30}(aument[oó]|creci[oó]|subi[oó]|alerta|r[eé]cord)\b"
    )
    mask_cs = nt.str.contains(_crime_stats, regex=True, na=False)
    df.loc[mask_cs & (df["economic_score"].fillna(0) < 55), "economic_score"] = 55
    n_cs = int(mask_cs.sum())
    if n_cs > 0:
        logger.info("post_filter_scores: crime statistics eco>=55 on %d articles", n_cs)

    # US Federal Reserve rates → eco=50, pol=0
    # (Fed decisions affect Peru via capital flows / exchange rate; NOT a Peru political event)
    _fed_rates = (
        r"(fed\b|reserva federal).{0,80}"
        r"(tasa|tipos? de inter[eé]s|pol[ií]tica monetaria).{0,60}"
        r"(mantiene|mantener|sube|subir|baja|bajar|recorta|recortar|eleva|elevar|"
        r"sin cambios|pausa|pausar|aumenta|aumentar|dispone a|se espera)\b|"
        r"(tasa|tipos? de inter[eé]s).{0,40}(fed\b|reserva federal)"
    )
    mask_fed = nt.str.contains(_fed_rates, regex=True, na=False)
    # Zero pol (Fed is NOT a Peru political event)
    n_fed_pol = int((mask_fed & (df["political_score"].fillna(0) > 0)).sum())
    df.loc[mask_fed, "political_score"] = 0
    # Boost eco to 50 minimum
    df.loc[mask_fed & (df["economic_score"].fillna(0) < 50), "economic_score"] = 50
    n_fed_eco = int(mask_fed.sum())
    if n_fed_pol + n_fed_eco > 0:
        logger.info(
            "post_filter_scores: Fed rates pol zeroed=%d eco>=50 on %d articles",
            n_fed_pol, n_fed_eco,
        )

    # Large energy / utility M&A in Peru (>$100M explicitly stated) → eco=65 if under-scored
    _energy_ma = (
        r"(compra|adquiere|adquisici[oó]n|autoriza.{0,20}comprar|recibe autorizaci[oó]n.{0,20}comprar)"
        r".{0,80}"
        r"(us\$\s*\d{3,}|\d{3,} millones de d[oó]lares|\d{3,} mill\.? us\$)"
        r".{0,60}"
        r"(filial|subsidiaria|empresa|generadora|distribuidora|concesi[oó]n|activo)"
        r".{0,40}"
        r"(electricidad|energ[ií]a el[eé]ctrica|gas natural|agua potable|telecomunicaciones|minera)\b"
    )
    mask_ema = nt.str.contains(_energy_ma, regex=True, na=False)
    df.loc[mask_ema & (df["economic_score"].fillna(0) < 65), "economic_score"] = 65
    n_ema = int(mask_ema.sum())
    if n_ema > 0:
        logger.info("post_filter_scores: energy M&A eco>=65 on %d articles", n_ema)

    # Security emergency declarations → eco=40 if under-scored (crime crisis affects investment/activity)
    _security_emergency = (
        r"(plan|estado).{0,20}emergencia.{0,30}(terrorismo|extorsi[oó]n|crimen organizado|inseguridad)\b|"
        r"(terrorismo urbano|crimen organizado).{0,40}(emergencia|declarar|activar|combatir)\b|"
        r"(extorsi[oó]n|sicariato).{0,30}(afecta|paraliza|cierra|impide).{0,30}"
        r"(negocio|comercio|empresa|mercado|actividad econ[oó]mica)\b"
    )
    mask_secem = nt.str.contains(_security_emergency, regex=True, na=False)
    df.loc[mask_secem & (df["economic_score"].fillna(0) < 40), "economic_score"] = 40
    n_secem = int(mask_secem.sum())
    if n_secem > 0:
        logger.info("post_filter_scores: security emergency eco>=40 on %d articles", n_secem)

    # ── POLITICAL CONTAMINATION FIXES (classifier leaks political into economic) ─

    # Presidential salary battle → eco=0, pol≥70 (political confrontation, not economic disruption)
    _pres_salary = (
        r"(sueldo|salario|remuneraci[oó]n).{0,30}(del|de la|de) (presidenta?|presidencial)\b|"
        r"(proyectos?|iniciativas?|ley).{0,30}(derogar|eliminar|reducir|congelar).{0,30}"
        r"(sueldo|salario|incremento).{0,20}(presidencial|presidenta?)\b|"
        r"incremento.{0,20}sueldo.{0,20}presidencial\b"
    )
    mask_ps = nt.str.contains(_pres_salary, regex=True, na=False)
    df.loc[mask_ps, "economic_score"] = 0
    df.loc[mask_ps & (df["political_score"].fillna(0) < 70), "political_score"] = 70
    if mask_ps.sum() > 0:
        logger.info("post_filter_scores: presidential salary corrected eco=0 pol>=70 on %d", int(mask_ps.sum()))

    # Ley de amnistía / impunidad → eco=0, pol≥65 (rule of law crisis, not economic)
    _amnesty = (
        r"ley de amnist[ií]a\b|"
        r"amnist[ií]a.{0,40}(aprobada|aprueba|congreso|vulnera|derechos|impunidad|militar)\b|"
        r"(amnist[ií]a|indulto).{0,30}(militares?|polic[ií]as?|funcionarios?).{0,30}"
        r"(congreso|aprueba|aprobó|debate)\b"
    )
    mask_am = nt.str.contains(_amnesty, regex=True, na=False)
    df.loc[mask_am, "economic_score"] = 0
    df.loc[mask_am & (df["political_score"].fillna(0) < 65), "political_score"] = 65
    if mask_am.sum() > 0:
        logger.info("post_filter_scores: amnesty law corrected eco=0 pol>=65 on %d", int(mask_am.sum()))

    # Congressional reform / MP reorganization proposals → eco=0, pol≥50
    _mp_reform = (
        r"(propone[nm]?|propuesta|proyecto de ley).{0,40}"
        r"(reorganizar?|reorganizaci[oó]n|disolver?|depurar?).{0,40}"
        r"(ministerio p[uú]blico|fiscal[ií]a|poder judicial|jnj|junta de fiscales)\b|"
        r"(per[uú] libre|fuerza popular|renovaci[oó]n popular|acci[oó]n popular).{0,60}"
        r"(reorganizar?|remover|destituir).{0,40}(fiscal|jnj|ministerio p[uú]blico)\b"
    )
    mask_mpr = nt.str.contains(_mp_reform, regex=True, na=False)
    df.loc[mask_mpr, "economic_score"] = 0
    df.loc[mask_mpr & (df["political_score"].fillna(0) < 50), "political_score"] = 50
    if mask_mpr.sum() > 0:
        logger.info("post_filter_scores: MP/judiciary reform corrected eco=0 pol>=50 on %d", int(mask_mpr.sum()))

    # EsSalud coverage expansion → eco=0 (social policy, not economic risk)
    _essalud_expand = (
        r"essalud.{0,60}(asegure[nm]?|cubra[nm]?|afilié[nm]?|incorpore[nm]?|extienda|amplíe).{0,40}"
        r"(padres?|familiar|dependientes?|adultos? mayores?|cónyuge|conviviente)\b"
    )
    mask_es = nt.str.contains(_essalud_expand, regex=True, na=False)
    n_es = int((mask_es & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_es, "economic_score"] = 0
    if n_es > 0:
        logger.info("post_filter_scores: EsSalud expansion zeroed eco=%d", n_es)

    # ── ECONOMIC UNDER-SCORING FIXES ─────────────────────────────────────────────

    # Trump / US tariffs directly threatening Peru's copper/mineral exports → eco≥70
    _trump_tariffs_peru = (
        r"(trump|eeuu|estados unidos|washington).{0,60}"
        r"(arancel|tarifa|impuesto.{0,10}importaci[oó]n).{0,60}"
        r"(cobre|minerales?|exportaciones? peruanas?|per[uú]|latinoam[eé]rica)\b|"
        r"arancel.{0,20}(50|60|70|80|100)\%.{0,40}cobre\b|"
        r"exportadores? peruanos?.{0,60}(perder[ií]an?|afectar[ií]an?|impacto).{0,40}"
        r"(arancel|tarifa|trump)\b"
    )
    mask_tt = nt.str.contains(_trump_tariffs_peru, regex=True, na=False)
    df.loc[mask_tt & (df["economic_score"].fillna(0) < 70), "economic_score"] = 70
    if mask_tt.sum() > 0:
        logger.info("post_filter_scores: Trump Peru tariffs eco>=70 on %d", int(mask_tt.sum()))

    # Road blockage cutting critical supplies (oxygen, medicine, food) → eco≥55
    _supply_blockage = (
        r"bloqueo.{0,50}(ox[ií]geno|medicamentos?|insumos? m[eé]dicos?|f[aá]rmacos?|alimentos?|"
        r"combustible|gas|gasolina).{0,30}(hospital|cl[ií]nica|poblaci[oó]n)\b|"
        r"(ox[ií]geno|medicamentos?|insumos? m[eé]dicos?).{0,30}(escasez|desabastecimiento|"
        r"no llega|cortado|bloqueado).{0,30}(hospital|cl[ií]nica|regi[oó]n)\b"
    )
    mask_sb = nt.str.contains(_supply_blockage, regex=True, na=False)
    df.loc[mask_sb & (df["economic_score"].fillna(0) < 55), "economic_score"] = 55
    if mask_sb.sum() > 0:
        logger.info("post_filter_scores: supply blockage eco>=55 on %d", int(mask_sb.sum()))

    # ── JULY 2025 FALSE-POSITIVE CLEANUP ─────────────────────────────────────────

    # A. Ministerio de Cultura — patrimony recovery, Machu Picchu tickets → eco=0
    _cultura_pat = (
        r"ministerio de cultura.{0,80}(recuper|pieza|virreinal|hist[oó]ric|patrimonio|arte|restitui|devolu)|"
        r"cultura.{0,40}(pieza virreinal|objeto hist|recuper[oó].{0,20}(pieza|objeto|bien))|"
        r"ministerio de cultura.{0,60}(boletos|entradas|machu picchu|venta de (boletos|entradas)|nuevas medidas)"
    )
    mask_cpa = nt.str.contains(_cultura_pat, regex=True, na=False) & ~mask_crisis
    n_cpa = int((mask_cpa & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_cpa, "economic_score"] = 0
    if n_cpa:
        logger.info("post_filter_scores: cultura patrimonio eco=0 on %d", n_cpa)

    # B. JNJ / Ministerio Público procedural politics (not fiscal disruption) → eco=0, pol≥45
    _jnj_mp_politics = (
        r"(jnj|junta nacional de justicia).{0,60}(firma|acta|minimizan?|procedimental|desacato|suspensión|inhibición)|"
        r"(del castillo|aldo v[aá]squez|dr[ou]e|integrante.{0,10}jnj).{0,60}(minimizan?|firma|acta)|"
        r"delia espinoza.{0,60}(denuncia|desacato|jnj|cargo)|"
        r"(regresa?|retorna?|vuelve|asume como?).{0,40}"
        r"(fiscal provincial|fiscal supremo|fiscal superior|fiscalía.{0,15}derechos humanos)"
    )
    mask_jnj = nt.str.contains(_jnj_mp_politics, regex=True, na=False) & ~mask_crisis
    n_jnj_eco = int((mask_jnj & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_jnj, "economic_score"] = 0
    df.loc[mask_jnj & (df["political_score"].fillna(0) < 45), "political_score"] = 45
    if n_jnj_eco:
        logger.info("post_filter_scores: JNJ/MP politics eco=0 on %d", n_jnj_eco)

    # C. Mesa Directiva del Congreso — elections, candidacies, audits → eco=0
    _mesa_directiva = (
        r"mesa directiva.{0,70}(auditar|negociaci[oó]n|candidatos?|lista|conformar|integrar|aspira|nueva mesa|gestión)|"
        r"(candidatos?|primera lista|segunda lista|lista.{0,15}candidatos?).{0,30}(mesa directiva|congreso)|"
        r"(c[oó]mo van|dudas.{0,20}opciones).{0,30}(negociaci[oó]n|nueva mesa directiva)"
    )
    mask_md = nt.str.contains(_mesa_directiva, regex=True, na=False) & ~mask_crisis
    n_md = int((mask_md & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_md, "economic_score"] = 0
    if n_md:
        logger.info("post_filter_scores: mesa directiva eco=0 on %d", n_md)

    # D. Congressional institutional distrust framing → eco=0, pol≥60
    _inst_distrust = (
        r"desconfianza institucional.{0,60}(ministerio p[uú]blico|estado|crisis|fiscal[ií]a)|"
        r"crisis del ministerio p[uú]blico.{0,60}(estado|inversi[oó]n|econom[ií]a|afecta)"
    )
    mask_id = nt.str.contains(_inst_distrust, regex=True, na=False)
    n_id = int((mask_id & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_id, "economic_score"] = 0
    df.loc[mask_id & (df["political_score"].fillna(0) < 60), "political_score"] = 60
    if n_id:
        logger.info("post_filter_scores: inst distrust eco=0, pol>=60 on %d", n_id)

    # E. Somos Perú / congressional-leadership audit of Mesa Directiva → eco=0
    _congreso_audit = (
        r"somos per[uú].{0,60}(auditar|gesti[oó]n.{0,20}salhuana|mesa directiva)|"
        r"(salhuana|presidente del congreso).{0,60}(viaje|amiga|par[ií]s|financiamiento|justifica)"
    )
    mask_ca = nt.str.contains(_congreso_audit, regex=True, na=False) & ~mask_crisis
    n_ca = int((mask_ca & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_ca, "economic_score"] = 0
    if n_ca:
        logger.info("post_filter_scores: congreso audit/scandal eco=0 on %d", n_ca)

    # F. Presidential political statements (not policy) → eco=0, pol≥40
    _pres_pol_speech = (
        r"(boluarte|presidenta?).{0,60}(menosprecia|llama.{0,15}(miopes|sordos|cieg|ignorant)|"
        r"rompe su silencio|no le interesa|calla|silencio con la prensa)|"
        r"(boluarte|presidenta?).{0,40}(analicen bien.{0,20}voto|entreguen su voto|el voto es sagrado)|"
        r"(boluarte|presidenta?).{0,30}llama.{0,15}(miopes|sordos|ignorantes).{0,30}opositores"
    )
    mask_pps = nt.str.contains(_pres_pol_speech, regex=True, na=False) & ~mask_crisis
    n_pps = int((mask_pps & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_pps, "economic_score"] = 0
    df.loc[mask_pps & (df["political_score"].fillna(0) < 40), "political_score"] = 40
    if n_pps:
        logger.info("post_filter_scores: pres political speech eco=0 on %d", n_pps)

    # G. ONPE/electoral administrative payments (not disruption) → eco=0
    _onpe_admin = (
        r"onpe.{0,60}(pago|compensaci[oó]n|pagar|abona).{0,30}(miembro.{0,10}mesa|personal|colaboradores)|"
        r"(plazo|vence.{0,10}plazo|inscribir|inscripci[oó]n).{0,30}"
        r"(alianza.{0,10}electoral|organizaci[oó]n.{0,10}pol[ií]tic|partido.{0,10}pol[ií]tic).{0,50}"
        r"(elecciones|comicios|20\d{2})|"
        r"(elecciones\s+20\d{2}).{0,60}(plazo.{0,20}inscri|inscri.{0,20}alianza|alianza.{0,20}electoral)"
    )
    mask_oa = nt.str.contains(_onpe_admin, regex=True, na=False) & ~mask_crisis
    n_oa = int((mask_oa & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_oa, "economic_score"] = 0
    if n_oa:
        logger.info("post_filter_scores: ONPE/electoral admin eco=0 on %d", n_oa)

    # H. Congressional workers/employees union demands (not national labor) → eco=0
    _congreso_sindicato = (
        r"sindicato.{0,40}(trabajadores.{0,20}(congreso|parlamento|poder legislativo)|"
        r"congreso.{0,20}(trabajadores|empleados|personal))|"
        r"(congreso|poder legislativo).{0,60}sindicato.{0,40}(bono|exige|demanda|aumento|sueldo)"
    )
    mask_cs2 = nt.str.contains(_congreso_sindicato, regex=True, na=False) & ~mask_crisis
    n_cs2 = int((mask_cs2 & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_cs2, "economic_score"] = 0
    if n_cs2:
        logger.info("post_filter_scores: congreso sindicato eco=0 on %d", n_cs2)

    # I. Routine institutional appointments (migraciones, minor agencies) → eco=0
    _routine_appt = (
        r"(migraciones|reniec|sernanp|osinfor|indeci|conam|mincetur|minjusdh).{0,60}"
        r"(asumir[aá]|asume|asumirá temporalmente|jefatura temporal|encargado de la jefatura)|"
        r"(asumir[aá]|asume).{0,20}temporalmente.{0,20}(jefatura|gerencia|cargo).{0,40}"
        r"(de la entidad|de la instituci[oó]n|del organismo)"
    )
    mask_ra = nt.str.contains(_routine_appt, regex=True, na=False) & ~mask_crisis
    n_ra = int((mask_ra & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_ra, "economic_score"] = 0
    if n_ra:
        logger.info("post_filter_scores: routine appointment eco=0 on %d", n_ra)

    # J. Voluntariado law / school contests / educational awards → eco=0
    _voluntariado_edu = (
        r"ley.{0,20}(del|de) voluntariado|voluntariado.{0,30}(cr[eé]dito|reconocer|ley)|"
        r"premio nacional.{0,30}(escolar|estudiantes?|escolares?)|"
        r"(amplían?|extiende).{0,30}plazo.{0,30}(trabajos?|inscripci[oó]n).{0,30}(estudiantes?|escolares?|alumnos?)"
    )
    mask_ve = nt.str.contains(_voluntariado_edu, regex=True, na=False) & ~mask_crisis
    n_ve = int((mask_ve & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_ve, "economic_score"] = 0
    if n_ve:
        logger.info("post_filter_scores: voluntariado/edu eco=0 on %d", n_ve)

    # K. Corruption/scandal news for non-economic subjects → eco=0, pol boost
    _corruption_scandal = (
        r"caso qali warma.{0,60}(prisi[oó]n|imputado|investigado|detenido|formali|sentencia)|"
        r"(caso cofre|caso waykis|caso c[oó]cteles).{0,60}(aceptar?|desistimiento|cerrar?|fiscal[ií]a)|"
        r"(salhuana|congresista|funcionario).{0,60}(viaje.{0,20}(amiga|amigo|familiar)|"
        r"financiamiento.{0,20}viaje|justifica.{0,20}viaje)"
    )
    mask_cor = nt.str.contains(_corruption_scandal, regex=True, na=False) & ~mask_crisis
    n_cor = int((mask_cor & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_cor, "economic_score"] = 0
    df.loc[mask_cor & (df["political_score"].fillna(0) < 35), "political_score"] = 35
    if n_cor:
        logger.info("post_filter_scores: corruption/scandal eco=0 on %d", n_cor)

    # L. Local/municipal disputes with no systemic economic impact → eco=0
    _local_dispute = (
        r"(municipalidad|alcalde|alcaldesa|regidora?).{0,60}"
        r"(arena de lima|parque.{0,30}(surco|barranco|miraflores|san isidro)|"
        r"viabilidad.{0,15}proyecto.{0,20}(arena|estadio|coliseo))|"
        r"(niega.{0,20}haber|acuña.{0,20}parque|parque.{0,20}acuña)"
    )
    mask_ld = nt.str.contains(_local_dispute, regex=True, na=False) & ~mask_crisis
    n_ld = int((mask_ld & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_ld, "economic_score"] = 0
    if n_ld:
        logger.info("post_filter_scores: local dispute eco=0 on %d", n_ld)

    # M. MTC regulatory announcements (positive/neutral, not disruption) → cap eco at 20
    _mtc_regulatory = (
        r"mtc.{0,60}(modifica|modif.{0,20}reglamento|nuevo reglamento).{0,60}"
        r"(portuario|aeropuerto|telecomunicaciones|ferroviario|carreteras?)(?!.{0,40}(crisis|colapso|huelga|paraliza))|"
        r"reglamento portuario.{0,60}(modernizaci[oó]n|inversiones?|impulsar)"
    )
    mask_mtc = nt.str.contains(_mtc_regulatory, regex=True, na=False) & ~mask_crisis
    df.loc[mask_mtc & (df["economic_score"].fillna(0) > 20), "economic_score"] = 20
    if mask_mtc.sum():
        logger.info("post_filter_scores: MTC regulatory capped eco<=20 on %d", int(mask_mtc.sum()))

    # N. Fiestas Patrias home security/travel logistics → eco=0
    _fiestas_patrias_tips = (
        r"fiestas patrias.{0,60}(c[oó]mo proteger|tu casa|viajas|seguridad en casa|tips|consejos)|"
        r"feriado.{0,20}(c[oó]mo proteger|tu casa|seguridad en casa)|"
        r"(proteger|cuidar).{0,20}(tu|su) casa.{0,20}(feriado|fiestas patrias|viajes)"
    )
    mask_fpt = nt.str.contains(_fiestas_patrias_tips, regex=True, na=False)
    n_fpt = int((mask_fpt & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_fpt, "economic_score"] = 0
    if n_fpt:
        logger.info("post_filter_scores: fiestas patrias tips eco=0 on %d", n_fpt)

    # O. Congressional session entertainment/audio leaks → eco=0
    _congreso_gossip = (
        r"audio.{0,30}(partido de f[uú]tbol|f[uú]tbol|partido).{0,30}(filtra|comisi[oó]n|votaci[oó]n|congreso)|"
        r"(filtra|filt[ró]).{0,30}audio.{0,30}(partido|f[uú]tbol).{0,30}(congreso|comisi[oó]n)"
    )
    mask_cg = nt.str.contains(_congreso_gossip, regex=True, na=False) & ~mask_crisis
    n_cg = int((mask_cg & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_cg, "economic_score"] = 0
    if n_cg:
        logger.info("post_filter_scores: congreso gossip eco=0 on %d", n_cg)

    # P. Congressional legislative proposals by individual legislators (opinion, no vote) → eco=0
    _legislator_opinion = (
        r"congresista.{0,40}(considera|opina|propone|plantea|pide|exige|cree|sugiere).{0,40}"
        r"(miner[ií]a|ambiental|control|regulaci[oó]n|ley|reforma)(?!.{0,30}(aprobó|aprueba|promulg|entr[oó] en vigor))"
    )
    mask_lo = nt.str.contains(_legislator_opinion, regex=True, na=False) & ~mask_crisis
    n_lo = int((mask_lo & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_lo, "economic_score"] = 0
    if n_lo:
        logger.info("post_filter_scores: legislator opinion eco=0 on %d", n_lo)

    # ── CROSS-MONTH FALSE-POSITIVE CLEANUP ───────────────────────────────────────

    # Q. Commercial chain arrivals / gym / fitness openings in Peru → eco=0
    _commercial_arrival = (
        r"(smart fit|bio ritmo|biorit|gold.s gym|planet fitness|decathlon|primark|zara|h&m|ikea|"
        r"forever 21|pull.{0,3}bear|bershka|subway|domino|pizza hut|kfc|starbucks|"
        r"mcdonald|burger king|dunkin|popeyes).{0,60}(trae|llega|inaugura|abre|estrena).{0,30}per[uú]|"
        r"(trae a|llega a per[uú]|llega al mercado peruano|abre en).{0,40}"
        r"(centro comercial|el polo|miraflores|san isidro|jockey|la molina|surco)|"
        r"centro comercial.{0,60}(inaugura|abre|estrena|nuevo local|nueva tienda)"
    )
    mask_coa = nt.str.contains(_commercial_arrival, regex=True, na=False) & ~mask_crisis
    n_coa = int((mask_coa & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_coa, "economic_score"] = 0
    if n_coa:
        logger.info("post_filter_scores: commercial arrival eco=0 on %d", n_coa)

    # R. Consumer/lifestyle licensing & travel admin → eco=0
    _consumer_admin = (
        r"(licencia de conducir|brevete|brevete digital).{0,60}"
        r"(c[oó]mo (obtener|sacar|renovar)|requisitos|pasos|primera vez|tramitar)|"
        r"visas? para viajar.{0,40}(pa[ií]ses|exoneran|no necesitan|sin visa)|"
        r"(pa[ií]ses|destinos?).{0,40}(exoneran|no requieren|sin visa).{0,40}peruanos?|"
        r"multa.{0,30}(viajar dos|moto.{0,10}dos|dos en moto|tres en moto)|"
        r"(vacaciones|feriado).{0,20}(semana santa|fiestas patrias|a[ñn]o nuevo|navidad).{0,40}"
        r"(fechas|d[ií]as libres|horarios|calendario)"
    )
    mask_cad = nt.str.contains(_consumer_admin, regex=True, na=False) & ~mask_crisis
    n_cad = int((mask_cad & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_cad, "economic_score"] = 0
    if n_cad:
        logger.info("post_filter_scores: consumer/lifestyle admin eco=0 on %d", n_cad)

    # S. Security/crime opinion columns (advice, "cuídese", named columnists) → eco=0
    _crime_opinion = (
        r"cu[ií]date de (la|el) (delincuencia|crimen|robo|asalto)|"
        r"crimen sin castigo.{0,20}(por|,).{0,30}(anderson|mendoza|salazar|pease|rospigliosi|ledesma)|"
        r"c[oó]mo (protegerse|defenderse|evitar).{0,30}(delincuencia|robo|asalto|crimen)|"
        r"consejos para (no ser|evitar ser) v[ií]ctima"
    )
    mask_creo = nt.str.contains(_crime_opinion, regex=True, na=False) & ~mask_crisis
    n_creo = int((mask_creo & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_creo, "economic_score"] = 0
    if n_creo:
        logger.info("post_filter_scores: crime opinion eco=0 on %d", n_creo)

    # T. Diplomatic / ceremonial presidential meetings (no trade deal signed) → eco=0
    _diplomatic_ceremony = (
        r"(boluarte|presidenta?).{0,60}"
        r"(se re[uú]ne con|se encontr[oó] con|invita a|viaje.{0,10}a|visit[oó] a).{0,60}"
        r"(pr[ií]ncipe|rey|emir|sultan|sheij|jeque|l[ií]der.{0,10}(de|del)|"
        r"trump|macron|ursula|scholz|biden|obama|von der leyen).{0,30}"
        r"(?!.{0,40}(acuerdo|tratado|convenio|firma|protocolo|inversi[oó]n|millones|comercio))|"
        r"onu.{0,50}(boluarte|presidenta?).{0,50}(se re[uú]ne|visita|discurso|agenda)"
    )
    mask_dc = nt.str.contains(_diplomatic_ceremony, regex=True, na=False) & ~mask_crisis
    n_dc = int((mask_dc & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_dc, "economic_score"] = 0
    if n_dc:
        logger.info("post_filter_scores: diplomatic ceremony eco=0 on %d", n_dc)

    # U. Senamhi routine temperature/weather (sierra/coast) → eco=0
    _senamhi_routine_temp = (
        r"senamhi.{0,60}(seguir[aá]n|continuar[aá]n|soportar[aá]n|prevé|pronostica).{0,40}"
        r"(altas? temperaturas?|bajas? temperaturas?|lloviznas?|neblina|nublado)|"
        r"(sierra|costa|lima).{0,40}(seguir[aá]n|continuar[aá]n|soportar[aá]n).{0,30}"
        r"(altas? temperaturas?|bajas? temperaturas?|fr[ií]o|calor)"
    )
    mask_srt = nt.str.contains(_senamhi_routine_temp, regex=True, na=False) & ~mask_crisis
    n_srt = int((mask_srt & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_srt, "economic_score"] = 0
    if n_srt:
        logger.info("post_filter_scores: senamhi routine temp eco=0 on %d", n_srt)

    # V. Ibex / European / Asian stock market news → eco=0
    _foreign_bourse = (
        r"(ibex.{0,5}35|bolsa espa[ñn]ola|bolsa (de|en) madrid|bourse de paris|dax\b|"
        r"ftse\b|cac.{0,3}40|hang seng|nikkei|kospi|asx.{0,5}200|shanghai composite|"
        r"msci (emergente|europa|asia)|topix).{0,60}(cae|sube|cierra|cotiza|baja|abre)"
    )
    mask_fb = nt.str.contains(_foreign_bourse, regex=True, na=False) & ~mask_crisis
    n_fb = int((mask_fb & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_fb, "economic_score"] = 0
    if n_fb:
        logger.info("post_filter_scores: foreign bourse eco=0 on %d", n_fb)

    # W. ANGR / regional government routine admin elections → eco=0
    _regional_admin = (
        r"(angr|gobernadores regionales|congreso de gobernadores).{0,60}"
        r"(elegir[aá]|nueva junta|nuevo consejo|elecci[oó]n).{0,40}"
        r"(directivo|directiva|presidente|consejo)|"
        r"(nueva junta directiva|nuevo consejo directivo).{0,40}(angr|gobernadores)"
    )
    mask_ra2 = nt.str.contains(_regional_admin, regex=True, na=False) & ~mask_crisis
    n_ra2 = int((mask_ra2 & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_ra2, "economic_score"] = 0
    if n_ra2:
        logger.info("post_filter_scores: regional admin eco=0 on %d", n_ra2)

    # X. "Político candidato / candidatura" opinion pieces → eco=0
    _pol_candidate_opinion = (
        r"(vizcarra|alva|acuña|l[oó]pez aliaga|urresti|nicanor|keiko|fujimori|toledo|"
        r"villanueva|arce|forsyth|soto).{0,50}"
        r"(candidatura|candidato|postular|se presenta|confirmó que va|tiene intenciones)|"
        r"(sobre|sobre la|ante la) posible candidatura.{0,40}"
        r"(presidencial|al congreso|a la alcald[ií]a)"
    )
    mask_pco = nt.str.contains(_pol_candidate_opinion, regex=True, na=False) & ~mask_crisis
    n_pco = int((mask_pco & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_pco, "economic_score"] = 0
    if n_pco:
        logger.info("post_filter_scores: political candidacy opinion eco=0 on %d", n_pco)

    # ── SYSTEMATIC MONTH-BY-MONTH FALSE-POSITIVE CLEANUP ─────────────────────────

    # Y. Animal/pet news → both scores 0
    _animal_news = (
        r"(perro|perra|gato|gata|mascota|chihuahua|cachorro|animal).{0,50}"
        r"(multa|demanda|rescate|sobrevivi|salv[oó]|accidente|helic[oó]ptero|caída|aeronave)|"
        r"(veterinaria|cl[ií]nica veterinaria).{0,40}(multa|sanción|indecopi|infracción)"
    )
    mask_ani = nt.str.contains(_animal_news, regex=True, na=False)
    df.loc[mask_ani, "economic_score"] = 0
    df.loc[mask_ani, "political_score"] = 0
    if mask_ani.sum():
        logger.info("post_filter_scores: animal news zeroed on %d", int(mask_ani.sum()))

    # Z. Military accident (non-attack) → eco=0
    _mil_accident = (
        r"(helic[oó]ptero|avi[oó]n|aeronave).{0,50}(fap|ej[eé]rcito|ffaa|militar).{0,50}"
        r"(accidentado?|cay[oó]|estrellado?|siniestro|tripulante.{0,20}(muere|mueren|fallece))|"
        r"(fap|ej[eé]rcito).{0,50}(helic[oó]ptero|aeronave).{0,50}(muere|fallece|tripulante|accidente)"
    )
    mask_ma = nt.str.contains(_mil_accident, regex=True, na=False) & ~mask_crisis
    n_ma = int((mask_ma & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_ma, "economic_score"] = 0
    if n_ma:
        logger.info("post_filter_scores: military accident eco=0 on %d", n_ma)

    # AA. Human-interest flood/weather anecdote (not disruption) → eco=0
    _flood_anecdote = (
        r"(jóvenes?|niños?|personas?).{0,30}(cruzaron?|caminan?|vadean?).{0,30}"
        r"(inundad|anegad).{0,20}(colch[oó]n|tabla|flotando|nadando)|"
        r"(colch[oó]n|tabla).{0,20}(cruzar?|cruzaron?).{0,20}(calle|avenida|pista).{0,20}inundad"
    )
    mask_fa = nt.str.contains(_flood_anecdote, regex=True, na=False)
    df.loc[mask_fa, "economic_score"] = 0
    if mask_fa.sum():
        logger.info("post_filter_scores: flood anecdote eco=0 on %d", int(mask_fa.sum()))

    # BB. Consumer product marketing (brands, flavors, new product launches) → eco=0
    _brand_marketing = (
        r"(peruanos?|consumidores?).{0,40}(prefieren|eligen|optan por).{0,30}"
        r"(sabores?|marcas?|productos?|bebidas?|comida)|"
        r"(diageo|nestl[eé]|p&g|unilever|coca.cola|pepsi|ab inbev|backus|lindley).{0,60}"
        r"(refuerza|lanza|presenta|estrategia|nuevos? productos?)|"
        r"(abre sede|nueva sede|expande|expansi[oó]n).{0,30}(en|hacia).{0,30}"
        r"(costa rica|colombia|chile|m[eé]xico|brasil|ecuador|bolivia|argentina)"
        r"(?!.{0,30}(inversi[oó]n|millones|exporta))"
    )
    mask_bm = nt.str.contains(_brand_marketing, regex=True, na=False) & ~mask_crisis
    n_bm = int((mask_bm & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_bm, "economic_score"] = 0
    if n_bm:
        logger.info("post_filter_scores: brand marketing eco=0 on %d", n_bm)

    # CC. Business self-help / how-to advice columns → eco=0
    _business_advice = (
        r"c[oó]mo (fortalecer|mejorar|gestionar|hacer crecer|escalar|digitalizar).{0,30}"
        r"(negocio|empresa|emprendimiento|mype|pyme|restaurante|tienda)|"
        r"el per[uú] emprendedor.{0,30}(que llevamos|dentro)|"
        r"f[aá]bricas de ayer y de hoy"
    )
    mask_ba = nt.str.contains(_business_advice, regex=True, na=False) & ~mask_crisis
    n_ba = int((mask_ba & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_ba, "economic_score"] = 0
    if n_ba:
        logger.info("post_filter_scores: business advice eco=0 on %d", n_ba)

    # DD. Party discipline / internal party politics → eco=0, pol boost
    _party_discipline = (
        r"(acci[oó]n popular|podemos per[uú]|alianza para el progreso|fuerza popular|"
        r"per[uú] libre|partido morado|somos per[uú]|juntos por el per[uú]).{0,60}"
        r"(proceso disciplinario|expulsi[oó]n|expulsaron|suspend[ií]|renunci[oó]|"
        r"disoluci[oó]n|disuelv|disolvió|bancada.{0,20}(rota|disuelta))|"
        r"(disolvió|disoluci[oó]n).{0,30}bancada"
    )
    mask_pd = nt.str.contains(_party_discipline, regex=True, na=False) & ~mask_crisis
    n_pd_eco = int((mask_pd & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_pd, "economic_score"] = 0
    df.loc[mask_pd & (df["political_score"].fillna(0) < 30), "political_score"] = 30
    if n_pd_eco:
        logger.info("post_filter_scores: party discipline eco=0 on %d", n_pd_eco)

    # EE. Cabinet formation / ministerial composition announcements → eco=0
    _cabinet_formation = (
        r"(gabinete|consejo de ministros).{0,60}"
        r"(no (estar[aá]|incluir[aá]|tendr[aá]).{0,30}(congresista|ministro saliente|"
        r"ex.?ministro|politicos?)|conformaci[oó]n|composici[oó]n)|"
        r"(jer[ií]|boluarte|premier).{0,50}gabinete.{0,40}"
        r"(no integrar[aá]n|no estar[aá]n|no habr[aá]|no tendr[aá]n).{0,30}"
        r"(congresistas?|ministros? salientes?)"
    )
    mask_cf = nt.str.contains(_cabinet_formation, regex=True, na=False) & ~mask_crisis
    n_cf_eco = int((mask_cf & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_cf, "economic_score"] = 0
    if n_cf_eco:
        logger.info("post_filter_scores: cabinet formation eco=0 on %d", n_cf_eco)

    # FF. Congressional bonus/salary proposals for military/police/prosecutors → eco=0, pol boost
    _congress_bonus_police = (
        r"(congreso|pleno).{0,50}(aprueba|propone|debate|presenta).{0,50}"
        r"(bonificaci[oó]n|bono).{0,50}"
        r"(defensor.{0,10}patria|polic[ií]a|efectivos?|fiscal|magistrado|juez|docente|maestro)"
        r"(?!.{0,40}(% del pib|millones de d[oó]lares|impacto fiscal))"
    )
    mask_cbp = nt.str.contains(_congress_bonus_police, regex=True, na=False) & ~mask_crisis
    n_cbp = int((mask_cbp & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_cbp, "economic_score"] = 0
    df.loc[mask_cbp & (df["political_score"].fillna(0) < 40), "political_score"] = 40
    if n_cbp:
        logger.info("post_filter_scores: congress salary/bonus eco=0 on %d", n_cbp)

    # GG. Prosecutorial raids, political figure legal actions (non-economic) → eco=0, pol boost
    _judicial_raid_political = (
        r"fiscal[ií]a (allana|realiza allanamiento|realiz[oó] operativo).{0,50}"
        r"(boluarte|nicanor|santiv[aá][ñn]ez|jer[ií]|vill[aá]|congresist|ministro)|"
        r"(jer[ií]|ministro.{0,10}interior|premier).{0,50}"
        r"(particip[oó]|estuvo presente|visit[oó]).{0,50}(penal|requis[a]|operativo policial)"
    )
    mask_jrp = nt.str.contains(_judicial_raid_political, regex=True, na=False) & ~mask_crisis
    n_jrp = int((mask_jrp & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_jrp, "economic_score"] = 0
    df.loc[mask_jrp & (df["political_score"].fillna(0) < 40), "political_score"] = 40
    if n_jrp:
        logger.info("post_filter_scores: judicial/political raid eco=0 on %d", n_jrp)

    # HH. Security equipment / police gear announcements → eco=0, pol boost
    _security_equipment = (
        r"(polic[ií]a|efectivos?|agentes?).{0,60}"
        r"(chalecos?|equipos?|armamentos?|patrulleros?|c[aá]maras?|drones?).{0,40}"
        r"(insuficiente|faltan|sin|no tienen|requieren|necesitan|incomunicados?)|"
        r"\d[\d,]+.{0,20}chalecos?.{0,30}\d[\d,]+.{0,20}(polic[ií]a|efectivos?)"
    )
    mask_se = nt.str.contains(_security_equipment, regex=True, na=False) & ~mask_crisis
    n_se = int((mask_se & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_se, "economic_score"] = 0
    if n_se:
        logger.info("post_filter_scores: security equipment eco=0 on %d", n_se)

    # II. Foreign political statements / opinions on foreign events → eco=0, pol=0
    _foreign_pol_statement = (
        r"(jer[ií]|boluarte|premier|canciller|foreign minister).{0,60}"
        r"(sobre (la |el )?(captura|detenci[oó]n|arresto) de|sobre (maduro|ortega|bukele|milei)|"
        r"venezuela (es|no es|fue)|declaraci[oó]n.{0,20}venezuela)|"
        r"(captura|detenci[oó]n) de (maduro|nicolas maduro).{0,60}"
        r"(jer[ií]|boluarte|premier|per[uú]|gobierno peruano)"
    )
    mask_fps = nt.str.contains(_foreign_pol_statement, regex=True, na=False) & ~mask_crisis
    n_fps = int((mask_fps & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_fps, "economic_score"] = 0
    if n_fps:
        logger.info("post_filter_scores: foreign pol statement eco=0 on %d", n_fps)

    # JJ. Political analysis/essay titles (opinion columns) → eco=0, pol boost
    _political_essay = (
        r"(despolarizar|desfujimorizar|descastillizar|democratizar).{0,30}(implica|significa|requiere)|"
        r"las medallas que (pesan|cuelgan|pesan sobre).{0,20}(boluarte|presidenta?|premier)|"
        r"(el|la|los|las) (per[uú]|peruanos?) que (llevamos|tenemos|somos).{0,10}dentro"
    )
    mask_pe = nt.str.contains(_political_essay, regex=True, na=False) & ~mask_crisis
    n_pe_eco = int((mask_pe & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_pe, "economic_score"] = 0
    df.loc[mask_pe & (df["political_score"].fillna(0) < 45), "political_score"] = 45
    if n_pe_eco:
        logger.info("post_filter_scores: political essay eco=0 on %d", n_pe_eco)

    # KK. Court/TC rulings on party registration or telecom → eco=0, pol boost
    _party_court = (
        r"(corte superior|tc|tribunal constitucional).{0,60}"
        r"(no acata|sigue tr[aá]mite|continua?r[aá]|tramita?r[aá]).{0,50}"
        r"(partido|organizaci[oó]n pol[ií]tica|unidad nacional|inscripci[oó]n)|"
        r"congreso.{0,50}(activaci[oó]n ilegal|l[ií]neas? telef[oó]nicas?).{0,30}"
        r"(ley|sanci[oó]n|castigar[aá]|penalizar[aá]|a[ñn]os de prisi[oó]n)"
    )
    mask_pct = nt.str.contains(_party_court, regex=True, na=False) & ~mask_crisis
    n_pct = int((mask_pct & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_pct, "economic_score"] = 0
    df.loc[mask_pct & (df["political_score"].fillna(0) < 35), "political_score"] = 35
    if n_pct:
        logger.info("post_filter_scores: party/court ruling eco=0 on %d", n_pct)

    # LL. Administrative funeral/social benefits procedures → eco=0
    _admin_benefits = (
        r"prestaci[oó]n.{0,20}(s[\\/]\s*1[\.,]?000|mil soles).{0,30}(sepelio|entierro|funeral)|"
        r"(sepelio|funeral).{0,20}(s[\\/]\s*1[\.,]?000|mil soles).{0,30}(requisito|plazo|solicitar|cómo)|"
        r"(requisitos?|c[oó]mo solicitar?|pasos? para).{0,30}(sepelio|defunci[oó]n|fallecimiento).{0,30}"
        r"(s[\\/]\s*1[\.,]?000|bono|prestaci[oó]n)"
    )
    mask_ab = nt.str.contains(_admin_benefits, regex=True, na=False) & ~mask_crisis
    n_ab = int((mask_ab & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_ab, "economic_score"] = 0
    if n_ab:
        logger.info("post_filter_scores: admin benefits eco=0 on %d", n_ab)

    # MM. "Betssy Chávez / Nicanor Boluarte / El Monstruo" political-legal commentary → eco=0
    _fugitive_political = (
        r"betssy ch[aá]vez.{0,60}(por qu[eé]|castillo|redact[oó]|mensaje|presi[oó]n|declaraci[oó]n)|"
        r"noblecilla.{0,60}(betssy|comunicaci[oó]n|perd[ií]|fugitiva?)|"
        r"el monstruo.{0,60}(partidos?|suspend[ií]|militantes?|pol[ií]ticos?)"
    )
    mask_fp = nt.str.contains(_fugitive_political, regex=True, na=False) & ~mask_crisis
    n_fp_eco = int((mask_fp & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_fp, "economic_score"] = 0
    df.loc[mask_fp & (df["political_score"].fillna(0) < 40), "political_score"] = 40
    if n_fp_eco:
        logger.info("post_filter_scores: fugitive/political commentary eco=0 on %d", n_fp_eco)

    # NN. Prosecutorial reorganization by incoming AG → eco=0, pol boost
    _mp_reorg = (
        r"(tom[aá]s g[aá]lvez|nuevo fiscal de la naci[oó]n|nuevo jefe del mp).{0,60}"
        r"(desactiva?|desmantelan?|disuelv|reemplaza|reorganiza|equipos especiales)|"
        r"(desactiva?|desmantelan?).{0,30}(equipos? especiales?|unidades? especiales?).{0,30}"
        r"(ministerio p[uú]blico|mp\b|fiscal[ií]a)"
    )
    mask_mr2 = nt.str.contains(_mp_reorg, regex=True, na=False) & ~mask_crisis
    n_mr2 = int((mask_mr2 & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_mr2, "economic_score"] = 0
    df.loc[mask_mr2 & (df["political_score"].fillna(0) < 50), "political_score"] = 50
    if n_mr2:
        logger.info("post_filter_scores: MP reorganization eco=0 on %d", n_mr2)

    # OO. TC/judicial rulings on institutional laws (PNP, police powers) → eco=0, pol boost
    _tc_institutional = (
        r"(tc|tribunal constitucional).{0,60}"
        r"(demanda|falla|resuelve|declara (infundada|fundada|inconstitucional)).{0,60}"
        r"(ley.{0,20}(pnp|fiscal[ií]a|investigaciones|policiales?|penal)|"
        r"facultades? (policiales?|de la pnp|investigadora))"
    )
    mask_tci = nt.str.contains(_tc_institutional, regex=True, na=False) & ~mask_crisis
    n_tci = int((mask_tci & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_tci, "economic_score"] = 0
    df.loc[mask_tci & (df["political_score"].fillna(0) < 40), "political_score"] = 40
    if n_tci:
        logger.info("post_filter_scores: TC institutional ruling eco=0 on %d", n_tci)

    # PP. Electoral violation rulings (JEE/JNE candidates) → eco=0, pol boost
    _electoral_violation = (
        r"(jee|jne|jurado nacional de elecciones).{0,60}"
        r"(infringement|vulneraci[oó]n|violaci[oó]n|infracci[oó]n).{0,40}"
        r"(neutralidad|veda|ley electoral|norma electoral|congresista|candidato)|"
        r"(congresista|candidato).{0,40}(vulneró|infringi[oó]|violó).{0,40}(ley electoral|neutralidad|veda)"
    )
    mask_ev = nt.str.contains(_electoral_violation, regex=True, na=False) & ~mask_crisis
    n_ev_eco = int((mask_ev & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_ev, "economic_score"] = 0
    df.loc[mask_ev & (df["political_score"].fillna(0) < 40), "political_score"] = 40
    if n_ev_eco:
        logger.info("post_filter_scores: electoral violation eco=0 on %d", n_ev_eco)

    # ── ENCODING-ROBUST FIXES (accented chars replaced with  in parquet) ────

    # QQ. Presidential food card (S/5k tarjeta) → eco=0, pol≥40
    _pres_tarjeta = (
        r"tarjeta.{0,30}(s[\\/]?\s*5[\.,]?\s*mil|cinco mil).{0,60}(boluarte|presidenta?|presidencia|personal|alimento)|"
        r"(boluarte|presidenta?|presidencia).{0,60}tarjeta.{0,30}(s[\\/]?\s*5[\.,]?\s*mil|cinco mil|alimento)"
    )
    mask_pt = nt.str.contains(_pres_tarjeta, regex=True, na=False) & ~mask_crisis
    n_pt = int((mask_pt & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_pt, "economic_score"] = 0
    df.loc[mask_pt & (df["political_score"].fillna(0) < 40), "political_score"] = 40
    if n_pt:
        logger.info("post_filter_scores: pres tarjeta eco=0 on %d", n_pt)

    # RR. Vehicle/crime individual sentences / rammings → eco=0
    _vehicle_crime = (
        r"(embisti[o]|atropell[o]).{0,50}"
        r"(restaurante|negocio|tienda|local|establecimiento|personas?|transeúntes?)|"
        r"dictan?.{0,20}(meses?|a[n]os?).{0,20}(pris[i][o]n preventiva|condena).{0,40}"
        r"(sujeto|individuo|hombre|mujer|acusado|imputado).{0,60}(embisti[o]|rob[o]|asalt[o])"
    )
    mask_vc = nt.str.contains(_vehicle_crime, regex=True, na=False) & ~mask_crisis
    n_vc = int((mask_vc & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_vc, "economic_score"] = 0
    if n_vc:
        logger.info("post_filter_scores: vehicle/crime individual eco=0 on %d", n_vc)

    # SS. Congressional procedure timelines (15 days to investigate X) → eco=0, pol boost
    _congress_proc = (
        r"(comisi[o]n permanente|subcomisi[o]n|congreso).{0,60}"
        r"(15|treinta|noventa|10|20|30).{0,10}d[i]as?.{0,30}"
        r"(investigar|indagar|revisar|analizar).{0,40}"
        r"(denuncia|acusaci[o]n constitucional|caso|informe)|"
        r"(parlamento|congreso).{0,30}(queda con|tiene|suma).{0,10}\d{1,2} bancadas?"
    )
    mask_cp = nt.str.contains(_congress_proc, regex=True, na=False) & ~mask_crisis
    n_cp = int((mask_cp & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_cp, "economic_score"] = 0
    df.loc[mask_cp & (df["political_score"].fillna(0) < 40), "political_score"] = 40
    if n_cp:
        logger.info("post_filter_scores: congress procedure eco=0 on %d", n_cp)

    # TT. Crime advice columns (encoding-robust, no accents needed) → eco=0
    _crime_advice_col = (
        r"cu.{0,3}dese de la delincuencia|"
        r"cr[i]menes en [a-z]+ [|] edicion|"
        r"consejos? para no ser v[i]ctima|"
        r"como protegerse.{0,30}(robo|asalto|delincuencia)|"
        r"red de prostituci[o]n en el congreso"
    )
    mask_cac = nt.str.contains(_crime_advice_col, regex=True, na=False)
    n_cac = int((mask_cac & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_cac, "economic_score"] = 0
    if n_cac:
        logger.info("post_filter_scores: crime advice col eco=0 on %d", n_cac)

    # UU. Political proposals (named legislators proposing, no vote) → eco=0 or cap at 10
    _named_pol_proposal = (
        r"n.{0,4}stor de la rosa propone|"
        r"(congresista|legislador|parlamentario).{0,50}propone.{0,50}construcci[o]n de|"
        r"delegaci[o]n.{0,20}facultades.{0,30}shock desregulatorio"
    )
    mask_npp = nt.str.contains(_named_pol_proposal, regex=True, na=False) & ~mask_crisis
    df.loc[mask_npp & (df["economic_score"].fillna(0) > 15), "economic_score"] = 15
    if mask_npp.sum():
        logger.info("post_filter_scores: named pol proposal eco<=15 on %d", int(mask_npp.sum()))

    # VV. Fintech/banking product marketing for youth → eco=0
    _fintech_marketing = (
        r"(cuenta|tarjeta).{0,20}(juvenil|joven|chill\b|kids?).{0,40}"
        r"(primer paso|generaci[o]n financieramente|inteligente|lanz|nuevo producto)|"
        r"generaci[o]n financieramente inteligente"
    )
    mask_fm = nt.str.contains(_fintech_marketing, regex=True, na=False) & ~mask_crisis
    n_fm = int((mask_fm & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_fm, "economic_score"] = 0
    if n_fm:
        logger.info("post_filter_scores: fintech marketing eco=0 on %d", n_fm)

    # WW. Billionaire/wealth profiles of foreign nationals → eco=0
    _billionaire_profile = (
        r"(qui[e]n es|c[o]mo se hizo|el hombre|la mujer).{0,30}"
        r"(m[a]s rico|m[a]s rica).{0,40}(am[e]rica latina|mundo)|"
        r"(millonario|multimillonario).{0,30}(mexicano|colombiano|brasile[n]o|argentino|chileno).{0,30}"
        r"(fortuna|patrimonio|riqueza).{0,20}(supera|alcanza|vale)"
    )
    mask_bp = nt.str.contains(_billionaire_profile, regex=True, na=False) & ~mask_crisis
    n_bp = int((mask_bp & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_bp, "economic_score"] = 0
    if n_bp:
        logger.info("post_filter_scores: billionaire profile eco=0 on %d", n_bp)

    # XX. Individual political resignation from party → eco=0, pol boost
    _pol_resignation = (
        r"(carlos anderson|pol[i]tico|congresista|militante).{0,40}"
        r"renuncia.{0,30}(al partido|del partido|al grupo|de la bancada).{0,30}"
        r"(per[u] moderno|acci[o]n popular|fuerza popular|podemos|alianza)|"
        r"(renuncia|se retira).{0,20}(al|del|de la).{0,10}(partido|bancada).{0,30}"
        r"(las razones|sus motivos|por diferencias|discrepancias)"
    )
    mask_pr2 = nt.str.contains(_pol_resignation, regex=True, na=False) & ~mask_crisis
    n_pr2 = int((mask_pr2 & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_pr2, "economic_score"] = 0
    df.loc[mask_pr2 & (df["political_score"].fillna(0) < 25), "political_score"] = 25
    if n_pr2:
        logger.info("post_filter_scores: political resignation eco=0 on %d", n_pr2)

    # YY. Machu Picchu tickets/management controversy → cap eco at 10
    _machu_tickets = (
        r"(machu picchu|mapi|santuario hist[o]rico).{0,60}"
        r"(caos|cupos|boletos|entradas|venta).{0,40}"
        r"(alcalde|mincul|problema|demanda|responsabiliza)"
    )
    mask_mt = nt.str.contains(_machu_tickets, regex=True, na=False) & ~mask_crisis
    df.loc[mask_mt & (df["economic_score"].fillna(0) > 10), "economic_score"] = 10
    if mask_mt.sum():
        logger.info("post_filter_scores: Machu Picchu tickets eco<=10 on %d", int(mask_mt.sum()))

    # ZZ. Senamhi routine dust/sand storms forecast → eco=0
    _senamhi_dust = (
        r"(tormentas? de arena|vientos? fuertes?|nublado|lloviznas? ligeras?).{0,30}senamhi|"
        r"senamhi.{0,60}(tormentas? de arena|qu[e] pronostica|c[o]mo estar[a]|"
        r"vientos? fuertes?|d[i]as? siguientes?)"
    )
    mask_sd = nt.str.contains(_senamhi_dust, regex=True, na=False) & ~mask_crisis
    n_sd = int((mask_sd & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_sd, "economic_score"] = 0
    if n_sd:
        logger.info("post_filter_scores: senamhi dust/forecast eco=0 on %d", n_sd)

    # AAA. Consumer taste / food lifestyle articles ("sabores dulces", beverage profiles) → eco=0
    _food_lifestyle = (
        r"sabores? (dulces?|salados?|picantes?|agridulces?|reconfortantes?)|"
        r"(mejores?|top \d+).{0,30}(sabores?|helados?|postres?|cocteles?|bebidas?).{0,30}(per[uú]|lima|verano)|"
        r"(cual|cu[aá]l).{0,20}es el mejor sabor|"
        r"tendencias? gastronomicas?.{0,40}(20\d\d|per[uú]|lima)"
    )
    mask_fl = nt.str.contains(_food_lifestyle, regex=True, na=False) & ~mask_crisis
    n_fl = int((mask_fl & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_fl, "economic_score"] = 0
    if n_fl:
        logger.info("post_filter_scores: food lifestyle eco=0 on %d", n_fl)

    # BBB. Education/school/university non-economic content → eco=0
    _education_non_eco = (
        r"escuela bicentenario|"
        r"tiktok.{0,30}(universidad|clases|educacion|aprendizaje)|"
        r"(generacion|generaci[oó]n)\s*z.{0,60}(trabajo|estudio|empresa|carrera|"
        r"educacion|finanzas|laboral|millennial)|"
        r"(generacion|generaci[oó]n)\s*(z\b|centennial|alpha).{0,30}(peru|joven|estudio)|"
        r"universidad.{0,30}(ofrece|lanza|presenta).{0,30}(nueva carrera|maestria|diploma|certificado)"
        r"(?!.{0,30}(millon|inversion|presupuesto))"
    )
    mask_edu = nt.str.contains(_education_non_eco, regex=True, na=False) & ~mask_crisis
    n_edu = int((mask_edu & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_edu, "economic_score"] = 0
    if n_edu:
        logger.info("post_filter_scores: education non-eco eco=0 on %d", n_edu)

    # CCC. Broad crime incidents (individual robbery/murder/assault) → eco=0
    # Exception: if title mentions empresa/sector/millones/comercio/industria → keep eco score
    _crime_incident = (
        r"\b(roban?|robaron|asaltan?|asaltaron|asesinan?|asesinaron|matan?|mataron|"
        r"secuestran?|secuestraron|extorsionan?|sicario\b|ajusticiaron?|"
        r"balean?|balearon|apunalan?|apunalaron).{0,80}"
        r"(hombre|mujer|joven|ancian[oa]|pareja|familia|conductor|chofer|"
        r"mototaxista|taxista|comerciante|vecino|vecina|escolar|estudiante|"
        r"trabajador|persona|individuo|sujeto|ladron|ladrones)\b|"
        r"\b(hombre|mujer|joven|ancian[oa]).{0,30}"
        r"(roban?|robaron|asaltan?|asaltaron|asesinan?|matan?|secuestran?|balean?)"
    )
    _crime_eco_exception = (
        r"empresa|sector|millones|industria|comercio|negocio|mype|pyme|"
        r"economia|economica|inversiones?|exporta|produccion"
    )
    mask_ci = (
        nt.str.contains(_crime_incident, regex=True, na=False) &
        ~nt.str.contains(_crime_eco_exception, regex=True, na=False) &
        ~mask_crisis
    )
    n_ci = int((mask_ci & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_ci, "economic_score"] = 0
    if n_ci:
        logger.info("post_filter_scores: crime incident eco=0 on %d", n_ci)

    # DDD. Broad pet/animal human-interest news (no disaster qualifier needed) → eco=0
    _animal_broad = (
        r"\b(perrito|gatito|chihuahua|pitbull|rottweiler|bulldog|"
        r"golden retriever|labrador|husky|canino|felino).{0,80}"
        r"(rescatado?|adoptado?|viral|tierno|conmueve|hero|abandono|"
        r"regalo|disfraz|concurso|campeon|gana|llora|busca|"
        r"muere|fallece|atropellado?|herido?|salvo?)|"
        r"(veterinaria|clinica veterinaria).{0,60}(mascota|perro|gato|animal)|"
        r"(indecopi|multa|sancion).{0,40}(veterinaria|clinica.{0,10}veterinaria)|"
        r"mascota.{0,40}(adopcion|adoptar|rescate|viral|tierno)|"
        r"(perrito|gatito|chihuahua).{0,30}(video|foto|imagen)"
    )
    mask_ab2 = nt.str.contains(_animal_broad, regex=True, na=False) & ~mask_crisis
    n_ab2 = int((mask_ab2 & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_ab2, "economic_score"] = 0
    if n_ab2:
        logger.info("post_filter_scores: broad animal/pet eco=0 on %d", n_ab2)

    # EEE. Remaining targeted FP fixes
    # Investment opinion "Sueño con un Perú donde el 5% invierta en acciones" → eco=15
    _sueno_opinion = r"sueno con un peru.{0,40}(invierta|invertir|acciones)"
    mask_sueno = nt.str.contains(_sueno_opinion, regex=True, na=False) & ~mask_crisis
    df.loc[mask_sueno & (df["economic_score"].fillna(0) > 15), "economic_score"] = 15
    if mask_sueno.sum():
        logger.info("post_filter_scores: sueno opinion eco<=15 on %d", int(mask_sueno.sum()))

    _targeted_fp = (
        # Crime section header "Crímenes en [region] | EDICION"
        r"cr.{0,3}men.{0,3} en [a-z].{1,20}[|].*edicion|"
        r"cr.{0,3}men.{0,3} en [a-z].{1,20} edicion|"
        # Petroperu opinion column "Petroperú: una verdad incómoda"
        r"petroperu.{0,5}[:]?.{0,10}(verdad incomoda|verdad incomodo|opinion|columna)|"
        # Political denial/minimization "X minimiza denuncia"
        r"(minimiza|descarta|niega).{0,30}denuncia.{0,40}(boluarte|premier|presidente|congres)|"
        r"(boluarte|premier|presidente|congresista).{0,60}(minimiza|descarta|niega).{0,30}denuncia"
    )
    mask_tfp = nt.str.contains(_targeted_fp, regex=True, na=False) & ~mask_crisis
    n_tfp = int((mask_tfp & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_tfp, "economic_score"] = 0
    df.loc[mask_tfp & (df["political_score"].fillna(0) < 30), "political_score"] = 30
    if n_tfp:
        logger.info("post_filter_scores: targeted FP eco=0 on %d", n_tfp)

    # ── END NEW FILTERS ──────────────────────────────────────────────────────────

    n_zeroed = 0
    for pattern, use_crisis_exception in rules_eco_zero:
        mask = nt.str.contains(pattern, regex=True, na=False)
        if use_crisis_exception:
            mask = mask & ~mask_crisis
        changed = mask & (df["economic_score"].fillna(0) > 0)
        n_zeroed += int(changed.sum())
        df.loc[mask, "economic_score"] = 0

    # FX routine cap at 10
    _fx_routine = r"(precio del d[oó]lar|d[oó]lar hoy|divisa cierra|tipo de cambio hoy|sol se cotiza)"
    _fx_large_move = r"(puntos b[aá]sicos|pierde|gana|dispara|desploma|hunde|sube.*%|baja.*%)"
    mask_fx = (
        nt.str.contains(_fx_routine, regex=True, na=False) &
        ~nt.str.contains(_fx_large_move, regex=True, na=False)
    )
    df.loc[mask_fx, "economic_score"] = df.loc[mask_fx, "economic_score"].clip(upper=10)

    logger.info("post_filter_scores: zeroed eco on %d articles", n_zeroed)

    # ── ECONOMIC WHITELIST CAP (last rule — catches all remaining garbage) ────────
    # If eco >= 35 but title has ZERO economic keywords → cap at 15.
    # This is the catch-all for opinion columns, crime blotters, gym openings,
    # pet clinics, travel tips, political scandals, etc. that slip past blacklists.
    _ECO_WHITELIST_TERMS = [
        # Macro/fiscal
        "pbi", "pib", "inflacion", "deflacion", "recesion", "crecimiento",
        "desacelera", "recuperacion", "economia", "economico", "economica",
        # Trade
        "arancel", "exportacion", "importacion", "comercio exterior",
        "balanza", "tlc", "dumping",
        # Monetary/financial
        "bcrp", "banco central", "tasa de interes", "tipo de cambio",
        "dolar", "divisa", "devaluacion", "reservas",
        "bolsa de valores", "bvl", "bono",
        # Fiscal
        "mef", "sunat", "tributar", "impuesto", "igv", "renta",
        "fiscal", "presupuesto", "deuda", "deficit", "superavit",
        "uit",
        # Energy/commodities
        "petroperu", "petroper", "gas", "gnv", "glp", "gasolina",
        "petroleo", "combustible", "electricidad", "energia",
        "camisea", "oleoducto", "refineria",
        # Mining
        "mineri", "minero", "minera", "cobre", "oro", "plata", "zinc",
        "reinfo", "mape", "concesion minera",
        # Corporate
        "telefonica", "concursal", "quiebra", "insolvencia",
        "corpac", "latam", "interbank",
        # Labor
        "empleo", "desempleo", "salario", "sueldo", "remuneracion",
        "pension", "afp", "onp", "essalud", "gratificacion", "cts",
        "huelga", "paro", "bloqueo",
        # Investment/infrastructure
        "inversion", "inversiones", "majes", "siguas", "chinchero",
        "ciadi", "arbitraje", "indemnizacion", "licitacion",
        "concesion", "obra publica",
        # Agriculture/fishing
        "agro", "agricul", "cosecha", "pesca", "pota", "anchoveta",
        "harina de pescado",
        # Banking/insurance
        "banco", "financier", "credito", "morosidad", "microfinancier",
        "caja municipal", "seguro", "sbs", "smv",
        # Prices/costs
        "precio", "costo", "tarifa", "canasta", "pobreza",
        # Industry/transport
        "manufactura", "industrial", "fabrica", "produccion",
        "aeropuerto", "puerto", "ferrocarr",
        # Business groups / regulators
        "confiep", "adex", "ccl", "sni", "snmpe", "comex", "indecopi",
        # Formalization
        "formalizacion", "mypes", "mype",
        # Real estate
        "vivienda", "construccion", "inmobiliari", "hipotec",
        "arrendamiento", "alquiler",
        # Natural disasters (economic impact)
        "huaico", "inundacion", "desborde", "emergencia", "damnificado",
        "sequia", "helada", "fenomeno el nino",
        # Retail/consumer economy
        "retail", "supermercado", "consumo", "consumidor",
        # Telecom
        "telecom", "internet",
        # Tourism
        "turismo", "turista", "hoteleria",
        # Corporate M&A / numbers
        "millones", "miles de millones", "us$", "s/.", "ingresos",
        "utilidades", "facturacion", "rentabilidad",
    ]
    # Build efficient pattern: any keyword present → has eco signal
    _eco_wl_pattern = "|".join(_ECO_WHITELIST_TERMS)
    mask_has_eco_kw = nt.str.contains(_eco_wl_pattern, regex=True, na=False)
    mask_wl_cap = (df["economic_score"].fillna(0) >= 35) & ~mask_has_eco_kw
    n_wl_cap = int(mask_wl_cap.sum())
    df.loc[mask_wl_cap, "economic_score"] = df.loc[mask_wl_cap, "economic_score"].clip(upper=15)
    if n_wl_cap:
        logger.info("post_filter_scores: whitelist cap eco<=15 on %d articles", n_wl_cap)

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

    _pre_sports = _SPORTS_PATTERN
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

    # 2. Sports (all leagues, clubs, competitions, athletes) → both=0
    _sports = _SPORTS_PATTERN
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

    # 23. Celebrity gossip / tabloid / entertainment → BOTH scores 0
    _celeb = (
        r"maju mantilla|magaly medina|far[aá]ndula|reconciliaci[oó]n.*salcedo|"
        r"se agarran a golpes|pelea en combi|"
        r"supuesta reconciliaci[oó]n|infidelidad|ampay\b|esc[aá]ndalo sentimental|"
        r"romance entre|separaci[oó]n de|confirma su embarazo|boda de|"
        r"chollywood|showbiz|telenovela|reality (de amor|sentimental)|"
        r"ale fuller|mayra go[ñn]i|flavia laos|ma[ñn]ana me caso|esto es guerra|"
        r"al fondo hay sitio|gran hermano|la voz per[uú]|yo soy\b|el gran chef|"
        r"natti natasha|bad bunny|taylor swift|reggaet[oó]n|concierto de|gira musical|"
        r"primer adelanto de|estreno de.*serie|nueva temporada de|reaparecen con|"
        r"asaltan a|roban a|accidente de tr[aá]nsito|choque vehicular|atropell[oa]\b|"
        r"preferencias religiosas|creencias religiosas|fe de los j[oó]venes|"
        r"religi[oó]n.*encuesta|encuesta.*religi[oó]n"
    )
    mask_celeb = titles.str.contains(_celeb, regex=True, na=False)
    df.loc[mask_celeb, "economic_score"] = 0
    df.loc[mask_celeb, "political_score"] = 0

    # 24. Foreign weather / local crime in foreign locations → BOTH scores 0
    # "Alerta de lluvias en Florida", "tormenta en Texas", etc. are not Peru events
    _foreign_weather_news = (
        r"(alerta|tormenta|hurac[aá]n|nieve|lluvia|calor|fr[ií]o|nevada|inundaci[oó]n)"
        r".{0,40}(florida|texas|california|nueva york|miami|new york|chicago|"
        r"m[eé]xico d\.?f\.?|buenos aires|bogot[aá]|santiago de chile|caracas|madrid|barcelona)"
        r"|"
        r"(florida|texas|california|nueva york|miami|chicago).{0,40}"
        r"(alerta|tormenta|hurac[aá]n|sismo|terremoto|incendio forestal)"
    )
    mask_foreign_weather = titles.str.contains(_foreign_weather_news, regex=True, na=False)
    df.loc[mask_foreign_weather, "political_score"] = 0
    df.loc[mask_foreign_weather, "economic_score"] = 0

    masks = [mask_farandula, mask_sports, mask_fx_routine, mask_sismo, mask_weather,
             mask_inhab, mask_pol_action, mask_electoral, mask_mercados, mask_gastro,
             mask_horoscope, mask_lottery, mask_reality, mask_lifestyle, mask_personal_fin,
             mask_foreign_markets, mask_foreign_econ, mask_market_summary,
             mask_corp_earnings, mask_sports_biz, mask_candidacy, mask_consumer_info,
             mask_celeb, mask_foreign_weather]
    labels = ["farándula", "sports", "fx_routine", "sismo", "weather",
              "inhabilitacion", "pol_action", "electoral", "mercados", "gastro",
              "horoscope", "lottery", "reality", "lifestyle_tips", "personal_finance",
              "foreign_markets", "foreign_econ", "market_summary",
              "corp_earnings", "sports_biz", "candidacy", "consumer_info", "celeb_gossip",
              "foreign_weather"]
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
