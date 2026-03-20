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


_POLITICAL_SYSTEM_PROMPT = """Eres un analista de riesgo político peruano. Tu tarea: evaluar si este artículo de prensa señala INCERTIDUMBRE política institucional en el Perú.

NO es riesgo político:
- Deportes (fútbol, vóley, tenis, retiros de deportistas, resultados deportivos) → 0
- Farándula, celebridades, entretenimiento, TV → 0
- Clima, alertas meteorológicas, lluvias, vientos, oleajes → 0
- Tecnología, gadgets, aplicaciones, dispositivos médicos → 0
- Noticias internacionales sin impacto directo en la política peruana → 0
- Recetas, turismo, estilo de vida, salud general → 0
- Obituarios de personas no políticas → 0

SÍ es riesgo político (y debe puntuar ALTO):
- Vacancia presidencial, moción de vacancia → 75-90
- Voto de confianza (cualquier mención) → 65-80
- Interpelación o censura de ministros → 55-70
- Cambio de gabinete bajo presión → 60-75
- Conflicto entre Ejecutivo y Congreso → 55-70
- Detención, arresto, orden de captura de presidente o ex-presidente → 80+
- Corrupción de funcionarios activos, allanamientos, casos judiciales → 55-70
- Asesinato de figura política, funcionario, asesor → 65-80
- Crimen organizado infiltrado en instituciones políticas → 70+
- Fallos del Tribunal Constitucional sobre temas políticos → 55-65
- Protestas masivas que amenazan continuidad del gobierno → 65-80
- Estado de emergencia por conflicto social → 60-70

MODERADO (campañas electorales 2026):
- Candidatos presidenciales haciendo propuestas de campaña → 15-25
- Debates, foros de candidatos → 20-30
- Encuestas electorales → 15-25
- Candidato con problemas judiciales activos → 55-70

EJEMPLOS (estudia estos antes de responder):

Título: "Chiquito Romero se retiró como futbolista profesional"
→ pol=0 (deportes, completamente irrelevante)

Título: "Alerta amarilla por fuertes vientos en la costa sur"
→ pol=0 (clima, irrelevante)

Título: "Pastillero inteligente: dispositivo electrónico para tratamientos médicos"
→ pol=0 (tecnología/salud, irrelevante)

Título: "Hinchas celebran gol del Once Caldas desde el cable aéreo de Manizales"
→ pol=0 (deportes extranjeros, irrelevante)

Título: "Trump dice a la OTAN que EEUU no necesita ayuda de nadie"
→ pol=0 (internacional sin impacto directo en política peruana)

Título: "Denisse Miralles no tiene seguro el voto de confianza"
→ pol=72 (crisis institucional: gabinete puede caer)

Título: "Keiko Fujimori considera poco probable que FP otorgue voto de confianza"
→ pol=70 (principal fuerza parlamentaria amenaza con negar confianza al gabinete)

Título: "Sigue en suspenso decisión del TC sobre Vladimir Cerrón"
→ pol=70 (TC decidiendo sobre líder prófugo sentenciado)

Título: "Gobierno inyectará más de S/500 millones a Petroperú"
→ pol=15 (decisión fiscal, no crisis política directa)

Título: "Keiko Fujimori propone bono a comedores populares"
→ pol=20 (propuesta de campaña, no crisis)

Título: "Capturan a implicado en asesinato de asesora del Congreso Andrea Vidal"
→ pol=70 (asesinato vinculado a institución política)

Título: "Forsyth propone gobernar desde regiones"
→ pol=20 (propuesta genérica de campaña)

Título: "Candidatos al Congreso registran multas de tránsito impagadas"
→ pol=10 (nota menor sobre candidatos, no genera incertidumbre)

Título: "Reniec amplía vigencia de DNI para adultos mayores"
→ pol=0 (servicio estatal rutinario)

Escala:
0 = Sin relevancia política
1-20 = Señal política débil (campaña rutinaria, burocracia menor)
21-40 = Tensión baja-moderada (investigaciones, huelgas locales)
41-60 = Inestabilidad significativa (interpelaciones, conflictos entre poderes)
61-80 = Crisis mayor (vacancia, censuras, protestas nacionales)
81-100 = Emergencia (vacancia consumada, golpe, renuncia presidencial)

REGLA: Si un artículo usa una crisis económica como CONTEXTO para una maniobra política (ej: "oposición condiciona voto de confianza por crisis energética"), el puntaje POLÍTICO debe ser alto. Lo que importa es la ACCIÓN política, no el tema económico."""

_ECONOMIC_SYSTEM_PROMPT = """Eres un analista de riesgo económico peruano. Tu tarea: evaluar si este artículo señala INCERTIDUMBRE sobre la actividad económica del Perú.

NO es riesgo económico:
- Deportes (fútbol, vóley, retiros de deportistas, resultados) → 0
- Farándula, celebridades, entretenimiento → 0
- Clima rutinario (alertas de viento, lluvia, oleaje) → 0
- Tecnología, gadgets, apps, dispositivos → 0
- Desastres internacionales sin conexión a Perú (terremoto en X, tornado en Y) → 0
- Noticias internacionales sin impacto en economía peruana → 0
- Datos económicos POSITIVOS o RUTINARIOS (crecimiento normal, inversión anunciada, récords de exportación) → 0
- Noticias de candidatos sobre temas NO económicos → 0
- Crimen común sin impacto económico sistémico → 0

SÍ es riesgo económico (y debe puntuar ALTO):
- Petroperú: rescate, inyección fiscal, quiebra, deuda, crisis → 75-90
- Camisea / gas natural: interrupción, escasez, corte de suministro → 75-90
- Conflictos mineros (Las Bambas, Tía María, Cuajone, Southern, Quellaveco) → 55-70
- Paro nacional / huelga general → 60-70
- Bloqueo de carreteras / interrupción de cadena de suministro → 40-55
- Consejo Fiscal advierte sobre gasto excesivo → 50-60
- Gasto público excesivo aprobado por Congreso / MEF → 55-65
- Escasez de productos básicos / desabastecimiento → 60-70
- Inflación acelerando / BCRP subiendo tasa → 45-55
- Tipo de cambio bajo presión / sol depreciándose fuerte → 45-55
- Fuga de capitales / salida de inversión extranjera → 65-75
- Interpelación a ministro de Economía / Energía → 50-60
- Inversión cancelada por conflicto social → 50-60

EJEMPLOS (estudia estos antes de responder):

Título: "Chiquito Romero se retiró como futbolista profesional"
→ eco=0 (deportes, irrelevante)

Título: "India: tornado deja dos muertos y cientos de heridos"
→ eco=0 (desastre extranjero sin conexión a Perú)

Título: "Trump dice a la OTAN que EEUU no necesita ayuda de nadie"
→ eco=0 (geopolítica sin impacto directo en economía peruana)

Título: "Reniec amplía vigencia de DNI"
→ eco=0 (servicio estatal rutinario)

Título: "PBI creció 3.2% en enero, superando expectativas"
→ eco=0 (dato positivo, no genera incertidumbre)

Título: "Gobierno inyectará más de S/500 millones a Petroperú para reactivar operaciones"
→ eco=80 (rescate fiscal masivo a empresa estatal en crisis)

Título: "Este ha sido el costo de 13 días sin gas natural"
→ eco=80 (crisis energética prolongada con impacto en producción)

Título: "Alonso Segura, presidente del Consejo Fiscal: Lo recientemente aprobado hipotecará las finanzas públicas"
→ eco=60 (advertencia fiscal institucional sobre sostenibilidad)

Título: "Transportistas anuncian paro nacional por precio del combustible"
→ eco=65 (paralización nacional afecta cadenas de suministro)

Título: "Comunidades bloquean acceso a mina Las Bambas por tercer día"
→ eco=60 (paralización minera, principal sector exportador)

Título: "Plantean interpelar al ministro de Economía por crisis energética"
→ eco=60 (interpelación sobre crisis sectorial, señal institucional)

Título: "Keiko Fujimori propone bono a comedores populares"
→ eco=0 (propuesta de campaña, no genera incertidumbre económica)

Título: "Rafael Belaunde promete gas natural para el sur"
→ eco=0 (promesa de campaña, no crisis actual)

Título: "Sicario asesina a conductor de combi en el Callao"
→ eco=0 (crimen común, sin impacto económico sistémico)

Escala:
0 = Sin riesgo económico O noticia económica positiva/rutinaria
1-20 = Señal débil (volatilidad normal, advertencias menores)
21-40 = Estrés moderado (indicadores deteriorándose, alzas en canasta básica)
41-60 = Vulnerabilidad significativa (disrupciones sectoriales, bloqueos, reformas fiscales)
61-80 = Crisis severa (colapso sectorial, paralización energética, rescate fiscal)
81-100 = Emergencia (default soberano, colapso financiero, hiperinflación)

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
        r"renuncia al cargo|es reelegido|asume el cargo)|"
        # Bolivian internal politics (judicial, arrests, party conflicts — no Peru nexus)
        r"(hijo|familiar|aliado|partido).{0,40}(arce|evo morales|morales evo).{0,60}"
        r"(detenido|arrestado|imputado|acusado|investigado|formalizado)|"
        r"(arce|evo morales).{0,40}(hijo|familiar|aliado).{0,60}"
        r"(detenido|arrestado|imputado|acusado)|"
        r"(detienen|arrestan|imputan|acusan|investigan).{0,60}"
        r"(hijo|familiar|aliado|funcionario).{0,40}(arce|evo morales)\b|"
        r"justicia boliviana|fiscal[ií]a de bolivia|gobierno boliviano.{0,60}"
        r"(detiene|arresta|imputa|investiga)|"
        r"crisis pol[ií]tica (en|de) bolivia\b|"
        r"bolivia.{0,30}(golpe|crisis institucional|estado de emergencia)|"
        # Foreign wire dispatch format: "Country: ..." — foreign internal news
        r"^(bolivia|ecuador|argentina|colombia|venezuela|brasil|paraguay|uruguay|"
        r"chile|m[eé]xico|cuba|nicaragua|el salvador|honduras|guatemala)\s*:\s*.{0,200}"
        r"(detenido|arrestado|imputado|acusado|golpe|crisis|protestas|manifestantes|"
        r"huelga|paro nacional|estado de emergencia|renuncia|destituido|vacancia)"
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
        r"interpelaci[oó]n.{0,30}(ministro|premier).{0,30}(aprobada|debate|hoy|este\s+\w+es)\b|"
        r"(plantean?|aprobaron?|presentan?|anuncian?|proponen?).{0,40}interpelar?.{0,40}(ministro|premier)\b|"
        r"interpelar?.{0,40}(ministro|premier).{0,40}(congreso|pleno|bancada|legislador)\b"
    )
    mask_mr = nt.str.contains(_minister_removal, regex=True, na=False)
    df.loc[mask_mr & (df["political_score"].fillna(0) < 55), "political_score"] = 55
    n_mr = int(mask_mr.sum())
    if n_mr > 0:
        logger.info("post_filter_scores: minister removal pol>=55 on %d articles", n_mr)

    # Interpelación + energy/economic crisis → eco ≥ 55
    # An interpelación about energy/economic crisis has both pol AND eco signal
    _interpelacion_energy = (
        r"(interpelar?|interpelaci[oó]n).{0,80}"
        r"(energ[eé]tica?|crisis.{0,20}gas|gasoducto|petro|combustible|"
        r"econom[ií]a|fiscal|mef|deuda|gasto|presupuesto)\b|"
        r"(energ[eé]tica?|crisis.{0,20}gas|petro|combustible).{0,80}"
        r"(interpelar?|interpelaci[oó]n)\b"
    )
    mask_ie = nt.str.contains(_interpelacion_energy, regex=True, na=False)
    df.loc[mask_ie & (df["economic_score"].fillna(0) < 55), "economic_score"] = 55
    n_ie = int(mask_ie.sum())
    if n_ie > 0:
        logger.info("post_filter_scores: interpelacion energy eco>=55 on %d articles", n_ie)

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

    # Petroperú crisis / bankruptcy / bailout → eco ≥ 80 (spec section 3e: ≥80)
    _petroperu = (
        r"petroper[uú].{0,60}"
        r"(quiebra|concurso de acreedores|rescate|bail.?out|crisis|colapso|cierre|"
        r"p[eé]rdida|deuda|insolvencia|intervenida|liquidaci[oó]n|subsidio|aporte del estado|"
        r"salvavidas|reduce producci[oó]n|reactiv|millones|inyecci[oó]n|inyectar)\b"
    )
    mask_pp = nt.str.contains(_petroperu, regex=True, na=False)
    df.loc[mask_pp & (df["economic_score"].fillna(0) < 80), "economic_score"] = 80
    n_pp = int(mask_pp.sum())
    if n_pp > 0:
        logger.info("post_filter_scores: Petroperu crisis eco>=80 on %d articles", n_pp)

    # Petroperú / state company fiscal injection (gobierno inyecta millones) → eco ≥ 80
    # A direct Treasury/government cash injection into a state company is a high fiscal risk signal.
    _petroperu_injection = (
        r"(inyectar[aá]|inyect[oó]|inyecci[oó]n).{0,80}"
        r"(petroper[uú]|empresa estatal|empresa p[uú]blica).{0,80}"
        r"(s[\\/]\s*\d|millones|mil millones)|"
        r"(petroper[uú]|empresa estatal|empresa p[uú]blica).{0,80}"
        r"(inyectar[aá]|inyect[oó]|inyecci[oó]n).{0,80}"
        r"(s[\\/]\s*\d|millones|mil millones)|"
        r"(gobierno|estado).{0,40}(inyectar[aá]|inyect[oó]).{0,40}"
        r"(s[\\/]\s*\d{3,}|[0-9]{3,}.*millones).{0,40}"
        r"(petroper[uú]|petrolera estatal|empresa estatal|empresa p[uú]blica)"
    )
    mask_ppi = nt.str.contains(_petroperu_injection, regex=True, na=False)
    df.loc[mask_ppi & (df["economic_score"].fillna(0) < 80), "economic_score"] = 80
    n_ppi = int(mask_ppi.sum())
    if n_ppi > 0:
        logger.info("post_filter_scores: Petroperú/state injection eco>=80 on %d articles", n_ppi)

    # TGP / Camisea regulatory threat — license revocation, MINEM sanctions → eco ≥ 55
    # Physical crisis already handled above (eco ≥ 80). This covers regulatory risk.
    _tgp_regulatory = (
        r"(tgp|transportadora de gas).{0,80}"
        r"(licencia|concesi[oó]n|sanci[oó]n|multa|incumplimiento|normas|negligencia|"
        r"revocar|cancelar|suspensi[oó]n|investigaci[oó]n|fiscalizaci[oó]n)\b|"
        r"(minem|osinergmin).{0,60}(tgp|camisea|gasoducto|ducto).{0,60}"
        r"(licencia|concesi[oó]n|sanci[oó]n|multa|negligencia|normas|cancel)\b"
    )
    mask_tgp = nt.str.contains(_tgp_regulatory, regex=True, na=False)
    df.loc[mask_tgp & (df["economic_score"].fillna(0) < 55), "economic_score"] = 55
    n_tgp = int(mask_tgp.sum())
    if n_tgp > 0:
        logger.info("post_filter_scores: TGP regulatory threat eco>=55 on %d articles", n_tgp)

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

    # L2. NIMBY / neighborhood disputes (crematorio, antenas, relleno sanitario, etc.) → eco=0
    # These are purely local land-use conflicts with no macroeconomic significance.
    # Note: no ~mask_crisis exception here — "vecinos protestan" always triggers the crisis
    # guard's 'protesta' token, but a neighborhood crematorium dispute is NEVER a real crisis.
    _nimby_dispute = (
        r"vecinos?.{0,40}(protestan?|rechazan?|se oponen?|en contra de|impiden?).{0,60}"
        r"(crematorio|antena|relleno sanitario|planta de tratamiento|cementerio|"
        r"estaci[oó]n de gas|grifos?|cancha sintetica|discoteca|bar\b|"
        r"torre.{0,10}(alta tensi[oó]n|telecomunicaciones))|"
        r"(crematorio|relleno sanitario|planta de tratamiento).{0,60}"
        r"(vecinos?|protesta|rechazo|oposici[oó]n|municipalidad|surco|barranco|"
        r"miraflores|san isidro|san juan|villa maria|villa el salvador|comas|"
        r"los olivos|independencia|callao)|"
        r"controversia.{0,20}(surco|barranco|miraflores|san isidro|san juan).{0,60}"
        r"(crematorio|relleno|antena|construccion)"
    )
    mask_nimby = nt.str.contains(_nimby_dispute, regex=True, na=False)
    n_nimby = int((mask_nimby & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_nimby, "economic_score"] = 0
    if n_nimby:
        logger.info("post_filter_scores: NIMBY/neighborhood dispute eco=0 on %d", n_nimby)

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

    # UU2. Candidate proposals about security/prisons/military (no fiscal specifics) → eco≤15
    # "Carlos Álvarez propone que las Fuerzas Armadas controlen los penales" is a political
    # campaign proposal. It has no direct fiscal or economic disruption implication.
    _candidate_security_proposal = (
        r"(candidato|candidata|propone|plantea).{0,60}"
        r"(fuerzas armadas|ejercito|marina|militares?|ffaa).{0,60}"
        r"(penales?|prisiones?|carcel|reclusos?|inpec|penitenciar|seguridad\b)|"
        r"(penales?|prisiones?|carcel|reclusos?|penitenciar).{0,60}"
        r"(fuerzas armadas|ejercito|marina|militares?|ffaa).{0,60}"
        r"(propone|plantea|candidato|candidata|propuesta)|"
        r"(propone|plantea).{0,80}"
        r"(control.{0,20}(penales?|prisiones?|carcel)|"
        r"penales?.{0,20}a cargo de|penales?.{0,20}bajo mando).{0,60}"
        r"(fuerzas armadas|militares?|ejercito)"
    )
    mask_csp = nt.str.contains(_candidate_security_proposal, regex=True, na=False) & ~mask_crisis
    n_csp = int((mask_csp & (df["economic_score"].fillna(0) > 15)).sum())
    df.loc[mask_csp & (df["economic_score"].fillna(0) > 15), "economic_score"] = 15
    if n_csp:
        logger.info("post_filter_scores: candidate security/prison proposal eco<=15 on %d", n_csp)

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

    # CCC2. Organized/targeted crime (sicario, hitman attacks in public transport) → eco=30
    # After CCC zeroes all crime incidents, restore a moderate signal for sicario/hitman
    # attacks in public spaces (combi, bus, market). These signal urban insecurity
    # with real economic cost, but below the threshold of systemic disruption.
    _organized_crime_hit = (
        r"\bsicario\b.{0,80}(combi|bus\b|mototaxi|mercado|taxi\b|transporte|pasajero)|"
        r"(combi|bus\b|mototaxi|taxi\b).{0,60}\bsicario\b|"
        r"\bsicario\b.{0,60}(mata|matan|mat[oó]).{0,30}(chofer|conductor|pasajero|tres|dos)"
    )
    mask_och = (
        nt.str.contains(_organized_crime_hit, regex=True, na=False) &
        (df["economic_score"].fillna(0) < 30)
    )
    n_och = int(mask_och.sum())
    df.loc[mask_och, "economic_score"] = 30
    if n_och:
        logger.info("post_filter_scores: organized crime hit eco=30 on %d", n_och)

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

    # ── FALSE POSITIVE FIXES (new rules F1–F5) ───────────────────────────────────

    # F1. Foreign natural disasters with no Peru connection → eco=0, pol=0
    _foreign_disaster = (
        r"(india|indonesia|bangladesh|pakistan|filipinas|vietnam|myanmar).{0,60}"
        r"(tornado|ciclon|terremoto|inundacion|tsunami|huracan|tifon)|"
        r"(tornado|ciclon|terremoto|inundacion|tsunami|huracan|tifon).{0,60}"
        r"(india|indonesia|bangladesh|pakistan|filipinas|vietnam|myanmar)"
    )
    _has_peru_f1 = r"peru|peruana?|peruanos?"
    mask_f1 = (
        nt.str.contains(_foreign_disaster, regex=True, na=False) &
        ~nt.str.contains(_has_peru_f1, regex=True, na=False)
    )
    n_f1_eco = int((mask_f1 & (df["economic_score"].fillna(0) > 0)).sum())
    n_f1_pol = int((mask_f1 & (df["political_score"].fillna(0) > 0)).sum())
    df.loc[mask_f1, "economic_score"] = 0
    df.loc[mask_f1, "political_score"] = 0
    if n_f1_eco + n_f1_pol > 0:
        logger.info("post_filter_scores: F1 foreign disaster zeroed eco=%d pol=%d articles", n_f1_eco, n_f1_pol)

    # F2. Gadgets / consumer medical tech → pol=0
    _gadget_health = (
        r"(pastillero|wearable|smartwatch|aplicacion movil|app para|dispositivo (para|que)|gadget).{0,60}"
        r"(salud|tratamiento|medicamento|dieta|ejercicio)"
    )
    mask_f2 = nt.str.contains(_gadget_health, regex=True, na=False)
    n_f2_pol = int((mask_f2 & (df["political_score"].fillna(0) > 0)).sum())
    df.loc[mask_f2, "political_score"] = 0
    if n_f2_pol > 0:
        logger.info("post_filter_scores: F2 gadget/health tech pol zeroed on %d articles", n_f2_pol)

    # F3. Routine weather alerts (no national emergency language) → pol=0
    _routine_alert = (
        r"(alerta amarilla|alerta verde|alerta naranja).{0,60}"
        r"(vientos|lluvia|frio|helada|neblina)"
    )
    _emergency_lang = r"estado de emergencia|evacuacion masiva|decenas de muertos|colapso"
    mask_f3 = (
        nt.str.contains(_routine_alert, regex=True, na=False) &
        ~nt.str.contains(_emergency_lang, regex=True, na=False)
    )
    n_f3_pol = int((mask_f3 & (df["political_score"].fillna(0) > 0)).sum())
    df.loc[mask_f3, "political_score"] = 0
    if n_f3_pol > 0:
        logger.info("post_filter_scores: F3 routine weather alert pol zeroed on %d articles", n_f3_pol)

    # F4. Traffic fines on candidates / minor candidate gossip → cap pol at 20
    _candidate_fines = (
        r"(candidatos?|candidato|candidata).{0,60}"
        r"(multa|multas|papeleta|infraccion|deuda de transito)"
    )
    mask_f4 = nt.str.contains(_candidate_fines, regex=True, na=False)
    n_f4_pol = int((mask_f4 & (df["political_score"].fillna(0) > 20)).sum())
    df.loc[mask_f4 & (df["political_score"].fillna(0) > 20), "political_score"] = 20
    if n_f4_pol > 0:
        logger.info("post_filter_scores: F4 candidate traffic fines pol capped at 20 on %d articles", n_f4_pol)

    # F5. Foreign military/geopolitics without Peru impact → pol=0, eco≤15
    _foreign_mil = (
        r"(otan|nato|trump|estados unidos|eeuu|rusia|china|israel|gaza|ucrania).{0,60}"
        r"(necesita|aliados|tropas|misiles|acuerdo|ejercicios militares|ataque)"
    )
    _has_peru_f5 = r"peru|peruanos?|exportaciones peruanas|impacto en peru|afecta.{0,10}peru"
    mask_f5 = (
        nt.str.contains(_foreign_mil, regex=True, na=False) &
        ~nt.str.contains(_has_peru_f5, regex=True, na=False)
    )
    n_f5_pol = int((mask_f5 & (df["political_score"].fillna(0) > 0)).sum())
    n_f5_eco = int((mask_f5 & (df["economic_score"].fillna(0) > 15)).sum())
    df.loc[mask_f5, "political_score"] = 0
    df.loc[mask_f5 & (df["economic_score"].fillna(0) > 15), "economic_score"] = 15
    if n_f5_pol + n_f5_eco > 0:
        logger.info("post_filter_scores: F5 foreign military/geopolitics pol zeroed=%d eco capped=%d", n_f5_pol, n_f5_eco)

    # F6. Traffic accidents (tráiler vuelca, choque vehicular, colisión) → pol=0, eco=0
    # These are accident news, not political/economic disruption events.
    # Exception: if paro/bloqueo/huelga is also present, it may be a supply-chain disruption.
    _traffic_accident = (
        r"tr[aá]iler.{0,60}(vuelca|volc[oó]|choca|accidente)|"
        r"(vuelca|volc[oó]).{0,60}tr[aá]iler|"
        r"(choque|colisi[oó]n|chocaron|colisionaron).{0,60}"
        r"(v[eé]hiculos?|autos?|camion|bus\b|camioneta|moto\b|tren)|"
        r"(v[eé]hiculos?|autos?|camion|bus\b|camioneta).{0,60}"
        r"(chocaron|colisionaron|se impact|impact[oó])"
    )
    _blockade_exc = r"(paro|huelga|bloqueo|bloquearon|manifestantes|protesta)"
    mask_f6 = (
        nt.str.contains(_traffic_accident, regex=True, na=False) &
        ~nt.str.contains(_blockade_exc, regex=True, na=False)
    )
    n_f6_pol = int((mask_f6 & (df["political_score"].fillna(0) > 0)).sum())
    n_f6_eco = int((mask_f6 & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_f6, "political_score"] = 0
    df.loc[mask_f6, "economic_score"] = 0
    if n_f6_pol + n_f6_eco > 0:
        logger.info("post_filter_scores: F6 traffic accident zeroed pol=%d eco=%d", n_f6_pol, n_f6_eco)

    # F7. Traffic congestion / routine road closures → pol=0, eco=0
    # Congestión vehicular, conos, cierre de vía por obras: not political/economic events.
    _traffic_congestion = (
        r"congesti[oó]n vehicular.{0,60}(conos|obra|cierre|desvio|av\.|avenida)|"
        r"(conos|desvio|cierre de via).{0,60}congesti[oó]n vehicular|"
        r"conductores?.{0,40}reportan?.{0,40}(congesti[oó]n|tr[aá]fico)|"
        r"(colocaci[oó]n de conos|conos de seguridad).{0,60}"
        r"(avenida|av\.|via|congesti[oó]n|tr[aá]fico)|"
        r"tr[aá]fico.{0,40}(lento|pesado|caos|colapso).{0,40}(avenida|av\.|carretera|via)"
    )
    mask_f7 = (
        nt.str.contains(_traffic_congestion, regex=True, na=False) &
        ~nt.str.contains(_blockade_exc, regex=True, na=False)
    )
    n_f7_pol = int((mask_f7 & (df["political_score"].fillna(0) > 0)).sum())
    n_f7_eco = int((mask_f7 & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_f7, "political_score"] = 0
    df.loc[mask_f7, "economic_score"] = 0
    if n_f7_pol + n_f7_eco > 0:
        logger.info("post_filter_scores: F7 traffic congestion zeroed pol=%d eco=%d", n_f7_pol, n_f7_eco)

    # F8. Community lifestyle / free workshops → pol=0, eco=0
    # "Talleres gratuitos beneficiaron a vecinos", craft workshops, cooking classes
    _lifestyle_workshop = (
        r"talleres? gratuitos?.{0,60}(vecinos?|beneficiar|comunidad|barrio)|"
        r"(vecinos?|comunidad).{0,60}talleres? gratuitos?|"
        r"talleres? (de cocina|de tejido|de pintura|de manualidades|libre[s]?).{0,40}"
        r"(vecinos?|comunidad|beneficiar|gratuitos?)"
    )
    mask_f8 = nt.str.contains(_lifestyle_workshop, regex=True, na=False)
    n_f8 = int((mask_f8 & ((df["political_score"].fillna(0) > 0) | (df["economic_score"].fillna(0) > 0))).sum())
    df.loc[mask_f8, "political_score"] = 0
    df.loc[mask_f8, "economic_score"] = 0
    if n_f8 > 0:
        logger.info("post_filter_scores: F8 lifestyle workshop zeroed on %d articles", n_f8)

    # F9. US consumer/lifestyle articles (shopping tips, spending habits) → pol=0, eco=0
    # "Lista de compras impulsivas que están vaciando la cuenta de los estadounidenses"
    _us_consumer = (
        r"compras? impulsivas?.{0,80}(estadounidenses?|americanos?|cuenta[s]?)|"
        r"(estadounidenses?|americanos?).{0,80}compras? impulsivas?|"
        r"lista de compras.{0,60}(vacian?|gastan?|estadounidenses?|americanos?)|"
        r"(vacian?|gastan?).{0,60}cuenta[s]?.{0,40}(estadounidenses?|americanos?)"
    )
    _has_peru_f9 = r"peru|peruana?|peruanos?"
    mask_f9 = (
        nt.str.contains(_us_consumer, regex=True, na=False) &
        ~nt.str.contains(_has_peru_f9, regex=True, na=False)
    )
    n_f9 = int((mask_f9 & ((df["political_score"].fillna(0) > 0) | (df["economic_score"].fillna(0) > 0))).sum())
    df.loc[mask_f9, "political_score"] = 0
    df.loc[mask_f9, "economic_score"] = 0
    if n_f9 > 0:
        logger.info("post_filter_scores: F9 US consumer zeroed on %d articles", n_f9)

    # F10. Foreign electoral politics (US Latino voters, Trump, etc.) → pol=0
    # "Votantes latinos se alejan de Trump, y su política para Venezuela y Cuba"
    _foreign_electoral = (
        r"votantes? latinos?.{0,60}(trump|biden|harris|demócrata|republican|eeuu|estados unidos)|"
        r"(trump|biden|harris).{0,60}votantes? latinos?|"
        r"latinos?.{0,40}(se alejan?|apoyan?|rechazan?).{0,40}(trump|biden|harris|partido)"
    )
    _has_peru_f10 = r"peru|peruanos?|candidato peruano"
    mask_f10 = (
        nt.str.contains(_foreign_electoral, regex=True, na=False) &
        ~nt.str.contains(_has_peru_f10, regex=True, na=False)
    )
    n_f10 = int((mask_f10 & (df["political_score"].fillna(0) > 0)).sum())
    df.loc[mask_f10, "political_score"] = 0
    df.loc[mask_f10 & (df["economic_score"].fillna(0) > 15), "economic_score"] = 15
    if n_f10 > 0:
        logger.info("post_filter_scores: F10 foreign electoral zeroed pol=%d articles", n_f10)

    # F11. Digital data privacy (comercialización de datos, privacidad digital) → pol≤15, eco=0
    # Not a political crisis unless it involves a government agency breach.
    _data_privacy = (
        r"comercializaci[oó]n ilegal de datos|"
        r"venta de datos (personales|privados|ilegales?)|"
        r"(hackeo|phishing|ciberataque).{0,60}(datos|privacidad|cuenta[s]?)|"
        r"tus datos (personales|privados|en internet)"
    )
    _govt_breach_exc = r"gobierno|estado|ministerio|congres|sunat|reniec|onp"
    mask_f11 = (
        nt.str.contains(_data_privacy, regex=True, na=False) &
        ~nt.str.contains(_govt_breach_exc, regex=True, na=False)
    )
    n_f11_pol = int((mask_f11 & (df["political_score"].fillna(0) > 15)).sum())
    n_f11_eco = int((mask_f11 & (df["economic_score"].fillna(0) > 0)).sum())
    df.loc[mask_f11 & (df["political_score"].fillna(0) > 15), "political_score"] = 15
    df.loc[mask_f11, "economic_score"] = 0
    if n_f11_pol + n_f11_eco > 0:
        logger.info("post_filter_scores: F11 data privacy capped pol=%d eco=0 on %d", n_f11_pol, n_f11_eco)

    # ── BOOST RULES (B1–B14) — applied BEFORE whitelist cap ─────────────────────

    # B1. New cabinet / toma de juramento → pol≥65
    _toma_juramento = (
        r"toma de juramento.{0,60}(gabinete|premier|primer ministro)|"
        r"nuevo gabinete.{0,60}(jura|jurament|boluarte|presidenta?|premier)|"
        r"(boluarte|presidenta?|premier).{0,60}nuevo gabinete"
    )
    mask_b1 = nt.str.contains(_toma_juramento, regex=True, na=False)
    n_b1 = int((mask_b1 & (df["political_score"].fillna(0) < 65)).sum())
    df.loc[mask_b1 & (df["political_score"].fillna(0) < 65), "political_score"] = 65
    if n_b1 > 0:
        logger.info("post_filter_scores: B1 new cabinet/juramento pol>=65 on %d articles", n_b1)

    # B2. Voto de confianza (any mention) → pol≥70
    # Per spec section D: ANY article containing "voto de confianza" is a top-priority
    # institutional crisis signal. Flat floor — no tiering.
    # Crisis context overrides candidate floors: "Keiko says FP won't give voto de confianza"
    # is a voto de confianza article (pol=70), not a candidate article (pol=15-25).
    _confianza_any = r"voto de confianza"
    mask_b2 = nt.str.contains(_confianza_any, regex=True, na=False)
    n_b2 = int((mask_b2 & (df["political_score"].fillna(0) < 70)).sum())
    df.loc[mask_b2 & (df["political_score"].fillna(0) < 70), "political_score"] = 70
    if n_b2 > 0:
        logger.info("post_filter_scores: B2 voto de confianza pol>=70 on %d articles", n_b2)

    # B3. Assassination of political/public figure → pol≥70, eco≥20
    _assassination = (
        r"(asesinato|asesinaron|matan|ejecutan).{0,60}"
        r"(congresista|asesora|asesor|regidor|alcalde|gobernador|fiscal|juez|"
        r"periodista|candidato|funcionario|andrea vidal)"
    )
    mask_b3 = nt.str.contains(_assassination, regex=True, na=False)
    n_b3_pol = int((mask_b3 & (df["political_score"].fillna(0) < 70)).sum())
    n_b3_eco = int((mask_b3 & (df["economic_score"].fillna(0) < 20)).sum())
    df.loc[mask_b3 & (df["political_score"].fillna(0) < 70), "political_score"] = 70
    df.loc[mask_b3 & (df["economic_score"].fillna(0) < 20), "economic_score"] = 20
    if n_b3_pol + n_b3_eco > 0:
        logger.info("post_filter_scores: B3 political assassination pol>=70 on %d eco>=20 on %d", n_b3_pol, n_b3_eco)

    # B4. Corruption arrests of public officials → pol≥60
    _corrupt_arrest = (
        r"(detienen|capturan|arrestan|allanamiento|allanaron|incautan|incautaron).{0,60}"
        r"(congresista|ministro|gobernador|alcalde|fiscal|juez|funcionario)"
    )
    # Exclude sports/entertainment context
    _corrupt_sports_exc = r"(jugador|deportista|futbolista|cantante|artista)"
    mask_b4 = (
        nt.str.contains(_corrupt_arrest, regex=True, na=False) &
        ~nt.str.contains(_corrupt_sports_exc, regex=True, na=False)
    )
    n_b4 = int((mask_b4 & (df["political_score"].fillna(0) < 60)).sum())
    df.loc[mask_b4 & (df["political_score"].fillna(0) < 60), "political_score"] = 60
    if n_b4 > 0:
        logger.info("post_filter_scores: B4 corruption arrest pol>=60 on %d articles", n_b4)

    # B5. Destitución / suspensión / remoción of official → pol≥60
    _destitucion = (
        r"(destituy|destituido|destituida|suspendido|suspendida|removido|removida|"
        r"inhabilitado|inhabilitada).{0,60}"
        r"(ministro|fiscal|juez|congresista|gobernador|alcalde|funcionario|director)"
    )
    mask_b5 = nt.str.contains(_destitucion, regex=True, na=False)
    n_b5 = int((mask_b5 & (df["political_score"].fillna(0) < 60)).sum())
    df.loc[mask_b5 & (df["political_score"].fillna(0) < 60), "political_score"] = 60
    if n_b5 > 0:
        logger.info("post_filter_scores: B5 official removal/suspension pol>=60 on %d articles", n_b5)

    # B6. Campaign foro / candidatos presidenciales → cap pol≤40, eco≤10
    # These are electoral campaign events, NOT crises — cap if over-scored
    _campaign_foro = (
        r"(foro de candidatos|foro electoral|candidatos presidenciales|candidatos al congreso).{0,60}"
        r"(propone|plantea|expone|expondran|expuso|debate|debatr|presentan|presentaran)"
    )
    mask_b6 = nt.str.contains(_campaign_foro, regex=True, na=False)
    n_b6_pol = int((mask_b6 & (df["political_score"].fillna(0) > 40)).sum())
    n_b6_eco = int((mask_b6 & (df["economic_score"].fillna(0) > 10)).sum())
    df.loc[mask_b6 & (df["political_score"].fillna(0) > 40), "political_score"] = 40
    df.loc[mask_b6 & (df["economic_score"].fillna(0) > 10), "economic_score"] = 10
    if n_b6_pol + n_b6_eco > 0:
        logger.info("post_filter_scores: B6 campaign foro capped pol<=40 on %d eco<=10 on %d", n_b6_pol, n_b6_eco)

    # B7. Camisea / gas natural disruption → eco≥80
    _camisea_disruption = (
        r"(camisea|gasoducto|gas natural).{0,60}"
        r"(paraliza|paralizado|corte|cortado|ruptura|rotura|sin gas|dias sin gas|"
        r"semanas sin gas|interrupcion|emergencia|colapso)"
    )
    mask_b7 = nt.str.contains(_camisea_disruption, regex=True, na=False)
    n_b7 = int((mask_b7 & (df["economic_score"].fillna(0) < 80)).sum())
    df.loc[mask_b7 & (df["economic_score"].fillna(0) < 80), "economic_score"] = 80
    if n_b7 > 0:
        logger.info("post_filter_scores: B7 Camisea/gas disruption eco>=80 on %d articles", n_b7)

    # B8. Gasto público / deuda aprobada por Congreso → eco≥60
    _fiscal_congress = (
        r"(congreso aprueba|congreso aprobo).{0,60}"
        r"(gasto|deuda|prestamo|credito suplementario|modificacion presupuestal|endeudamiento)|"
        r"(hipotecara|deficit fiscal|deuda publica).{0,60}(finanzas|presupuesto|estado)|"
        r"consejo fiscal.{0,60}(advierte|alerta|hipotecara|insostenible|riesgo fiscal)"
    )
    mask_b8 = nt.str.contains(_fiscal_congress, regex=True, na=False)
    n_b8 = int((mask_b8 & (df["economic_score"].fillna(0) < 60)).sum())
    df.loc[mask_b8 & (df["economic_score"].fillna(0) < 60), "economic_score"] = 60
    if n_b8 > 0:
        logger.info("post_filter_scores: B8 Congress fiscal expansion eco>=60 on %d articles", n_b8)

    # B9. Petroperú préstamo / rescate → eco≥75
    _petroperu_loan = (
        r"(petroperu|petroper).{0,60}"
        r"(prestamo|rescate|salvavidas|credito|deuda|millones|inyeccion|inyectara|capitalizacion)"
    )
    mask_b9 = nt.str.contains(_petroperu_loan, regex=True, na=False)
    n_b9 = int((mask_b9 & (df["economic_score"].fillna(0) < 75)).sum())
    df.loc[mask_b9 & (df["economic_score"].fillna(0) < 75), "economic_score"] = 75
    if n_b9 > 0:
        logger.info("post_filter_scores: B9 Petroperu loan/rescue eco>=75 on %d articles", n_b9)

    # B10. PBI / crecimiento recortado por shock → eco≥55
    _pbi_shock = (
        r"(triple choque|doble choque|choque adverso|frena|desacelera).{0,60}"
        r"(pbi|pib|crecimiento|economia peruana|peru)|"
        r"pierde.{0,30}(puntos de crecimiento|% de pbi|% del pbi)"
    )
    mask_b10 = nt.str.contains(_pbi_shock, regex=True, na=False)
    n_b10 = int((mask_b10 & (df["economic_score"].fillna(0) < 55)).sum())
    df.loc[mask_b10 & (df["economic_score"].fillna(0) < 55), "economic_score"] = 55
    if n_b10 > 0:
        logger.info("post_filter_scores: B10 PBI shock eco>=55 on %d articles", n_b10)

    # B11. Inflación acelera + monetary implications → eco≥50
    _inflation_accel = (
        r"(inflacion|inflacion).{0,60}"
        r"(acelera|sube|aumenta|dispara|presiona).{0,60}"
        r"(bcrp|banco central|tasas|dolar|tipo de cambio|bcp|bbva|scotiabank)"
    )
    mask_b11 = nt.str.contains(_inflation_accel, regex=True, na=False)
    n_b11 = int((mask_b11 & (df["economic_score"].fillna(0) < 50)).sum())
    df.loc[mask_b11 & (df["economic_score"].fillna(0) < 50), "economic_score"] = 50
    if n_b11 > 0:
        logger.info("post_filter_scores: B11 inflation acceleration eco>=50 on %d articles", n_b11)

    # B12. Paro de transporte por combustible → eco≥50
    _transport_strike = (
        r"(paro|huelga|paralizacion).{0,60}"
        r"(transporte|transportistas|camioneros|conductores).{0,60}"
        r"(combustible|gasolina|diesel|glp|precio)"
    )
    mask_b12 = nt.str.contains(_transport_strike, regex=True, na=False)
    n_b12 = int((mask_b12 & (df["economic_score"].fillna(0) < 50)).sum())
    df.loc[mask_b12 & (df["economic_score"].fillna(0) < 50), "economic_score"] = 50
    if n_b12 > 0:
        logger.info("post_filter_scores: B12 transport strike fuel eco>=50 on %d articles", n_b12)

    # B13. Multiple carreteras cortadas (supply chain impact) → eco≥45
    _roads_cut = (
        r"(carreteras|vias).{0,60}"
        r"(cortadas|bloqueadas|interrumpidas|cerradas).{0,60}"
        r"(lluvias|huaico|desborde|bloqueo).{0,60}"
        r"(decena|varias|multiples|norte|sur|sierra|selva)"
    )
    mask_b13 = nt.str.contains(_roads_cut, regex=True, na=False)
    n_b13 = int((mask_b13 & (df["economic_score"].fillna(0) < 45)).sum())
    df.loc[mask_b13 & (df["economic_score"].fillna(0) < 45), "economic_score"] = 45
    if n_b13 > 0:
        logger.info("post_filter_scores: B13 multiple roads cut eco>=45 on %d articles", n_b13)

    # B14. Mining blockage / conflict → eco≥55
    _mining_blockage = (
        r"(comunidades|pobladores|manifestantes).{0,60}"
        r"(bloquean|bloquearon|paralizan|paralizaron|toman|tomaron).{0,60}"
        r"(mina|minera|proyecto minero|operaciones)|"
        r"(mina|minera).{0,60}"
        r"(suspende|suspendio|paraliza|paralizo|detiene|detuvo).{0,60}"
        r"(operaciones|produccion)"
    )
    mask_b14 = nt.str.contains(_mining_blockage, regex=True, na=False)
    n_b14 = int((mask_b14 & (df["economic_score"].fillna(0) < 55)).sum())
    df.loc[mask_b14 & (df["economic_score"].fillna(0) < 55), "economic_score"] = 55
    if n_b14 > 0:
        logger.info("post_filter_scores: B14 mining blockage eco>=55 on %d articles", n_b14)

    # ── END BOOST RULES ──────────────────────────────────────────────────────────

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
        "camisea", "oleoducto", "refineria", "tgp", "transportadora de gas",
        "osinergmin", "minem",
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
        # Global supply chain / geopolitical energy (Ormuz, trade routes, etc.)
        "ormuz", "suministro", "cadena de suministro", "energetico",
        "precios internacionales", "petroquimico", "flete",
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

    # ── IMPROVEMENTS 4 & 5: Soft floors, Peru-specific rules ─────────────────

    # Improvement 4: Soft floors — routine bureaucratic / minor local events
    # previously zeroed get a small floor score instead of hard 0.
    # These rules apply AFTER all zeroing rules above; they can only raise from 0.
    _routine_bureaucratic = (
        r"(presidenta?|premier|ministro|congresista|gobernador|alcalde).{0,60}"
        r"(inauguró|inaugura|visita.{0,20}obra|lanza.{0,20}programa|"
        r"entrega.{0,20}(bono|kit|donación|canasta)|"
        r"suscribe.{0,20}convenio|firma.{0,20}convenio|"
        r"se reúne con.{0,40}(delegación|embajada|gremio|sector)|"
        r"supervisa.{0,40}(obra|proyecto|avance))"
    )
    mask_bureaucratic = nt.str.contains(_routine_bureaucratic, regex=True, na=False)
    # Only raise from 0 to 5 — never lower existing scores
    df.loc[mask_bureaucratic & (df["political_score"].fillna(0) == 0), "political_score"] = 5
    n_bureaucratic = int(mask_bureaucratic.sum())
    if n_bureaucratic > 0:
        logger.info("post_filter_scores: soft floor pol=5 for routine bureaucratic on %d", n_bureaucratic)

    # Improvement 5a: Presidential candidates April 2026 → pol = 20
    # if article mentions them in campaign context AND pol is currently < 15 or in (15-40).
    # Exceptions: Cerrón judicial news, Keiko judicial/corruption news, arrest keywords.
    _candidates_2026 = (
        r"\b(lopez chau|chau\b.*candidat|atencio sotomayor|acu[nñ]a\b|williams zapata|"
        r"paz de la barra|fujimori\b|molinelli\b|sanchez palomino|"
        r"belaunde llosa|valderrama\b|belmont\b|becerra garcia|nieto montesinos|"
        r"carrasco salazar|gonzales castillo|masse fernandez|forsyth\b|"
        r"olivera vega|guevara amasifuen|alvarez loayza|caller\b|lescano\b|"
        r"grozo costa|cerron rojas|diez.canseco|vizcarra cornejo|chirinos purizaga|"
        r"espa.{0,5}garces|jaico carranza|luna galvez|perez tello)\b"
    )
    _campaign_context = (
        r"\b(foro|candidato|candidata|propone|plantea|expone|debate|"
        r"propuesta electoral|plan de gobierno|campa[nñ]a|programa electoral|"
        r"inscripci[oó]n de candidatura|JNE|ONPE|segunda vuelta|primera vuelta)\b"
    )
    # Exceptions: judicial/legal crisis context takes priority
    _candidate_crisis_exc = (
        r"\b(prisi[oó]n|prisión|detenido|detenida|captura|arrest|sentencia|"
        r"condenad|condena|enjuiciad|investigad|fiscal[ií]a|penal|imputad|"
        r"formaliz|prisi[oó]n preventiva)\b"
    )
    mask_cand_2026 = (
        nt.str.contains(_candidates_2026, regex=True, na=False) &
        nt.str.contains(_campaign_context, regex=True, na=False) &
        ~nt.str.contains(_candidate_crisis_exc, regex=True, na=False)
    )
    # Apply: set pol=20 if currently < 15, or if in (15,40) but no crisis context
    mask_cand_low = mask_cand_2026 & (df["political_score"].fillna(0) < 15)
    mask_cand_mid = mask_cand_2026 & (df["political_score"].fillna(0) >= 15) & (df["political_score"].fillna(0) <= 40)
    df.loc[mask_cand_low, "political_score"] = 20
    df.loc[mask_cand_mid, "political_score"] = 20
    n_cand_2026 = int((mask_cand_low | mask_cand_mid).sum())
    if n_cand_2026 > 0:
        logger.info("post_filter_scores: 5a candidate 2026 pol=20 on %d articles", n_cand_2026)

    # Improvement 5b: Key political figures — minimum pol scores
    _antauro = r"\bantauro humala\b"
    _ollanta = r"\bollanta humala\b"
    _castillo = r"\bpedro castillo\b"
    _vizcarra_m = r"\bmart[ií]n vizcarra\b"
    _ppk = r"\b(ppk|kuczynski)\b"
    _toledo = r"\balejandro toledo\b"
    _nicanor = r"\bnicanor boluarte\b"
    _dina = r"\b(dina boluarte|presidenta boluarte)\b"
    _montesinos = r"\b(vladimiro montesinos|montesinos\b)"
    _cerron = r"\b(vladimir cerr[oó]n|cerr[oó]n rojas)\b"

    def _floor_pol(mask, floor):
        """Raise pol to floor where mask matches AND pol > 0."""
        cond = mask & (df["political_score"].fillna(0) > 0) & (df["political_score"].fillna(0) < floor)
        df.loc[cond, "political_score"] = floor
        return int(cond.sum())

    def _floor_pol_always(mask, floor):
        """Raise pol to floor where mask matches regardless of current value."""
        cond = mask & (df["political_score"].fillna(0) < floor)
        df.loc[cond, "political_score"] = floor
        return int(cond.sum())

    n5b = 0
    n5b += _floor_pol(nt.str.contains(_antauro, regex=True, na=False), 45)
    n5b += _floor_pol(nt.str.contains(_ollanta, regex=True, na=False), 45)
    # Pedro Castillo always high
    n5b += _floor_pol_always(nt.str.contains(_castillo, regex=True, na=False), 65)
    n5b += _floor_pol(nt.str.contains(_vizcarra_m, regex=True, na=False), 55)
    n5b += _floor_pol(nt.str.contains(_ppk, regex=True, na=False), 50)
    n5b += _floor_pol(nt.str.contains(_toledo, regex=True, na=False), 55)
    n5b += _floor_pol(nt.str.contains(_nicanor, regex=True, na=False), 55)
    # Dina Boluarte — always relevant (current president)
    n5b += _floor_pol_always(nt.str.contains(_dina, regex=True, na=False), 40)
    n5b += _floor_pol(nt.str.contains(_montesinos, regex=True, na=False), 50)
    # Cerrón always very high
    n5b += _floor_pol_always(nt.str.contains(_cerron, regex=True, na=False), 70)
    if n5b > 0:
        logger.info("post_filter_scores: 5b key figures pol floors applied on %d articles", n5b)

    # Improvement 5c: Presidential arrests/imprisonment → pol=82
    _pres_arrest = (
        r"(detiene|detienen|captura|capturan|arresta|arrestan|"
        r"orden de captura|prisi[oó]n preventiva|enviado a prisi[oó]n|encarcel)"
        r".{0,60}"
        r"(presidente|ex.?presidente|castillo|vizcarra|fujimori|toledo|kuczynski|ppk|boluarte|humala)|"
        r"(presidente|ex.?presidente|castillo|vizcarra|fujimori|toledo|kuczynski|ppk|boluarte|humala)"
        r".{0,60}"
        r"(detiene|detienen|captura|capturan|arresta|arrestan|"
        r"orden de captura|prisi[oó]n preventiva|enviado a prisi[oó]n|encarcel)"
    )
    mask_pres_arrest = nt.str.contains(_pres_arrest, regex=True, na=False)
    n_pa = int((mask_pres_arrest & (df["political_score"].fillna(0) < 82)).sum())
    df.loc[mask_pres_arrest & (df["political_score"].fillna(0) < 82), "political_score"] = 82
    if n_pa > 0:
        logger.info("post_filter_scores: 5c presidential arrest pol=82 on %d articles", n_pa)

    # Improvement 5d: Paros/huelgas weighted by scale (floor logic only)
    # 1. Paro/huelga nacional → eco=65, pol=55
    _paro_nacional = (
        r"(paro|huelga|paralizaci[oó]n).{0,30}(nacional|a nivel nacional|en todo el pa[ií]s|en todo peru)|"
        r"(paro nacional|huelga nacional|paralizaci[oó]n nacional)"
    )
    mask_pn = nt.str.contains(_paro_nacional, regex=True, na=False)
    df.loc[mask_pn & (df["economic_score"].fillna(0) < 65), "economic_score"] = 65
    df.loc[mask_pn & (df["political_score"].fillna(0) < 55), "political_score"] = 55
    n_pn = int(mask_pn.sum())
    if n_pn > 0:
        logger.info("post_filter_scores: 5d paro nacional eco>=65 pol>=55 on %d", n_pn)

    # 2. Huelga indefinida (key sectors) → eco=60, pol=45
    _huelga_indef = (
        r"huelga indefinida.{0,60}"
        r"(maestros|docentes|m[eé]dicos|enfermeros|trabajadores judiciales|"
        r"poder judicial|empleados p[uú]blicos|servidores civiles)"
    )
    mask_hi = nt.str.contains(_huelga_indef, regex=True, na=False) & ~mask_pn
    df.loc[mask_hi & (df["economic_score"].fillna(0) < 60), "economic_score"] = 60
    df.loc[mask_hi & (df["political_score"].fillna(0) < 45), "political_score"] = 45
    n_hi = int(mask_hi.sum())
    if n_hi > 0:
        logger.info("post_filter_scores: 5d huelga indefinida eco>=60 pol>=45 on %d", n_hi)

    # 3. Paro regional → eco=55, pol=35
    _paro_regional = (
        r"paro.{0,40}(regi[oó]n|regional|arequipa|puno|cusco|loreto|piura|cajamarca|"
        r"jun[ií]n|ayacucho|huancavelica|apur[ií]mac|madre de dios|san mart[ií]n|"
        r"[aá]ncash|la libertad|lambayeque|ica|tacna|moquegua|tumbes|amazonas|"
        r"ucayali|hu[aá]nuco|lima metropolitana|lima provincias|callao|pasco)"
    )
    mask_pr3 = nt.str.contains(_paro_regional, regex=True, na=False) & ~mask_pn & ~mask_hi
    df.loc[mask_pr3 & (df["economic_score"].fillna(0) < 55), "economic_score"] = 55
    df.loc[mask_pr3 & (df["political_score"].fillna(0) < 35), "political_score"] = 35
    n_pr3 = int(mask_pr3.sum())
    if n_pr3 > 0:
        logger.info("post_filter_scores: 5d paro regional eco>=55 pol>=35 on %d", n_pr3)

    # 4. Huelga sectorial / paro gremial → eco=55, pol=25
    _huelga_sectorial = (
        r"(huelga|paro).{0,50}"
        r"(minero|miner[ií]a|transporte|transportistas|camioneros|agrario|pesquero|"
        r"construcci[oó]n civil|bancario|docente|m[eé]dico|enfermero)"
    )
    mask_hs = nt.str.contains(_huelga_sectorial, regex=True, na=False) & ~mask_pn & ~mask_hi & ~mask_pr3
    df.loc[mask_hs & (df["economic_score"].fillna(0) < 55), "economic_score"] = 55
    df.loc[mask_hs & (df["political_score"].fillna(0) < 25), "political_score"] = 25
    n_hs = int(mask_hs.sum())
    if n_hs > 0:
        logger.info("post_filter_scores: 5d huelga sectorial eco>=55 pol>=25 on %d", n_hs)

    # 5. Paralización local → eco=35, pol=15
    _paraliz_local = (
        r"(paralizaci[oó]n|paraliza).{0,40}"
        r"(provincia|distrito|ciudad|localidad|comunidad|zona)"
    )
    mask_pl = nt.str.contains(_paraliz_local, regex=True, na=False) & ~mask_pn & ~mask_hi & ~mask_pr3 & ~mask_hs
    df.loc[mask_pl & (df["economic_score"].fillna(0) < 35), "economic_score"] = 35
    df.loc[mask_pl & (df["political_score"].fillna(0) < 15), "political_score"] = 15
    n_pl = int(mask_pl.sum())
    if n_pl > 0:
        logger.info("post_filter_scores: 5d paralización local eco>=35 pol>=15 on %d", n_pl)

    # Improvement 5e: Additional crisis patterns

    # Narcoestado / crimen organizado en política → pol=75
    _narco_estado = (
        r"(narco|narcotr[aá]fico|cartel|crimen organizado|organizaci[oó]n criminal)"
        r".{0,60}"
        r"(congresista|ministro|gobierno|estado peruano|funcionario|partido|"
        r"candidato|alcalde|gobernador)"
    )
    mask_narco = nt.str.contains(_narco_estado, regex=True, na=False)
    df.loc[mask_narco & (df["political_score"].fillna(0) < 75), "political_score"] = 75
    n_narco = int(mask_narco.sum())
    if n_narco > 0:
        logger.info("post_filter_scores: 5e narcoestado pol>=75 on %d", n_narco)

    # Conflictos mineros específicos → eco=62, pol=50
    _mining_conflict_specific = (
        r"(las bambas|t[ií]a mar[ií]a|cuajone|antamina|quellaveco|toromocho|"
        r"cerro verde|constancia)"
        r".{0,80}"
        r"(bloqueo|protesta|paraliz|conflicto|comunidad|enfrentamiento|herido|muerto)"
    )
    mask_mcs = nt.str.contains(_mining_conflict_specific, regex=True, na=False)
    df.loc[mask_mcs & (df["economic_score"].fillna(0) < 62), "economic_score"] = 62
    df.loc[mask_mcs & (df["political_score"].fillna(0) < 50), "political_score"] = 50
    n_mcs = int(mask_mcs.sum())
    if n_mcs > 0:
        logger.info("post_filter_scores: 5e mining conflict specific eco>=62 pol>=50 on %d", n_mcs)

    # Escasez / desabastecimiento → eco=65
    _escasez = (
        r"(escasez|desabastecimiento|desabastecen|falta de|sin stock)"
        r".{0,40}"
        r"(alimentos|combustible|gas|medicamentos|ox[ií]geno|agua potable|"
        r"productos b[aá]sicos|canasta)"
    )
    mask_esc = nt.str.contains(_escasez, regex=True, na=False)
    df.loc[mask_esc & (df["economic_score"].fillna(0) < 65), "economic_score"] = 65
    n_esc = int(mask_esc.sum())
    if n_esc > 0:
        logger.info("post_filter_scores: 5e escasez eco>=65 on %d", n_esc)

    # Inversión cancelada por conflicto social → eco=58
    _inversion_cancel = (
        r"(cancela|cancel[oó]|suspende|suspendi[oó]|paraliza|paraliz[oó]|retira|retir[oó])"
        r".{0,60}"
        r"(inversi[oó]n|proyecto minero|proyecto de inversi[oó]n|concesi[oó]n)"
        r".{0,60}"
        r"(conflicto|protesta|comunidad|bloqueo|violencia)"
    )
    mask_ic = nt.str.contains(_inversion_cancel, regex=True, na=False)
    df.loc[mask_ic & (df["economic_score"].fillna(0) < 58), "economic_score"] = 58
    n_ic = int(mask_ic.sum())
    if n_ic > 0:
        logger.info("post_filter_scores: 5e inversión cancelada eco>=58 on %d", n_ic)

    # Corrupción/judicialización de funcionarios → pol=62
    _corruption_official = (
        r"(detienen|capturan|formalizan|sentencian|condenan|prisi[oó]n preventiva)"
        r".{0,50}"
        r"(ministro|viceministro|congresista|gobernador regional|alcalde|"
        r"director|funcionario p[uú]blico|asesor del gobierno)"
    )
    mask_co = nt.str.contains(_corruption_official, regex=True, na=False)
    df.loc[mask_co & (df["political_score"].fillna(0) < 62), "political_score"] = 62
    n_co = int(mask_co.sum())
    if n_co > 0:
        logger.info("post_filter_scores: 5e corruption official pol>=62 on %d", n_co)

    # ── END IMPROVEMENTS 4 & 5 ───────────────────────────────────────────────

    # ── NEW RULES (GPR redesign, March 2026) ──────────────────────────────────

    # 3f. Conflicto Ejecutivo vs Congreso → pol ≥ 60
    # Explicit confrontation between branches of government is a political crisis signal.
    _exec_congress_conflict = (
        r"(ejecutivo|gobierno|presidenta?|premier).{0,60}"
        r"(enfrenta|confronta|choca|tensión|pugna|conflicto|disputa).{0,60}"
        r"(congreso|poder legislativo|parlamento)|"
        r"(congreso|poder legislativo|parlamento).{0,60}"
        r"(enfrenta|confronta|choca|tensión|pugna|conflicto|disputa).{0,60}"
        r"(ejecutivo|gobierno|presidenta?|premier)|"
        r"(crisis|ruptura|fractura).{0,30}(entre el ejecutivo y el congreso|"
        r"entre el gobierno y el congreso|entre los poderes del estado)|"
        r"(poderes del estado|ejecutivo y legislativo).{0,40}"
        r"(crisis|conflicto|ruptura|enfrentamiento|pugna)"
    )
    mask_ecc = nt.str.contains(_exec_congress_conflict, regex=True, na=False)
    n_ecc = int((mask_ecc & (df["political_score"].fillna(0) < 60)).sum())
    df.loc[mask_ecc & (df["political_score"].fillna(0) < 60), "political_score"] = 60
    if n_ecc > 0:
        logger.info("post_filter_scores: Ejecutivo vs Congreso conflict pol>=60 on %d articles", n_ecc)

    # 3h. Eco soft floor for routine bureaucratic news → eco=5
    # The routine bureaucratic rule already sets pol=5. Apply a symmetric eco=5
    # for the same pattern (routine gov activity has minimal economic signal too).
    mask_eco_bureaucratic = nt.str.contains(_routine_bureaucratic, regex=True, na=False)
    n_eco_bur = int((mask_eco_bureaucratic & (df["economic_score"].fillna(0) == 0)).sum())
    df.loc[mask_eco_bureaucratic & (df["economic_score"].fillna(0) == 0), "economic_score"] = 5
    if n_eco_bur > 0:
        logger.info("post_filter_scores: eco soft floor=5 for routine bureaucratic on %d articles", n_eco_bur)

    # 3h. Minor local government news → pol = 10 (not 0)
    # Local council decisions, municipal ordinances, district-level governance
    # are weakly relevant political signals — not zero.
    _minor_local_gov = (
        r"(municipalidad|alcald[ií]a|municipio|consejo municipal|"
        r"regidor[ea]?|alcalde|alcaldesa).{0,60}"
        r"(acuerdo|ordenanza|sesión|aprobó|aprueba|debate|propone|"
        r"inaugura|acto oficial|comisión)\b"
        r"(?!.{0,30}(huelga|protesta|crisis|denuncia|corrupci[oó]n|"
        r"destituid|suspendid|arresta|captura))"
    )
    mask_mlg = nt.str.contains(_minor_local_gov, regex=True, na=False)
    n_mlg = int((mask_mlg & (df["political_score"].fillna(0) == 0)).sum())
    df.loc[mask_mlg & (df["political_score"].fillna(0) == 0), "political_score"] = 10
    if n_mlg > 0:
        logger.info("post_filter_scores: minor local gov pol=10 on %d articles", n_mlg)

    # ── FINAL OVERRIDE: re-zero foreign domestic articles ─────────────────────
    # Some boost rules later in this function (e.g. arrest/removal patterns) can
    # re-boost articles that were already zeroed by mask_foreign_dom.
    # Re-apply the foreign domestic mask unconditionally at the end.
    mask_fd_recheck = nt.str.contains(_foreign_domestic, regex=True, na=False) & ~mask_crisis
    n_fd_recheck = int((mask_fd_recheck & (
        (df["political_score"].fillna(0) > 0) | (df["economic_score"].fillna(0) > 0)
    )).sum())
    if n_fd_recheck > 0:
        df.loc[mask_fd_recheck, "political_score"] = 0
        df.loc[mask_fd_recheck, "economic_score"] = 0
        logger.info("post_filter_scores: FD re-zero override cleared %d articles", n_fd_recheck)
    # ── END NEW RULES ─────────────────────────────────────────────────────────

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

    # 25. NIMBY / neighborhood disputes (crematorio, antenas, etc.) → eco=0
    # Note: no crisis exception — "vecinos protestan" triggers the crisis guard's 'protesta'
    # token, but local NIMBY disputes are never genuine economic crises.
    _nimby_inline = (
        r"vecinos?.{0,40}(protestan?|rechazan?|se oponen?|en contra de|impiden?).{0,60}"
        r"(crematorio|antena|relleno sanitario|planta de tratamiento|cementerio)|"
        r"(crematorio|relleno sanitario|planta de tratamiento).{0,60}"
        r"(vecinos?|protesta|rechazo|oposici[oó]n|municipalidad)|"
        r"controversia.{0,20}(surco|barranco|miraflores|san isidro|san juan).{0,60}"
        r"(crematorio|relleno|antena|construccion)"
    )
    mask_nimby_inline = titles.str.contains(_nimby_inline, regex=True, na=False)
    df.loc[mask_nimby_inline, "economic_score"] = 0

    # 26. Candidate security/prison proposals without fiscal specifics → cap eco at 15
    # Political proposals to use military for prison control have no direct economic disruption.
    _cand_sec_prop_inline = (
        r"(candidato|candidata|propone|plantea).{0,60}"
        r"(fuerzas armadas|ejercito|marina|militares?|ffaa).{0,60}"
        r"(penales?|prisiones?|carcel|reclusos?|penitenciar|seguridad\b)|"
        r"(penales?|prisiones?|carcel).{0,60}"
        r"(fuerzas armadas|ejercito|marina|militares?|ffaa).{0,60}"
        r"(propone|plantea|candidato|candidata)"
    )
    mask_csp_inline = (
        titles.str.contains(_cand_sec_prop_inline, regex=True, na=False) &
        ~mask_crisis_exception &
        (df["economic_score"].fillna(0) > 15)
    )
    df.loc[mask_csp_inline, "economic_score"] = 15

    # 27. Government fiscal injection into state company → eco≥80
    # "Gobierno inyectará S/500 millones a Petroperú" is a high fiscal risk event.
    _fiscal_injection_inline = (
        r"(inyectar[aá]|inyect[oó]|inyecci[oó]n).{0,80}"
        r"(petroper[uú]|empresa estatal|empresa p[uú]blica|petrolera estatal).{0,80}"
        r"(s[\\/]\s*\d|millones|mil millones)|"
        r"(petroper[uú]|empresa estatal|empresa p[uú]blica|petrolera estatal).{0,80}"
        r"(inyectar[aá]|inyect[oó]|inyecci[oó]n).{0,80}"
        r"(s[\\/]\s*\d|millones|mil millones)|"
        r"(gobierno|estado).{0,40}(inyectar[aá]|inyect[oó]).{0,40}"
        r"(s[\\/]\s*\d{3,}|[0-9]{3,}.*millones).{0,40}"
        r"(petroper[uú]|petrolera estatal|empresa estatal)"
    )
    mask_fii = (
        titles.str.contains(_fiscal_injection_inline, regex=True, na=False) &
        (df["economic_score"].fillna(0) < 80)
    )
    df.loc[mask_fii, "economic_score"] = 80

    # 28. Urban individual crime incidents (sicario/shooting targeting individuals) → cap eco at 35
    # Single criminal acts (combi shootings, targeted killings) signal insecurity but are not
    # macroeconomic disruption events. Cap at 35, not zero (there IS some economic risk signal).
    _urban_crime_incident = (
        r"\b(sicario|pistolero|asesino a sueldo).{0,60}"
        r"(dispara|balea|ataca|mata|hiere|ejecuta).{0,60}"
        r"(combi|bus\b|mototaxi|taxista|chofer|pasajero|comerciante|vecino)|"
        r"\b(dispara|balea|ataca).{0,30}(combi|mototaxi|taxi\b|bus\b).{0,60}"
        r"(mata|hiere|muere|muerto|herido|pasajero|chofer)"
    )
    mask_uci = (
        titles.str.contains(_urban_crime_incident, regex=True, na=False) &
        ~mask_crisis_exception &
        (df["economic_score"].fillna(0) > 35)
    )
    df.loc[mask_uci, "economic_score"] = 35

    # 29. F1: Foreign natural disasters (no Peru connection) → eco=0, pol=0
    _foreign_disaster_inline = (
        r"(india|indonesia|bangladesh|pakistan|filipinas|vietnam|myanmar).{0,60}"
        r"(tornado|ciclon|terremoto|inundacion|tsunami|huracan|tifon)|"
        r"(tornado|ciclon|terremoto|inundacion|tsunami|huracan|tifon).{0,60}"
        r"(india|indonesia|bangladesh|pakistan|filipinas|vietnam|myanmar)"
    )
    _has_peru_inline = r"peru|peruana?|peruanos?"
    mask_f1_inline = (
        titles.str.contains(_foreign_disaster_inline, regex=True, na=False) &
        ~titles.str.contains(_has_peru_inline, regex=True, na=False)
    )
    df.loc[mask_f1_inline, "economic_score"] = 0
    df.loc[mask_f1_inline, "political_score"] = 0

    # 30. F2: Gadgets / consumer medical tech → pol=0
    _gadget_health_inline = (
        r"(pastillero|wearable|smartwatch|aplicacion movil|app para|dispositivo (para|que)|gadget).{0,60}"
        r"(salud|tratamiento|medicamento|dieta|ejercicio)"
    )
    mask_f2_inline = titles.str.contains(_gadget_health_inline, regex=True, na=False)
    df.loc[mask_f2_inline, "political_score"] = 0

    # 31. F3: Routine weather alerts (no emergency language) → pol=0
    _routine_alert_inline = (
        r"(alerta amarilla|alerta verde|alerta naranja).{0,60}"
        r"(vientos|lluvia|frio|helada|neblina)"
    )
    _emergency_lang_inline = r"estado de emergencia|evacuacion masiva|decenas de muertos|colapso"
    mask_f3_inline = (
        titles.str.contains(_routine_alert_inline, regex=True, na=False) &
        ~titles.str.contains(_emergency_lang_inline, regex=True, na=False)
    )
    df.loc[mask_f3_inline, "political_score"] = 0

    # 32. F4: Traffic fines on candidates → cap pol at 20
    _candidate_fines_inline = (
        r"(candidatos?|candidato|candidata).{0,60}"
        r"(multa|multas|papeleta|infraccion|deuda de transito)"
    )
    mask_f4_inline = titles.str.contains(_candidate_fines_inline, regex=True, na=False)
    df.loc[mask_f4_inline & (df["political_score"].fillna(0) > 20), "political_score"] = 20

    # 33. F5: Foreign military/geopolitics without Peru impact → pol=0, eco≤15
    _foreign_mil_inline = (
        r"(otan|nato|trump|estados unidos|eeuu|rusia|china|israel|gaza|ucrania).{0,60}"
        r"(necesita|aliados|tropas|misiles|acuerdo|ejercicios militares|ataque)"
    )
    _has_peru_f5_inline = r"peru|peruanos?|exportaciones peruanas|impacto en peru|afecta.{0,10}peru"
    mask_f5_inline = (
        titles.str.contains(_foreign_mil_inline, regex=True, na=False) &
        ~titles.str.contains(_has_peru_f5_inline, regex=True, na=False)
    )
    df.loc[mask_f5_inline, "political_score"] = 0
    df.loc[mask_f5_inline & (df["economic_score"].fillna(0) > 15), "economic_score"] = 15

    # 34. B1: New cabinet / toma de juramento → pol≥65
    _toma_juramento_inline = (
        r"toma de juramento.{0,60}(gabinete|premier|primer ministro)|"
        r"nuevo gabinete.{0,60}(jura|jurament|boluarte|presidenta?|premier)|"
        r"(boluarte|presidenta?|premier).{0,60}nuevo gabinete"
    )
    mask_b1_inline = titles.str.contains(_toma_juramento_inline, regex=True, na=False)
    df.loc[mask_b1_inline & (df["political_score"].fillna(0) < 65), "political_score"] = 65

    # 35. B2: Voto de confianza (any mention in title) → pol≥70
    mask_b2_inline = titles.str.contains(r"voto de confianza", regex=True, na=False)
    df.loc[mask_b2_inline & (df["political_score"].fillna(0) < 70), "political_score"] = 70

    # 36. B3: Assassination of political/public figure → pol≥70, eco≥20
    _assassination_inline = (
        r"(asesinato|asesinaron|matan|ejecutan).{0,60}"
        r"(congresista|asesora|asesor|regidor|alcalde|gobernador|fiscal|juez|"
        r"periodista|candidato|funcionario|andrea vidal)"
    )
    mask_b3_inline = titles.str.contains(_assassination_inline, regex=True, na=False)
    df.loc[mask_b3_inline & (df["political_score"].fillna(0) < 70), "political_score"] = 70
    df.loc[mask_b3_inline & (df["economic_score"].fillna(0) < 20), "economic_score"] = 20

    # 37. B4: Corruption arrests of public officials → pol≥60
    _corrupt_arrest_inline = (
        r"(detienen|capturan|arrestan|allanamiento|allanaron|incautan|incautaron).{0,60}"
        r"(congresista|ministro|gobernador|alcalde|fiscal|juez|funcionario)"
    )
    _corrupt_sports_exc_inline = r"(jugador|deportista|futbolista|cantante|artista)"
    mask_b4_inline = (
        titles.str.contains(_corrupt_arrest_inline, regex=True, na=False) &
        ~titles.str.contains(_corrupt_sports_exc_inline, regex=True, na=False)
    )
    df.loc[mask_b4_inline & (df["political_score"].fillna(0) < 60), "political_score"] = 60

    # 38. B5: Destitución / suspensión / remoción of official → pol≥60
    _destitucion_inline = (
        r"(destituy|destituido|destituida|suspendido|suspendida|removido|removida|"
        r"inhabilitado|inhabilitada).{0,60}"
        r"(ministro|fiscal|juez|congresista|gobernador|alcalde|funcionario|director)"
    )
    mask_b5_inline = titles.str.contains(_destitucion_inline, regex=True, na=False)
    df.loc[mask_b5_inline & (df["political_score"].fillna(0) < 60), "political_score"] = 60

    # 39. B6: Campaign foro / candidatos presidenciales → cap pol≤40, eco≤10
    _campaign_foro_inline = (
        r"(foro de candidatos|foro electoral|candidatos presidenciales|candidatos al congreso).{0,60}"
        r"(propone|plantea|expone|expondran|expuso|debate|debatr|presentan|presentaran)"
    )
    mask_b6_inline = titles.str.contains(_campaign_foro_inline, regex=True, na=False)
    df.loc[mask_b6_inline & (df["political_score"].fillna(0) > 40), "political_score"] = 40
    df.loc[mask_b6_inline & (df["economic_score"].fillna(0) > 10), "economic_score"] = 10

    # 40. B7: Camisea / gas natural disruption → eco≥80
    _camisea_disruption_inline = (
        r"(camisea|gasoducto|gas natural).{0,60}"
        r"(paraliza|paralizado|corte|cortado|ruptura|rotura|sin gas|dias sin gas|"
        r"semanas sin gas|interrupcion|emergencia|colapso)"
    )
    mask_b7_inline = titles.str.contains(_camisea_disruption_inline, regex=True, na=False)
    df.loc[mask_b7_inline & (df["economic_score"].fillna(0) < 80), "economic_score"] = 80

    # 41. B8: Gasto público / deuda aprobada por Congreso → eco≥60
    _fiscal_congress_inline = (
        r"(congreso aprueba|congreso aprobo).{0,60}"
        r"(gasto|deuda|prestamo|credito suplementario|modificacion presupuestal|endeudamiento)|"
        r"(hipotecara|deficit fiscal|deuda publica).{0,60}(finanzas|presupuesto|estado)|"
        r"consejo fiscal.{0,60}(advierte|alerta|hipotecara|insostenible|riesgo fiscal)"
    )
    mask_b8_inline = titles.str.contains(_fiscal_congress_inline, regex=True, na=False)
    df.loc[mask_b8_inline & (df["economic_score"].fillna(0) < 60), "economic_score"] = 60

    # 42. B9: Petroperú préstamo / rescate → eco≥75
    _petroperu_loan_inline = (
        r"(petroperu|petroper).{0,60}"
        r"(prestamo|rescate|salvavidas|credito|deuda|millones|inyeccion|inyectara|capitalizacion)"
    )
    mask_b9_inline = titles.str.contains(_petroperu_loan_inline, regex=True, na=False)
    df.loc[mask_b9_inline & (df["economic_score"].fillna(0) < 75), "economic_score"] = 75

    # 43. B10: PBI / crecimiento recortado por shock → eco≥55
    _pbi_shock_inline = (
        r"(triple choque|doble choque|choque adverso|frena|desacelera).{0,60}"
        r"(pbi|pib|crecimiento|economia peruana|peru)|"
        r"pierde.{0,30}(puntos de crecimiento|% de pbi|% del pbi)"
    )
    mask_b10_inline = titles.str.contains(_pbi_shock_inline, regex=True, na=False)
    df.loc[mask_b10_inline & (df["economic_score"].fillna(0) < 55), "economic_score"] = 55

    # 44. B11: Inflación acelera + monetary implications → eco≥50
    _inflation_accel_inline = (
        r"(inflacion|inflaci[oó]n).{0,60}"
        r"(acelera|sube|aumenta|dispara|presiona).{0,60}"
        r"(bcrp|banco central|tasas|dolar|tipo de cambio|bcp|bbva|scotiabank)"
    )
    mask_b11_inline = titles.str.contains(_inflation_accel_inline, regex=True, na=False)
    df.loc[mask_b11_inline & (df["economic_score"].fillna(0) < 50), "economic_score"] = 50

    # 45. B12: Paro de transporte por combustible → eco≥50
    _transport_strike_inline = (
        r"(paro|huelga|paralizacion).{0,60}"
        r"(transporte|transportistas|camioneros|conductores).{0,60}"
        r"(combustible|gasolina|diesel|glp|precio)"
    )
    mask_b12_inline = titles.str.contains(_transport_strike_inline, regex=True, na=False)
    df.loc[mask_b12_inline & (df["economic_score"].fillna(0) < 50), "economic_score"] = 50

    # 46. B13: Multiple carreteras cortadas (supply chain impact) → eco≥45
    _roads_cut_inline = (
        r"(carreteras|vias).{0,60}"
        r"(cortadas|bloqueadas|interrumpidas|cerradas).{0,60}"
        r"(lluvias|huaico|desborde|bloqueo).{0,60}"
        r"(decena|varias|multiples|norte|sur|sierra|selva)"
    )
    mask_b13_inline = titles.str.contains(_roads_cut_inline, regex=True, na=False)
    df.loc[mask_b13_inline & (df["economic_score"].fillna(0) < 45), "economic_score"] = 45

    # 47. B14: Mining blockage / conflict → eco≥55
    _mining_blockage_inline = (
        r"(comunidades|pobladores|manifestantes).{0,60}"
        r"(bloquean|bloquearon|paralizan|paralizaron|toman|tomaron).{0,60}"
        r"(mina|minera|proyecto minero|operaciones)|"
        r"(mina|minera).{0,60}"
        r"(suspende|suspendio|paraliza|paralizo|detiene|detuvo).{0,60}"
        r"(operaciones|produccion)"
    )
    mask_b14_inline = titles.str.contains(_mining_blockage_inline, regex=True, na=False)
    df.loc[mask_b14_inline & (df["economic_score"].fillna(0) < 55), "economic_score"] = 55

    # 48. Traffic accidents / vehicle crashes → pol=0, eco=0
    # Tráiler vuelca, choque vehicular, colisión, etc. are not political/economic events.
    # Exception: if the title also mentions paro/bloqueo/huelga (could be a blockade context).
    _traffic_accident_inline = (
        r"tr[aá]iler.{0,60}(vuelca|volc[oó]|choca|crash|accidente)|"
        r"(vuelca|volc[oó]).{0,60}tr[aá]iler|"
        r"(choque|colisi[oó]n|chocaron|colisionaron).{0,60}"
        r"(v[eé]hiculos?|autos?|camion|bus\b|camioneta|moto\b|tren)|"
        r"(v[eé]hiculos?|autos?|camion|bus\b|camioneta).{0,60}"
        r"(chocaron|colisionaron|se impact|impact[oó])"
    )
    _traffic_blockade_exc = r"(paro|huelga|bloqueo|bloquearon|manifestantes|protesta)"
    mask_traffic_acc_inline = (
        titles.str.contains(_traffic_accident_inline, regex=True, na=False) &
        ~titles.str.contains(_traffic_blockade_exc, regex=True, na=False)
    )
    df.loc[mask_traffic_acc_inline, "political_score"] = 0
    df.loc[mask_traffic_acc_inline, "economic_score"] = 0

    # 49. Traffic congestion / routine road incidents → pol=0, eco=0
    # Congestión vehicular, colocación de conos, cierre de vías por obra, etc.
    _traffic_congestion_inline = (
        r"congesti[oó]n vehicular.{0,60}(conos|obra|cierre|desvio|av\.|avenida)|"
        r"(conos|desvio|cierre de via).{0,60}congesti[oó]n vehicular|"
        r"conductores?.{0,40}reportan?.{0,40}(congesti[oó]n|tr[aá]fico)|"
        r"(colocaci[oó]n de conos|conos de seguridad).{0,60}"
        r"(avenida|av\.|via|congesti[oó]n|tr[aá]fico)|"
        r"tr[aá]fico.{0,40}(lento|pesado|caos|colapso).{0,40}(avenida|av\.|carretera|via)"
    )
    mask_traffic_cong_inline = (
        titles.str.contains(_traffic_congestion_inline, regex=True, na=False) &
        ~titles.str.contains(_traffic_blockade_exc, regex=True, na=False)
    )
    df.loc[mask_traffic_cong_inline, "political_score"] = 0
    df.loc[mask_traffic_cong_inline, "economic_score"] = 0

    # 50. F8: Lifestyle workshops (talleres gratuitos a vecinos) → pol=0, eco=0
    _workshop_inline = (
        r"talleres? gratuitos?.{0,60}(vecinos?|beneficiar|comunidad)|"
        r"(vecinos?|comunidad).{0,60}talleres? gratuitos?"
    )
    mask_workshop_inline = titles.str.contains(_workshop_inline, regex=True, na=False)
    df.loc[mask_workshop_inline, "political_score"] = 0
    df.loc[mask_workshop_inline, "economic_score"] = 0

    # 51. F9: US consumer lifestyle (compras impulsivas, estadounidenses) → pol=0, eco=0
    _us_consumer_inline = (
        r"compras? impulsivas?.{0,80}(estadounidenses?|americanos?)|"
        r"lista de compras.{0,60}(vacian?|estadounidenses?)"
    )
    _has_peru_inline2 = r"peru|peruana?|peruanos?"
    mask_us_consumer_inline = (
        titles.str.contains(_us_consumer_inline, regex=True, na=False) &
        ~titles.str.contains(_has_peru_inline2, regex=True, na=False)
    )
    df.loc[mask_us_consumer_inline, "political_score"] = 0
    df.loc[mask_us_consumer_inline, "economic_score"] = 0

    # 52. F10: Foreign electoral (votantes latinos, Trump) → pol=0
    _foreign_elect_inline = (
        r"votantes? latinos?.{0,60}(trump|biden|harris|eeuu|estados unidos)|"
        r"(trump|biden|harris).{0,60}votantes? latinos?"
    )
    _has_peru_inline3 = r"peru|peruanos?"
    mask_foreign_elect_inline = (
        titles.str.contains(_foreign_elect_inline, regex=True, na=False) &
        ~titles.str.contains(_has_peru_inline3, regex=True, na=False)
    )
    df.loc[mask_foreign_elect_inline, "political_score"] = 0
    df.loc[mask_foreign_elect_inline & (df["economic_score"].fillna(0) > 15), "economic_score"] = 15

    # 53. F11: Data privacy non-govt (comercialización ilegal de datos) → pol≤15
    _data_priv_inline = r"comercializaci[oó]n ilegal de datos|venta de datos (personales|privados)"
    _govt_exc_inline = r"gobierno|estado|ministerio|congres|sunat|reniec"
    mask_data_priv_inline = (
        titles.str.contains(_data_priv_inline, regex=True, na=False) &
        ~titles.str.contains(_govt_exc_inline, regex=True, na=False)
    )
    df.loc[mask_data_priv_inline & (df["political_score"].fillna(0) > 15), "political_score"] = 15
    df.loc[mask_data_priv_inline, "economic_score"] = 0

    masks = [mask_farandula, mask_sports, mask_fx_routine, mask_sismo, mask_weather,
             mask_inhab, mask_pol_action, mask_electoral, mask_mercados, mask_gastro,
             mask_horoscope, mask_lottery, mask_reality, mask_lifestyle, mask_personal_fin,
             mask_foreign_markets, mask_foreign_econ, mask_market_summary,
             mask_corp_earnings, mask_sports_biz, mask_candidacy, mask_consumer_info,
             mask_celeb, mask_foreign_weather, mask_nimby_inline,
             mask_f1_inline, mask_f2_inline, mask_f3_inline, mask_f4_inline, mask_f5_inline,
             mask_b1_inline, mask_b2_inline, mask_b3_inline, mask_b4_inline, mask_b5_inline,
             mask_b6_inline, mask_b7_inline, mask_b8_inline, mask_b9_inline, mask_b10_inline,
             mask_b11_inline, mask_b12_inline, mask_b13_inline, mask_b14_inline,
             mask_traffic_acc_inline, mask_traffic_cong_inline,
             mask_workshop_inline, mask_us_consumer_inline, mask_foreign_elect_inline, mask_data_priv_inline]
    labels = ["farándula", "sports", "fx_routine", "sismo", "weather",
              "inhabilitacion", "pol_action", "electoral", "mercados", "gastro",
              "horoscope", "lottery", "reality", "lifestyle_tips", "personal_finance",
              "foreign_markets", "foreign_econ", "market_summary",
              "corp_earnings", "sports_biz", "candidacy", "consumer_info", "celeb_gossip",
              "foreign_weather", "nimby_dispute",
              "F1_foreign_disaster", "F2_gadget_health", "F3_routine_alert", "F4_candidate_fines", "F5_foreign_mil",
              "B1_new_cabinet", "B2_confianza_riesgo", "B3_assassination", "B4_corrupt_arrest", "B5_destitucion",
              "B6_campaign_foro", "B7_camisea", "B8_fiscal_congress", "B9_petroperu_loan", "B10_pbi_shock",
              "B11_inflation", "B12_transport_strike", "B13_roads_cut", "B14_mining_blockage",
              "traffic_accident", "traffic_congestion",
              "workshop_fp", "us_consumer_fp", "foreign_electoral_fp", "data_privacy_fp"]
    n_filtered = masks[0].copy()
    for m in masks[1:]:
        n_filtered = n_filtered | m
    if n_filtered.sum() > 0:
        counts = ", ".join(f"{lbl}={m.sum()}" for lbl, m in zip(labels, masks) if m.sum() > 0)
        logger.info("Post-filter applied to %d articles (%s)", n_filtered.sum(), counts)

    # ── IMPROVEMENTS 4 & 5 inline (same rules as post_filter_scores) ─────────
    # Use accent-stripped titles (nt_inline) for consistent matching.
    nt_inline = df["title"].fillna("").apply(_normalize_title)

    # 5b: Key political figures — minimum pol scores
    def _inline_floor_pol(mask_ser, floor):
        cond = mask_ser & (df["political_score"].fillna(0) > 0) & (df["political_score"].fillna(0) < floor)
        df.loc[cond, "political_score"] = floor

    def _inline_floor_pol_always(mask_ser, floor):
        cond = mask_ser & (df["political_score"].fillna(0) < floor)
        df.loc[cond, "political_score"] = floor

    _inline_floor_pol(nt_inline.str.contains(r"\bantauro humala\b", regex=True, na=False), 45)
    _inline_floor_pol(nt_inline.str.contains(r"\bollanta humala\b", regex=True, na=False), 45)
    _inline_floor_pol_always(nt_inline.str.contains(r"\bpedro castillo\b", regex=True, na=False), 65)
    _inline_floor_pol(nt_inline.str.contains(r"\bmart[ií]n vizcarra\b", regex=True, na=False), 55)
    _inline_floor_pol(nt_inline.str.contains(r"\b(ppk|kuczynski)\b", regex=True, na=False), 50)
    _inline_floor_pol(nt_inline.str.contains(r"\balejandro toledo\b", regex=True, na=False), 55)
    _inline_floor_pol(nt_inline.str.contains(r"\bnicanor boluarte\b", regex=True, na=False), 55)
    _inline_floor_pol_always(nt_inline.str.contains(r"\b(dina boluarte|presidenta boluarte)\b", regex=True, na=False), 40)
    _inline_floor_pol(nt_inline.str.contains(r"\b(vladimiro montesinos|montesinos\b)", regex=True, na=False), 50)
    _inline_floor_pol_always(nt_inline.str.contains(r"\b(vladimir cerr[oó]n|cerr[oó]n rojas)\b", regex=True, na=False), 70)

    # 5c: Presidential arrests → pol=82
    _pres_arrest_inline = (
        r"(detiene|detienen|captura|capturan|arresta|arrestan|"
        r"orden de captura|prision preventiva|enviado a prision|encarcel)"
        r".{0,60}"
        r"(presidente|ex.?presidente|castillo|vizcarra|fujimori|toledo|kuczynski|ppk|boluarte|humala)|"
        r"(presidente|ex.?presidente|castillo|vizcarra|fujimori|toledo|kuczynski|ppk|boluarte|humala)"
        r".{0,60}"
        r"(detiene|detienen|captura|capturan|arresta|arrestan|"
        r"orden de captura|prision preventiva|enviado a prision|encarcel)"
    )
    mask_pa_i = nt_inline.str.contains(_pres_arrest_inline, regex=True, na=False)
    df.loc[mask_pa_i & (df["political_score"].fillna(0) < 82), "political_score"] = 82

    # 5d: Paros/huelgas
    _paro_nac_i = (
        r"(paro|huelga|paralizacion).{0,30}(nacional|a nivel nacional|en todo el pais|en todo peru)|"
        r"(paro nacional|huelga nacional|paralizacion nacional)"
    )
    mask_pn_i = nt_inline.str.contains(_paro_nac_i, regex=True, na=False)
    df.loc[mask_pn_i & (df["economic_score"].fillna(0) < 65), "economic_score"] = 65
    df.loc[mask_pn_i & (df["political_score"].fillna(0) < 55), "political_score"] = 55

    _huelga_indef_i = (
        r"huelga indefinida.{0,60}"
        r"(maestros|docentes|medicos|enfermeros|trabajadores judiciales|"
        r"poder judicial|empleados publicos|servidores civiles)"
    )
    mask_hi_i = nt_inline.str.contains(_huelga_indef_i, regex=True, na=False) & ~mask_pn_i
    df.loc[mask_hi_i & (df["economic_score"].fillna(0) < 60), "economic_score"] = 60
    df.loc[mask_hi_i & (df["political_score"].fillna(0) < 45), "political_score"] = 45

    _paro_reg_i = (
        r"paro.{0,40}(region|regional|arequipa|puno|cusco|loreto|piura|cajamarca|"
        r"junin|ayacucho|huancavelica|apurimac|madre de dios|san martin|"
        r"ancash|la libertad|lambayeque|ica|tacna|moquegua|tumbes|amazonas|"
        r"ucayali|huanuco|lima metropolitana|lima provincias|callao|pasco)"
    )
    mask_pr_i = nt_inline.str.contains(_paro_reg_i, regex=True, na=False) & ~mask_pn_i & ~mask_hi_i
    df.loc[mask_pr_i & (df["economic_score"].fillna(0) < 55), "economic_score"] = 55
    df.loc[mask_pr_i & (df["political_score"].fillna(0) < 35), "political_score"] = 35

    _huelga_sec_i = (
        r"(huelga|paro).{0,50}"
        r"(minero|mineria|transporte|transportistas|camioneros|agrario|pesquero|"
        r"construccion civil|bancario|docente|medico|enfermero)"
    )
    mask_hs_i = nt_inline.str.contains(_huelga_sec_i, regex=True, na=False) & ~mask_pn_i & ~mask_hi_i & ~mask_pr_i
    df.loc[mask_hs_i & (df["economic_score"].fillna(0) < 55), "economic_score"] = 55
    df.loc[mask_hs_i & (df["political_score"].fillna(0) < 25), "political_score"] = 25

    # 5e: Additional crisis patterns
    _narco_i = (
        r"(narco|narcotrafico|cartel|crimen organizado|organizacion criminal)"
        r".{0,60}"
        r"(congresista|ministro|gobierno|estado peruano|funcionario|partido|"
        r"candidato|alcalde|gobernador)"
    )
    mask_narco_i = nt_inline.str.contains(_narco_i, regex=True, na=False)
    df.loc[mask_narco_i & (df["political_score"].fillna(0) < 75), "political_score"] = 75

    _mining_conf_i = (
        r"(las bambas|tia maria|cuajone|antamina|quellaveco|toromocho|cerro verde|constancia)"
        r".{0,80}"
        r"(bloqueo|protesta|paraliz|conflicto|comunidad|enfrentamiento|herido|muerto)"
    )
    mask_mci = nt_inline.str.contains(_mining_conf_i, regex=True, na=False)
    df.loc[mask_mci & (df["economic_score"].fillna(0) < 62), "economic_score"] = 62
    df.loc[mask_mci & (df["political_score"].fillna(0) < 50), "political_score"] = 50

    _escasez_i = (
        r"(escasez|desabastecimiento|desabastecen|falta de|sin stock)"
        r".{0,40}"
        r"(alimentos|combustible|gas|medicamentos|oxigeno|agua potable|"
        r"productos basicos|canasta)"
    )
    mask_esc_i = nt_inline.str.contains(_escasez_i, regex=True, na=False)
    df.loc[mask_esc_i & (df["economic_score"].fillna(0) < 65), "economic_score"] = 65

    _inv_cancel_i = (
        r"(cancela|cancelo|suspende|suspendio|paraliza|paralizo|retira|retiro)"
        r".{0,60}"
        r"(inversion|proyecto minero|proyecto de inversion|concesion)"
        r".{0,60}"
        r"(conflicto|protesta|comunidad|bloqueo|violencia)"
    )
    mask_ic_i = nt_inline.str.contains(_inv_cancel_i, regex=True, na=False)
    df.loc[mask_ic_i & (df["economic_score"].fillna(0) < 58), "economic_score"] = 58

    _corrup_off_i = (
        r"(detienen|capturan|formalizan|sentencian|condenan|prision preventiva)"
        r".{0,50}"
        r"(ministro|viceministro|congresista|gobernador regional|alcalde|"
        r"director|funcionario publico|asesor del gobierno)"
    )
    mask_co_i = nt_inline.str.contains(_corrup_off_i, regex=True, na=False)
    df.loc[mask_co_i & (df["political_score"].fillna(0) < 62), "political_score"] = 62

    # ── END IMPROVEMENTS 4 & 5 inline ────────────────────────────────────────

    # ── J. NaN scores → 0 ─────────────────────────────────────────────────────
    # FINAL OVERRIDE: re-apply foreign domestic zeroing to catch any articles
    # that were re-boosted by inline rules (e.g. _pres_arrest_inline matching
    # "detienen...expresidente" in Bolivia articles).
    nt_final = df["title"].fillna("").apply(_normalize_title)
    mask_crisis_final = nt_final.str.contains(_crisis_guard, regex=True, na=False)
    _foreign_domestic_final = (
        r"elecciones (en |de )(colombia|argentina|brasil|m[eé]xico|chile|venezuela|"
        r"bolivia|ecuador|estados unidos|eeuu|reino unido|espa[ñn]a|francia|alemania|"
        r"canada|australia|india)\b|"
        r"(hijo|familiar|aliado|partido).{0,40}(arce|evo morales|morales evo).{0,60}"
        r"(detenido|arrestado|imputado|acusado|investigado|formalizado)|"
        r"(arce|evo morales).{0,40}(hijo|familiar|aliado).{0,60}"
        r"(detenido|arrestado|imputado|acusado)|"
        r"(detienen|arrestan|imputan|acusan|investigan).{0,60}"
        r"(hijo|familiar|aliado|funcionario).{0,40}(arce|evo morales)\b|"
        r"justicia boliviana|fiscal[ií]a de bolivia|gobierno boliviano.{0,60}"
        r"(detiene|arresta|imputa|investiga)|"
        r"crisis pol[ií]tica (en|de) bolivia\b|"
        r"bolivia.{0,30}(golpe|crisis institucional|estado de emergencia)|"
        r"^(bolivia|ecuador|argentina|colombia|venezuela|brasil|paraguay|uruguay|"
        r"chile|m[eé]xico|cuba|nicaragua|el salvador|honduras|guatemala)\s*:\s*.{0,200}"
        r"(detenido|arrestado|imputado|acusado|golpe|crisis|protestas|manifestantes|"
        r"huelga|paro nacional|estado de emergencia|renuncia|destituido|vacancia)"
    )
    mask_fd_final = nt_final.str.contains(_foreign_domestic_final, regex=True, na=False) & ~mask_crisis_final
    n_fd_final = int((mask_fd_final & ((df["political_score"].fillna(0) > 0) | (df["economic_score"].fillna(0) > 0))).sum())
    if n_fd_final > 0:
        df.loc[mask_fd_final, "political_score"] = 0
        df.loc[mask_fd_final, "economic_score"] = 0
        logger.info("post_filter_scores: final FD override re-zeroed %d articles boosted by inline rules", n_fd_final)

    # Articles never classified get NaN political_score or economic_score.
    # Must be set to 0 to avoid contaminating the intensity sum in the index formula.
    n_nan_pol = int(df["political_score"].isna().sum())
    n_nan_eco = int(df["economic_score"].isna().sum())
    if n_nan_pol > 0 or n_nan_eco > 0:
        df["political_score"] = df["political_score"].fillna(0.0)
        df["economic_score"] = df["economic_score"].fillna(0.0)
        logger.info("post_filter_scores: J NaN->0 pol=%d eco=%d", n_nan_pol, n_nan_eco)

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
