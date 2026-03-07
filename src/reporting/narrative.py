"""Automated narrative generation for Qhawarina reports via Claude API.

Generates coherent, professional analytical text from structured index data.
Produces: executive summary, week-in-review, what-to-watch sections.
All output in Spanish, written as a professional risk analyst.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("nexus.reporting.narrative")


# ── Prompt templates ─────────────────────────────────────────────────────────

WEEKLY_EXECUTIVE_PROMPT = """\
Eres un analista senior de riesgo político-económico para Perú, escribiendo para
el boletín semanal "Qhawarina" dirigido a ejecutivos, inversores y tomadores de
decisiones. Tu análisis debe ser objetivo, preciso y útil para la toma de decisiones.

CONTEXTO DEL ÍNDICE:
- El Índice de Incertidumbre Qhawarina mide la proporción de cobertura mediática
  dedicada a eventos de inestabilidad política y económica en Perú.
- Base: media=100. Un valor de 150 significa 50% más cobertura de inestabilidad que el promedio.
- Fuentes: El Comercio, Gestión, La República.

DATOS DE LA SEMANA ({start_date} al {end_date}):
- Índice Político: promedio={pol_index:.0f} (suavizado={pol_smooth:.0f})
  Cambio vs semana anterior: {pol_change:+.0f} puntos
  Rango: mín={pol_min:.0f}, máx={pol_max:.0f}
- Índice Económico: promedio={econ_index:.0f} (suavizado={econ_smooth:.0f})
  Cambio vs semana anterior: {econ_change:+.0f} puntos
  Rango: mín={econ_min:.0f}, máx={econ_max:.0f}
- Artículos analizados: {n_total} total ({n_pol} políticos, {n_econ} económicos)
- Fuentes activas: {n_sources} promedio/día

PRINCIPALES NOTICIAS POLÍTICAS (por severidad):
{pol_articles}

PRINCIPALES NOTICIAS ECONÓMICAS (por severidad):
{econ_articles}

INSTRUCCIONES:
Escribe un resumen ejecutivo de exactamente 3-4 oraciones. Estructura:
1. Oración de apertura: nivel general de incertidumbre y tendencia (subió/bajó/estable)
2. Factor político principal que impulsó el índice
3. Factor económico principal
4. Perspectiva o implicación para la semana entrante

REGLAS:
- Sé específico: menciona nombres, instituciones, cifras cuando sea relevante
- No uses superlativos innecesarios ni lenguaje alarmista
- No repitas títulos de noticias textualmente
- Interpreta los números: qué significan para el empresario o inversor
- Si el índice está bajo (<80), reconócelo como un período de relativa calma

Responde SOLO con el resumen ejecutivo (3-4 oraciones, español, texto plano sin markdown)."""

WEEKLY_REVIEW_PROMPT = """\
Eres un analista de riesgo político-económico para Perú. Escribe la sección
"Semana en Revisión" del boletín semanal Qhawarina.

PERÍODO: {start_date} al {end_date}

DATOS DIARIOS DEL ÍNDICE:
{daily_table}

PRINCIPALES NOTICIAS POLÍTICAS:
{pol_articles}

PRINCIPALES NOTICIAS ECONÓMICAS:
{econ_articles}

INSTRUCCIONES:
Escribe 4-6 viñetas (bullet points) describiendo los eventos más importantes
de la semana, en orden cronológico. Cada viñeta debe:
- Empezar con la fecha (ej: "Lunes 3: ...")
- Describir el evento y su impacto en 1-2 oraciones
- Clasificar implícitamente como político o económico

Usa formato:
• Lun X: [descripción del evento y su relevancia]
• Mar X: [descripción]
...

Responde SOLO con las viñetas (4-6, español, texto plano sin markdown). No incluyas encabezados."""

WEEKLY_OUTLOOK_PROMPT = """\
Eres un analista de riesgo político-económico para Perú. Escribe la sección
"Qué Vigilar" del boletín semanal Qhawarina para la semana siguiente.

CONTEXTO RECIENTE ({start_date} al {end_date}):
- Índice político promedio: {pol_index:.0f} (tendencia: {pol_trend})
- Índice económico promedio: {econ_index:.0f} (tendencia: {econ_trend})

EVENTOS RECIENTES MÁS RELEVANTES:
{recent_events}

INSTRUCCIONES:
Escribe 3-4 puntos sobre qué vigilar la próxima semana. Cada punto debe:
- Identificar un riesgo o evento potencial basado en las tendencias actuales
- Ser específico y accionable (no genérico)
- Indicar dirección probable del impacto (alcista/bajista para incertidumbre)

Formato:
▸ [Evento/riesgo a vigilar]: [por qué importa, 1-2 oraciones]

Responde SOLO con los puntos (3-4, español, texto plano sin markdown). No incluyas encabezados."""


# ── Risk level labels ────────────────────────────────────────────────────────

def _risk_label(index_value: float) -> str:
    """Convert index value (mean=100) to risk label."""
    if index_value >= 200:
        return "MUY ALTO"
    if index_value >= 150:
        return "ALTO"
    if index_value >= 100:
        return "MODERADO"
    if index_value >= 50:
        return "BAJO"
    return "MUY BAJO"


def _risk_color(index_value: float) -> str:
    """Return hex color for risk level."""
    if index_value >= 200:
        return "#C62828"
    if index_value >= 150:
        return "#FF8F00"
    if index_value >= 100:
        return "#FDD835"
    if index_value >= 50:
        return "#66BB6A"
    return "#2E7D32"


def _trend_label(change: float) -> str:
    """Describe trend direction."""
    if change > 15:
        return "al alza significativa"
    if change > 5:
        return "al alza moderada"
    if change > -5:
        return "estable"
    if change > -15:
        return "a la baja moderada"
    return "a la baja significativa"


# ── Article formatting ───────────────────────────────────────────────────────

SEVERITY_LABELS = {0: "IRRELEVANTE", 1: "BAJA", 2: "MEDIA", 3: "ALTA"}


def _format_articles(articles_df: pd.DataFrame, max_n: int = 10) -> str:
    """Format top articles as text list for prompt."""
    if articles_df.empty:
        return "(ninguna noticia relevante)"
    lines = []
    for _, row in articles_df.head(max_n).iterrows():
        sev = SEVERITY_LABELS.get(row.get("article_severity", 0), "?")
        source = row.get("source", "?")
        title = str(row.get("title", ""))[:100]
        lines.append(f"- [{sev}] ({source}) {title}")
    return "\n".join(lines)


# ── Data preparation ─────────────────────────────────────────────────────────

def prepare_weekly_data(
    index_v2: pd.DataFrame,
    articles: pd.DataFrame,
    end_date: str,
) -> dict:
    """Prepare structured data for weekly report narrative generation.

    Parameters
    ----------
    index_v2 : V2 index DataFrame (daily_index_v2.parquet)
    articles : Classified articles DataFrame
    end_date : Last day of the week (ISO string)

    Returns
    -------
    Dict with all fields needed for narrative prompts + report rendering.
    """
    idx = index_v2.copy()
    idx["date"] = pd.to_datetime(idx["date"])

    end = pd.Timestamp(end_date)
    start = end - pd.Timedelta(days=6)

    # This week
    mask = (idx["date"] >= start) & (idx["date"] <= end)
    week = idx[mask].copy()

    # Previous week for comparison
    prev_end = start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=6)
    prev_mask = (idx["date"] >= prev_start) & (idx["date"] <= prev_end)
    prev_week = idx[prev_mask]

    if week.empty:
        return None

    # Articles for this week
    arts = articles.copy()
    arts["published"] = pd.to_datetime(arts["published"], utc=True)
    arts["_date"] = arts["published"].dt.normalize().dt.tz_localize(None)
    week_arts = arts[(arts["_date"] >= start) & (arts["_date"] <= end)]

    pol_arts = week_arts[week_arts["article_category"].isin(["political", "both"])]
    econ_arts = week_arts[week_arts["article_category"].isin(["economic", "both"])]
    pol_arts = pol_arts.sort_values("article_severity", ascending=False)
    econ_arts = econ_arts.sort_values("article_severity", ascending=False)

    # Compute stats
    pol_index_mean = week["political_index"].mean()
    econ_index_mean = week["economic_index"].mean()
    pol_smooth_mean = week["political_smooth"].mean()
    econ_smooth_mean = week["economic_smooth"].mean()

    prev_pol = prev_week["political_index"].mean() if not prev_week.empty else pol_index_mean
    prev_econ = prev_week["economic_index"].mean() if not prev_week.empty else econ_index_mean

    pol_change = pol_index_mean - prev_pol
    econ_change = econ_index_mean - prev_econ

    # Daily table for review prompt
    daily_rows = []
    days_es = {0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie", 5: "Sáb", 6: "Dom"}
    for _, row in week.iterrows():
        d = row["date"]
        day_name = days_es.get(d.dayofweek, "?")
        daily_rows.append(
            f"{day_name} {d.day}: pol={row['political_index']:.0f} "
            f"econ={row['economic_index']:.0f} arts={int(row['n_articles_total'])}"
        )
    daily_table = "\n".join(daily_rows)

    return {
        "start_date": start.strftime("%d/%m/%Y"),
        "end_date": end.strftime("%d/%m/%Y"),
        "start_iso": start.strftime("%Y-%m-%d"),
        "end_iso": end.strftime("%Y-%m-%d"),
        "week_number": end.isocalendar()[1],
        "year": end.year,
        # Index stats
        "pol_index": pol_index_mean,
        "pol_smooth": pol_smooth_mean,
        "pol_change": pol_change,
        "pol_min": week["political_index"].min(),
        "pol_max": week["political_index"].max(),
        "pol_risk": _risk_label(pol_index_mean),
        "pol_risk_color": _risk_color(pol_index_mean),
        "pol_trend": _trend_label(pol_change),
        "econ_index": econ_index_mean,
        "econ_smooth": econ_smooth_mean,
        "econ_change": econ_change,
        "econ_min": week["economic_index"].min(),
        "econ_max": week["economic_index"].max(),
        "econ_risk": _risk_label(econ_index_mean),
        "econ_risk_color": _risk_color(econ_index_mean),
        "econ_trend": _trend_label(econ_change),
        # Article counts
        "n_pol": len(pol_arts),
        "n_econ": len(econ_arts),
        "n_total": len(week_arts),
        "n_sources": week["n_sources"].mean(),
        # Formatted articles for prompts
        "pol_articles_text": _format_articles(pol_arts),
        "econ_articles_text": _format_articles(econ_arts),
        # Raw data for charts
        "week_daily": week,
        "pol_articles": pol_arts,
        "econ_articles": econ_arts,
        "daily_table": daily_table,
        # Previous week for comparison
        "prev_pol": prev_pol,
        "prev_econ": prev_econ,
        # Full index for trend chart
        "index_full": idx,
    }


# ── Claude API narrative generation ──────────────────────────────────────────

def generate_weekly_narrative(
    data: dict,
    model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """Generate all narrative sections for the weekly report.

    Returns dict with keys: executive_summary, week_review, outlook.
    """
    from src.nlp.classifier import _get_client

    client = _get_client()

    # 1. Executive Summary
    exec_prompt = WEEKLY_EXECUTIVE_PROMPT.format(
        start_date=data["start_date"],
        end_date=data["end_date"],
        pol_index=data["pol_index"],
        pol_smooth=data["pol_smooth"],
        pol_change=data["pol_change"],
        pol_min=data["pol_min"],
        pol_max=data["pol_max"],
        econ_index=data["econ_index"],
        econ_smooth=data["econ_smooth"],
        econ_change=data["econ_change"],
        econ_min=data["econ_min"],
        econ_max=data["econ_max"],
        n_total=data["n_total"],
        n_pol=data["n_pol"],
        n_econ=data["n_econ"],
        n_sources=data["n_sources"],
        pol_articles=data["pol_articles_text"],
        econ_articles=data["econ_articles_text"],
    )

    logger.info("Generating executive summary...")
    resp1 = client.messages.create(
        model=model, max_tokens=400,
        messages=[{"role": "user", "content": exec_prompt}],
    )
    executive_summary = resp1.content[0].text.strip()

    # 2. Week in Review
    review_prompt = WEEKLY_REVIEW_PROMPT.format(
        start_date=data["start_date"],
        end_date=data["end_date"],
        daily_table=data["daily_table"],
        pol_articles=data["pol_articles_text"],
        econ_articles=data["econ_articles_text"],
    )

    logger.info("Generating week review...")
    resp2 = client.messages.create(
        model=model, max_tokens=500,
        messages=[{"role": "user", "content": review_prompt}],
    )
    week_review = resp2.content[0].text.strip()

    # 3. Outlook
    # Get recent high-severity events
    all_arts = pd.concat([data["pol_articles"], data["econ_articles"]])
    if not all_arts.empty:
        all_arts = all_arts.drop_duplicates(subset=["url"])
    top_events = _format_articles(
        all_arts.sort_values("article_severity", ascending=False), max_n=8,
    )

    outlook_prompt = WEEKLY_OUTLOOK_PROMPT.format(
        start_date=data["start_date"],
        end_date=data["end_date"],
        pol_index=data["pol_index"],
        pol_trend=data["pol_trend"],
        econ_index=data["econ_index"],
        econ_trend=data["econ_trend"],
        recent_events=top_events,
    )

    logger.info("Generating outlook...")
    resp3 = client.messages.create(
        model=model, max_tokens=400,
        messages=[{"role": "user", "content": outlook_prompt}],
    )
    outlook = resp3.content[0].text.strip()

    return {
        "executive_summary": _md_to_html(executive_summary),
        "week_review": _md_to_html(week_review),
        "outlook": _md_to_html(outlook),
    }


def _md_to_html(text: str) -> str:
    """Convert basic markdown formatting to HTML.

    Handles: **bold**, *italic*, bullet lists (- or •), ▸ arrows.
    Strips markdown headers (##).
    """
    import re
    # Strip markdown headers
    text = re.sub(r'^#{1,3}\s+.*$', '', text, flags=re.MULTILINE)
    # Bold: **text** → <strong>text</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic: *text* → <em>text</em>
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    # Clean up extra blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
