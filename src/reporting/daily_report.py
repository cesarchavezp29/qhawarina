"""Daily instability report generator.

Produces a markdown report for a given date explaining what drove the
political and economic instability scores, with top articles and an
optional Claude-generated narrative summary.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger("nexus.reporting.daily")

SEVERITY_LABELS = {1: "LOW", 2: "MEDIUM", 3: "HIGH"}

NARRATIVE_PROMPT = """Eres un analista de riesgo político-económico para Perú.

A partir de las noticias del día, escribe un resumen ejecutivo de 3-4 oraciones
en español explicando los principales eventos que impulsaron la inestabilidad
política y económica. Sé conciso y objetivo. No repitas títulos textuales.

FECHA: {date}
SCORE POLÍTICO: {pol_score:.1f} (v2={pol_v2:.2f}, {pol_level})
SCORE ECONÓMICO: {econ_score:.1f} (v2={econ_v2:.2f}, {econ_level})
ARTÍCULOS POLÍTICOS: {n_pol}  |  ECONÓMICOS: {n_econ}  |  TOTAL: {n_total}

TOP NOTICIAS POLÍTICAS:
{pol_articles}

TOP NOTICIAS ECONÓMICAS:
{econ_articles}

Responde SOLO con el resumen ejecutivo (3-4 oraciones, español)."""


def _level_label(v2: float) -> str:
    """Convert v2 score to human-readable level."""
    if v2 >= 0.8:
        return "MUY ALTO"
    if v2 >= 0.6:
        return "ALTO"
    if v2 >= 0.4:
        return "MODERADO"
    if v2 >= 0.2:
        return "BAJO"
    return "MUY BAJO"


def _format_article_line(row: pd.Series) -> str:
    """Format a single article as a bullet line."""
    sev = SEVERITY_LABELS.get(row.get("article_severity", 0), "?")
    source = row.get("source", "?")
    title = row.get("title", "Sin título")
    return f"- [{sev}] ({source}) {title}"


def get_day_data(
    date: str,
    index_path: Path,
    classified_path: Path,
) -> tuple[pd.Series | None, pd.DataFrame, pd.DataFrame]:
    """Load index row and articles for a specific date.

    Returns (index_row, political_articles, economic_articles).
    """
    idx = pd.read_parquet(index_path)
    idx["date"] = pd.to_datetime(idx["date"])
    target = pd.Timestamp(date)

    row = idx[idx["date"] == target]
    if row.empty:
        return None, pd.DataFrame(), pd.DataFrame()
    row = row.iloc[0]

    articles = pd.read_parquet(classified_path)
    articles["published"] = pd.to_datetime(articles["published"], utc=True)
    articles["_date"] = articles["published"].dt.date

    day_articles = articles[articles["_date"] == target.date()]

    pol = day_articles[day_articles["article_category"].isin(["political", "both"])]
    econ = day_articles[day_articles["article_category"].isin(["economic", "both"])]

    pol = pol.sort_values("article_severity", ascending=False)
    econ = econ.sort_values("article_severity", ascending=False)

    return row, pol, econ


def generate_report(
    date: str,
    index_path: Path,
    classified_path: Path,
    narrative: str | None = None,
) -> str:
    """Generate a markdown report for a given date.

    Parameters
    ----------
    date : ISO date string (e.g. "2026-01-21")
    index_path : Path to daily_index.parquet
    classified_path : Path to articles_classified.parquet
    narrative : Optional Claude-generated narrative summary

    Returns
    -------
    Markdown string with the full report.
    """
    row, pol, econ = get_day_data(date, index_path, classified_path)

    if row is None:
        return f"# Daily Instability Report — {date}\n\nNo data available for this date.\n"

    pol_level = _level_label(row["political_v2"])
    econ_level = _level_label(row["economic_v2"])

    lines = []
    lines.append(f"# Daily Instability Report — {date}")
    lines.append("")
    lines.append("## Scores")
    lines.append("")
    lines.append("| Index | Score | Z-score | Level | V2 | Articles |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        f"| Political | {row['political_score']:.1f} | {row['political_zscore']:.2f} "
        f"| {pol_level} | {row['political_v2']:.2f} | {int(row['n_articles_political'])} |"
    )
    lines.append(
        f"| Economic | {row['economic_score']:.1f} | {row['economic_zscore']:.2f} "
        f"| {econ_level} | {row['economic_v2']:.2f} | {int(row['n_articles_economic'])} |"
    )
    lines.append(
        f"| **Total articles** | | | | | **{int(row['n_articles_total'])}** |"
    )
    lines.append("")

    # Narrative summary
    if narrative:
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(narrative)
        lines.append("")

    # Top political articles
    lines.append(f"## Political Articles ({len(pol)})")
    lines.append("")
    if pol.empty:
        lines.append("No political articles for this date.")
    else:
        # Show severity breakdown
        sev_counts = pol["article_severity"].value_counts().sort_index(ascending=False)
        sev_parts = [f"{SEVERITY_LABELS.get(s, '?')}: {c}" for s, c in sev_counts.items()]
        lines.append(f"Severity breakdown: {' | '.join(sev_parts)}")
        lines.append("")
        for _, a in pol.head(15).iterrows():
            lines.append(_format_article_line(a))
        if len(pol) > 15:
            lines.append(f"- ... and {len(pol) - 15} more")
    lines.append("")

    # Top economic articles
    lines.append(f"## Economic Articles ({len(econ)})")
    lines.append("")
    if econ.empty:
        lines.append("No economic articles for this date.")
    else:
        sev_counts = econ["article_severity"].value_counts().sort_index(ascending=False)
        sev_parts = [f"{SEVERITY_LABELS.get(s, '?')}: {c}" for s, c in sev_counts.items()]
        lines.append(f"Severity breakdown: {' | '.join(sev_parts)}")
        lines.append("")
        for _, a in econ.head(15).iterrows():
            lines.append(_format_article_line(a))
        if len(econ) > 15:
            lines.append(f"- ... and {len(econ) - 15} more")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*Generated by NEXUS Daily Instability Pipeline*")

    return "\n".join(lines)


def generate_narrative(
    date: str,
    index_path: Path,
    classified_path: Path,
    model: str = "claude-haiku-4-5-20251001",
) -> str:
    """Use Claude to generate a narrative summary for a given date."""
    row, pol, econ = get_day_data(date, index_path, classified_path)

    if row is None:
        return ""

    pol_lines = "\n".join(_format_article_line(a) for _, a in pol.head(10).iterrows())
    econ_lines = "\n".join(_format_article_line(a) for _, a in econ.head(10).iterrows())

    if not pol_lines:
        pol_lines = "(ninguna)"
    if not econ_lines:
        econ_lines = "(ninguna)"

    prompt = NARRATIVE_PROMPT.format(
        date=date,
        pol_score=row["political_score"],
        pol_v2=row["political_v2"],
        pol_level=_level_label(row["political_v2"]),
        econ_score=row["economic_score"],
        econ_v2=row["economic_v2"],
        econ_level=_level_label(row["economic_v2"]),
        n_pol=int(row["n_articles_political"]),
        n_econ=int(row["n_articles_economic"]),
        n_total=int(row["n_articles_total"]),
        pol_articles=pol_lines,
        econ_articles=econ_lines,
    )

    from src.nlp.classifier import _get_client

    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


PERIOD_NARRATIVE_PROMPT = """Eres un analista de riesgo político-económico para Perú.

A partir de las noticias del período {period_type} ({start_date} al {end_date}),
escribe un resumen ejecutivo de 4-5 oraciones en español explicando los principales
eventos que impulsaron la inestabilidad política y económica. Sé conciso y objetivo.

PERÍODO: {start_date} al {end_date}
SCORE POLÍTICO PROMEDIO: {pol_score:.1f} (v2={pol_v2:.2f}, {pol_level})
SCORE ECONÓMICO PROMEDIO: {econ_score:.1f} (v2={econ_v2:.2f}, {econ_level})
ARTÍCULOS POLÍTICOS: {n_pol}  |  ECONÓMICOS: {n_econ}  |  TOTAL: {n_total}

TOP NOTICIAS POLÍTICAS:
{pol_articles}

TOP NOTICIAS ECONÓMICAS:
{econ_articles}

Responde SOLO con el resumen ejecutivo (4-5 oraciones, español)."""


def generate_period_narrative(
    start_date: str,
    end_date: str,
    index_path: Path,
    classified_path: Path,
    period_type: str = "semanal",
    model: str = "claude-haiku-4-5-20251001",
) -> str:
    """Use Claude to generate a narrative summary for a date range."""
    idx = pd.read_parquet(index_path)
    idx["date"] = pd.to_datetime(idx["date"])

    mask = (idx["date"] >= pd.Timestamp(start_date)) & \
           (idx["date"] <= pd.Timestamp(end_date))
    period = idx[mask]

    if period.empty:
        return ""

    articles = pd.read_parquet(classified_path)
    articles["published"] = pd.to_datetime(articles["published"], utc=True)
    articles["_date"] = articles["published"].dt.date

    start_d = pd.Timestamp(start_date).date()
    end_d = pd.Timestamp(end_date).date()
    day_arts = articles[(articles["_date"] >= start_d) & (articles["_date"] <= end_d)]

    pol = day_arts[day_arts["article_category"].isin(["political", "both"])]
    econ = day_arts[day_arts["article_category"].isin(["economic", "both"])]
    pol = pol.sort_values("article_severity", ascending=False)
    econ = econ.sort_values("article_severity", ascending=False)

    pol_lines = "\n".join(_format_article_line(a) for _, a in pol.head(12).iterrows())
    econ_lines = "\n".join(_format_article_line(a) for _, a in econ.head(12).iterrows())

    if not pol_lines:
        pol_lines = "(ninguna)"
    if not econ_lines:
        econ_lines = "(ninguna)"

    avg_pol_v2 = period["political_v2"].mean()
    avg_econ_v2 = period["economic_v2"].mean()

    prompt = PERIOD_NARRATIVE_PROMPT.format(
        period_type=period_type,
        start_date=start_date,
        end_date=end_date,
        pol_score=period["political_score"].mean(),
        pol_v2=avg_pol_v2,
        pol_level=_level_label(avg_pol_v2),
        econ_score=period["economic_score"].mean(),
        econ_v2=avg_econ_v2,
        econ_level=_level_label(avg_econ_v2),
        n_pol=len(pol),
        n_econ=len(econ),
        n_total=len(pol) + len(econ),
        pol_articles=pol_lines,
        econ_articles=econ_lines,
    )

    from src.nlp.classifier import _get_client

    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()
