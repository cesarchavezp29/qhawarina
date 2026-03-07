"""Qhawarina Weekly Report v2 — HTML+CSS → PDF via xhtml2pdf.

Generates professional 4-page weekly uncertainty reports with:
  - Automated Claude API narrative generation
  - Matplotlib charts embedded as base64 PNGs
  - 3 template designs (A=Institucional, B=Moderno, C=Ejecutivo)
  - All content in Spanish
"""

import base64
import logging
from io import BytesIO
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from xhtml2pdf import pisa

from src.reporting.narrative import (
    generate_weekly_narrative,
    prepare_weekly_data,
)
from src.reporting.charts import (
    weekly_trend_chart,
    weekly_bar_chart,
    article_count_chart,
)

logger = logging.getLogger("nexus.reporting.weekly_v2")

TEMPLATE_DIR = Path(__file__).parent / "templates"
LOGO_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "logo.png"


def _load_logo() -> str:
    """Load logo as base64 string."""
    if LOGO_PATH.exists():
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


def generate_weekly_report(
    end_date: str,
    index_path: str | Path,
    classified_path: str | Path,
    output_path: str | Path,
    design: str = "a",
    generate_narrative: bool = True,
    model: str = "claude-haiku-4-5-20251001",
) -> Path:
    """Generate a weekly PDF report.

    Parameters
    ----------
    end_date : ISO date (last day of the week, e.g. "2026-02-08")
    index_path : Path to daily_index_v2.parquet
    classified_path : Path to articles_classified.parquet
    output_path : Where to save the PDF
    design : "a" (Institucional), "b" (Moderno), or "c" (Ejecutivo)
    generate_narrative : If True, call Claude API for narrative sections
    model : Claude model for narrative generation

    Returns
    -------
    Path to generated PDF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────
    logger.info("Loading data for weekly report ending %s...", end_date)
    index_v2 = pd.read_parquet(index_path)
    articles = pd.read_parquet(classified_path)

    data = prepare_weekly_data(index_v2, articles, end_date)
    if data is None:
        logger.warning("No data available for week ending %s", end_date)
        return output_path

    # ── Generate narrative ───────────────────────────────────────
    if generate_narrative:
        logger.info("Generating narrative via Claude API...")
        narrative = generate_weekly_narrative(data, model=model)
    else:
        narrative = {
            "executive_summary": "Resumen ejecutivo no disponible (generación desactivada).",
            "week_review": "Revisión semanal no disponible.",
            "outlook": "Perspectiva no disponible.",
        }

    # ── Generate charts ──────────────────────────────────────────
    logger.info("Generating charts...")
    charts = {
        "trend": weekly_trend_chart(data["index_full"], end_date),
        "daily_bars": weekly_bar_chart(data["week_daily"]),
        "article_counts": article_count_chart(data["week_daily"]),
    }

    # ── Render HTML from template ────────────────────────────────
    template_file = f"weekly_design_{design.lower()}.html"
    logger.info("Rendering template: %s", template_file)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=False,
    )
    template = env.get_template(template_file)

    html = template.render(
        data=data,
        narrative=narrative,
        charts=charts,
        logo_b64=_load_logo(),
    )

    # ── Convert HTML → PDF ───────────────────────────────────────
    logger.info("Converting to PDF: %s", output_path)
    buf = BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=buf, encoding="utf-8")

    if pisa_status.err:
        logger.error("PDF generation failed with %d errors", pisa_status.err)
    else:
        with open(output_path, "wb") as f:
            f.write(buf.getvalue())
        logger.info("Weekly report saved: %s (%d KB)",
                     output_path, len(buf.getvalue()) // 1024)

    return output_path


def generate_all_samples(
    end_date: str,
    index_path: str | Path,
    classified_path: str | Path,
    output_dir: str | Path,
    generate_narrative: bool = True,
    model: str = "claude-haiku-4-5-20251001",
) -> list[Path]:
    """Generate all 3 design samples for comparison.

    Returns list of paths to generated PDFs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data once, share across all designs
    logger.info("Loading shared data...")
    index_v2 = pd.read_parquet(index_path)
    articles = pd.read_parquet(classified_path)
    data = prepare_weekly_data(index_v2, articles, end_date)

    if data is None:
        logger.warning("No data for week ending %s", end_date)
        return []

    # Generate narrative once
    if generate_narrative:
        logger.info("Generating shared narrative...")
        narrative = generate_weekly_narrative(data, model=model)
    else:
        narrative = {
            "executive_summary": "Resumen ejecutivo no disponible.",
            "week_review": "Revisión semanal no disponible.",
            "outlook": "Perspectiva no disponible.",
        }

    # Generate charts once
    logger.info("Generating shared charts...")
    charts = {
        "trend": weekly_trend_chart(data["index_full"], end_date),
        "daily_bars": weekly_bar_chart(data["week_daily"]),
        "article_counts": article_count_chart(data["week_daily"]),
    }

    logo_b64 = _load_logo()

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=False,
    )

    paths = []
    designs = {
        "a": "Institucional",
        "b": "Moderno",
        "c": "Ejecutivo",
    }

    for design_key, design_name in designs.items():
        template_file = f"weekly_design_{design_key}.html"
        out_path = output_dir / f"sample_{design_key}_{design_name.lower()}.pdf"

        logger.info("Rendering design %s (%s)...", design_key.upper(), design_name)
        template = env.get_template(template_file)
        html = template.render(
            data=data,
            narrative=narrative,
            charts=charts,
            logo_b64=logo_b64,
        )

        buf = BytesIO()
        pisa_status = pisa.CreatePDF(html, dest=buf, encoding="utf-8")

        if pisa_status.err:
            logger.error("Design %s: PDF failed with %d errors",
                         design_key.upper(), pisa_status.err)
        else:
            with open(out_path, "wb") as f:
                f.write(buf.getvalue())
            logger.info("Design %s: %s (%d KB)",
                         design_key.upper(), out_path, len(buf.getvalue()) // 1024)
            paths.append(out_path)

    return paths
