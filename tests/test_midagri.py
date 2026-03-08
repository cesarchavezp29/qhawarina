"""Tests for MIDAGRI wholesale food price pipeline."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest


def test_product_category_mapping():
    """Known products should map to correct categories."""
    from src.ingestion.midagri import classify_product

    # CNPA-based classification (3-digit prefix)
    assert classify_product("01122", "MAIZ MORADO") == "cereals"
    assert classify_product("01211", "ESPARRAGO FRESCO") == "vegetables"
    assert classify_product("01319", "MARACUYA COSTA") == "fruits"
    assert classify_product("01411", "PAPA YUNGAY") == "roots_tubers"
    assert classify_product("01511", "ARVEJA VERDE") == "legumes"

    # Keyword fallback for unknown CNPA prefixes
    assert classify_product("09999", "TOMATE ITALIANO") == "vegetables"
    assert classify_product("09999", "MANZANA DELICIA") == "fruits"
    assert classify_product("09999", "PAPA CANCHAN") == "roots_tubers"
    assert classify_product("09999", "ALGO DESCONOCIDO") == "other"


def test_extract_month_year():
    """Spanish month names in bulletin titles should extract correctly."""
    from src.ingestion.midagri import extract_month_year

    assert extract_month_year("Boletin ... enero 2025") == "2025-01"
    assert extract_month_year("Boletin ... febrero 2026") == "2026-02"
    assert extract_month_year("Boletin ... setiembre 2024") == "2024-09"
    assert extract_month_year("Boletin ... septiembre 2024") == "2024-09"
    assert extract_month_year("Boletin ... diciembre 2023") == "2023-12"
    assert extract_month_year("random text without month") is None


def test_extract_date_from_filename():
    """Date should be extracted from bulletin PDF filenames."""
    from src.ingestion.midagri import extract_date_from_filename

    fn = "7685472-boletin-de-abastecimiento-y-precios-mayoristas-gmml-y-mm-n-2-02-02-2026.pdf"
    assert extract_date_from_filename(fn) == "2026-02-02"

    fn2 = "6342760-boletin-15-06-2024.pdf"
    assert extract_date_from_filename(fn2) == "2024-06-15"

    assert extract_date_from_filename("no-date-here.pdf") is None


def test_extract_date_from_text():
    """Date should be extracted from bulletin first-page text."""
    from src.ingestion.midagri import extract_date_from_text

    assert extract_date_from_text("Lima, 02 de febrero del 2026") == "2026-02-02"
    assert extract_date_from_text("Lima, 15 de enero del 2025") == "2025-01-15"
    assert extract_date_from_text("Lima, 3 de marzo del 2024") == "2024-03-03"
    assert extract_date_from_text("No date here") is None


def test_monthly_aggregation():
    """Synthetic daily prices should produce correct monthly averages and var%."""
    from src.ingestion.midagri import MidagriAggregator

    # Create synthetic daily price data
    dates = pd.date_range("2024-01-01", "2024-03-31", freq="B")  # business days
    rows = []
    for d in dates:
        for product, category, price in [
            ("PAPA YUNGAY", "roots_tubers", 1.50),
            ("TOMATE", "vegetables", 2.00),
            ("MANZANA", "fruits", 3.00),
        ]:
            # Add small variation
            month_factor = 1.0 + (d.month - 1) * 0.1  # 10% increase per month
            rows.append({
                "date": d,
                "cnpa": "01411",
                "product": product,
                "market": "GMML",
                "price_today": price * month_factor,
                "price_week_avg": price * month_factor,
                "category": category,
                "volume_tons": 100,
                "filename": "test.pdf",
            })

    daily_df = pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "monthly.parquet"
        agg = MidagriAggregator(output_path=output_path)
        monthly_df = agg.build_monthly_series(daily_df)

    assert not monthly_df.empty

    # Check we have 8 series
    series_ids = set(monthly_df["series_id"])
    assert "MIDAGRI_ALL_AVG" in series_ids
    assert "MIDAGRI_VEG_AVG" in series_ids
    assert "MIDAGRI_FRUIT_AVG" in series_ids
    assert "MIDAGRI_TUBER_AVG" in series_ids
    assert "MIDAGRI_ALL_VAR" in series_ids

    # Check monthly counts: 3 months of AVG data
    all_avg = monthly_df[monthly_df["series_id"] == "MIDAGRI_ALL_AVG"]
    assert len(all_avg) == 3  # Jan, Feb, Mar

    # Check var% has 2 entries (Feb and Mar, relative to previous month)
    all_var = monthly_df[monthly_df["series_id"] == "MIDAGRI_ALL_VAR"]
    assert len(all_var) == 2

    # Var% should be positive (prices increase each month)
    assert (all_var["value_raw"] > 0).all()

    # Check panel schema columns
    assert "source" in monthly_df.columns
    assert "frequency_original" in monthly_df.columns
    assert monthly_df["source"].iloc[0] == "MIDAGRI"


def test_parse_manifest_tracking():
    """Parsed files should be tracked, re-run should skip them."""
    from src.ingestion.midagri import MidagriPDFParser

    with tempfile.TemporaryDirectory() as tmpdir:
        extracted_dir = Path(tmpdir) / "extracted"
        extracted_dir.mkdir()

        parser = MidagriPDFParser(
            bulletins_dir=Path(tmpdir) / "bulletins",
            extracted_dir=extracted_dir,
        )

        # Initially empty
        parsed = parser._load_parsed_manifest()
        assert len(parsed) == 0

        # Save some parsed files
        parser._save_parsed_manifest({"file1.pdf", "file2.pdf"})

        # Should load them back
        parsed = parser._load_parsed_manifest()
        assert "file1.pdf" in parsed
        assert "file2.pdf" in parsed
        assert len(parsed) == 2


def test_scraper_incremental():
    """Manifest-based skip logic should prevent re-downloading."""
    from src.ingestion.midagri import MidagriBulletinScraper

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "manifest.csv"
        scraper = MidagriBulletinScraper(
            output_dir=Path(tmpdir) / "bulletins",
            manifest_path=manifest_path,
        )

        # Initially empty manifest
        manifest = scraper._load_manifest()
        assert len(manifest) == 0

        # Save a manifest with some entries
        df = pd.DataFrame([{
            "title": "Test Bulletin",
            "month": "2025-01",
            "url": "https://example.com",
            "pdf_url": "https://example.com/test.pdf",
            "filename": "test.pdf",
            "status": "downloaded",
        }])
        scraper._save_manifest(df)

        # Should load it back
        manifest = scraper._load_manifest()
        assert len(manifest) == 1
        assert manifest.iloc[0]["filename"] == "test.pdf"


def test_category_map_coverage():
    """All CNPA 3-digit prefixes used in bulletins should be mapped."""
    from src.ingestion.midagri import CATEGORY_MAP

    # These are the known prefixes from actual bulletins
    expected_prefixes = {"011", "012", "013", "014", "015", "016"}
    for prefix in expected_prefixes:
        assert prefix in CATEGORY_MAP, f"Missing CNPA prefix: {prefix}"


# ── Poultry pipeline tests ─────────────────────────────────────────────────


def test_extract_last_price():
    """Price extraction from poultry table rows."""
    from src.ingestion.midagri import _extract_last_price

    # Typical row: label + 7d-avg + day1 + day2 + var%
    assert _extract_last_price("7.34 7.59 7.58 -0.1%") == pytest.approx(7.58)
    assert _extract_last_price("7.54 7.80 7.80 0.0%") == pytest.approx(7.80)
    assert _extract_last_price("4.11 4.35 - -") == pytest.approx(4.35)

    # Only one number
    assert _extract_last_price("7.50") == pytest.approx(7.50)

    # No valid prices
    assert _extract_last_price("- - -") is None
    assert _extract_last_price("no numbers here") is None


def test_poultry_label_patterns():
    """Poultry label regex patterns should match bulletin text."""
    from src.ingestion.midagri import _POULTRY_LABELS

    assert _POULTRY_LABELS["chicken_wholesale"].search(
        "Centros de Distribucion (pollo vivo)"
    )
    assert _POULTRY_LABELS["chicken_wholesale"].search(
        "Centro de Distribución (pollo vivo)"
    )
    assert _POULTRY_LABELS["chicken_farm"].search("Granja (pollo vivo)")
    assert _POULTRY_LABELS["chicken_retail"].search(
        "Mercados Minoristas (pollo eviscerado)"
    )
    assert _POULTRY_LABELS["chicken_weight"].search(
        "Peso promedio por pollo (kg x unid.)"
    )
    assert _POULTRY_LABELS["egg"].search("Huevo rosado")
    assert _POULTRY_LABELS["hen_colored"].search("Gallina colorada")


def test_poultry_monthly_aggregation():
    """Synthetic poultry daily prices should aggregate to monthly series."""
    from src.ingestion.midagri import PoultryAggregator

    dates = pd.date_range("2024-06-01", "2024-08-31", freq="B")
    rows = []
    for d in dates:
        month_factor = 1.0 + (d.month - 6) * 0.05
        rows.append({
            "date": d,
            "chicken_wholesale": 7.50 * month_factor,
            "chicken_farm": 7.00 * month_factor,
            "chicken_retail": 11.50 * month_factor,
            "chicken_weight": 2.65,
            "egg": 4.20 * month_factor,
            "hen_colored": 7.00,
            "filename": "test.pdf",
        })

    daily_df = pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "poultry_monthly.parquet"
        agg = PoultryAggregator(output_path=output_path)
        monthly_df = agg.build_monthly_series(daily_df)

    assert not monthly_df.empty

    series_ids = set(monthly_df["series_id"])
    assert "MIDAGRI_CHICKEN_AVG" in series_ids
    assert "MIDAGRI_CHICKEN_VAR" in series_ids
    assert "MIDAGRI_EGG_AVG" in series_ids
    assert "MIDAGRI_EGG_VAR" in series_ids

    # 3 months of AVG data
    chicken_avg = monthly_df[monthly_df["series_id"] == "MIDAGRI_CHICKEN_AVG"]
    assert len(chicken_avg) == 3

    # 2 months of VAR data (Jul, Aug relative to previous)
    chicken_var = monthly_df[monthly_df["series_id"] == "MIDAGRI_CHICKEN_VAR"]
    assert len(chicken_var) == 2
    assert (chicken_var["value_raw"] > 0).all()  # prices increase

    # Schema columns
    assert monthly_df["source"].iloc[0] == "MIDAGRI"


def test_poultry_scraper_manifest():
    """Poultry scraper manifest tracking should work."""
    from src.ingestion.midagri import PoultryBulletinScraper

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "manifest.csv"
        scraper = PoultryBulletinScraper(
            output_dir=Path(tmpdir) / "bulletins",
            manifest_path=manifest_path,
        )

        # Initially empty
        manifest = scraper._load_manifest()
        assert len(manifest) == 0

        # Save and reload
        df = pd.DataFrame([{
            "title": "Poultry Feb 2026",
            "month": "2026-02",
            "url": "https://example.com",
            "pdf_url": "https://example.com/poultry.pdf",
            "filename": "poultry.pdf",
            "status": "downloaded",
        }])
        scraper._save_manifest(df)
        manifest = scraper._load_manifest()
        assert len(manifest) == 1


def test_poultry_parser_manifest():
    """Poultry PDF parser manifest tracking should work."""
    from src.ingestion.midagri import PoultryPDFParser

    with tempfile.TemporaryDirectory() as tmpdir:
        extracted_dir = Path(tmpdir) / "extracted"
        extracted_dir.mkdir()

        parser = PoultryPDFParser(
            bulletins_dir=Path(tmpdir) / "bulletins",
            extracted_dir=extracted_dir,
        )

        parsed = parser._load_parsed_manifest()
        assert len(parsed) == 0

        parser._save_parsed_manifest({"poultry1.pdf", "poultry2.pdf"})
        parsed = parser._load_parsed_manifest()
        assert "poultry1.pdf" in parsed
        assert len(parsed) == 2


def test_poultry_date_from_filename():
    """Date extraction from poultry bulletin filenames."""
    from src.ingestion.midagri import PoultryPDFParser

    # Typical poultry filename
    fn = "7585633-boletin-de-abastecimiento-y-precios-de-aves-30-01-2026.pdf"
    assert PoultryPDFParser._extract_date_from_filename(fn) == "2026-01-30"

    fn2 = "5141955-boletin-aves-15-02-2024.pdf"
    assert PoultryPDFParser._extract_date_from_filename(fn2) == "2024-02-15"

    assert PoultryPDFParser._extract_date_from_filename("no-date.pdf") is None
