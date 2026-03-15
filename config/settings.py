"""Global configuration for NEXUS project."""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

# ── Data directories ──────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TARGETS_DIR = DATA_DIR / "targets"
RESULTS_DIR = DATA_DIR / "results"

RAW_BCRP_DIR = RAW_DIR / "bcrp"
RAW_INEI_DIR = RAW_DIR / "inei"
RAW_SUNAT_DIR = RAW_DIR / "sunat"
RAW_VIIRS_DIR = RAW_DIR / "viirs"
RAW_GEO_DIR = RAW_DIR / "geo"

PROCESSED_NATIONAL_DIR = PROCESSED_DIR / "national"
PROCESSED_DEPARTMENTAL_DIR = PROCESSED_DIR / "departmental"
PROCESSED_DAILY_DIR = PROCESSED_DIR / "daily_instability"
RAW_RSS_DIR = RAW_DIR / "rss"
RAW_MIDAGRI_DIR = RAW_DIR / "midagri"
MIDAGRI_BULLETINS_DIR = RAW_MIDAGRI_DIR / "bulletins"
MIDAGRI_EXTRACTED_DIR = RAW_MIDAGRI_DIR / "extracted"
MIDAGRI_POULTRY_BULLETINS_DIR = RAW_MIDAGRI_DIR / "poultry_bulletins"
MIDAGRI_POULTRY_EXTRACTED_DIR = RAW_MIDAGRI_DIR / "poultry_extracted"

RAW_SUPERMARKET_DIR = RAW_DIR / "supermarket"
SUPERMARKET_SNAPSHOTS_DIR = RAW_SUPERMARKET_DIR / "snapshots"

# ── BCRP API ──────────────────────────────────────────────────────────────────
BCRP_API_BASE = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"
BCRP_MAX_SERIES_PER_REQUEST = 10
BCRP_REQUEST_DELAY_SECONDS = 1.0
BCRP_MAX_RETRIES = 3
BCRP_RETRY_BACKOFF_FACTOR = 2.0
BCRP_TIMEOUT_SECONDS = 30

# ── Date ranges ───────────────────────────────────────────────────────────────
DEFAULT_START_YEAR = 2004
DEFAULT_START_MONTH = 1
DEFAULT_END_YEAR = 2025
DEFAULT_END_MONTH = 12

# For backtesting: initial training window
BACKTEST_TRAIN_START = "2007-01"
BACKTEST_EVAL_START = "2010-01"
BACKTEST_EVAL_END = "2026-01"  # Extended through Jan 2026

# ── ENAHO / INEI ─────────────────────────────────────────────────────────────
RAW_ENAHO_DIR = RAW_DIR / "inei" / "enaho"
ENAHO_START_YEAR = 2004
ENAHO_SUMARIA_MODULE = "sumaria"

# ── Series catalog path ──────────────────────────────────────────────────────
SERIES_CATALOG_PATH = CONFIG_DIR / "series_catalog.yaml"
ENAHO_MODULES_PATH = CONFIG_DIR / "enaho_modules.yaml"

# ── Political instability index ──────────────────────────────────────────
RAW_POLITICAL_DIR = RAW_DIR / "political"
PROCESSED_POLITICAL_DIR = PROCESSED_DIR / "political_instability"
POLITICAL_SOURCES_PATH = CONFIG_DIR / "political_sources.yaml"
RSS_FEEDS_PATH = CONFIG_DIR / "rss_feeds.yaml"

# ── Output formats ────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_FORMAT = "parquet"

# ── UBIGEO department codes ───────────────────────────────────────────────────
DEPARTMENTS = {
    "01": "Amazonas",
    "02": "Áncash",
    "03": "Apurímac",
    "04": "Arequipa",
    "05": "Ayacucho",
    "06": "Cajamarca",
    "07": "Callao",
    "08": "Cusco",
    "09": "Huancavelica",
    "10": "Huánuco",
    "11": "Ica",
    "12": "Junín",
    "13": "La Libertad",
    "14": "Lambayeque",
    "15": "Lima",
    "16": "Loreto",
    "17": "Madre de Dios",
    "18": "Moquegua",
    "19": "Pasco",
    "20": "Piura",
    "21": "Puno",
    "22": "San Martín",
    "23": "Tacna",
    "24": "Tumbes",
    "25": "Ucayali",
}
