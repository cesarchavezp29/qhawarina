"""Geodata availability check for NEXUS visualization pipeline.

Checks whether Peru boundary GeoJSON files are available locally.
These files are required for map generation but must be downloaded manually
from the Humanitarian Data Exchange (HDX) or INEI.
"""

import logging
from pathlib import Path

from config.settings import RAW_GEO_DIR

logger = logging.getLogger(__name__)

GEO_BOUNDARIES_DIR = RAW_GEO_DIR / "peru_boundaries"

REQUIRED_FILES = [
    "departamentos.geojson",
    "provincias.geojson",
    "distritos.geojson",
]

HDX_URL = "https://data.humdata.org/dataset/cod-ab-per"


def check_geodata_available() -> bool:
    """Check whether all required GeoJSON boundary files exist.

    Returns
    -------
    bool
        True if all files are present.
    """
    missing = []
    for fname in REQUIRED_FILES:
        if not (GEO_BOUNDARIES_DIR / fname).exists():
            missing.append(fname)

    if missing:
        logger.warning(
            "Missing geodata files in %s: %s",
            GEO_BOUNDARIES_DIR, ", ".join(missing),
        )
        return False

    logger.info("All %d geodata files present in %s", len(REQUIRED_FILES), GEO_BOUNDARIES_DIR)
    return True


def geodata_download_instructions() -> str:
    """Return human-readable instructions for obtaining geodata."""
    return (
        f"Peru boundary GeoJSON files are required for map generation.\n"
        f"Expected location: {GEO_BOUNDARIES_DIR}/\n"
        f"Required files: {', '.join(REQUIRED_FILES)}\n\n"
        f"Download from HDX (Humanitarian Data Exchange):\n"
        f"  {HDX_URL}\n\n"
        f"Or from INEI geoportal:\n"
        f"  https://www.inei.gob.pe/\n\n"
        f"Place the .geojson files in the directory above."
    )
