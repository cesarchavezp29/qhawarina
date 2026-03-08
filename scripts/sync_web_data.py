"""
Sync exported data to qhawarina Next.js project.

This script copies the exported JSON/CSV files from exports/data
to the Next.js public/assets/data directory for web consumption.
"""

import shutil
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("nexus.sync_web")

# Paths
EXPORT_DIR = Path("D:/Nexus/nexus/exports/data")
WEB_DATA_DIR = Path("D:/qhawarina/public/assets/data")


def sync_data():
    """Copy all data files to web project."""

    if not EXPORT_DIR.exists():
        logger.error(f"Export directory not found: {EXPORT_DIR}")
        logger.error("Run 'python scripts/export_web_data.py' first!")
        return False

    if not WEB_DATA_DIR.exists():
        logger.info(f"Creating web data directory: {WEB_DATA_DIR}")
        WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SYNCING DATA TO WEB PROJECT")
    logger.info("=" * 60)

    # Copy all files
    copied = 0
    for file_path in EXPORT_DIR.glob("*"):
        if file_path.is_file():
            dest = WEB_DATA_DIR / file_path.name
            shutil.copy2(file_path, dest)
            logger.info(f"Copied: {file_path.name} ({file_path.stat().st_size / 1024:.1f} KB)")
            copied += 1

    logger.info("=" * 60)
    logger.info(f"SYNC COMPLETE - {copied} files copied")
    logger.info(f"Source: {EXPORT_DIR}")
    logger.info(f"Destination: {WEB_DATA_DIR}")
    logger.info("=" * 60)

    return True


if __name__ == "__main__":
    success = sync_data()
    exit(0 if success else 1)
