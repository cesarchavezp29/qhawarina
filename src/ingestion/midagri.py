"""MIDAGRI wholesale food price pipeline.

Downloads daily wholesale price bulletins from gob.pe (MIDAGRI),
extracts structured price data via pdfplumber, and aggregates into
monthly series for the national panel.

Collection URL:
  https://www.gob.pe/institucion/midagri/colecciones/338-boletin-de-abastecimiento-y-precios-en-el-mercado-mayorista-de-lima-gmml-y-mercado-de-frutas-n-2-mm-n-2
"""

import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

from config.settings import (
    MIDAGRI_BULLETINS_DIR,
    MIDAGRI_EXTRACTED_DIR,
    MIDAGRI_POULTRY_BULLETINS_DIR,
    MIDAGRI_POULTRY_EXTRACTED_DIR,
    PROCESSED_NATIONAL_DIR,
    RAW_MIDAGRI_DIR,
)

logger = logging.getLogger("nexus.midagri")

# ── Constants ────────────────────────────────────────────────────────────────

COLLECTION_URL = (
    "https://www.gob.pe/institucion/midagri/colecciones/"
    "338-boletin-de-abastecimiento-y-precios-en-el-mercado-mayorista"
    "-de-lima-gmml-y-mercado-de-frutas-n-2-mm-n-2"
)
POULTRY_COLLECTION_URL = (
    "https://www.gob.pe/institucion/midagri/colecciones/"
    "4-boletin-diario-de-comercializacion-y-precio-de-aves"
)
USER_AGENT = "NEXUS/1.0 (academic research)"
REQUEST_DELAY = 1.5  # seconds between requests

MONTH_MAP = {
    "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
    "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
    "septiembre": "09", "setiembre": "09", "octubre": "10",
    "noviembre": "11", "diciembre": "12",
}

MONTH_MAP_REVERSE = {v: k for k, v in MONTH_MAP.items() if k != "setiembre"}

# Product category classification based on CNPA 3-digit prefix
# 011=CEREALES, 012=HORTALIZAS, 013=FRUTAS, 014=TUBERCULOS Y RAICES,
# 015=MENESTRAS/LEGUMINOSAS, 016=CONDIMENTOS Y ESPECIAS, 019=OTROS
CATEGORY_MAP = {
    "011": "cereals",
    "012": "vegetables",
    "013": "fruits",
    "014": "roots_tubers",
    "015": "legumes",
    "016": "condiments",
    "019": "other",
}

# Keyword-based fallback for category classification
CATEGORY_KEYWORDS = {
    "vegetables": [
        "lechuga", "espinaca", "brocoli", "coliflor", "col ", "repollo",
        "apio", "perejil", "cebolla", "ajo", "tomate", "pimiento",
        "aji", "pepinillo", "berenjena", "zapallo", "calabaza",
        "zanahoria", "nabo", "betarraga", "vainita", "arveja",
        "haba", "frijol", "holantao", "alcachofa", "esparrago",
        "acelga", "choclo", "maiz",
    ],
    "fruits": [
        "manzana", "naranja", "mandarina", "limon", "platano",
        "mango", "papaya", "palta", "uva", "fresa", "melon",
        "sandia", "maracuya", "chirimoya", "lucuma", "granada",
        "durazno", "pera", "membrillo", "toronja", "pina",
        "guanabana", "pepino", "coco", "higo", "tuna",
        "granadilla", "carambola", "zapote", "tangelo",
    ],
    "roots_tubers": [
        "papa", "camote", "yuca", "olluco", "oca", "mashua",
    ],
}

# ── Bulletin date/filename parsing ───────────────────────────────────────────

# Pattern: dd-mm-yyyy at end of filename (before .pdf)
_DATE_FROM_FILENAME = re.compile(r"(\d{2})-(\d{2})-(\d{4})\.pdf$", re.IGNORECASE)

# Pattern: "Lima, DD de MONTH del YYYY"
_DATE_FROM_TEXT = re.compile(
    r"Lima,?\s+(\d{1,2})\s+de\s+(\w+)\s+del?\s+(\d{4})", re.IGNORECASE
)

# Volume pattern: "ingreso de N toneladas"
_VOLUME_PATTERN = re.compile(r"ingreso de\s+([\d\s]+)\s+toneladas", re.IGNORECASE)

# Regex for text-based product row parsing (fallback)
_PRODUCT_ROW = re.compile(
    r"^(\d{5})\s+"       # CNPA code
    r"(.+?)\s+"          # Product name
    r"(GMML|MMF2)\s+"    # Market
    r"(.+)$"             # Remaining numbers
)

# Table extraction settings for pdfplumber
_TABLE_SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
}


def extract_date_from_filename(filename: str) -> str | None:
    """Extract YYYY-MM-DD from bulletin PDF filename.

    Filenames like: '...-02-02-2026.pdf' -> '2026-02-02'
    """
    m = _DATE_FROM_FILENAME.search(filename)
    if m:
        dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
        return f"{yyyy}-{mm}-{dd}"
    return None


def extract_date_from_text(text: str) -> str | None:
    """Extract YYYY-MM-DD from bulletin first-page text.

    Text like: 'Lima, 02 de febrero del 2026' -> '2026-02-02'
    """
    m = _DATE_FROM_TEXT.search(text)
    if m:
        dd, month_name, yyyy = m.group(1), m.group(2).lower(), m.group(3)
        mm = MONTH_MAP.get(month_name)
        if mm:
            return f"{yyyy}-{mm}-{int(dd):02d}"
    return None


def extract_month_year(title: str) -> str | None:
    """Extract YYYY-MM from bulletin title (Spanish month name)."""
    title_lower = title.lower()
    for month_name, month_num in MONTH_MAP.items():
        if month_name in title_lower:
            year_match = re.search(r"20[2-3]\d", title_lower)
            if year_match:
                return f"{year_match.group()}-{month_num}"
    return None


def classify_product(cnpa: str, product_name: str) -> str:
    """Classify a product into a category based on CNPA code and name."""
    prefix_3 = cnpa[:3] if len(cnpa) >= 3 else ""
    cat = CATEGORY_MAP.get(prefix_3)
    if cat:
        return cat

    name_lower = product_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return category
    return "other"


# =====================================================================
# Class 1: MidagriBulletinScraper
# =====================================================================


class MidagriBulletinScraper:
    """Download MIDAGRI wholesale price bulletins from gob.pe."""

    def __init__(
        self,
        output_dir: Path | None = None,
        manifest_path: Path | None = None,
    ):
        self.output_dir = output_dir or MIDAGRI_BULLETINS_DIR
        self.manifest_path = manifest_path or (RAW_MIDAGRI_DIR / "bulletin_manifest.csv")
        self._session = requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT

    def _load_manifest(self) -> pd.DataFrame:
        if self.manifest_path.exists():
            return pd.read_csv(self.manifest_path)
        return pd.DataFrame(columns=[
            "title", "month", "url", "pdf_url", "filename", "status",
        ])

    def _save_manifest(self, df: pd.DataFrame) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.manifest_path, index=False)

    def _fetch_page(self, url: str, params: dict | None = None) -> str | None:
        try:
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.warning("Failed to fetch %s: %s", url, e)
            return None

    def _extract_bulletin_links(self, html: str) -> list[dict]:
        """Extract monthly bulletin page links from collection HTML."""
        bulletins = []
        pattern = re.compile(
            r'<a\s+class="leading-6 font-bold"[^>]*'
            r'href="(/institucion/midagri/informes-publicaciones/[^"]+)"[^>]*>'
            r"\s*(.*?)\s*</a>",
            re.DOTALL,
        )
        for match in pattern.finditer(html):
            path, title = match.group(1), match.group(2)
            title = re.sub(r"<[^>]+>", "", title).strip()
            if not title:
                continue
            url = f"https://www.gob.pe{path}"
            bulletins.append({"url": url, "title": title})
        return bulletins

    def _extract_pdf_links(self, html: str) -> list[str]:
        """Extract PDF download links from a monthly bulletin page."""
        pattern = re.compile(
            r"https://cdn\.www\.gob\.pe/uploads/document/file/[^\s\"']+\.pdf",
            re.IGNORECASE,
        )
        return sorted(set(pattern.findall(html)))

    def _download_pdf(self, url: str, dest: Path) -> bool:
        try:
            resp = self._session.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except requests.RequestException as e:
            logger.warning("Failed to download %s: %s", url, e)
            return False

    def _get_all_bulletin_pages(
        self, max_pages: int = 10, since_date: str | None = None
    ) -> list[dict]:
        """Fetch all monthly bulletin page links from paginated collection."""
        all_bulletins = []
        for page in range(1, max_pages + 1):
            logger.info("Fetching collection page %d/%d...", page, max_pages)
            params = {"page": page} if page > 1 else None
            html = self._fetch_page(COLLECTION_URL, params=params)
            if html is None:
                logger.warning("Failed to fetch page %d, stopping", page)
                break

            bulletins = self._extract_bulletin_links(html)
            if not bulletins:
                logger.info("No bulletins on page %d, stopping", page)
                break

            all_bulletins.extend(bulletins)
            logger.info("  Found %d bulletin months on page %d", len(bulletins), page)
            time.sleep(REQUEST_DELAY)

        if since_date:
            filtered = []
            for b in all_bulletins:
                month = extract_month_year(b["title"])
                if month and month >= since_date:
                    filtered.append(b)
            logger.info("Filtered to %d months (since %s)", len(filtered), since_date)
            return filtered

        return all_bulletins

    def count_new_bulletins(self, max_pages: int = 10) -> int:
        """Dry-run: count how many new PDFs would be downloaded."""
        manifest = self._load_manifest()
        downloaded = set(manifest.get("filename", pd.Series()).dropna())

        bulletin_pages = self._get_all_bulletin_pages(max_pages=max_pages)
        new_count = 0

        for bp in bulletin_pages:
            html = self._fetch_page(bp["url"])
            time.sleep(REQUEST_DELAY)
            if html is None:
                continue
            pdf_links = self._extract_pdf_links(html)
            for pdf_url in pdf_links:
                filename = pdf_url.split("/")[-1].split("?")[0]
                if filename not in downloaded:
                    new_count += 1

        return new_count

    def scrape_new_bulletins(
        self,
        max_pages: int = 10,
        since_date: str | None = None,
        force: bool = False,
    ) -> list[dict]:
        """Download new bulletin PDFs.

        Parameters
        ----------
        max_pages : int
            Max collection pages to fetch.
        since_date : str, optional
            Only process months >= this YYYY-MM.
        force : bool
            Re-download even if file exists.

        Returns
        -------
        list[dict]
            Manifest entries for newly downloaded PDFs.
        """
        manifest_df = self._load_manifest()
        downloaded = set(manifest_df.get("filename", pd.Series()).dropna())

        bulletin_pages = self._get_all_bulletin_pages(
            max_pages=max_pages, since_date=since_date
        )
        logger.info("Total monthly bulletin pages: %d", len(bulletin_pages))

        new_entries = []
        for i, bp in enumerate(bulletin_pages):
            title = bp["title"]
            month_str = extract_month_year(title) or "unknown"
            logger.info(
                "[%d/%d] %s -> %s", i + 1, len(bulletin_pages), title[:60], month_str
            )

            html = self._fetch_page(bp["url"])
            time.sleep(REQUEST_DELAY)

            if html is None:
                new_entries.append({
                    "title": title, "month": month_str,
                    "url": bp["url"], "pdf_url": "", "filename": "",
                    "status": "fetch_failed",
                })
                continue

            pdf_links = self._extract_pdf_links(html)
            if not pdf_links:
                logger.warning("  No PDFs found for: %s", title)
                continue

            for pdf_url in pdf_links:
                filename = pdf_url.split("/")[-1].split("?")[0]

                if not force and filename in downloaded:
                    logger.debug("  Skipping (exists): %s", filename)
                    continue

                dest = self.output_dir / month_str / filename
                success = self._download_pdf(pdf_url, dest)
                entry = {
                    "title": title, "month": month_str,
                    "url": bp["url"], "pdf_url": pdf_url,
                    "filename": filename,
                    "status": "downloaded" if success else "download_failed",
                }
                new_entries.append(entry)
                downloaded.add(filename)
                time.sleep(REQUEST_DELAY)

        # Merge with existing manifest
        if new_entries:
            new_df = pd.DataFrame(new_entries)
            manifest_df = pd.concat([manifest_df, new_df], ignore_index=True)
            self._save_manifest(manifest_df)
            n_ok = sum(1 for e in new_entries if e["status"] == "downloaded")
            logger.info("Downloaded %d new PDFs (%d total entries)", n_ok, len(new_entries))

        return new_entries


# =====================================================================
# Class 2: MidagriPDFParser
# =====================================================================


class MidagriPDFParser:
    """Extract price tables from MIDAGRI daily bulletin PDFs.

    Uses pdfplumber table extraction (text strategy) as primary method,
    with text-based regex parsing as fallback. Whichever method extracts
    more product rows wins.
    """

    def __init__(
        self,
        bulletins_dir: Path | None = None,
        extracted_dir: Path | None = None,
    ):
        self.bulletins_dir = bulletins_dir or MIDAGRI_BULLETINS_DIR
        self.extracted_dir = extracted_dir or MIDAGRI_EXTRACTED_DIR
        self.parsed_manifest_path = self.extracted_dir / "parsed_manifest.csv"
        self.daily_prices_path = self.extracted_dir / "daily_prices.parquet"

    def _load_parsed_manifest(self) -> set[str]:
        if self.parsed_manifest_path.exists():
            df = pd.read_csv(self.parsed_manifest_path)
            return set(df["filename"].dropna())
        return set()

    def _save_parsed_manifest(self, parsed: set[str]) -> None:
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"filename": sorted(parsed)})
        df.to_csv(self.parsed_manifest_path, index=False)

    def _extract_date(self, pdf) -> str | None:
        try:
            text = pdf.pages[0].extract_text() or ""
            date = extract_date_from_text(text)
            if date:
                return date
        except Exception:
            pass
        return None

    def _extract_volume(self, pdf) -> float | None:
        try:
            text = pdf.pages[0].extract_text() or ""
            m = _VOLUME_PATTERN.search(text)
            if m:
                return float(m.group(1).replace(" ", "").replace("\xa0", ""))
        except Exception:
            pass
        return None

    @staticmethod
    def _clean_spaced_text(s: str) -> str:
        """Fix text with inserted spaces (e.g. 'GM M L' -> 'GMML')."""
        return re.sub(r"\s+", "", s) if s else ""

    @staticmethod
    def _standardize_product_names(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize product names across bulletins.

        Some PDFs have OCR-spaced names ('M AIZ M ARLO') while others
        have clean names ('MAIZ MARLO'). For each (cnpa, market) pair,
        pick the shortest name (clean names have no extra spaces).
        """
        if df.empty or "product" not in df.columns:
            return df

        # For each (cnpa, market), find the shortest name (fewest chars = no OCR spaces)
        canonical = (
            df.groupby(["cnpa", "market"])["product"]
            .agg(lambda names: min(names, key=len))
            .reset_index()
            .rename(columns={"product": "product_clean"})
        )

        df = df.merge(canonical, on=["cnpa", "market"], how="left")
        df["product"] = df["product_clean"].fillna(df["product"])
        df = df.drop(columns=["product_clean"])
        return df

    @staticmethod
    def _find_market_col(row: list) -> int | None:
        """Find the column index containing the market identifier (GMML/MMF2).

        The column layout varies across PDFs — market can be at col 2, 3, or 4.
        """
        for i in range(1, min(len(row), 6)):
            cleaned = re.sub(r"\s+", "", str(row[i] or "")).upper()
            if cleaned in ("GMML", "MMF2"):
                return i
        return None

    def _parse_via_tables(self, pdf) -> list[dict]:
        """Parse price rows using pdfplumber table extraction (primary).

        Adaptively detects column layout since PDF format varies:
        - Finds market column dynamically (GMML/MMF2, usually col 2-4)
        - Product name = all columns between CNPA and market
        - Today's price = second-to-last numeric column (last is daily var%)
        - Current week avg = first numeric column after market + 1
        """
        rows = []
        current_category = "other"

        for page in pdf.pages:
            try:
                tables = page.extract_tables(table_settings=_TABLE_SETTINGS)
            except Exception:
                continue

            for table in tables:
                for row in table:
                    if not row or len(row) < 8:
                        continue

                    cnpa = self._clean_spaced_text(str(row[0] or ""))

                    # Category header: 3-digit code
                    if re.match(r"^\d{3}$", cnpa):
                        current_category = CATEGORY_MAP.get(cnpa, "other")
                        continue

                    # Product row: 5-digit CNPA
                    if not re.match(r"^\d{5}$", cnpa):
                        continue

                    # Find market column dynamically
                    market_col = self._find_market_col(row)
                    if market_col is None:
                        continue
                    market = self._clean_spaced_text(str(row[market_col] or "")).upper()

                    # Product name: all cols between CNPA (col 0) and market
                    name_parts = []
                    for c in row[1:market_col]:
                        if c and str(c).strip():
                            name_parts.append(str(c).strip())
                    product = " ".join(name_parts)
                    product = re.sub(r"\s{2,}", " ", product)

                    # Extract numeric values from columns after market
                    # Layout: ...market, [prev_avg], [cur_avg], [var%], d1..d8, [daily_var%]
                    # The last column is daily var% (ends with %), second-to-last is today's price
                    after_market = row[market_col + 1:]

                    # Collect all parseable floats (skip empty, var% cells)
                    numeric_vals = []
                    for cell in after_market:
                        cell_str = str(cell or "").strip()
                        if not cell_str or cell_str.endswith("%"):
                            continue
                        try:
                            numeric_vals.append(float(cell_str))
                        except ValueError:
                            continue

                    # Need at least: prev_avg, cur_avg, + some daily prices
                    if len(numeric_vals) < 4:
                        continue

                    # Today's price = last numeric value (daily prices end last)
                    today_price = numeric_vals[-1]
                    # Current week average = second numeric value
                    cur_avg = numeric_vals[1] if len(numeric_vals) > 1 else today_price

                    if today_price <= 0 or today_price > 500:
                        continue

                    rows.append({
                        "cnpa": cnpa,
                        "product": product,
                        "market": market,
                        "price_today": today_price,
                        "price_week_avg": cur_avg,
                        "category": classify_product(cnpa, product),
                    })

        return rows

    def _parse_via_text(self, pdf) -> list[dict]:
        """Parse price rows using text extraction (fallback)."""
        rows = []
        current_category = "other"

        for page in pdf.pages:
            text = page.extract_text()
            if not text or ("GMML" not in text and "MMF2" not in text):
                continue

            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Category header (3-digit code)
                cat_match = re.match(r"^(\d{3})\s+(.+)$", line)
                if cat_match and not re.search(r"\d\.\d", line):
                    current_category = CATEGORY_MAP.get(cat_match.group(1), "other")
                    continue

                # Product row (5-digit CNPA with prices)
                m = _PRODUCT_ROW.match(line)
                if m:
                    cnpa, product, market = m.group(1), m.group(2).strip(), m.group(3)
                    nums = re.findall(r"-?[\d]+\.[\d]+|-?[\d]+", m.group(4))

                    if len(nums) >= 4:
                        try:
                            cur_avg = float(nums[1])
                            today_price = float(nums[-2])
                            if today_price <= 0 or today_price > 500:
                                continue
                            rows.append({
                                "cnpa": cnpa,
                                "product": product,
                                "market": market,
                                "price_today": today_price,
                                "price_week_avg": cur_avg,
                                "category": classify_product(cnpa, product),
                            })
                        except (ValueError, IndexError):
                            continue

        return rows

    def _parse_price_rows(self, pdf) -> list[dict]:
        """Parse product price rows using table extraction with text fallback."""
        rows_table = self._parse_via_tables(pdf)
        rows_text = self._parse_via_text(pdf)

        if len(rows_table) >= len(rows_text):
            return rows_table
        logger.debug(
            "Text extraction found more rows (%d vs %d), using text",
            len(rows_text), len(rows_table),
        )
        return rows_text

    def parse_bulletin(self, pdf_path: Path) -> pd.DataFrame:
        """Parse a single bulletin PDF into a DataFrame.

        Returns DataFrame with columns:
        [date, cnpa, product, market, price_today, price_week_avg,
         volume_tons, category, filename]
        """
        import pdfplumber

        try:
            with pdfplumber.open(pdf_path) as pdf:
                date_str = self._extract_date(pdf)
                if not date_str:
                    date_str = extract_date_from_filename(pdf_path.name)
                if not date_str:
                    logger.warning("Could not extract date from %s", pdf_path.name)
                    return pd.DataFrame()

                volume = self._extract_volume(pdf)
                rows = self._parse_price_rows(pdf)

                if not rows:
                    logger.warning("No price rows extracted from %s", pdf_path.name)
                    return pd.DataFrame()

                df = pd.DataFrame(rows)
                df["date"] = pd.Timestamp(date_str)
                df["volume_tons"] = volume
                df["filename"] = pdf_path.name

                return df

        except Exception as e:
            logger.warning("Failed to parse %s: %s", pdf_path.name, e)
            return pd.DataFrame()

    def parse_all_bulletins(self, force: bool = False) -> pd.DataFrame:
        """Parse all unparsed bulletin PDFs.

        Parameters
        ----------
        force : bool
            Re-parse already parsed PDFs.

        Returns
        -------
        pd.DataFrame
            Concatenated daily prices from all bulletins.
        """
        parsed = set() if force else self._load_parsed_manifest()
        pdf_files = sorted(self.bulletins_dir.rglob("*.pdf"))

        if not pdf_files:
            logger.warning("No PDF files found in %s", self.bulletins_dir)
            return pd.DataFrame()

        to_parse = [f for f in pdf_files if f.name not in parsed]
        logger.info(
            "PDFs: %d total, %d already parsed, %d to parse",
            len(pdf_files), len(parsed), len(to_parse),
        )

        new_dfs = []
        for i, pdf_path in enumerate(to_parse):
            if (i + 1) % 50 == 0 or i == 0:
                logger.info("Parsing PDF %d/%d: %s", i + 1, len(to_parse), pdf_path.name)
            df = self.parse_bulletin(pdf_path)
            if not df.empty:
                new_dfs.append(df)
                parsed.add(pdf_path.name)

        self._save_parsed_manifest(parsed)

        # Load existing extracted data and merge
        existing_df = pd.DataFrame()
        if not force and self.daily_prices_path.exists():
            existing_df = pd.read_parquet(self.daily_prices_path)

        if new_dfs:
            new_df = pd.concat(new_dfs, ignore_index=True)
            if not existing_df.empty:
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=["date", "cnpa", "product", "market"], keep="last"
                )
            else:
                combined = new_df
        else:
            combined = existing_df

        if not combined.empty:
            # Standardize OCR-spaced product names across bulletins
            combined = self._standardize_product_names(combined)
            # Re-deduplicate after name standardization
            combined = combined.drop_duplicates(
                subset=["date", "cnpa", "product", "market"], keep="last"
            )
            self.extracted_dir.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(self.daily_prices_path, index=False)
            logger.info(
                "Daily prices: %d records, %d unique dates",
                len(combined), combined["date"].nunique(),
            )

        return combined


# =====================================================================
# Class 3: MidagriAggregator
# =====================================================================


class MidagriAggregator:
    """Aggregate daily wholesale prices into monthly series."""

    def __init__(self, output_path: Path | None = None):
        self.output_path = output_path or (
            PROCESSED_NATIONAL_DIR / "midagri_monthly_prices.parquet"
        )

    def build_monthly_series(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate daily prices into monthly series.

        Generates 8 series:
        - MIDAGRI_ALL_AVG / MIDAGRI_VEG_AVG / MIDAGRI_FRUIT_AVG / MIDAGRI_TUBER_AVG
        - MIDAGRI_ALL_VAR / MIDAGRI_VEG_VAR / MIDAGRI_FRUIT_VAR / MIDAGRI_TUBER_VAR

        Returns long-format DataFrame matching panel schema.
        """
        if daily_df.empty:
            logger.warning("Empty daily DataFrame, cannot aggregate")
            return pd.DataFrame()

        df = daily_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year_month"] = df["date"].dt.to_period("M")

        series_defs = [
            ("MIDAGRI_ALL_AVG", "Wholesale avg price - all products", None),
            ("MIDAGRI_VEG_AVG", "Wholesale avg price - vegetables", "vegetables"),
            ("MIDAGRI_FRUIT_AVG", "Wholesale avg price - fruits", "fruits"),
            ("MIDAGRI_TUBER_AVG", "Wholesale avg price - roots & tubers", "roots_tubers"),
        ]

        monthly_rows = []
        for series_id, series_name, cat_filter in series_defs:
            subset = df if cat_filter is None else df[df["category"] == cat_filter]
            if subset.empty:
                continue

            monthly_avg = (
                subset.groupby("year_month")["price_today"]
                .mean()
                .sort_index()
            )

            for period, value in monthly_avg.items():
                monthly_rows.append({
                    "date": period.to_timestamp(),
                    "series_id": series_id,
                    "series_name": series_name,
                    "category": "wholesale_prices",
                    "value_raw": round(value, 4),
                })

        if not monthly_rows:
            return pd.DataFrame()

        avg_df = pd.DataFrame(monthly_rows)

        var_series_map = {
            "MIDAGRI_ALL_AVG": ("MIDAGRI_ALL_VAR", "Wholesale price var% - all products"),
            "MIDAGRI_VEG_AVG": ("MIDAGRI_VEG_VAR", "Wholesale price var% - vegetables"),
            "MIDAGRI_FRUIT_AVG": ("MIDAGRI_FRUIT_VAR", "Wholesale price var% - fruits"),
            "MIDAGRI_TUBER_AVG": ("MIDAGRI_TUBER_VAR", "Wholesale price var% - roots & tubers"),
        }

        var_rows = []
        for avg_id, (var_id, var_name) in var_series_map.items():
            subset = avg_df[avg_df["series_id"] == avg_id].sort_values("date")
            if len(subset) < 2:
                continue
            values = subset["value_raw"].values
            dates = subset["date"].values
            for i in range(1, len(values)):
                if values[i - 1] != 0:
                    pct_change = ((values[i] - values[i - 1]) / values[i - 1]) * 100
                    var_rows.append({
                        "date": pd.Timestamp(dates[i]),
                        "series_id": var_id,
                        "series_name": var_name,
                        "category": "wholesale_prices",
                        "value_raw": round(pct_change, 4),
                    })

        all_rows = monthly_rows + var_rows
        result = pd.DataFrame(all_rows)
        result["source"] = "MIDAGRI"
        result["frequency_original"] = "D"
        result["publication_lag_days"] = 1

        result = result.sort_values(["series_id", "date"]).reset_index(drop=True)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(self.output_path, index=False)
        logger.info(
            "Monthly series saved: %s (%d rows, %d series, %s to %s)",
            self.output_path,
            len(result),
            result["series_id"].nunique(),
            result["date"].min().strftime("%Y-%m"),
            result["date"].max().strftime("%Y-%m"),
        )

        return result

    @staticmethod
    def export_to_excel(
        daily_df: pd.DataFrame,
        monthly_df: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        """Export clean data to Excel with multiple sheets.

        Sheets:
        - daily_prices: all daily product prices
        - monthly_series: aggregated monthly series
        - product_catalog: unique products with categories
        """
        daily = daily_df.copy()
        daily["date"] = pd.to_datetime(daily["date"])

        # Clean daily table: drop internal columns, sort
        daily_out = daily[
            ["date", "cnpa", "product", "category", "market",
             "price_today", "price_week_avg", "volume_tons"]
        ].sort_values(["date", "category", "product"]).reset_index(drop=True)

        daily_out.columns = [
            "Fecha", "CNPA", "Producto", "Categoria", "Mercado",
            "Precio_Hoy_SolKg", "Precio_Prom_Semanal_SolKg", "Volumen_Ton",
        ]

        # Monthly series
        monthly_out = monthly_df[
            ["date", "series_id", "series_name", "value_raw"]
        ].copy()
        monthly_out.columns = ["Fecha", "Serie_ID", "Serie_Nombre", "Valor"]

        # Product catalog
        catalog = (
            daily[["cnpa", "product", "category", "market"]]
            .drop_duplicates()
            .sort_values(["category", "cnpa", "product"])
            .reset_index(drop=True)
        )
        catalog.columns = ["CNPA", "Producto", "Categoria", "Mercado"]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            daily_out.to_excel(writer, sheet_name="precios_diarios", index=False)
            monthly_out.to_excel(writer, sheet_name="series_mensuales", index=False)
            catalog.to_excel(writer, sheet_name="catalogo_productos", index=False)

        logger.info(
            "Excel exported: %s (%d daily rows, %d monthly rows, %d products)",
            output_path, len(daily_out), len(monthly_out), len(catalog),
        )
        return output_path


# =====================================================================
# Class 4: PoultryBulletinScraper
# =====================================================================


class PoultryBulletinScraper:
    """Download MIDAGRI daily poultry price bulletins from gob.pe.

    Two-level hierarchy:
    - Collection page lists monthly container pages (one per month)
    - Each container page holds ~20 daily PDF bulletins (business days)

    Data starts Feb 2024, covering Lima and Arequipa wholesale markets.
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        manifest_path: Path | None = None,
    ):
        self.output_dir = output_dir or MIDAGRI_POULTRY_BULLETINS_DIR
        self.manifest_path = manifest_path or (RAW_MIDAGRI_DIR / "poultry_manifest.csv")
        self._session = requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT

    def _load_manifest(self) -> pd.DataFrame:
        if self.manifest_path.exists():
            return pd.read_csv(self.manifest_path)
        return pd.DataFrame(columns=[
            "title", "month", "url", "pdf_url", "filename", "status",
        ])

    def _save_manifest(self, df: pd.DataFrame) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.manifest_path, index=False)

    def _fetch_page(self, url: str, params: dict | None = None) -> str | None:
        try:
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.warning("Failed to fetch %s: %s", url, e)
            return None

    def _extract_container_links(self, html: str) -> list[dict]:
        """Extract monthly container page links from collection HTML."""
        containers = []
        pattern = re.compile(
            r'<a\s+class="leading-6 font-bold"[^>]*'
            r'href="(/institucion/midagri/informes-publicaciones/[^"]+)"[^>]*>'
            r"\s*(.*?)\s*</a>",
            re.DOTALL,
        )
        for match in pattern.finditer(html):
            path, title = match.group(1), match.group(2)
            title = re.sub(r"<[^>]+>", "", title).strip()
            if not title:
                continue
            url = f"https://www.gob.pe{path}"
            containers.append({"url": url, "title": title})
        return containers

    def _extract_pdf_links(self, html: str) -> list[str]:
        """Extract PDF download links from a monthly container page."""
        pattern = re.compile(
            r"https://cdn\.www\.gob\.pe/uploads/document/file/[^\s\"']+\.pdf",
            re.IGNORECASE,
        )
        return sorted(set(pattern.findall(html)))

    def _download_pdf(self, url: str, dest: Path) -> bool:
        try:
            resp = self._session.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except requests.RequestException as e:
            logger.warning("Failed to download %s: %s", url, e)
            return False

    def _get_all_container_pages(
        self, max_pages: int = 5, since_date: str | None = None,
    ) -> list[dict]:
        """Fetch all monthly container page links from paginated collection."""
        all_containers = []
        for page in range(1, max_pages + 1):
            logger.info("Fetching poultry collection page %d/%d...", page, max_pages)
            params = {"page": page} if page > 1 else None
            html = self._fetch_page(POULTRY_COLLECTION_URL, params=params)
            if html is None:
                break

            containers = self._extract_container_links(html)
            if not containers:
                break

            all_containers.extend(containers)
            logger.info("  Found %d monthly containers on page %d", len(containers), page)
            time.sleep(REQUEST_DELAY)

        if since_date:
            filtered = []
            for c in all_containers:
                month = extract_month_year(c["title"])
                if month and month >= since_date:
                    filtered.append(c)
            logger.info("Filtered to %d months (since %s)", len(filtered), since_date)
            return filtered

        return all_containers

    def scrape_new_bulletins(
        self,
        max_pages: int = 5,
        since_date: str | None = None,
        force: bool = False,
    ) -> list[dict]:
        """Download new poultry bulletin PDFs.

        Parameters
        ----------
        max_pages : int
            Max collection pages to fetch.
        since_date : str, optional
            Only process months >= this YYYY-MM.
        force : bool
            Re-download even if file exists.

        Returns
        -------
        list[dict]
            Manifest entries for newly downloaded PDFs.
        """
        manifest_df = self._load_manifest()
        downloaded = set(manifest_df.get("filename", pd.Series()).dropna())

        containers = self._get_all_container_pages(
            max_pages=max_pages, since_date=since_date,
        )
        logger.info("Total monthly containers: %d", len(containers))

        new_entries = []
        for i, container in enumerate(containers):
            title = container["title"]
            month_str = extract_month_year(title) or "unknown"
            logger.info(
                "[%d/%d] %s -> %s", i + 1, len(containers), title[:60], month_str,
            )

            html = self._fetch_page(container["url"])
            time.sleep(REQUEST_DELAY)

            if html is None:
                new_entries.append({
                    "title": title, "month": month_str,
                    "url": container["url"], "pdf_url": "", "filename": "",
                    "status": "fetch_failed",
                })
                continue

            pdf_links = self._extract_pdf_links(html)
            if not pdf_links:
                logger.warning("  No PDFs found for: %s", title)
                continue

            logger.info("  Found %d daily PDFs", len(pdf_links))

            for pdf_url in pdf_links:
                filename = pdf_url.split("/")[-1].split("?")[0]

                if not force and filename in downloaded:
                    continue

                dest = self.output_dir / month_str / filename
                success = self._download_pdf(pdf_url, dest)
                entry = {
                    "title": title, "month": month_str,
                    "url": container["url"], "pdf_url": pdf_url,
                    "filename": filename,
                    "status": "downloaded" if success else "download_failed",
                }
                new_entries.append(entry)
                downloaded.add(filename)
                time.sleep(REQUEST_DELAY)

        if new_entries:
            new_df = pd.DataFrame(new_entries)
            manifest_df = pd.concat([manifest_df, new_df], ignore_index=True)
            self._save_manifest(manifest_df)
            n_ok = sum(1 for e in new_entries if e["status"] == "downloaded")
            logger.info(
                "Downloaded %d new poultry PDFs (%d total entries)", n_ok, len(new_entries),
            )

        return new_entries


# =====================================================================
# Class 5: PoultryPDFParser
# =====================================================================

# Row label patterns for the poultry bulletin tables
_POULTRY_LABELS = {
    "chicken_wholesale": re.compile(
        r"Centro[s]?\s+de\s+Distribuc", re.IGNORECASE,
    ),
    "chicken_farm": re.compile(
        r"Granja\s*\(?\s*pollo\s+vivo", re.IGNORECASE,
    ),
    "chicken_retail": re.compile(
        r"Mercados\s+Minoristas", re.IGNORECASE,
    ),
    "chicken_weight": re.compile(
        r"Peso\s+promedio", re.IGNORECASE,
    ),
    "egg": re.compile(
        r"Huevo\s+rosado", re.IGNORECASE,
    ),
    "hen_colored": re.compile(
        r"Gallina\s+colorada", re.IGNORECASE,
    ),
}

# Poultry bulletin date: "Lima, DD de MONTH del YYYY"
_POULTRY_DATE = re.compile(
    r"Lima,?\s+(\d{1,2})\s+de\s+(\w+)\s+del?\s+(\d{4})", re.IGNORECASE,
)

# Date from poultry filename: ...-DD-MM-YYYY.pdf
_POULTRY_FILENAME_DATE = re.compile(
    r"(\d{2})-(\d{2})-(\d{4})\.pdf$", re.IGNORECASE,
)


def _extract_last_price(line: str) -> float | None:
    """Extract the most recent price from a poultry table row.

    Row format: Label  7d-avg  day1  day2  Var%
    We want the last non-percentage numeric value (today's price).
    Falls back to 7-day average if daily cells are dashes.
    """
    # Split on whitespace and filter numeric tokens
    tokens = re.findall(r"-?[\d]+[.,][\d]+%?|(?<!\w)-(?!\w)", line)
    prices = []
    for tok in tokens:
        if tok == "-" or tok.endswith("%"):
            continue
        try:
            prices.append(float(tok.replace(",", ".")))
        except ValueError:
            continue
    if not prices:
        return None
    # Filter reasonable poultry prices: 0.50 to 50 S/kg
    # (chicken ~5-12, egg ~3-6, weight ~2-3)
    valid = [p for p in prices if 0.3 < p < 50]
    return valid[-1] if valid else None


class PoultryPDFParser:
    """Extract poultry price data from MIDAGRI daily bulletin PDFs.

    Each bulletin is a 2-page PDF:
    - Page 1: Lima wholesale chicken prices, egg prices, monthly summary
    - Page 2: Arequipa wholesale prices

    Extracts per-day: wholesale chicken, farm-gate chicken, retail chicken,
    egg price, hen price, and average bird weight for Lima market.
    """

    def __init__(
        self,
        bulletins_dir: Path | None = None,
        extracted_dir: Path | None = None,
    ):
        self.bulletins_dir = bulletins_dir or MIDAGRI_POULTRY_BULLETINS_DIR
        self.extracted_dir = extracted_dir or MIDAGRI_POULTRY_EXTRACTED_DIR
        self.parsed_manifest_path = self.extracted_dir / "parsed_manifest.csv"
        self.daily_prices_path = self.extracted_dir / "daily_poultry_prices.parquet"

    def _load_parsed_manifest(self) -> set[str]:
        if self.parsed_manifest_path.exists():
            df = pd.read_csv(self.parsed_manifest_path)
            return set(df["filename"].dropna())
        return set()

    def _save_parsed_manifest(self, parsed: set[str]) -> None:
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"filename": sorted(parsed)})
        df.to_csv(self.parsed_manifest_path, index=False)

    @staticmethod
    def _extract_date_from_pdf(pdf) -> str | None:
        """Extract date from first page text."""
        try:
            text = pdf.pages[0].extract_text() or ""
            m = _POULTRY_DATE.search(text)
            if m:
                dd, month_name, yyyy = m.group(1), m.group(2).lower(), m.group(3)
                mm = MONTH_MAP.get(month_name)
                if mm:
                    return f"{yyyy}-{mm}-{int(dd):02d}"
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_date_from_filename(filename: str) -> str | None:
        """Extract YYYY-MM-DD from poultry bulletin filename."""
        m = _POULTRY_FILENAME_DATE.search(filename)
        if m:
            dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
            return f"{yyyy}-{mm}-{dd}"
        return None

    def _parse_page_text(self, pdf, page_idx: int = 0) -> dict[str, float | None]:
        """Parse poultry prices from page text using label matching.

        Returns dict of {price_type: value} for each detected row.
        """
        try:
            text = pdf.pages[page_idx].extract_text() or ""
        except (IndexError, Exception):
            return {}

        results = {}
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            for label_name, pattern in _POULTRY_LABELS.items():
                if pattern.search(line):
                    price = _extract_last_price(line)
                    if price is not None:
                        results[label_name] = price
                    break

        return results

    def _parse_via_tables(self, pdf, page_idx: int = 0) -> dict[str, float | None]:
        """Parse poultry prices using pdfplumber table extraction.

        Falls back to text parsing if table extraction yields no data.
        """
        try:
            tables = pdf.pages[page_idx].extract_tables(
                table_settings=_TABLE_SETTINGS,
            )
        except Exception:
            return {}

        results = {}
        for table in tables:
            for row in table:
                if not row or len(row) < 3:
                    continue
                # Join first cells to form the row label
                label_text = " ".join(str(c or "") for c in row[:2])
                for label_name, pattern in _POULTRY_LABELS.items():
                    if label_name in results:
                        continue
                    if pattern.search(label_text):
                        # Extract numbers from remaining cells
                        prices = []
                        for cell in row[2:]:
                            cell_str = str(cell or "").strip()
                            if not cell_str or cell_str == "-" or cell_str.endswith("%"):
                                continue
                            try:
                                prices.append(float(cell_str.replace(",", ".")))
                            except ValueError:
                                continue
                        valid = [p for p in prices if 0.3 < p < 50]
                        if valid:
                            results[label_name] = valid[-1]
                        break

        return results

    def parse_bulletin(self, pdf_path: Path) -> pd.DataFrame:
        """Parse a single poultry bulletin PDF.

        Returns DataFrame with columns:
        [date, chicken_wholesale, chicken_farm, chicken_retail,
         chicken_weight, egg, hen_colored, filename]
        """
        import pdfplumber

        try:
            with pdfplumber.open(pdf_path) as pdf:
                date_str = self._extract_date_from_pdf(pdf)
                if not date_str:
                    date_str = self._extract_date_from_filename(pdf_path.name)
                if not date_str:
                    logger.warning("Could not extract date from %s", pdf_path.name)
                    return pd.DataFrame()

                # Try table extraction first, fall back to text
                prices = self._parse_via_tables(pdf, page_idx=0)
                text_prices = self._parse_page_text(pdf, page_idx=0)

                # Merge: prefer table values, fill gaps from text
                for key, val in text_prices.items():
                    if key not in prices or prices[key] is None:
                        prices[key] = val

                if not prices:
                    logger.warning("No prices extracted from %s", pdf_path.name)
                    return pd.DataFrame()

                row = {
                    "date": pd.Timestamp(date_str),
                    "chicken_wholesale": prices.get("chicken_wholesale"),
                    "chicken_farm": prices.get("chicken_farm"),
                    "chicken_retail": prices.get("chicken_retail"),
                    "chicken_weight": prices.get("chicken_weight"),
                    "egg": prices.get("egg"),
                    "hen_colored": prices.get("hen_colored"),
                    "filename": pdf_path.name,
                }

                return pd.DataFrame([row])

        except Exception as e:
            logger.warning("Failed to parse poultry bulletin %s: %s", pdf_path.name, e)
            return pd.DataFrame()

    def parse_all_bulletins(self, force: bool = False) -> pd.DataFrame:
        """Parse all unparsed poultry bulletin PDFs.

        Returns concatenated daily poultry prices.
        """
        parsed = set() if force else self._load_parsed_manifest()
        pdf_files = sorted(self.bulletins_dir.rglob("*.pdf"))

        if not pdf_files:
            logger.warning("No poultry PDF files found in %s", self.bulletins_dir)
            return pd.DataFrame()

        to_parse = [f for f in pdf_files if f.name not in parsed]
        logger.info(
            "Poultry PDFs: %d total, %d already parsed, %d to parse",
            len(pdf_files), len(parsed), len(to_parse),
        )

        new_dfs = []
        for i, pdf_path in enumerate(to_parse):
            if (i + 1) % 50 == 0 or i == 0:
                logger.info(
                    "Parsing poultry PDF %d/%d: %s", i + 1, len(to_parse), pdf_path.name,
                )
            df = self.parse_bulletin(pdf_path)
            if not df.empty:
                new_dfs.append(df)
                parsed.add(pdf_path.name)

        self._save_parsed_manifest(parsed)

        # Load existing and merge
        existing_df = pd.DataFrame()
        if not force and self.daily_prices_path.exists():
            existing_df = pd.read_parquet(self.daily_prices_path)

        if new_dfs:
            new_df = pd.concat(new_dfs, ignore_index=True)
            if not existing_df.empty:
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["date"], keep="last")
            else:
                combined = new_df
        else:
            combined = existing_df

        if not combined.empty:
            combined = combined.sort_values("date").reset_index(drop=True)
            self.extracted_dir.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(self.daily_prices_path, index=False)
            logger.info(
                "Poultry daily prices: %d records, %d unique dates",
                len(combined), combined["date"].nunique(),
            )

        return combined


# =====================================================================
# Class 6: PoultryAggregator
# =====================================================================


class PoultryAggregator:
    """Aggregate daily poultry prices into monthly panel series."""

    def __init__(self, output_path: Path | None = None):
        self.output_path = output_path or (
            PROCESSED_NATIONAL_DIR / "midagri_poultry_monthly.parquet"
        )

    def build_monthly_series(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate daily poultry prices into monthly series.

        Generates 4 series:
        - MIDAGRI_CHICKEN_AVG: monthly avg wholesale chicken price (S/kg)
        - MIDAGRI_CHICKEN_VAR: MoM % change in wholesale chicken price
        - MIDAGRI_EGG_AVG: monthly avg wholesale egg price (S/kg)
        - MIDAGRI_EGG_VAR: MoM % change in egg price

        Returns long-format DataFrame matching panel schema.
        """
        if daily_df.empty:
            logger.warning("Empty poultry DataFrame, cannot aggregate")
            return pd.DataFrame()

        df = daily_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year_month"] = df["date"].dt.to_period("M")

        series_defs = [
            ("MIDAGRI_CHICKEN_AVG", "Wholesale chicken price Lima (S/kg)", "chicken_wholesale"),
            ("MIDAGRI_EGG_AVG", "Wholesale egg price Lima (S/kg)", "egg"),
        ]

        monthly_rows = []
        for series_id, series_name, col in series_defs:
            subset = df[df[col].notna()].copy() if col in df.columns else pd.DataFrame()
            if subset.empty:
                continue

            monthly_avg = (
                subset.groupby("year_month")[col]
                .mean()
                .sort_index()
            )

            for period, value in monthly_avg.items():
                monthly_rows.append({
                    "date": period.to_timestamp(),
                    "series_id": series_id,
                    "series_name": series_name,
                    "category": "wholesale_prices",
                    "value_raw": round(value, 4),
                })

        if not monthly_rows:
            return pd.DataFrame()

        avg_df = pd.DataFrame(monthly_rows)

        # Compute MoM variation series
        var_series_map = {
            "MIDAGRI_CHICKEN_AVG": ("MIDAGRI_CHICKEN_VAR", "Wholesale chicken price var% Lima"),
            "MIDAGRI_EGG_AVG": ("MIDAGRI_EGG_VAR", "Wholesale egg price var% Lima"),
        }

        var_rows = []
        for avg_id, (var_id, var_name) in var_series_map.items():
            subset = avg_df[avg_df["series_id"] == avg_id].sort_values("date")
            if len(subset) < 2:
                continue
            values = subset["value_raw"].values
            dates = subset["date"].values
            for i in range(1, len(values)):
                if values[i - 1] != 0:
                    pct_change = ((values[i] - values[i - 1]) / values[i - 1]) * 100
                    var_rows.append({
                        "date": pd.Timestamp(dates[i]),
                        "series_id": var_id,
                        "series_name": var_name,
                        "category": "wholesale_prices",
                        "value_raw": round(pct_change, 4),
                    })

        all_rows = monthly_rows + var_rows
        result = pd.DataFrame(all_rows)
        result["source"] = "MIDAGRI"
        result["frequency_original"] = "D"
        result["publication_lag_days"] = 1

        result = result.sort_values(["series_id", "date"]).reset_index(drop=True)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(self.output_path, index=False)
        logger.info(
            "Poultry monthly series saved: %s (%d rows, %d series, %s to %s)",
            self.output_path,
            len(result),
            result["series_id"].nunique(),
            result["date"].min().strftime("%Y-%m"),
            result["date"].max().strftime("%Y-%m"),
        )

        return result

    @staticmethod
    def export_to_excel(
        daily_df: pd.DataFrame,
        monthly_df: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        """Export clean poultry data to Excel with multiple sheets.

        Sheets:
        - precios_diarios: daily chicken/egg prices
        - series_mensuales: aggregated monthly series
        """
        daily = daily_df.copy()
        daily["date"] = pd.to_datetime(daily["date"])

        # Clean daily table
        daily_cols = ["date", "chicken_wholesale", "chicken_farm",
                      "chicken_retail", "chicken_weight", "egg", "hen_colored"]
        daily_out = daily[[c for c in daily_cols if c in daily.columns]].copy()
        daily_out = daily_out.sort_values("date").reset_index(drop=True)

        col_rename = {
            "date": "Fecha",
            "chicken_wholesale": "Pollo_Mayorista_SolKg",
            "chicken_farm": "Pollo_Granja_SolKg",
            "chicken_retail": "Pollo_Minorista_SolKg",
            "chicken_weight": "Peso_Promedio_Kg",
            "egg": "Huevo_Rosado_SolKg",
            "hen_colored": "Gallina_Colorada_SolKg",
        }
        daily_out = daily_out.rename(columns=col_rename)

        # Monthly series
        monthly_out = monthly_df[
            ["date", "series_id", "series_name", "value_raw"]
        ].copy()
        monthly_out.columns = ["Fecha", "Serie_ID", "Serie_Nombre", "Valor"]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            daily_out.to_excel(writer, sheet_name="precios_diarios", index=False)
            monthly_out.to_excel(writer, sheet_name="series_mensuales", index=False)

        logger.info(
            "Poultry Excel exported: %s (%d daily rows, %d monthly rows)",
            output_path, len(daily_out), len(monthly_out),
        )
        return output_path
