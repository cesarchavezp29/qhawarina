"""BCRP (Banco Central de Reserva del Perú) API client.

Downloads macroeconomic time series from the BCRP statistical API.

API endpoint pattern:
    https://estadisticas.bcrp.gob.pe/estadisticas/series/api/{codes}/{format}/{start}/{end}/{lang}

Constraints:
    - Maximum 10 series per request
    - Date format: {year}-{month} (e.g., '2007-1')
    - Response dates in Spanish: 'Ene.2024', 'Feb.2024', etc.
    - Missing values encoded as 'n.d.'
    - Rate limiting: 1-second delay between requests
    - API may return HTML error pages instead of JSON
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import pandas as pd
import yaml

from src.utils.dates import format_bcrp_date, parse_bcrp_period
from src.utils.io import save_parquet, ensure_dir

logger = logging.getLogger(__name__)


class BCRPClient:
    """Synchronous client for the BCRP statistical API."""

    BASE_URL = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
    }

    def __init__(
        self,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        request_delay: float = 1.5,
        timeout: float = 30.0,
        lang: str = "esp",
    ):
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.request_delay = request_delay
        self.timeout = timeout
        self.lang = lang
        self._last_request_time: float = 0.0

    def _build_url(self, codes: list[str], start: str, end: str) -> str:
        """Build the API URL for a batch of series codes.

        Args:
            codes: List of series codes (max 10).
            start: Start date as '{year}-{month}' (e.g., '2007-1').
            end: End date as '{year}-{month}'.
        """
        codes_str = "-".join(codes)
        return f"{self.BASE_URL}/{codes_str}/json/{start}/{end}/{self.lang}"

    def _rate_limit(self) -> None:
        """Enforce minimum delay between requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)

    def _fetch_json(self, url: str) -> dict[str, Any]:
        """Fetch JSON from URL with retry logic.

        Handles:
            - HTTP errors with exponential backoff
            - HTML error pages (non-JSON responses)
            - Connection timeouts
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            self._rate_limit()
            try:
                with httpx.Client(timeout=self.timeout, headers=self.DEFAULT_HEADERS) as client:
                    response = client.get(url)
                    self._last_request_time = time.monotonic()

                    response.raise_for_status()

                    # BCRP API returns JSON with Content-Type: text/html.
                    # Try to parse as JSON first; only raise if it's
                    # actually an HTML error page.
                    text = response.text.strip()
                    if text.startswith("{"):
                        import json
                        return json.loads(text)

                    # Genuine HTML error page
                    raise ValueError(
                        f"BCRP returned HTML error page (not JSON). "
                        f"URL may be invalid: {url}"
                    )

            except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as e:
                last_error = e
                wait = self.retry_backoff ** attempt
                logger.warning(
                    "BCRP request failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1, self.max_retries, e, wait,
                )
                time.sleep(wait)

        raise ConnectionError(
            f"Failed to fetch from BCRP after {self.max_retries} attempts: {last_error}"
        )

    def _parse_response(self, data: dict[str, Any]) -> pd.DataFrame:
        """Parse BCRP JSON response into a DataFrame.

        Response structure:
        {
            "config": {"series": [{"name": "...", ...}]},
            "periods": [
                {"name": "Ene.2007", "values": ["1234.56", "n.d."]},
                ...
            ]
        }
        """
        config = data.get("config", {})
        series_meta = config.get("series", [])
        periods = data.get("periods", [])

        if not periods:
            logger.warning("BCRP response contains no periods.")
            return pd.DataFrame()

        # Extract series names from config
        series_names = [s.get("name", f"series_{i}") for i, s in enumerate(series_meta)]

        rows = []
        for period in periods:
            period_name = period.get("name", "")
            parsed_date = parse_bcrp_period(period_name)

            if parsed_date is None:
                logger.warning("Could not parse period: '%s'. Skipping.", period_name)
                continue

            values = period.get("values", [])
            for i, raw_value in enumerate(values):
                series_name = series_names[i] if i < len(series_names) else f"series_{i}"

                # Handle missing values: "n.d.", empty strings, None
                value = _parse_value(raw_value)

                rows.append({
                    "date": parsed_date,
                    "series_name": series_name,
                    "series_index": i,
                    "value": value,
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def fetch_series(
        self,
        codes: list[str],
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
    ) -> pd.DataFrame:
        """Fetch one or more series from BCRP API.

        Downloads each series individually to avoid BCRP's reordering
        of series within batch requests. This is slower but guarantees
        correct code-to-data mapping.

        Args:
            codes: Series codes.
            start_year: Start year.
            start_month: Start month.
            end_year: End year.
            end_month: End month.

        Returns:
            DataFrame with columns: [date, series_name, series_code, value]
        """
        start = format_bcrp_date(start_year, start_month)
        end = format_bcrp_date(end_year, end_month)

        all_frames = []
        for code in codes:
            url = self._build_url([code], start, end)
            logger.info("Fetching series %s from BCRP", code)

            try:
                data = self._fetch_json(url)
            except ConnectionError as e:
                logger.error("Failed to fetch %s: %s", code, e)
                continue

            df = self._parse_response(data)

            if not df.empty:
                df["series_code"] = code
                all_frames.append(df)

        if not all_frames:
            return pd.DataFrame(columns=["date", "series_name", "series_code", "value"])

        return pd.concat(all_frames, ignore_index=True)

    def verify_series(
        self,
        code: str,
        test_year: int = 2023,
        test_month: int = 1,
    ) -> dict[str, Any]:
        """Test a single series code with a small date range.

        Returns dict with keys: code, valid, name, sample_count, error
        """
        start = format_bcrp_date(test_year, test_month)
        end = format_bcrp_date(test_year, 6)

        try:
            url = self._build_url([code], start, end)
            data = self._fetch_json(url)
            df = self._parse_response(data)

            if df.empty:
                return {"code": code, "valid": False, "name": None,
                        "sample_count": 0, "error": "No data returned"}

            name = df["series_name"].iloc[0] if not df.empty else None
            non_null = df["value"].notna().sum()

            return {"code": code, "valid": True, "name": name,
                    "sample_count": int(non_null), "error": None}

        except Exception as e:
            return {"code": code, "valid": False, "name": None,
                    "sample_count": 0, "error": str(e)}

    def verify_all_series(self, catalog_path: Path) -> pd.DataFrame:
        """Verify all series codes in the catalog YAML file.

        Updates the catalog with verification status and returns a summary.
        """
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = yaml.safe_load(f)

        results = []
        national = catalog.get("national", {})

        for category_key, category in national.items():
            series_list = category.get("series", [])
            for entry in series_list:
                code = entry["code"]
                logger.info("Verifying series %s (%s)...", code, entry.get("name", ""))
                result = self.verify_series(code)
                result["category"] = category_key
                result["expected_name"] = entry.get("name", "")
                results.append(result)

                # Update status in catalog
                entry["status"] = "verified" if result["valid"] else "invalid"
                if result["name"]:
                    entry["api_name"] = result["name"]

        # Write updated catalog
        with open(catalog_path, "w", encoding="utf-8") as f:
            yaml.dump(catalog, f, allow_unicode=True, default_flow_style=False,
                      sort_keys=False)

        return pd.DataFrame(results)

    def download_national_series(
        self,
        catalog_path: Path,
        output_dir: Path,
        start_year: int = 2004,
        start_month: int = 1,
        end_year: int = 2025,
        end_month: int = 12,
    ) -> pd.DataFrame:
        """Download all verified national series from the catalog.

        Saves individual series as parquet files and returns combined DataFrame.
        """
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = yaml.safe_load(f)

        ensure_dir(output_dir)
        national = catalog.get("national", {})

        all_codes = []
        code_metadata = {}

        for category_key, category in national.items():
            for entry in category.get("series", []):
                code = entry["code"]
                all_codes.append(code)
                code_metadata[code] = {
                    "category": category_key,
                    "name": entry.get("name", ""),
                    "frequency": category.get("frequency", "monthly"),
                }

        logger.info("Downloading %d national series from BCRP...", len(all_codes))

        df = self.fetch_series(
            all_codes,
            start_year, start_month,
            end_year, end_month,
        )

        if df.empty:
            logger.warning("No data downloaded from BCRP.")
            return df

        # Enrich with metadata
        df["category"] = df["series_code"].map(
            lambda c: code_metadata.get(c, {}).get("category", "unknown")
        )
        df["source"] = "BCRP"
        df["frequency_original"] = "M"

        # Save combined file
        combined_path = output_dir / "bcrp_national_all.parquet"
        save_parquet(df, combined_path)
        logger.info("Saved combined national data to %s", combined_path)

        # Save per-category files
        for category_key in df["category"].unique():
            cat_df = df[df["category"] == category_key]
            cat_path = output_dir / f"bcrp_national_{category_key}.parquet"
            save_parquet(cat_df, cat_path)
            logger.info("Saved %s (%d rows) to %s", category_key, len(cat_df), cat_path)

        return df


class AsyncBCRPClient:
    """Asynchronous client for the BCRP API using httpx.AsyncClient."""

    BASE_URL = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"

    def __init__(
        self,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        request_delay: float = 1.0,
        timeout: float = 30.0,
        lang: str = "esp",
    ):
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.request_delay = request_delay
        self.timeout = timeout
        self.lang = lang
        self._last_request_time: float = 0.0

    def _build_url(self, codes: list[str], start: str, end: str) -> str:
        codes_str = "-".join(codes)
        return f"{self.BASE_URL}/{codes_str}/json/{start}/{end}/{self.lang}"

    async def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.request_delay:
            await asyncio.sleep(self.request_delay - elapsed)

    async def _fetch_json(self, url: str) -> dict[str, Any]:
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            await self._rate_limit()
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(url)
                    self._last_request_time = time.monotonic()

                    response.raise_for_status()

                    text = response.text.strip()
                    if text.startswith("{"):
                        import json
                        return json.loads(text)

                    raise ValueError(
                        f"BCRP returned HTML error page (not JSON). URL: {url}"
                    )

            except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as e:
                last_error = e
                wait = self.retry_backoff ** attempt
                logger.warning(
                    "Async BCRP request failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1, self.max_retries, e, wait,
                )
                await asyncio.sleep(wait)

        raise ConnectionError(
            f"Failed to fetch from BCRP after {self.max_retries} attempts: {last_error}"
        )

    async def fetch_series(
        self,
        codes: list[str],
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
    ) -> pd.DataFrame:
        """Fetch series asynchronously, one per request.

        Downloads each series individually to avoid BCRP's reordering.
        Requests are sequential (rate-limited).
        """
        start = format_bcrp_date(start_year, start_month)
        end = format_bcrp_date(end_year, end_month)

        all_frames = []
        for code in codes:
            url = self._build_url([code], start, end)
            logger.info("Async fetching series %s", code)

            try:
                data = await self._fetch_json(url)
            except ConnectionError as e:
                logger.error("Failed to fetch %s: %s", code, e)
                continue

            df = _parse_response_data(data)

            if not df.empty:
                df["series_code"] = code
                all_frames.append(df)

        if not all_frames:
            return pd.DataFrame(columns=["date", "series_name", "series_code", "value"])

        return pd.concat(all_frames, ignore_index=True)


# ── Helper functions (shared between sync and async clients) ──────────────────

def _parse_value(raw: Any) -> Optional[float]:
    """Parse a raw BCRP value string into a float.

    Handles: 'n.d.', empty strings, None, whitespace, commas as thousands sep.
    """
    if raw is None:
        return None

    if isinstance(raw, (int, float)):
        return float(raw)

    s = str(raw).strip()

    if not s or s.lower() == "n.d." or s == "-":
        return None

    # Remove thousands separators (BCRP sometimes uses commas or spaces)
    s = s.replace(",", "").replace(" ", "")

    try:
        return float(s)
    except ValueError:
        logger.debug("Could not parse value: '%s'", raw)
        return None


def _parse_response_data(data: dict[str, Any]) -> pd.DataFrame:
    """Parse BCRP JSON response into a DataFrame (standalone function)."""
    config = data.get("config", {})
    series_meta = config.get("series", [])
    periods = data.get("periods", [])

    if not periods:
        return pd.DataFrame()

    series_names = [s.get("name", f"series_{i}") for i, s in enumerate(series_meta)]

    rows = []
    for period in periods:
        period_name = period.get("name", "")
        parsed_date = parse_bcrp_period(period_name)

        if parsed_date is None:
            logger.warning("Could not parse period: '%s'", period_name)
            continue

        values = period.get("values", [])
        for i, raw_value in enumerate(values):
            series_name = series_names[i] if i < len(series_names) else f"series_{i}"
            value = _parse_value(raw_value)
            rows.append({
                "date": parsed_date,
                "series_name": series_name,
                "series_index": i,
                "value": value,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_series_codes(catalog_path: Path, section: str = "national") -> list[str]:
    """Load all series codes from the catalog YAML for a given section."""
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)

    codes = []
    section_data = catalog.get(section, {})
    for category in section_data.values():
        if isinstance(category, dict):
            for entry in category.get("series", []):
                codes.append(entry["code"])
    return codes
