"""Peru fiscal calendar helpers and date parsing utilities."""

import re
from datetime import date, datetime
from typing import Optional

# BCRP API returns dates in Spanish abbreviated format
SPANISH_MONTHS = {
    "Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Ago": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dic": 12,
}

SPANISH_MONTHS_FULL = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
    "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
    "Septiembre": 9, "Setiembre": 9, "Octubre": 10,
    "Noviembre": 11, "Diciembre": 12,
}

# Reverse mapping: month number → BCRP API format
MONTH_TO_BCRP = {v: k for k, v in SPANISH_MONTHS.items()}


def parse_bcrp_period(period_str: str) -> Optional[date]:
    """Parse a BCRP period string like 'Ene.2024' or 'T1.2024' into a date.

    Monthly format: 'Ene.2024' → 2024-01-01
    Quarterly format: 'T1.2024' → 2024-01-01 (start of quarter)

    Returns None if parsing fails.
    """
    period_str = period_str.strip()

    # Monthly: "Ene.2024" or "Ene2024" or "Ene.24"
    monthly_match = re.match(r"([A-Za-záéíóú]+)\.?(\d{2,4})", period_str)
    if monthly_match:
        month_str, year_str = monthly_match.groups()
        month_str = month_str.capitalize()

        month = SPANISH_MONTHS.get(month_str)
        if month is None:
            month = SPANISH_MONTHS_FULL.get(month_str)
        if month is None:
            return None

        year = int(year_str)
        if year < 100:
            year += 2000 if year < 50 else 1900

        return date(year, month, 1)

    # Quarterly: "T1.2024" or "T1.24"
    quarterly_match = re.match(r"T(\d)\.?(\d{2,4})", period_str)
    if quarterly_match:
        quarter, year = int(quarterly_match.group(1)), int(quarterly_match.group(2))
        if year < 100:
            year += 2000 if year < 50 else 1900
        month = (quarter - 1) * 3 + 1
        return date(year, month, 1)

    return None


def format_bcrp_date(year: int, month: int) -> str:
    """Format a date for BCRP API requests.

    BCRP API expects: '{year}-{month}' e.g., '2007-1' for January 2007.
    """
    return f"{year}-{month}"


def quarter_of(d: date) -> int:
    """Return the quarter (1-4) for a given date."""
    return (d.month - 1) // 3 + 1


def start_of_quarter(year: int, quarter: int) -> date:
    """Return the first day of the given quarter."""
    month = (quarter - 1) * 3 + 1
    return date(year, month, 1)


def end_of_quarter(year: int, quarter: int) -> date:
    """Return the last day of the given quarter."""
    if quarter == 4:
        return date(year, 12, 31)
    next_q_start = start_of_quarter(year, quarter + 1)
    return date(next_q_start.year, next_q_start.month, 1) - __import__("datetime").timedelta(days=1)


def fiscal_year(d: date) -> int:
    """Peru fiscal year equals calendar year."""
    return d.year


def generate_monthly_range(start_year: int, start_month: int,
                           end_year: int, end_month: int) -> list[date]:
    """Generate a list of first-of-month dates for the given range (inclusive)."""
    dates = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        dates.append(date(year, month, 1))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return dates
