"""INEI data ingestion — ENAHO microdata download and poverty computation.

Downloads ENAHO Sumaria module via the `enahodata` package and computes
departmental poverty indicators for use as target variables.

Output: data/targets/poverty_departmental.parquet
Schema: [year, department_code, department_name, poverty_rate,
         extreme_poverty_rate, mean_consumption, gini]
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    RAW_ENAHO_DIR, ENAHO_START_YEAR, TARGETS_DIR, DEPARTMENTS,
)
from src.utils.io import save_parquet

logger = logging.getLogger(__name__)

# ENAHO Sumaria module number (contains household income/consumption aggregates)
SUMARIA_MODULE = "34"

# Available years (2004-2024; enahodata covers 2004-2023, 2024 added manually)
ENAHO_AVAILABLE_YEARS = list(range(2004, 2025))  # 2004-2024


# ── Statistical helpers ──────────────────────────────────────────────────────

def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted mean, ignoring NaN values.

    Args:
        values: Array of values.
        weights: Array of corresponding weights.

    Returns:
        Weighted mean as float. Returns NaN if no valid observations.
    """
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def weighted_gini(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted Gini coefficient.

    Uses the covariance formula: G = (2 * cov(y, F(y))) / mean(y)
    where F(y) is the weighted cumulative distribution.

    Args:
        values: Array of income/consumption values.
        weights: Array of corresponding weights.

    Returns:
        Gini coefficient in [0, 1]. Returns NaN if insufficient data.
    """
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    v = values[mask]
    w = weights[mask]

    if len(v) < 2 or v.sum() <= 0:
        return float("nan")

    # Sort by values
    order = np.argsort(v)
    v = v[order]
    w = w[order]

    # Cumulative weight fractions
    cum_w = np.cumsum(w)
    total_w = cum_w[-1]
    # Midpoint ranks (fraction of population at or below)
    ranks = (cum_w - w / 2) / total_w

    # Weighted mean
    mu = np.average(v, weights=w)
    if mu <= 0:
        return float("nan")

    # Gini = 2 * cov(v, ranks) / mu
    mean_rank = np.average(ranks, weights=w)
    cov = np.average((v - mu) * (ranks - mean_rank), weights=w)
    gini = 2 * cov / mu

    return float(np.clip(gini, 0.0, 1.0))


# ── Expenditure groups per deflator (from INEI do-file) ─────────────────────

# Each key is a temporal deflator (i01–i08); values are sumaria columns to sum.
EXPENDITURE_GROUPS = {
    "i01": [  # Food, beverages, tobacco
        "gru11hd", "gru12hd1", "gru12hd2", "gru13hd1", "gru13hd2", "gru13hd3",
        "g05hd", "g05hd1", "g05hd2", "g05hd3", "g05hd4", "g05hd5", "g05hd6",
        "ig06hd", "sg23", "sig24", "sg25", "sig26", "sg27", "sig28",
        "gru14hd", "gru14hd1", "gru14hd2", "gru14hd3", "gru14hd4", "gru14hd5",
    ],
    "i02": [  # Clothing & footwear
        "gru21hd", "gru22hd1", "gru22hd2", "gru23hd1", "gru23hd2", "gru23hd3",
        "gru24hd",
    ],
    "i03": [  # Housing, rent & fuel
        "gru31hd", "gru32hd1", "gru32hd2", "gru33hd1", "gru33hd2", "gru33hd3",
        "gru34hd",
    ],
    "i04": [  # Furniture & household equipment
        "gru41hd", "gru42hd1", "gru42hd2", "gru43hd1", "gru43hd2", "gru43hd3",
        "gru44hd", "sg421", "sg42d1", "sg423", "sg42d3",
    ],
    "i05": [  # Health care
        "gru51hd", "gru52hd1", "gru52hd2", "gru53hd1", "gru53hd2", "gru53hd3",
        "gru54hd",
    ],
    "i06": [  # Transport & communications
        "gru61hd", "gru62hd1", "gru62hd2", "gru63hd1", "gru63hd2", "gru63hd3",
        "gru64hd", "g07hd", "ig08hd", "sg422", "sg42d2",
    ],
    "i07": [  # Recreation & culture
        "gru71hd", "gru72hd1", "gru72hd2", "gru73hd1", "gru73hd2", "gru73hd3",
        "gru74hd", "sg42", "sg42d",
    ],
    "i08": [  # Other goods & services
        "gru81hd", "gru82hd1", "gru82hd2", "gru83hd1", "gru83hd2", "gru83hd3",
        "gru84hd",
    ],
}


def _compute_dominioa(dominio: pd.Series, estrato: pd.Series,
                      dpto: pd.Series) -> pd.Series:
    """Compute INEI's 17-level geographic domain (dominioA).

    Replicates the Stata do-file logic exactly.
    """
    dom = dominio.astype(int)
    est = estrato.astype(int)
    dpt = dpto.astype(int)

    # area: 1=urban (estrato<6), 2=rural
    # Lima Metropolitana (dominio=8) is forced urban
    area = pd.Series(2, index=dom.index)
    area[est < 6] = 1
    area[dom == 8] = 1

    result = pd.Series(np.nan, index=dom.index)
    result[(dom == 1) & (area == 1)] = 1   # Costa norte urbana
    result[(dom == 1) & (area == 2)] = 2   # Costa norte rural
    result[(dom == 2) & (area == 1)] = 3   # Costa centro urbana
    result[(dom == 2) & (area == 2)] = 4   # Costa centro rural
    result[(dom == 3) & (area == 1)] = 5   # Costa sur urbana
    result[(dom == 3) & (area == 2)] = 6   # Costa sur rural
    result[(dom == 4) & (area == 1)] = 7   # Sierra norte urbana
    result[(dom == 4) & (area == 2)] = 8   # Sierra norte rural
    result[(dom == 5) & (area == 1)] = 9   # Sierra centro urbana
    result[(dom == 5) & (area == 2)] = 10  # Sierra centro rural
    result[(dom == 6) & (area == 1)] = 11  # Sierra sur urbana
    result[(dom == 6) & (area == 2)] = 12  # Sierra sur rural
    result[(dom == 7) & (area == 1)] = 13  # Selva alta urbana
    result[(dom == 7) & (area == 2)] = 14  # Selva alta rural
    # Selva baja: Loreto(16), Madre de Dios(17), Ucayali(25)
    selva_baja = dpt.isin([16, 17, 25])
    result[(dom == 7) & selva_baja & (area == 1)] = 15  # Selva baja urbana
    result[(dom == 7) & selva_baja & (area == 2)] = 16  # Selva baja rural
    result[dom == 8] = 17  # Lima Metropolitana

    return result.astype(int)


# ── ENAHO Client ─────────────────────────────────────────────────────────────

class ENAHOClient:
    """Client for downloading and processing ENAHO Sumaria data."""

    def __init__(self, output_dir: Path = RAW_ENAHO_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._deflators = None  # cached (temporal_df, spatial_df)

    def available_years(self) -> list[int]:
        """Return years available for download."""
        return list(ENAHO_AVAILABLE_YEARS)

    def _find_sumaria_dir(self, year: int) -> Optional[Path]:
        """Find the best directory containing Sumaria data for a year.

        Searches multiple patterns, preferring manual downloads (which may
        contain poverty lines missing from enahodata versions).
        """
        # Priority 1: manual download (sumaria_{year}/*)
        manual_dir = self.output_dir / f"sumaria_{year}"
        if manual_dir.exists():
            for subdir in sorted(manual_dir.iterdir()):
                if subdir.is_dir() and any(subdir.glob("*.dta")):
                    return subdir

        # Priority 2: enahodata format (modulo_34_{year})
        enaho_dir = self.output_dir / f"modulo_{SUMARIA_MODULE}_{year}"
        if enaho_dir.exists() and any(enaho_dir.glob("*.dta")):
            return enaho_dir

        return None

    def downloaded_years(self) -> list[int]:
        """Return years already downloaded locally."""
        years = []
        for y in ENAHO_AVAILABLE_YEARS:
            if self._find_sumaria_dir(y) is not None:
                years.append(y)
        return sorted(years)

    def missing_years(self) -> list[int]:
        """Return years available but not yet downloaded."""
        downloaded = set(self.downloaded_years())
        return [y for y in ENAHO_AVAILABLE_YEARS if y not in downloaded]

    def download_new_years(self, force: bool = False) -> list[int]:
        """Download any missing ENAHO Sumaria years.

        Args:
            force: If True, re-download all years.

        Returns:
            List of years that were downloaded.
        """
        try:
            from enahodata.enahodata import enahodata
        except ImportError:
            logger.error(
                "enahodata package not installed. "
                "Install with: pip install enahodata"
            )
            return []

        if force:
            to_download = self.available_years()
        else:
            to_download = self.missing_years()

        if not to_download:
            logger.info("All ENAHO years already downloaded.")
            return []

        logger.info("Downloading ENAHO Sumaria for years: %s", to_download)

        years_str = [str(y) for y in to_download]
        enahodata(
            modulos=[SUMARIA_MODULE],
            anios=years_str,
            output_dir=str(self.output_dir),
            descomprimir=True,
            only_dta=True,
            overwrite=force,
            verbose=True,
        )

        # Return which years were actually downloaded
        downloaded = []
        for y in to_download:
            if self._find_sumaria_dir(y) is not None:
                downloaded.append(y)

        logger.info("Successfully downloaded %d years.", len(downloaded))
        return downloaded

    def read_sumaria(self, year: int) -> Optional[pd.DataFrame]:
        """Read the Sumaria .dta file for a given year.

        Returns DataFrame with at minimum: ubigeo, factor07 (weight),
        gashog2d (household consumption), linea/linpe (poverty lines).
        Returns None if file not found.
        """
        year_dir = self._find_sumaria_dir(year)
        if year_dir is None:
            logger.warning("ENAHO directory not found for year %d", year)
            return None

        dta_files = list(year_dir.glob("*.dta"))
        if not dta_files:
            logger.warning("No .dta files found for year %d in %s", year, year_dir)
            return None

        # Find the sumaria file (usually named like 'sumaria-*.dta' or 'enaho01-sumaria*.dta')
        # Prefer the main file over deflator variants (-12, -12g) which may
        # lack poverty-line columns.
        sumaria_files = [f for f in dta_files if "sumaria" in f.name.lower()]
        sumaria_file = None
        if sumaria_files:
            # Sort by name length — main file (sumaria-YYYY.dta) is shortest
            sumaria_files.sort(key=lambda f: len(f.name))
            sumaria_file = sumaria_files[0]

        if sumaria_file is None:
            # Fall back to first .dta file
            sumaria_file = dta_files[0]
            logger.info(
                "No file with 'sumaria' in name for year %d, using %s",
                year, sumaria_file.name,
            )

        logger.info("Reading ENAHO Sumaria for %d: %s", year, sumaria_file.name)
        return pd.read_stata(sumaria_file, convert_categoricals=False)

    # ── Deflation infrastructure ────────────────────────────────────────────

    def _load_deflators(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load INEI temporal and spatial deflator files.

        Returns (temporal_df, spatial_df).  Cached after first load.
        """
        if self._deflators is not None:
            return self._deflators

        # Search for deflator files in sumaria_* directories
        bases_dirs = list(self.output_dir.glob("sumaria_*/*/Gasto*/Bases"))
        if not bases_dirs:
            bases_dirs = list(self.output_dir.glob("sumaria_*/*/Bases"))
        if not bases_dirs:
            raise FileNotFoundError(
                "Deflator files not found.  Expected in "
                "<enaho_dir>/sumaria_YYYY/*/Gasto*/Bases/"
            )

        bases = sorted(bases_dirs)[-1]  # latest year's deflators

        temporal_file = list(bases.glob("deflactores_base*.dta"))
        spatial_file = list(bases.glob("despacial_ld*.dta"))
        if not temporal_file or not spatial_file:
            raise FileNotFoundError(
                f"Deflator .dta files not found in {bases}"
            )

        temporal = pd.read_stata(temporal_file[0])
        spatial = pd.read_stata(spatial_file[0])
        logger.info(
            "Loaded deflators from %s: temporal=%d rows, spatial=%d rows",
            bases, len(temporal), len(spatial),
        )
        self._deflators = (temporal, spatial)
        return self._deflators

    def _compute_real_consumption(self, df: pd.DataFrame, year: int) -> pd.Series:
        """Compute real per-capita monthly consumption following INEI do-file.

        Applies temporal (i01-i08) and spatial (ld) deflators to 8
        expenditure groups, sums to total real per-capita monthly
        consumption at constant Lima Metropolitan prices.

        Returns a Series of real per-capita monthly consumption aligned
        with df's index.
        """
        temporal, spatial = self._load_deflators()

        # Department code (numeric, Callao=7→15)
        dpto = df["ubigeo"].astype(str).str[:2].astype(int)
        dpto = dpto.where(dpto != 7, 15)

        # Merge temporal deflators by (year, dpto)
        defl_year = temporal[temporal["aniorec"] == year].copy()
        if defl_year.empty:
            logger.warning("No temporal deflators for year %d; using nominal", year)
            return df["gashog2d"] / (df["mieperho"].astype(float).clip(lower=1) * 12)

        defl_map = defl_year.set_index("dpto")
        i_cols = {col: defl_map[col] for col in defl_map.columns
                  if col.startswith("i0")}

        # Merge spatial deflator by dominioA
        dominioa = _compute_dominioa(df["dominio"], df["estrato"], dpto)
        ld_map = spatial.set_index("dominioA")["ld"]
        ld = dominioa.map(ld_map).fillna(1.0)

        # Map temporal deflators to each household via dpto
        i_values = {}
        for col_name in i_cols:
            i_values[col_name] = dpto.map(i_cols[col_name]).fillna(1.0)

        # Denominator per household: p * mieperho * ld * deflator
        p = 12.0
        mie = df["mieperho"].astype(float).clip(lower=1)
        base_denom = p * mie * ld

        # Sum deflated expenditure groups
        total = pd.Series(0.0, index=df.index)
        for deflator_name, col_list in EXPENDITURE_GROUPS.items():
            defl = i_values.get(deflator_name, pd.Series(1.0, index=df.index))
            denom = base_denom * defl
            group_sum = pd.Series(0.0, index=df.index)
            for col in col_list:
                if col in df.columns:
                    group_sum += df[col].fillna(0)
            total += group_sum / denom

        return total

    def _compute_real_income(self, df: pd.DataFrame, year: int,
                              income_var: str = "inghog1d") -> pd.Series:
        """Compute real per-capita monthly income following INEI methodology.

        Uses the i00 general deflator and spatial deflator ld.
        Formula: income_var / (12 * mieperho * ld * i00)

        Parameters
        ----------
        income_var : str
            Income variable to deflate.  ``inghog1d`` (default) matches
            INEI published income tables (I.15/I.19).  ``inghog2d`` is
            used in the poverty do-file for Gini computation.

        Returns a Series of real per-capita monthly income.
        """
        if income_var not in df.columns:
            income_var = "inghog2d"  # fallback
        temporal, spatial = self._load_deflators()

        dpto = df["ubigeo"].astype(str).str[:2].astype(int)
        dpto = dpto.where(dpto != 7, 15)

        # Temporal deflator i00
        defl_year = temporal[temporal["aniorec"] == year].copy()
        if defl_year.empty:
            logger.warning("No temporal deflators for year %d; using nominal income", year)
            return df[income_var] / (df["mieperho"].astype(float).clip(lower=1) * 12)

        i00 = dpto.map(defl_year.set_index("dpto")["i00"]).fillna(1.0)

        # Spatial deflator ld
        dominioa = _compute_dominioa(df["dominio"], df["estrato"], dpto)
        ld = dominioa.map(spatial.set_index("dominioA")["ld"]).fillna(1.0)

        mie = df["mieperho"].astype(float).clip(lower=1)
        return df[income_var].fillna(0) / (12.0 * mie * ld * i00)

    def compute_poverty_year(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        """Compute departmental poverty indicators from a Sumaria DataFrame.

        Uses INEI's pre-computed ``pobreza`` variable for official poverty
        classification.  Falls back to manual comparison of per-capita
        consumption vs poverty lines only when ``pobreza`` is absent.

        Required columns (case-insensitive): ubigeo, factor07, mieperho,
        gashog2d, and either pobreza or (linea + linpe).

        Returns DataFrame with schema:
            [year, department_code, department_name, poverty_rate,
             extreme_poverty_rate, poverty_gap, poverty_severity,
             mean_consumption, mean_income, gini]
        """
        # Normalize column names to lowercase
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Validate required columns
        required = ["ubigeo", "factor07", "gashog2d", "mieperho"]
        alt_names = {
            "mieperho": ["mietefam", "mieperho", "miession"],
            "factor07": ["factor07", "factor", "facpob07"],
        }
        for col in required:
            if col not in df.columns:
                alts = alt_names.get(col, [])
                found = False
                for alt in alts:
                    if alt in df.columns:
                        df[col] = df[alt]
                        found = True
                        break
                if not found:
                    raise ValueError(
                        f"Required column '{col}' not found in Sumaria data. "
                        f"Available columns: {sorted(df.columns.tolist())}"
                    )

        # Extract department code (first 2 digits of ubigeo)
        # INEI maps ubigeo "07" (Callao) to Lima (15)
        df["department_code"] = df["ubigeo"].astype(str).str[:2]
        df.loc[df["department_code"] == "07", "department_code"] = "15"

        # Real per-capita values (deflated)
        has_deflation_cols = "dominio" in df.columns and "estrato" in df.columns
        mie = df["mieperho"].astype(float).clip(lower=1)
        try:
            if has_deflation_cols:
                df["percapita_consumption"] = self._compute_real_consumption(df, year)
                # inghog1d for published income tables; inghog2d for Gini
                if "inghog1d" in df.columns:
                    df["percapita_income"] = self._compute_real_income(
                        df, year, income_var="inghog1d")
                    df["percapita_income_gini"] = self._compute_real_income(
                        df, year, income_var="inghog2d")
                elif "inghog2d" in df.columns:
                    df["percapita_income"] = self._compute_real_income(
                        df, year, income_var="inghog2d")
                    df["percapita_income_gini"] = df["percapita_income"]
                else:
                    df["percapita_income"] = df["percapita_consumption"]
                    df["percapita_income_gini"] = df["percapita_consumption"]
            else:
                raise FileNotFoundError("Missing dominio/estrato")
        except (FileNotFoundError, KeyError) as e:
            logger.warning("Deflation unavailable for %d (%s); using nominal", year, e)
            df["percapita_consumption"] = df["gashog2d"] / (mie * 12)
            if "inghog1d" in df.columns:
                df["percapita_income"] = df["inghog1d"] / (mie * 12)
            elif "inghog2d" in df.columns:
                df["percapita_income"] = df["inghog2d"] / (mie * 12)
            else:
                df["percapita_income"] = df["percapita_consumption"]
            if "inghog2d" in df.columns:
                df["percapita_income_gini"] = df["inghog2d"] / (mie * 12)
            else:
                df["percapita_income_gini"] = df["percapita_income"]

        # Poverty flags — prefer INEI's official classification
        # Coding: 1=pobre extremo, 2=pobre no extremo, 3=no pobre
        # (consistent across all years, both numeric and categorical)
        use_official = "pobreza" in df.columns
        if use_official:
            pob = df["pobreza"]
            if pob.dtype in ("int8", "int16", "int32", "int64", "float64"):
                # Numeric coding (convert_categoricals=False)
                df["is_poor"] = pob.isin([1, 2]).astype(float)
                df["is_extreme_poor"] = (pob == 1).astype(float)
            else:
                # String/categorical coding
                pobreza = pob.astype(str).str.lower().str.strip()
                df["is_poor"] = (
                    pobreza.str.contains("pobre", na=False)
                    & ~pobreza.str.contains("no pobre", na=False)
                ).astype(float)
                df["is_extreme_poor"] = (
                    pobreza == "pobre extremo"
                ).astype(float)
            logger.debug("Using official INEI pobreza variable for year %d", year)
        else:
            # Fallback: manual comparison (requires linea/linpe)
            for col in ("linea", "linpe"):
                if col not in df.columns:
                    raise ValueError(
                        f"Neither 'pobreza' nor '{col}' found in Sumaria data."
                    )
            df["is_poor"] = (df["percapita_consumption"] < df["linea"]).astype(float)
            df["is_extreme_poor"] = (df["percapita_consumption"] < df["linpe"]).astype(float)
            logger.warning("No 'pobreza' column for year %d; using manual fallback", year)

        # FGT measures: Foster-Greer-Thorbecke poverty indices
        # FGT_alpha = sum(w * max(0, (z - y)/z)^alpha) / sum(w)
        # Requires nominal per-capita consumption vs poverty line
        mie_f = df["mieperho"].astype(float).clip(lower=1)
        nominal_pc = df["gashog2d"] / (mie_f * 12)
        has_lines = "linea" in df.columns and "linpe" in df.columns
        if has_lines:
            gap_total = ((df["linea"] - nominal_pc) / df["linea"]).clip(lower=0)
            gap_extreme = ((df["linpe"] - nominal_pc) / df["linpe"]).clip(lower=0)
        else:
            gap_total = pd.Series(np.nan, index=df.index)
            gap_extreme = pd.Series(np.nan, index=df.index)
        df["fgt1_total"] = gap_total * df["is_poor"]
        df["fgt2_total"] = (gap_total ** 2) * df["is_poor"]
        df["fgt1_extreme"] = gap_extreme * df["is_extreme_poor"]
        df["fgt2_extreme"] = (gap_extreme ** 2) * df["is_extreme_poor"]

        # Weight = factor07 * mieperho (expand to person level)
        # Matches official: factornd07 = round(factor07 * mieperho, 1)
        df["weight"] = (df["factor07"] * mie_f).round(0)

        rows = []
        for dept_code, group in df.groupby("department_code"):
            if dept_code not in DEPARTMENTS:
                continue

            w = group["weight"].values
            consumption = group["percapita_consumption"].values
            income = group["percapita_income"].values
            income_gini = group["percapita_income_gini"].values
            poor = group["is_poor"].values
            extreme = group["is_extreme_poor"].values

            rows.append({
                "year": year,
                "department_code": dept_code,
                "department_name": DEPARTMENTS[dept_code],
                "poverty_rate": weighted_mean(poor, w),
                "extreme_poverty_rate": weighted_mean(extreme, w),
                "poverty_gap": weighted_mean(
                    group["fgt1_total"].values, w),
                "poverty_severity": weighted_mean(
                    group["fgt2_total"].values, w),
                "mean_consumption": weighted_mean(consumption, w),
                "mean_income": weighted_mean(income, w),
                "gini": weighted_gini(income_gini, w),
            })

        result = pd.DataFrame(rows)
        return result.sort_values(["year", "department_code"]).reset_index(drop=True)

    def compile_poverty_targets(
        self,
        output_dir: Path = TARGETS_DIR,
    ) -> dict:
        """Compile poverty targets from all downloaded ENAHO years.

        Returns dict with status info.
        """
        output_path = output_dir / "poverty_departmental.parquet"
        downloaded = self.downloaded_years()

        if not downloaded:
            logger.warning("No ENAHO data downloaded yet.")
            return {"status": "no_data", "rows": 0}

        all_frames = []
        for year in downloaded:
            try:
                df = self.read_sumaria(year)
                if df is None:
                    continue
                poverty_df = self.compute_poverty_year(df, year)
                if not poverty_df.empty:
                    all_frames.append(poverty_df)
                    logger.info(
                        "Year %d: %d departments, poverty range %.1f%%-%.1f%%",
                        year, len(poverty_df),
                        poverty_df["poverty_rate"].min() * 100,
                        poverty_df["poverty_rate"].max() * 100,
                    )
            except Exception as e:
                logger.error("Failed to process ENAHO year %d: %s", year, e)

        if not all_frames:
            logger.warning("No poverty data computed from any year.")
            return {"status": "no_data", "rows": 0}

        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.sort_values(["year", "department_code"]).reset_index(drop=True)

        save_parquet(combined, output_path)
        logger.info(
            "Saved poverty targets to %s (%d rows, years %d-%d)",
            output_path, len(combined),
            combined["year"].min(), combined["year"].max(),
        )

        return {
            "status": "updated",
            "rows": len(combined),
            "years": sorted(combined["year"].unique().tolist()),
            "output_path": str(output_path),
        }


# ── Top-level orchestrator function ──────────────────────────────────────────

def update_enaho(force: bool = False) -> dict:
    """Download new ENAHO years and compile poverty targets.

    This is the entry point called by the unified orchestrator.
    """
    client = ENAHOClient()

    # Step 1: Download any missing years
    new_years = client.download_new_years(force=force)
    logger.info("Downloaded %d new ENAHO years: %s", len(new_years), new_years)

    # Step 2: Compile poverty targets from all available data
    result = client.compile_poverty_targets()
    result["new_years_downloaded"] = new_years

    return result
