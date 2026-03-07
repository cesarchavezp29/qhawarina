"""Poverty nowcasting models: Panel Regression and DFM Hybrid.

Provides two approaches to nowcast departmental poverty:
  1. PanelPovertyNowcaster: Ridge panel regression with department fixed effects
  2. DFMPovertyNowcaster: PCA factor extraction + bridge regression

Both operate on annual aggregates of departmental panel data and produce
per-department nowcasts for poverty_rate, extreme_poverty_rate, and
mean_consumption.

Benchmarks: AR(1) per department and Random Walk (last year's value).
"""

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("nexus.poverty")

# Callao (07) is merged into Lima (15) in ENAHO poverty data
CALLAO_LIMA_MERGE = {"07": "15"}

# 24 departments with poverty data (excludes Callao which merges into Lima)
POVERTY_DEPARTMENTS = [
    "01", "02", "03", "04", "05", "06", "08", "09", "10", "11",
    "12", "13", "14", "15", "16", "17", "18", "19", "20", "21",
    "22", "23", "24", "25",
]

DEFAULT_TARGET_COLS = ["poverty_rate", "extreme_poverty_rate", "mean_consumption"]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _aggregate_dept_panel_annual(
    dept_panel: pd.DataFrame,
    value_col: str = "value_yoy",
) -> pd.DataFrame:
    """Aggregate departmental monthly panel to annual features per department.

    Parameters
    ----------
    dept_panel : pd.DataFrame
        Long-format departmental panel with date, ubigeo, category, value columns.
    value_col : str
        Which value column to aggregate (default: value_yoy).

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with index=(year, ubigeo) and columns=category names.
    """
    df = dept_panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    # Remap Callao → Lima
    df["ubigeo"] = df["ubigeo"].replace(CALLAO_LIMA_MERGE)

    # Exclude national aggregate
    df = df[df["ubigeo"] != "00"]

    # Filter to departments with poverty data
    df = df[df["ubigeo"].isin(POVERTY_DEPARTMENTS)]

    # Annual mean per (year, ubigeo, category)
    annual = (
        df.groupby(["year", "ubigeo", "category"])[value_col]
        .mean()
        .reset_index()
    )

    # Pivot to wide: one column per category
    wide = annual.pivot_table(
        index=["year", "ubigeo"],
        columns="category",
        values=value_col,
        aggfunc="mean",
    )
    wide.columns.name = None
    wide = wide.reset_index()

    return wide


def _aggregate_dept_panel_rolling(
    dept_panel: pd.DataFrame,
    end_date: pd.Timestamp,
    window_months: int = 12,
    value_col: str = "value_yoy",
) -> pd.DataFrame:
    """Aggregate departmental panel over a rolling window ending at end_date.

    Same preprocessing as annual (Callao merge, filter to 24 depts), but
    computes means over [end_date - window_months + 1, end_date] instead
    of calendar years. When end_date = Dec Y and window_months = 12,
    output matches _aggregate_dept_panel_annual() for year Y.

    Parameters
    ----------
    dept_panel : pd.DataFrame
        Long-format departmental panel with date, ubigeo, category, value columns.
    end_date : pd.Timestamp
        End of the rolling window (inclusive).
    window_months : int
        Window size in months (default: 12).
    value_col : str
        Which value column to aggregate (default: value_yoy).

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with columns: year, ubigeo, and one per category.
        One row per department.
    """
    df = dept_panel.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Remap Callao → Lima
    df["ubigeo"] = df["ubigeo"].replace(CALLAO_LIMA_MERGE)

    # Exclude national aggregate
    df = df[df["ubigeo"] != "00"]

    # Filter to departments with poverty data
    df = df[df["ubigeo"].isin(POVERTY_DEPARTMENTS)]

    # Rolling window bounds: start at 1st of the month, window_months-1 before end_date's month
    end_month_start = pd.Timestamp(end_date.year, end_date.month, 1)
    window_start = end_month_start - pd.DateOffset(months=window_months - 1)
    df = df[(df["date"] >= window_start) & (df["date"] <= end_date)]

    if df.empty:
        return pd.DataFrame(columns=["year", "ubigeo"])

    # Mean per (ubigeo, category) over the rolling window
    rolled = (
        df.groupby(["ubigeo", "category"])[value_col]
        .mean()
        .reset_index()
    )

    # Pivot to wide: one column per category
    wide = rolled.pivot_table(
        index="ubigeo",
        columns="category",
        values=value_col,
        aggfunc="mean",
    )
    wide.columns.name = None
    wide = wide.reset_index()
    wide["year"] = end_date.year

    return wide


# ── Panel Poverty Nowcaster ──────────────────────────────────────────────────


class PanelPovertyNowcaster:
    """Panel regression for poverty nowcasting via change prediction.

    Predicts year-on-year changes in poverty using annual aggregates of
    departmental economic indicators. Final prediction = last_observed + change.
    This approach avoids the level-prediction problem where department-specific
    heterogeneity dominates.

    Parameters
    ----------
    target_cols : list[str] or None
        Which poverty columns to nowcast. Defaults to all three targets.
    alpha : float
        Ridge regularization strength (only used when model_type='ridge').
    include_ar : bool
        If True, include lagged target as AR(1) feature.
    model_type : str
        'ridge' (default) or 'gbr' (GradientBoosting). GBR captures
        non-linear interactions between economic indicators.
    exclude_covid : bool
        If True, exclude 2020-2021 from training data to avoid COVID shock
        contamination. Default False.
    """

    def __init__(self, target_cols=None, alpha=10.0, include_ar=True,
                 model_type="ridge", exclude_covid=False):
        self.target_cols = target_cols or DEFAULT_TARGET_COLS
        self.alpha = alpha
        self.include_ar = include_ar
        self.model_type = model_type
        self.exclude_covid = exclude_covid
        self.models_ = {}
        self.scalers_ = {}
        self.feature_cols_ = None

    def fit(self, dept_features: pd.DataFrame, poverty_df: pd.DataFrame):
        """Fit models to predict poverty changes.

        Parameters
        ----------
        dept_features : pd.DataFrame
            Annual features from _aggregate_dept_panel_annual(), with year + ubigeo.
        poverty_df : pd.DataFrame
            Poverty targets with year, department_code, and target columns.
        """
        merged = self._merge_features_targets(dept_features, poverty_df)
        if merged.empty:
            logger.warning("No overlapping data for fit")
            return self

        # COVID exclusion filter: remove 2020-2021 from training
        if self.exclude_covid and "year" in merged.columns:
            pre_covid = merged[merged["year"] < 2020]
            post_covid = merged[merged["year"] > 2021]
            merged = pd.concat([pre_covid, post_covid], axis=0)
            logger.info("COVID filter: excluding 2020-2021 from training, using %d rows", len(merged))

        feature_cols = [
            c for c in merged.columns
            if c not in ["year", "ubigeo"] + self.target_cols
            and not c.endswith("_lag1") and not c.endswith("_change")
        ]

        # Fit model for each target — predict change, not level
        for tc in self.target_cols:
            if tc not in merged.columns:
                continue

            # Compute lag and change
            merged_sorted = merged.sort_values(["ubigeo", "year"])
            merged_sorted[f"{tc}_lag1"] = merged_sorted.groupby("ubigeo")[tc].shift(1)
            merged_sorted[f"{tc}_change"] = merged_sorted[tc] - merged_sorted[f"{tc}_lag1"]

            # Add AR(1) lag as explicit feature for GBR
            train_feature_cols = list(feature_cols)
            if self.include_ar and self.model_type == "gbr":
                train_feature_cols.append(f"{tc}_lag1")

            valid = merged_sorted.dropna(
                subset=[f"{tc}_change"] + train_feature_cols
            )
            if valid.empty:
                continue

            X = valid[train_feature_cols].values
            y = valid[f"{tc}_change"].values

            if self.model_type == "gbr":
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    min_samples_leaf=5,
                    random_state=42,
                )
                model.fit(X, y)
                self.models_[tc] = model
                self.scalers_[tc] = None  # GBR doesn't need scaling
            else:
                # Standardize features for Ridge
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                self.scalers_[tc] = scaler
                model = Ridge(alpha=self.alpha)
                model.fit(X_scaled, y)
                self.models_[tc] = model

            logger.info(
                "Poverty %s fitted for %s: %d obs, %d features",
                self.model_type, tc, len(valid), len(train_feature_cols),
            )

        self.feature_cols_ = feature_cols
        return self

    def nowcast(
        self, dept_features: pd.DataFrame, poverty_df: pd.DataFrame
    ) -> dict:
        """Generate per-department poverty nowcasts.

        Parameters
        ----------
        dept_features : pd.DataFrame
            Annual features including the nowcast year.
        poverty_df : pd.DataFrame
            Historical poverty data (used for AR lag).

        Returns
        -------
        dict
            {nowcast_value: float, dept_nowcasts: {dept: {target: val}}}
        """
        if not self.models_:
            return {"nowcast_value": np.nan, "dept_nowcasts": {}}

        # Identify nowcast year
        feature_years = set(dept_features["year"].unique())
        poverty_years = set(poverty_df["year"].unique())
        nowcast_years = feature_years - poverty_years
        if not nowcast_years:
            nowcast_year = max(feature_years)
        else:
            nowcast_year = max(nowcast_years)

        return self._predict_from_features(dept_features, poverty_df, nowcast_year)

    def nowcast_at_date(
        self,
        dept_panel: pd.DataFrame,
        poverty_df: pd.DataFrame,
        end_date: pd.Timestamp,
        window_months: int = 12,
        value_col: str = "value_yoy",
    ) -> dict:
        """Generate poverty nowcasts using rolling-window features at a given date.

        The model must already be fitted (via fit()). Features are computed
        over [end_date - window_months + 1, end_date], then run through
        the same prediction logic as nowcast().

        Parameters
        ----------
        dept_panel : pd.DataFrame
            Long-format departmental panel (monthly).
        poverty_df : pd.DataFrame
            Historical poverty data (used for AR lag).
        end_date : pd.Timestamp
            End of the rolling window.
        window_months : int
            Rolling window size (default: 12).
        value_col : str
            Which value column to aggregate.

        Returns
        -------
        dict
            Same as nowcast() plus 'end_date' and 'coverage' keys.
        """
        if not self.models_:
            return {"nowcast_value": np.nan, "dept_nowcasts": {}, "end_date": end_date}

        features = _aggregate_dept_panel_rolling(
            dept_panel, end_date, window_months, value_col,
        )
        if features.empty:
            return {"nowcast_value": np.nan, "dept_nowcasts": {}, "end_date": end_date}

        result = self._predict_from_features(features, poverty_df, end_date.year)
        result["end_date"] = end_date

        # Coverage info: non-NaN category count per department
        if self.feature_cols_ is not None:
            coverage = {}
            for _, row in features.iterrows():
                dept = row["ubigeo"]
                n_avail = row[
                    [c for c in self.feature_cols_ if c in features.columns]
                ].notna().sum()
                coverage[dept] = int(n_avail)
            result["coverage"] = coverage

        return result

    def _predict_from_features(
        self,
        dept_features: pd.DataFrame,
        poverty_df: pd.DataFrame,
        nowcast_year: int,
    ) -> dict:
        """Core prediction: look up AR lag, predict change, add to lag.

        Parameters
        ----------
        dept_features : pd.DataFrame
            Wide features with year, ubigeo, category columns.
        poverty_df : pd.DataFrame
            Historical poverty data (used for AR lag).
        nowcast_year : int
            Year to predict.

        Returns
        -------
        dict
            {nowcast_value, dept_nowcasts, ...}
        """
        if self.feature_cols_ is None:
            return {"nowcast_value": np.nan, "dept_nowcasts": {}}

        # Get nowcast year features
        nowcast_feats = dept_features[dept_features["year"] == nowcast_year]
        if nowcast_feats.empty:
            return {"nowcast_value": np.nan, "dept_nowcasts": {}}

        pov = poverty_df.copy()

        dept_nowcasts = {}
        for _, row in nowcast_feats.iterrows():
            dept = row["ubigeo"]
            available_cols = [c for c in self.feature_cols_ if c in row.index]
            x_raw = row[available_cols].values.astype(float)

            # Pad missing feature cols with NaN
            if len(available_cols) < len(self.feature_cols_):
                full_x = np.full(len(self.feature_cols_), np.nan)
                col_idx = [self.feature_cols_.index(c) for c in available_cols]
                for i, idx in enumerate(col_idx):
                    full_x[idx] = x_raw[i]
                x_raw = full_x

            # Fill NaN with training means (Ridge) or 0 (GBR handles missing better)
            nan_mask = np.isnan(x_raw)
            x_clean = x_raw.copy()
            if nan_mask.any():
                if self.model_type == "gbr":
                    x_clean[nan_mask] = 0.0
                else:
                    first_tc = next(iter(self.scalers_))
                    if self.scalers_[first_tc] is not None:
                        x_clean[nan_mask] = self.scalers_[first_tc].mean_[nan_mask]
                    else:
                        x_clean[nan_mask] = 0.0

            # Get last observed poverty for this department
            dept_pov = pov[pov["department_code"] == dept].sort_values("year")

            dept_nowcasts[dept] = {}
            for tc, model in self.models_.items():
                if dept_pov.empty or tc not in dept_pov.columns:
                    continue

                lag = float(dept_pov[tc].dropna().iloc[-1])

                if self.model_type == "gbr" and self.include_ar:
                    # Append AR lag as feature for GBR
                    x_input = np.concatenate([x_clean, [lag]]).reshape(1, -1)
                    predicted_change = model.predict(x_input)[0]
                elif self.model_type == "gbr":
                    predicted_change = model.predict(x_clean.reshape(1, -1))[0]
                else:
                    scaler = self.scalers_[tc]
                    x_scaled = scaler.transform(x_clean.reshape(1, -1))
                    predicted_change = model.predict(x_scaled)[0]

                pred = lag + predicted_change

                # Clip to valid range
                if tc in ("poverty_rate", "extreme_poverty_rate"):
                    pred = np.clip(pred, 0.0, 1.0)
                elif tc == "mean_consumption":
                    pred = max(0.0, pred)

                dept_nowcasts[dept][tc] = float(pred)

        # National average
        result = {"dept_nowcasts": dept_nowcasts}
        for tc in self.target_cols:
            vals = [d[tc] for d in dept_nowcasts.values() if tc in d]
            result[f"nowcast_{tc}"] = float(np.mean(vals)) if vals else np.nan

        primary = self.target_cols[0]
        result["nowcast_value"] = result.get(f"nowcast_{primary}", np.nan)

        return result

    def _merge_features_targets(
        self,
        dept_features: pd.DataFrame,
        poverty_df: pd.DataFrame,
        include_nowcast_year: bool = False,
    ) -> pd.DataFrame:
        """Merge annual features with poverty targets."""
        feats = dept_features.copy()
        pov = poverty_df.copy()
        pov = pov.rename(columns={"department_code": "ubigeo"})

        target_cols_available = [
            tc for tc in self.target_cols if tc in pov.columns
        ]
        merge_cols = ["year", "ubigeo"] + target_cols_available

        merged = feats.merge(
            pov[merge_cols],
            on=["year", "ubigeo"],
            how="left" if include_nowcast_year else "inner",
        )
        return merged


# ── DFM Poverty Hybrid ──────────────────────────────────────────────────────


class DFMPovertyNowcaster:
    """PCA-based factor model with bridge regression for poverty nowcasting.

    Extracts k_factors from the ultra-wide departmental panel (all series
    × all departments), aggregates monthly factors to annual, then fits
    per-department bridge regressions to poverty targets.

    Parameters
    ----------
    k_factors : int
        Number of PCA factors to extract.
    target_cols : list[str] or None
        Which poverty columns to nowcast.
    alpha : float
        Ridge regularization for bridge regressions.
    """

    def __init__(self, k_factors=3, target_cols=None, alpha=1.0):
        self.k_factors = k_factors
        self.target_cols = target_cols or DEFAULT_TARGET_COLS
        self.alpha = alpha
        self.models_ = {}
        self.scaler_ = None
        self.pca_ = None

    def fit(self, dept_panel: pd.DataFrame, poverty_df: pd.DataFrame):
        """Fit PCA + bridge regression.

        Parameters
        ----------
        dept_panel : pd.DataFrame
            Long-format departmental panel (monthly).
        poverty_df : pd.DataFrame
            Poverty targets with year, department_code, and target columns.
        """
        # Pivot to ultra-wide: columns = {category}_{ubigeo}
        wide = self._pivot_ultra_wide(dept_panel)
        if wide.empty:
            return self

        # Extract factors via PCA
        annual_factors = self._extract_annual_factors(wide)
        if annual_factors is None:
            return self

        # Bridge regression for each department × target
        pov = poverty_df.copy()
        pov = pov.rename(columns={"department_code": "ubigeo"})

        for tc in self.target_cols:
            if tc not in pov.columns:
                continue

            dept_models = {}
            for dept in POVERTY_DEPARTMENTS:
                dept_pov = pov[pov["ubigeo"] == dept][["year", tc]].dropna()
                if len(dept_pov) < 5:
                    continue

                # Merge factors with poverty
                merged = annual_factors.merge(dept_pov, on="year", how="inner")
                if len(merged) < 5:
                    continue

                factor_cols = [c for c in merged.columns if c.startswith("factor_")]

                # Add AR(1) lag
                merged = merged.sort_values("year")
                merged["ar_lag1"] = merged[tc].shift(1)
                merged = merged.dropna()

                if len(merged) < 3:
                    continue

                X = merged[factor_cols + ["ar_lag1"]].values
                y = merged[tc].values

                model = Ridge(alpha=self.alpha)
                model.fit(X, y)
                dept_models[dept] = {
                    "model": model,
                    "feature_cols": factor_cols + ["ar_lag1"],
                }

            self.models_[tc] = dept_models

        logger.info(
            "DFM poverty fitted: %d factors, %d targets",
            self.k_factors, len(self.models_),
        )
        return self

    def nowcast(
        self, dept_panel: pd.DataFrame, poverty_df: pd.DataFrame
    ) -> dict:
        """Generate per-department poverty nowcasts using factors."""
        if not self.models_:
            return {"nowcast_value": np.nan, "dept_nowcasts": {}}

        wide = self._pivot_ultra_wide(dept_panel)
        annual_factors = self._extract_annual_factors(wide)
        if annual_factors is None:
            return {"nowcast_value": np.nan, "dept_nowcasts": {}}

        pov = poverty_df.copy()
        pov = pov.rename(columns={"department_code": "ubigeo"})

        # Nowcast year
        factor_years = set(annual_factors["year"].unique())
        poverty_years = set(pov["year"].unique())
        nowcast_years = factor_years - poverty_years
        nowcast_year = max(nowcast_years) if nowcast_years else max(factor_years)

        nowcast_factors = annual_factors[annual_factors["year"] == nowcast_year]
        if nowcast_factors.empty:
            return {"nowcast_value": np.nan, "dept_nowcasts": {}}

        factor_cols = [c for c in nowcast_factors.columns if c.startswith("factor_")]
        factor_vals = nowcast_factors[factor_cols].values[0]

        dept_nowcasts = {}
        for tc, dept_models in self.models_.items():
            for dept, info in dept_models.items():
                # Get AR lag from last observed poverty
                dept_pov = pov[pov["ubigeo"] == dept].sort_values("year")
                if dept_pov.empty:
                    continue
                ar_lag = float(dept_pov[tc].iloc[-1])

                x = np.concatenate([factor_vals, [ar_lag]])
                pred = info["model"].predict(x.reshape(1, -1))[0]

                # Clip to valid range
                if tc in ("poverty_rate", "extreme_poverty_rate"):
                    pred = np.clip(pred, 0.0, 1.0)
                elif tc == "mean_consumption":
                    pred = max(0.0, pred)

                if dept not in dept_nowcasts:
                    dept_nowcasts[dept] = {}
                dept_nowcasts[dept][tc] = float(pred)

        result = {"dept_nowcasts": dept_nowcasts}
        for tc in self.target_cols:
            vals = [d[tc] for d in dept_nowcasts.values() if tc in d]
            result[f"nowcast_{tc}"] = float(np.mean(vals)) if vals else np.nan

        primary = self.target_cols[0]
        result["nowcast_value"] = result.get(f"nowcast_{primary}", np.nan)
        return result

    def _pivot_ultra_wide(self, dept_panel: pd.DataFrame) -> pd.DataFrame:
        """Pivot departmental panel to ultra-wide format."""
        df = dept_panel.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["ubigeo"] = df["ubigeo"].replace(CALLAO_LIMA_MERGE)
        df = df[df["ubigeo"] != "00"]
        df = df[df["ubigeo"].isin(POVERTY_DEPARTMENTS)]

        # Create unique column identifier
        df["col_id"] = df["category"] + "_" + df["ubigeo"]

        wide = df.pivot_table(
            index="date",
            columns="col_id",
            values="value_yoy",
            aggfunc="mean",
        )
        return wide

    def _extract_annual_factors(self, wide: pd.DataFrame):
        """Extract PCA factors and aggregate to annual."""
        if wide.empty or wide.shape[1] < self.k_factors:
            return None

        # Drop columns with >60% missing
        min_obs = 0.4 * len(wide)
        good_cols = wide.columns[wide.notna().sum() >= min_obs]
        data = wide[good_cols].copy()

        if data.shape[1] < self.k_factors:
            return None

        # Fill NaN with column means for PCA
        data_filled = data.fillna(data.mean())

        # Standardize
        self.scaler_ = StandardScaler()
        standardized = self.scaler_.fit_transform(data_filled)

        # PCA
        n_components = min(self.k_factors, standardized.shape[1])
        self.pca_ = PCA(n_components=n_components)
        factors = self.pca_.fit_transform(standardized)

        factor_df = pd.DataFrame(
            factors,
            index=data.index,
            columns=[f"factor_{i+1}" for i in range(n_components)],
        )

        # Aggregate to annual
        factor_df["year"] = factor_df.index.year
        annual_factors = factor_df.groupby("year").mean().reset_index()

        logger.info(
            "PCA factors: %d components, explained var = %.1f%%",
            n_components, self.pca_.explained_variance_ratio_.sum() * 100,
        )
        return annual_factors


# ── Benchmarks ──────────────────────────────────────────────────────────────


class PovertyAR1Benchmark:
    """AR(1) per department on historical poverty.

    Fits a simple AR(1) model per department: poverty_t = a + b * poverty_{t-1}.
    """

    def __init__(self, target_cols=None):
        self.target_cols = target_cols or DEFAULT_TARGET_COLS

    def fit(self, dept_features, poverty_df):
        """No-op: AR(1) is fitted in nowcast()."""
        pass

    def nowcast(self, dept_features, poverty_df) -> dict:
        """Predict using AR(1) per department."""
        pov = poverty_df.copy().sort_values(["department_code", "year"])
        dept_nowcasts = {}

        for dept in POVERTY_DEPARTMENTS:
            dept_pov = pov[pov["department_code"] == dept]
            if len(dept_pov) < 3:
                continue

            dept_nowcasts[dept] = {}
            for tc in self.target_cols:
                if tc not in dept_pov.columns:
                    continue
                y = dept_pov[tc].dropna().values
                if len(y) < 3:
                    continue

                # Simple AR(1): OLS on (y_t, y_{t-1})
                y_t = y[1:]
                y_lag = y[:-1]
                if len(y_t) < 2:
                    dept_nowcasts[dept][tc] = float(y[-1])
                    continue

                # OLS: y_t = a + b * y_lag
                X = np.column_stack([np.ones(len(y_lag)), y_lag])
                try:
                    beta = np.linalg.lstsq(X, y_t, rcond=None)[0]
                    pred = beta[0] + beta[1] * y[-1]
                except Exception:
                    pred = float(y[-1])

                dept_nowcasts[dept][tc] = float(pred)

        result = {"dept_nowcasts": dept_nowcasts}
        for tc in self.target_cols:
            vals = [d[tc] for d in dept_nowcasts.values() if tc in d]
            result[f"nowcast_{tc}"] = float(np.mean(vals)) if vals else np.nan

        primary = self.target_cols[0]
        result["nowcast_value"] = result.get(f"nowcast_{primary}", np.nan)
        return result


class PovertyRandomWalkBenchmark:
    """Last year's poverty = this year's poverty."""

    def __init__(self, target_cols=None):
        self.target_cols = target_cols or DEFAULT_TARGET_COLS

    def fit(self, dept_features, poverty_df):
        """No-op."""
        pass

    def nowcast(self, dept_features, poverty_df) -> dict:
        """Return last observed poverty for each department."""
        pov = poverty_df.copy().sort_values(["department_code", "year"])
        dept_nowcasts = {}

        for dept in POVERTY_DEPARTMENTS:
            dept_pov = pov[pov["department_code"] == dept]
            if dept_pov.empty:
                continue

            dept_nowcasts[dept] = {}
            for tc in self.target_cols:
                if tc not in dept_pov.columns:
                    continue
                last_val = dept_pov[tc].dropna()
                if last_val.empty:
                    continue
                dept_nowcasts[dept][tc] = float(last_val.iloc[-1])

        result = {"dept_nowcasts": dept_nowcasts}
        for tc in self.target_cols:
            vals = [d[tc] for d in dept_nowcasts.values() if tc in d]
            result[f"nowcast_{tc}"] = float(np.mean(vals)) if vals else np.nan

        primary = self.target_cols[0]
        result["nowcast_value"] = result.get(f"nowcast_{primary}", np.nan)
        return result
