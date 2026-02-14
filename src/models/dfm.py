"""Dynamic Factor Model for GDP and Inflation nowcasting.

Implements the Giannone-Reichlin-Small (2008) approach:
  1. Extract latent factors from a panel of monthly indicators via DFM
  2. GDP: bridge equation maps quarterly-aggregated factors to quarterly GDP growth
  3. Inflation: direct regression of monthly CPI variation on monthly factors

Uses statsmodels DynamicFactor with EM algorithm for missing values.
"""

import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("nexus.dfm")

# ── Series selections ────────────────────────────────────────────────────────

GDP_SERIES = [
    # gdp_indicators — sectoral PBI (already yoy via value_raw)
    # NOTE: PN01728AM (PBI total yoy) EXCLUDED — it is the target variable
    "PN01770AM",  # PBI global index (level, not yoy — safe to include)
    "PN01713AM",  # PBI agropecuario yoy
    "PN01716AM",  # PBI pesca yoy
    "PN01717AM",  # PBI minería yoy
    "PN01720AM",  # PBI manufactura yoy
    "PN01723AM",  # PBI electricidad yoy
    "PN01724AM",  # PBI construcción yoy
    "PN01725AM",  # PBI comercio yoy
    "PN01726AM",  # PBI otros servicios yoy
    # leading_indicators
    "PD37966AM",  # electricity production (GWh)
    "PD37967GM",  # cement consumption yoy
    "PN01246PM",  # exchange rate
    # credit_financial
    "PN00518MM",  # private credit
    "PN00027MM",  # international reserves
    "PN07807NM",  # TAMN lending rate
    "PN07816NM",  # TIPMN deposit rate
    # trade
    "PN38714BM",  # exports
    "PN38718BM",  # imports
    "PN38923BM",  # terms of trade
    # fiscal
    "PN02204FM",  # fiscal revenue
    "PN02409FM",  # fiscal spending
    # confidence
    "PD37981AM",  # business expectations
    # employment
    "PN38063GM",  # unemployment rate
    "PN31879GM",  # formal employment
    # monetary
    "PN00178MM",  # currency in circulation
    "PN01013MM",  # monetary base
    "PD04722MM",  # policy rate
    # commodity_prices
    "PN01652XM",  # copper
    "PN01660XM",  # oil WTI
    "PN01654XM",  # gold
    # payment_systems (leading indicators of economic activity)
    "PN08515EM",  # LBTR high-value payments
    "PN08521EM",  # CCE credit transfers
    "PN39936SM",  # low-value payments total
    "PN39951SM",  # digital payments indicator
    # nighttime_lights
    "NTL_SUM_NATIONAL",  # VIIRS nighttime lights national aggregate
    # poultry — MIDAGRI Lima wholesale chicken price level (r=0.83 vs GDP)
    "MIDAGRI_CHICKEN_AVG",  # wholesale chicken price S/kg - Lima
]

INFLATION_SERIES = [
    # inflation — NOTE: PN01273PM (CPI 12m var) EXCLUDED — it IS the target
    "PN01271PM",  # CPI monthly var (different from 12m target)
    "PN38706PM",  # core CPI index
    # food_prices
    "PN39445PM",  # food price index
    "PN01383PM",  # food price var
    "PN01286PM",  # wholesale price index (monthly var%)
    # monetary / rates
    "PD04722MM",  # policy rate
    "PN01013MM",  # monetary base
    "PN00178MM",  # currency in circulation
    # commodity_prices
    "PN01660XM",  # oil WTI
    "PN01652XM",  # copper
    "PN01654XM",  # gold
    "PN01246PM",  # exchange rate
    # confidence
    "PD12912AM",  # inflation expectations
    "PD37981AM",  # business expectations
    # credit
    "PN00518MM",  # private credit
    "PN07807NM",  # TAMN lending rate
    # payment_systems (demand-side inflation pressure)
    "PN39951SM",  # digital payments indicator
    "PN39936SM",  # low-value payments total
    # trade
    "PN38923BM",  # terms of trade
    # food_prices — MIDAGRI Lima wholesale (daily-frequency source)
    "MIDAGRI_ALL_VAR",    # wholesale food price var% - all products
    "MIDAGRI_VEG_VAR",    # wholesale food price var% - vegetables (most volatile)
    "MIDAGRI_FRUIT_VAR",  # wholesale food price var% - fruits
    "MIDAGRI_TUBER_VAR",  # wholesale food price var% - tubers (papa, yuca, camote)
    # poultry — MIDAGRI Lima wholesale egg price (r=0.73 vs CPI, strongest food signal)
    "MIDAGRI_EGG_VAR",      # wholesale egg price var% - Lima
    # supermarket — BPP-style high-frequency price indices (daily source)
    "SUPERMARKET_FOOD_VAR",  # supermarket food price variation MoM%
    "SUPERMARKET_ALL_VAR",   # supermarket all-products price variation MoM%
]


class NowcastDFM:
    """Dynamic Factor Model nowcaster for GDP or inflation.

    Parameters
    ----------
    k_factors : int
        Number of latent factors to extract.
    factor_order : int
        VAR order for factor dynamics.
    target : str
        'gdp' or 'inflation' — determines series selection and bridge method.
    inflation_col : str
        Target column for inflation nowcasting.
    bridge_method : str
        'ols' (default for backward compat) or 'ridge'. Ridge regularization
        prevents coefficient explosion in the bridge equation.
    bridge_alpha : float
        Ridge regularization strength (only used when bridge_method='ridge').
        Default 1.0 works well for GDP.
    rolling_window_years : int or None
        If set, only use the last N years of data for factor estimation.
        Helps avoid structural breaks (e.g., COVID) contaminating factors.
    include_factor_lags : int
        Number of lagged factor terms to include in bridge/direct regression.
        Default 0 (contemporaneous only). For inflation, 1 lag helps.
    include_target_ar : bool
        If True, include AR(1) of the target variable in the regression.
        Improves inflation nowcasting by capturing persistence.
    """

    def __init__(self, k_factors: int = 2, factor_order: int = 1, target: str = "gdp",
                 inflation_col: str = "ipc_monthly_var",
                 bridge_method: str = "ols", bridge_alpha: float = 1.0,
                 rolling_window_years: int | None = None,
                 include_factor_lags: int = 0,
                 include_target_ar: bool = False):
        self.k_factors = k_factors
        self.factor_order = factor_order
        self.target = target
        self.inflation_col = inflation_col
        self.bridge_method = bridge_method
        self.bridge_alpha = bridge_alpha
        self.rolling_window_years = rolling_window_years
        self.include_factor_lags = include_factor_lags
        self.include_target_ar = include_target_ar
        self.series_list = GDP_SERIES if target == "gdp" else INFLATION_SERIES
        self.scaler = StandardScaler()
        self._dfm_result = None
        self._factors = None
        self._bridge_model = None

    def _select_series(self, panel_wide: pd.DataFrame) -> pd.DataFrame:
        """Select series for the model from the wide panel.

        Only keeps series that actually exist in the panel and have
        enough non-missing observations.
        """
        available = [s for s in self.series_list if s in panel_wide.columns]
        if not available:
            raise ValueError(f"No series from {self.target} selection found in panel")

        selected = panel_wide[available].copy()

        # Drop series with < 40% non-missing (allow shorter series like payments)
        min_obs = 0.4 * len(selected)
        good_cols = selected.columns[selected.notna().sum() >= min_obs]
        selected = selected[good_cols]

        if selected.shape[1] < 3:
            raise ValueError(
                f"Only {selected.shape[1]} series with enough data (need >= 3)"
            )

        logger.info(
            "Selected %d/%d series for %s DFM",
            selected.shape[1], len(self.series_list), self.target,
        )
        return selected

    def _prepare_data(self, panel_selected: pd.DataFrame) -> np.ndarray:
        """Standardize columns to zero mean, unit variance.

        Stores scaler for reconstruction. Returns array with NaN preserved.
        """
        data = panel_selected.values.copy()

        # Fit scaler on non-NaN values per column
        self.scaler = StandardScaler()
        # We need to handle NaN carefully
        col_means = np.nanmean(data, axis=0)
        col_stds = np.nanstd(data, axis=0)
        col_stds[col_stds == 0] = 1.0  # avoid division by zero

        self._col_means = col_means
        self._col_stds = col_stds

        standardized = (data - col_means) / col_stds
        return standardized

    def fit(self, panel_wide: pd.DataFrame):
        """Fit the Dynamic Factor Model.

        Parameters
        ----------
        panel_wide : pd.DataFrame
            Wide-format panel (date index × series columns).
        """
        # Apply rolling window if configured
        if self.rolling_window_years is not None:
            cutoff = panel_wide.index.max() - pd.DateOffset(
                years=self.rolling_window_years
            )
            panel_wide = panel_wide.loc[panel_wide.index >= cutoff]
            logger.info(
                "Rolling window: using %d-year window (%s to %s)",
                self.rolling_window_years,
                panel_wide.index.min().strftime("%Y-%m"),
                panel_wide.index.max().strftime("%Y-%m"),
            )

        selected = self._select_series(panel_wide)
        self._selected_cols = selected.columns.tolist()
        self._selected_index = selected.index

        standardized = self._prepare_data(selected)

        # Create DataFrame for statsmodels
        endog = pd.DataFrame(
            standardized, index=selected.index, columns=selected.columns
        )

        # Fit DFM with EM for missing data
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = sm.tsa.DynamicFactor(
                    endog,
                    k_factors=self.k_factors,
                    factor_order=self.factor_order,
                    error_order=0,
                )
                # Try fit_em() first (statsmodels 0.14+), fall back to MLE
                if hasattr(mod, "fit_em"):
                    self._dfm_result = mod.fit_em(
                        maxiter=50, disp=False,
                    )
                else:
                    self._dfm_result = mod.fit(
                        method="lbfgs", maxiter=200, disp=False,
                    )

            # Extract smoothed factors
            factors = self._dfm_result.factors.filtered
            if isinstance(factors, pd.DataFrame):
                self._factors = factors
            else:
                factors = np.asarray(factors)
                # Handle shape: may be (k_factors, T) or (T, k_factors)
                if factors.shape[0] == self.k_factors and factors.shape[1] == len(selected):
                    factors = factors.T
                factor_cols = [f"factor_{i+1}" for i in range(self.k_factors)]
                self._factors = pd.DataFrame(
                    factors, index=selected.index, columns=factor_cols
                )

            logger.info(
                "DFM fit: %d factors, %d obs, %d series",
                self.k_factors, len(selected), selected.shape[1],
            )
        except Exception as e:
            logger.error("DFM fit failed: %s — falling back to PCA", e)
            self._fit_pca_fallback(standardized, selected.index)

    def _fit_pca_fallback(self, standardized: np.ndarray, index: pd.DatetimeIndex):
        """PCA-based factor extraction as fallback when DFM fails."""
        from sklearn.decomposition import PCA

        # Fill NaN with 0 for PCA (mean-imputation since data is standardized)
        data_filled = np.nan_to_num(standardized, nan=0.0)

        n_components = min(self.k_factors, data_filled.shape[1])
        pca = PCA(n_components=n_components)
        factors = pca.fit_transform(data_filled)

        factor_cols = [f"factor_{i+1}" for i in range(n_components)]
        self._factors = pd.DataFrame(factors, index=index, columns=factor_cols)
        self._dfm_result = None
        logger.info("PCA fallback: %d factors, explained var = %.1f%%",
                     n_components, pca.explained_variance_ratio_.sum() * 100)

    def get_factors(self) -> pd.DataFrame:
        """Return estimated monthly factors."""
        if self._factors is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return self._factors.copy()

    def _bridge_to_quarterly(
        self, factors: pd.DataFrame, gdp_quarterly: pd.DataFrame
    ):
        """Bridge equation: gdp_yoy_q ~ factor1_q + factor2_q.

        Quarterly-averages the monthly factors, then regresses on GDP.
        Supports OLS or Ridge bridge methods.
        """
        from sklearn.linear_model import Ridge as SkRidge

        # Quarterly-average factors
        factors_q = factors.resample("QS").mean().dropna()

        # Align with GDP target
        gdp = gdp_quarterly[["date", "gdp_yoy"]].copy()
        gdp = gdp.set_index("date")

        # Inner join on common dates
        common = factors_q.index.intersection(gdp.index)
        if len(common) < 10:
            raise ValueError(f"Only {len(common)} common quarterly obs (need >= 10)")

        X_df = factors_q.loc[common]
        y = gdp.loc[common, "gdp_yoy"]

        # Drop rows with NaN
        valid = X_df.join(y).dropna()
        X_df = valid[X_df.columns]
        y = valid["gdp_yoy"]

        if self.bridge_method == "ridge":
            ridge = SkRidge(alpha=self.bridge_alpha)
            ridge.fit(X_df.values, y.values)
            y_pred = ridge.predict(X_df.values)
            ss_res = np.sum((y.values - y_pred) ** 2)
            ss_tot = np.sum((y.values - y.values.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            self._bridge_model = ridge
            self._bridge_index = X_df.columns.tolist()
            self._bridge_r2 = r2
            logger.info(
                "Ridge bridge (alpha=%.1f) R² = %.3f, n = %d",
                self.bridge_alpha, r2, len(y),
            )
            return ridge
        else:
            X_ols = sm.add_constant(X_df)
            model = sm.OLS(y, X_ols, missing="drop").fit()
            self._bridge_model = model
            self._bridge_r2 = model.rsquared
            logger.info("OLS bridge R² = %.3f, n = %d", model.rsquared, len(y))
            return model

    def _direct_regression(
        self, factors: pd.DataFrame, inflation_monthly: pd.DataFrame
    ) -> sm.OLS:
        """Direct regression: inflation ~ factors + lagged_factors + AR(1).

        Optionally includes lagged factor terms and AR(1) of the target
        to capture delayed transmission and persistence effects.
        """
        target = inflation_monthly[["date", self.inflation_col]].copy()
        target = target.set_index("date")

        common = factors.index.intersection(target.index)
        if len(common) < 20:
            raise ValueError(f"Only {len(common)} common monthly obs (need >= 20)")

        # Build regressor matrix
        X_parts = [factors.loc[common]]

        # Add lagged factors
        for lag in range(1, self.include_factor_lags + 1):
            lagged = factors.shift(lag)
            lagged.columns = [f"{c}_lag{lag}" for c in factors.columns]
            X_parts.append(lagged.loc[common])

        # Add AR(1) of target
        if self.include_target_ar:
            ar_lag = target[self.inflation_col].shift(1).rename("target_ar1")
            X_parts.append(ar_lag.reindex(common))

        X = pd.concat(X_parts, axis=1).dropna()
        y = target.loc[X.index, self.inflation_col]
        X = sm.add_constant(X)

        model = sm.OLS(y, X, missing="drop").fit()
        self._bridge_model = model
        self._bridge_r2 = model.rsquared
        self._direct_reg_cols = X.columns.tolist()
        logger.info("Direct regression R² = %.3f, n = %d", model.rsquared, len(y))
        return model

    def nowcast(
        self, panel_wide: pd.DataFrame, target_df: pd.DataFrame
    ) -> dict:
        """Generate a nowcast for the latest available period.

        Parameters
        ----------
        panel_wide : pd.DataFrame
            Wide-format panel (used only for context; model must be fitted).
        target_df : pd.DataFrame
            Target DataFrame (GDP quarterly or inflation monthly).

        Returns
        -------
        dict
            nowcast_value, bridge_r2
        """
        if self._factors is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        factors = self._factors

        if self.target == "gdp":
            bridge = self._bridge_to_quarterly(factors, target_df)
            # Nowcast: use latest quarter's factor averages
            factors_q = factors.resample("QS").mean().dropna()
            if factors_q.empty:
                return {"nowcast_value": np.nan, "bridge_r2": np.nan}
            latest_q = factors_q.iloc[[-1]]

            if self.bridge_method == "ridge":
                nowcast_val = float(bridge.predict(latest_q.values)[0])
            else:
                X_new = sm.add_constant(latest_q, has_constant="add")
                nowcast_val = float(bridge.predict(X_new).iloc[0])

            return {
                "nowcast_value": nowcast_val,
                "bridge_r2": self._bridge_r2,
            }
        else:
            reg = self._direct_regression(factors, target_df)
            # Build prediction features matching training structure
            X_parts = [factors.iloc[[-1]]]

            for lag in range(1, self.include_factor_lags + 1):
                lagged = factors.shift(lag)
                lagged.columns = [f"{c}_lag{lag}" for c in factors.columns]
                X_parts.append(lagged.iloc[[-1]])

            if self.include_target_ar:
                inf_target = target_df.set_index("date")[self.inflation_col]
                last_inflation = inf_target.dropna().iloc[-1] if len(inf_target.dropna()) > 0 else 0.0
                ar_df = pd.DataFrame(
                    {"target_ar1": [last_inflation]},
                    index=factors.index[[-1]],
                )
                X_parts.append(ar_df)

            X_new = pd.concat(X_parts, axis=1)
            X_new = sm.add_constant(X_new, has_constant="add")

            # Align columns with training (handle missing cols gracefully)
            if hasattr(self, "_direct_reg_cols"):
                for col in self._direct_reg_cols:
                    if col not in X_new.columns:
                        X_new[col] = 0.0
                X_new = X_new[self._direct_reg_cols]

            nowcast_val = float(reg.predict(X_new).iloc[0])
            return {
                "nowcast_value": nowcast_val,
                "bridge_r2": self._bridge_r2,
            }


# ── Utility functions ────────────────────────────────────────────────────────


def reconstruct_12m_from_monthly(monthly_var: pd.Series) -> pd.Series:
    """Reconstruct 12-month inflation from monthly variation series.

    12m inflation = rolling 12-month product of (1 + m_i/100) - 1,
    expressed as percentage.
    """
    factors = 1 + monthly_var / 100.0
    rolling_product = factors.rolling(window=12, min_periods=12).apply(
        lambda x: x.prod(), raw=True
    )
    return (rolling_product - 1) * 100.0


# ── Phillips Curve model ────────────────────────────────────────────────────

PHILLIPS_SERIES = {
    "unemployment": "PN38063GM",   # output gap proxy
    "exchange_rate": "PN01246PM",  # PEN/USD level → pct_change
    "food_price": "PN01383PM",    # food price variation (already a rate)
    "oil_price": "PN01660XM",     # WTI level → pct_change
    "wholesale_price": "PN01286PM",  # IPM monthly var% (leads CPI by 2-3 weeks)
}


class PhillipsCurveNowcaster:
    """Phillips curve inflation nowcaster using economic structure.

    Features: lagged inflation, unemployment (output gap proxy),
    exchange rate change, food price variation, oil price change.

    Compatible with RollingBacktester interface (fit/nowcast).
    """

    def __init__(self, inflation_col: str = "ipc_monthly_var"):
        self.inflation_col = inflation_col
        self.target = "inflation"

    def fit(self, panel_wide: pd.DataFrame):
        """No-op: model is re-estimated in nowcast() (expanding window)."""
        pass

    def nowcast(self, panel_wide: pd.DataFrame, target_df: pd.DataFrame) -> dict:
        """Extract features, fit OLS, predict next period."""
        try:
            features = self._extract_features(panel_wide, target_df)
            if features is None or len(features) < 24:
                return {"nowcast_value": np.nan}

            y = features["inflation"]
            X = sm.add_constant(features.drop(columns=["inflation"]))

            model = sm.OLS(y, X, missing="drop").fit()

            # Forward-fill NaN from publication lags before prediction
            X_filled = X.ffill()
            last_row = X_filled.iloc[[-1]]
            if last_row.isna().any(axis=1).iloc[0]:
                return {"nowcast_value": np.nan}
            nowcast_val = float(model.predict(last_row).iloc[0])
            return {"nowcast_value": nowcast_val, "bridge_r2": model.rsquared}
        except Exception as e:
            logger.warning("Phillips curve failed: %s", e)
            return {"nowcast_value": np.nan}

    def _extract_features(
        self, panel_wide: pd.DataFrame, target_df: pd.DataFrame
    ) -> pd.DataFrame | None:
        """Build feature matrix from panel and target data."""
        # Inflation target (lagged)
        inf = target_df.set_index("date")[[self.inflation_col]].sort_index()
        inf.columns = ["inflation"]

        features = pd.DataFrame(index=inf.index)
        features["inflation"] = inf["inflation"]
        features["pi_lag1"] = inf["inflation"].shift(1)

        # Extract Phillips curve regressors from panel
        for name, series_id in PHILLIPS_SERIES.items():
            if series_id not in panel_wide.columns:
                continue
            s = panel_wide[series_id].copy()
            if name in ("exchange_rate", "oil_price"):
                # Level series → percentage change
                features[f"delta_{name}"] = s.pct_change() * 100
            elif name == "food_price":
                # Already a variation
                features["delta_food"] = s
            elif name == "wholesale_price":
                # Already a variation (monthly var%)
                features["delta_wholesale"] = s
            elif name == "unemployment":
                features["unemployment"] = s

        # Drop rows with all-NaN features (beyond inflation + lag)
        features = features.dropna(subset=["inflation", "pi_lag1"])

        if len(features) < 24:
            return None

        # Winsorize features during COVID (cap at ±3σ of pre-COVID period)
        covid_start = pd.Timestamp("2020-03-01")
        covid_end = pd.Timestamp("2021-12-01")
        pre_covid = features.loc[features.index < covid_start]

        for col in features.columns:
            if col == "inflation":  # don't winsorize the dependent variable
                continue
            if col not in pre_covid.columns or pre_covid[col].dropna().empty:
                continue
            mu = pre_covid[col].mean()
            sigma = pre_covid[col].std()
            if sigma > 0:
                mask = (features.index >= covid_start) & (features.index <= covid_end)
                features.loc[mask, col] = features.loc[mask, col].clip(
                    lower=mu - 3 * sigma, upper=mu + 3 * sigma
                )

        return features


class CombinationNowcaster:
    """Simple average of DFM and Phillips curve forecasts.

    Forecast combination reduces variance when component models
    make different errors (Timmermann 2006).
    """

    def __init__(self, k_factors=2, inflation_col="ipc_monthly_var"):
        self.dfm = NowcastDFM(k_factors=k_factors, target="inflation",
                              inflation_col=inflation_col)
        self.phillips = PhillipsCurveNowcaster(inflation_col=inflation_col)
        self.target = "inflation"

    def fit(self, panel_wide):
        self.dfm.fit(panel_wide)
        self.phillips.fit(panel_wide)  # no-op

    def nowcast(self, panel_wide, target_df):
        dfm_nc = self.dfm.nowcast(panel_wide, target_df)
        pc_nc = self.phillips.nowcast(panel_wide, target_df)

        d = dfm_nc.get("nowcast_value", np.nan)
        p = pc_nc.get("nowcast_value", np.nan)

        if np.isnan(d) and np.isnan(p):
            return {"nowcast_value": np.nan}
        elif np.isnan(d):
            return {"nowcast_value": p}
        elif np.isnan(p):
            return {"nowcast_value": d}

        return {"nowcast_value": (d + p) / 2}


# ── Benchmark models ─────────────────────────────────────────────────────────


class AR1Benchmark:
    """AR(1) benchmark on the target variable."""

    def __init__(self, target: str = "gdp", inflation_col: str = "ipc_monthly_var"):
        self.target = target
        self.inflation_col = inflation_col
        self._ar_model = None

    def fit(self, panel_wide: pd.DataFrame):
        """No-op: AR(1) is fitted on target in nowcast()."""
        pass

    def nowcast(self, panel_wide: pd.DataFrame, target_df: pd.DataFrame) -> dict:
        """Fit AR(1) on target history and forecast next period."""
        if self.target == "gdp":
            col = "gdp_yoy"
        else:
            col = self.inflation_col

        y = target_df.set_index("date")[col].dropna().sort_index()
        if len(y) < 4:
            return {"nowcast_value": np.nan}

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ar_model = sm.tsa.AutoReg(y, lags=1, trend="c").fit()
            forecast = ar_model.predict(start=len(y), end=len(y))
            return {"nowcast_value": float(forecast.iloc[0])}
        except Exception:
            return {"nowcast_value": float(y.iloc[-1])}


class RandomWalkBenchmark:
    """Naive random walk: y_hat(t+1) = y(t)."""

    def __init__(self, target: str = "gdp", inflation_col: str = "ipc_monthly_var"):
        self.target = target
        self.inflation_col = inflation_col

    def fit(self, panel_wide: pd.DataFrame):
        """No-op."""
        pass

    def nowcast(self, panel_wide: pd.DataFrame, target_df: pd.DataFrame) -> dict:
        """Return last observed target value."""
        if self.target == "gdp":
            col = "gdp_yoy"
        else:
            col = self.inflation_col

        y = target_df.set_index("date")[col].dropna().sort_index()
        if len(y) == 0:
            return {"nowcast_value": np.nan}
        return {"nowcast_value": float(y.iloc[-1])}


# ── ML Nowcaster (BCRP DT 003-2024 methodology) ─────────────────────────────


# Hyperparameter grids for TimeSeriesSplit CV (kept deliberately small for speed)
_ML_PARAM_GRIDS = {
    "lasso": {"alpha": [0.01, 0.1, 1.0, 10.0]},
    "ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "elastic_net": {
        "alpha": [0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.25, 0.5, 0.75],
    },
    "gbm": {
        "n_estimators": [100, 200],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.05, 0.1],
    },
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, None],
        "min_samples_leaf": [2, 5],
    },
}


class MLNowcaster:
    """ML-based nowcaster following BCRP Working Paper DT 003-2024.

    Replaces the DFM+bridge approach with direct regularized or tree-based
    models applied to the indicator panel.  The key advantage is automatic
    regularization (LASSO/Elastic Net) or nonlinearity (GBM/Random Forest).

    Supports the same fit(panel_wide) / nowcast(panel_wide, target_df)
    interface as NowcastDFM so it plugs directly into the backtester.

    Parameters
    ----------
    method : str
        One of 'lasso', 'ridge', 'elastic_net', 'gbm', 'random_forest'.
    target : str
        'gdp' or 'inflation'.
    inflation_col : str
        Target column when target='inflation'.
    cv_splits : int
        Number of TimeSeriesSplit folds for hyperparameter tuning.
    rolling_window_years : int or None
        If set, only use the last N years of panel data.
    include_target_ar : bool
        If True, include AR(1) lag of the target as a feature.
    """

    SUPPORTED_METHODS = ("lasso", "ridge", "elastic_net", "gbm", "random_forest")

    def __init__(
        self,
        method: str = "lasso",
        target: str = "gdp",
        inflation_col: str = "ipc_monthly_var",
        cv_splits: int = 5,
        rolling_window_years: int | None = None,
        include_target_ar: bool = True,
    ):
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Choose from {self.SUPPORTED_METHODS}"
            )
        self.method = method
        self.target = target
        self.inflation_col = inflation_col
        self.cv_splits = cv_splits
        self.rolling_window_years = rolling_window_years
        self.include_target_ar = include_target_ar

        self.series_list = GDP_SERIES if target == "gdp" else INFLATION_SERIES
        self._model = None
        self._best_params = None
        self._feature_cols: list[str] = []
        self._cv_score: float = np.nan
        self._train_r2: float = np.nan

    # ── Data preparation ─────────────────────────────────────────────────

    def _select_and_impute(self, panel_wide: pd.DataFrame) -> pd.DataFrame:
        """Select model series, forward-fill, then median-impute residual NaN."""
        available = [s for s in self.series_list if s in panel_wide.columns]
        if not available:
            raise ValueError(f"No series from {self.target} selection found in panel")

        selected = panel_wide[available].copy()

        # Drop series with < 40% non-missing (same threshold as NowcastDFM)
        min_obs = 0.4 * len(selected)
        good_cols = selected.columns[selected.notna().sum() >= min_obs]
        selected = selected[good_cols]

        if selected.shape[1] < 3:
            raise ValueError(
                f"Only {selected.shape[1]} series with enough data (need >= 3)"
            )

        # Forward-fill, then fill remaining NaN with column median
        selected = selected.ffill()
        medians = selected.median()
        selected = selected.fillna(medians)

        logger.info(
            "ML selected %d/%d series for %s (%s)",
            selected.shape[1], len(self.series_list), self.target, self.method,
        )
        return selected

    def _build_features_gdp(
        self,
        panel_wide: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build quarterly feature matrix for GDP nowcasting.

        Aggregates monthly indicators to quarterly means, aligns with
        GDP target, and optionally adds AR(1) lag.
        """
        selected = self._select_and_impute(panel_wide)

        # Quarterly-average the monthly indicators
        features_q = selected.resample("QS").mean().dropna(how="all")

        # GDP target
        gdp = target_df[["date", "gdp_yoy"]].copy()
        gdp = gdp.set_index("date")

        # Align
        common = features_q.index.intersection(gdp.index)
        if len(common) < 10:
            raise ValueError(f"Only {len(common)} quarterly obs (need >= 10)")

        X = features_q.loc[common].copy()
        y = gdp.loc[common, "gdp_yoy"]

        # Add AR(1) lag
        if self.include_target_ar:
            X["target_ar1"] = y.shift(1)

        # Drop rows with NaN (from AR lag or alignment)
        valid = X.join(y).dropna()
        X = valid[X.columns]
        y = valid["gdp_yoy"]

        return X, y

    def _build_features_inflation(
        self,
        panel_wide: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build monthly feature matrix for inflation nowcasting.

        Uses monthly indicators directly, optionally adds AR(1) lag.
        """
        selected = self._select_and_impute(panel_wide)

        # Inflation target
        inf = target_df[["date", self.inflation_col]].copy()
        inf = inf.set_index("date")

        # Align
        common = selected.index.intersection(inf.index)
        if len(common) < 20:
            raise ValueError(f"Only {len(common)} monthly obs (need >= 20)")

        X = selected.loc[common].copy()
        y = inf.loc[common, self.inflation_col]

        # Add AR(1) lag
        if self.include_target_ar:
            X["target_ar1"] = y.shift(1)

        # Drop NaN rows
        valid = X.join(y).dropna()
        X = valid[X.columns]
        y = valid[self.inflation_col]

        return X, y

    # ── Model building ───────────────────────────────────────────────────

    def _make_estimator(self, params: dict | None = None):
        """Instantiate an sklearn estimator for the configured method."""
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import ElasticNet, Lasso, Ridge

        p = params or {}
        if self.method == "lasso":
            return Lasso(max_iter=5000, **p)
        elif self.method == "ridge":
            return Ridge(**p)
        elif self.method == "elastic_net":
            return ElasticNet(max_iter=5000, **p)
        elif self.method == "gbm":
            return GradientBoostingRegressor(random_state=42, **p)
        elif self.method == "random_forest":
            return RandomForestRegressor(random_state=42, **p)

    def _tune_and_fit(self, X: pd.DataFrame, y: pd.Series):
        """Run TimeSeriesSplit CV grid search, then refit on all data.

        Stores the best model, parameters, and CV score.
        """
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        # For linear models, standardize features inside a pipeline so
        # regularization acts uniformly across features.
        # Tree models don't need scaling, but it doesn't hurt.
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", self._make_estimator()),
        ])

        # Prefix param names for Pipeline
        param_grid = {
            f"model__{k}": v
            for k, v in _ML_PARAM_GRIDS[self.method].items()
        }

        tscv = TimeSeriesSplit(n_splits=min(self.cv_splits, len(y) // 4))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search = GridSearchCV(
                pipe,
                param_grid,
                cv=tscv,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                error_score=np.nan,
            )
            search.fit(X.values, y.values)

        self._model = search.best_estimator_
        self._best_params = {
            k.replace("model__", ""): v
            for k, v in search.best_params_.items()
        }
        self._cv_score = float(np.sqrt(-search.best_score_))  # RMSE
        self._feature_cols = X.columns.tolist()

        # In-sample R2
        y_pred = self._model.predict(X.values)
        ss_res = np.sum((y.values - y_pred) ** 2)
        ss_tot = np.sum((y.values - y.values.mean()) ** 2)
        self._train_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        logger.info(
            "ML %s (%s) fitted: CV-RMSE=%.4f, train R2=%.3f, best=%s",
            self.method, self.target, self._cv_score, self._train_r2,
            self._best_params,
        )

    # ── Public interface (same as NowcastDFM) ────────────────────────────

    def fit(self, panel_wide: pd.DataFrame):
        """Fit the ML model (no-op; actual fitting happens in nowcast).

        We defer fitting to nowcast() because we need the target_df to
        build the supervised feature matrix.  This matches the existing
        pattern where NowcastDFM.fit() extracts factors but the bridge
        equation is fitted inside nowcast().
        """
        # Store panel reference for rolling-window application
        self._panel_wide = panel_wide

    def nowcast(
        self, panel_wide: pd.DataFrame, target_df: pd.DataFrame
    ) -> dict:
        """Build features, tune hyperparameters, fit, and predict.

        Parameters
        ----------
        panel_wide : pd.DataFrame
            Wide-format panel (date index x series columns).
        target_df : pd.DataFrame
            Target DataFrame with 'date' column.

        Returns
        -------
        dict
            nowcast_value, bridge_r2 (train R2), cv_rmse, best_params, method
        """
        # Apply rolling window if configured
        if self.rolling_window_years is not None:
            cutoff = panel_wide.index.max() - pd.DateOffset(
                years=self.rolling_window_years
            )
            panel_wide = panel_wide.loc[panel_wide.index >= cutoff]

        # Build features
        if self.target == "gdp":
            X, y = self._build_features_gdp(panel_wide, target_df)
        else:
            X, y = self._build_features_inflation(panel_wide, target_df)

        if len(y) < 10:
            return {"nowcast_value": np.nan, "bridge_r2": np.nan}

        # Tune + fit
        self._tune_and_fit(X, y)

        # Predict for the latest period
        selected = self._select_and_impute(panel_wide)

        if self.target == "gdp":
            # Quarterly-average, take last quarter
            latest_q = selected.resample("QS").mean().dropna(how="all")
            if latest_q.empty:
                return {"nowcast_value": np.nan, "bridge_r2": np.nan}
            X_new = latest_q.iloc[[-1]].copy()

            # Add AR(1) if configured
            if self.include_target_ar:
                gdp = target_df.set_index("date")["gdp_yoy"].dropna().sort_index()
                X_new["target_ar1"] = float(gdp.iloc[-1]) if len(gdp) > 0 else 0.0
        else:
            # Monthly: use latest month
            X_new = selected.iloc[[-1]].copy()

            if self.include_target_ar:
                inf = target_df.set_index("date")[self.inflation_col].dropna().sort_index()
                X_new["target_ar1"] = float(inf.iloc[-1]) if len(inf) > 0 else 0.0

        # Align columns with training (handle any missing/extra cols)
        for col in self._feature_cols:
            if col not in X_new.columns:
                X_new[col] = 0.0
        X_new = X_new[self._feature_cols]

        nowcast_val = float(self._model.predict(X_new.values)[0])

        return {
            "nowcast_value": nowcast_val,
            "bridge_r2": self._train_r2,
            "cv_rmse": self._cv_score,
            "best_params": self._best_params,
            "method": self.method,
        }
