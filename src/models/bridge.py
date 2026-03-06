"""Bridge equation models for quarterly GDP nowcasting from monthly indicators.

Bridge equations map high-frequency (monthly) indicators to low-frequency (quarterly)
targets using temporal aggregation + regression (Baffigi, Golinelli & Parigi, 2004).

Supports:
  - Single-indicator bridges (MIDAS-style with monthly averaging)
  - Multi-indicator bridges with OLS or Ridge regression
  - Automatic ragged-edge handling for mixed-frequency data
"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge

logger = logging.getLogger("nexus.bridge")


class BridgeEquation:
    """Bridge equation: map monthly indicators to quarterly GDP.

    Parameters
    ----------
    indicators : list of str
        Series IDs from the wide panel to use as bridge regressors.
    method : str
        'ols' or 'ridge'.
    alpha : float
        Ridge regularization strength (ignored for OLS).
    ar_lags : int
        Number of AR lags of the target to include (0 = none).
    """

    def __init__(
        self,
        indicators: list[str],
        method: str = "ridge",
        alpha: float = 1.0,
        ar_lags: int = 1,
    ):
        self.indicators = indicators
        self.method = method
        self.alpha = alpha
        self.ar_lags = ar_lags
        self.target = "gdp"
        self._model = None
        self._feature_cols = None
        self._r2 = None

    def fit(self, panel_wide: pd.DataFrame):
        """No-op for compatibility — model is fitted in nowcast()."""
        pass

    def _aggregate_to_quarterly(self, panel_wide: pd.DataFrame) -> pd.DataFrame:
        """Quarterly-average available monthly indicators.

        Handles ragged edges: if a quarter has at least 1 month of data
        for an indicator, it uses whatever is available.
        """
        available = [s for s in self.indicators if s in panel_wide.columns]
        if not available:
            raise ValueError("No bridge indicators found in panel")

        selected = panel_wide[available].copy()
        quarterly = selected.resample("QS").mean()

        # Drop quarters where ALL indicators are NaN
        quarterly = quarterly.dropna(how="all")

        logger.info(
            "Bridge: %d indicators, %d quarters",
            len(available), len(quarterly),
        )
        return quarterly

    def nowcast(
        self, panel_wide: pd.DataFrame, target_df: pd.DataFrame
    ) -> dict:
        """Fit bridge equation and generate nowcast for latest quarter.

        Parameters
        ----------
        panel_wide : DataFrame
            Wide-format monthly panel (date index x series columns).
        target_df : DataFrame
            GDP target with columns: date, gdp_yoy.

        Returns
        -------
        dict with nowcast_value, bridge_r2.
        """
        # Quarterly-aggregate indicators
        X_q = self._aggregate_to_quarterly(panel_wide)

        # Prepare target
        gdp = target_df[["date", "gdp_yoy"]].copy()
        gdp = gdp.set_index("date").sort_index()

        # Align on common quarters
        common = X_q.index.intersection(gdp.index)
        if len(common) < 8:
            logger.warning("Only %d common quarters — too few for bridge", len(common))
            return {"nowcast_value": np.nan, "bridge_r2": np.nan}

        X = X_q.loc[common].copy()
        y = gdp.loc[common, "gdp_yoy"]

        # Add AR lags of target
        if self.ar_lags > 0:
            for lag in range(1, self.ar_lags + 1):
                X[f"gdp_ar{lag}"] = y.shift(lag)

        # Drop NaN rows
        valid = X.join(y).dropna()
        X = valid.drop(columns=["gdp_yoy"])
        y = valid["gdp_yoy"]

        if len(y) < 8:
            return {"nowcast_value": np.nan, "bridge_r2": np.nan}

        self._feature_cols = X.columns.tolist()

        # Fit
        if self.method == "ridge":
            model = Ridge(alpha=self.alpha)
            model.fit(X.values, y.values)
            y_pred = model.predict(X.values)
            ss_res = np.sum((y.values - y_pred) ** 2)
            ss_tot = np.sum((y.values - y.values.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            self._model = model
        else:
            X_ols = sm.add_constant(X)
            model = sm.OLS(y, X_ols, missing="drop").fit()
            r2 = model.rsquared
            self._model = model

        self._r2 = r2
        logger.info("Bridge R2=%.3f, method=%s, n=%d", r2, self.method, len(y))

        # Nowcast: use latest quarter from indicators
        latest_q = X_q.iloc[[-1]].copy()
        if self.ar_lags > 0:
            # Use last known GDP for AR terms
            last_gdp = gdp["gdp_yoy"].dropna()
            for lag in range(1, self.ar_lags + 1):
                if len(last_gdp) >= lag:
                    latest_q[f"gdp_ar{lag}"] = last_gdp.iloc[-lag]
                else:
                    latest_q[f"gdp_ar{lag}"] = np.nan

        # Ensure column alignment
        for col in self._feature_cols:
            if col not in latest_q.columns:
                latest_q[col] = 0.0
        latest_q = latest_q[self._feature_cols]

        if latest_q.isna().any(axis=1).iloc[0]:
            # Fill remaining NaN with 0 (ragged edge: treat missing as neutral/no signal)
            # Do NOT use ffill(axis=1) — that would copy one series' value into another
            latest_q = latest_q.fillna(0.0)

        if self.method == "ridge":
            nowcast_val = float(self._model.predict(latest_q.values)[0])
        else:
            X_new = sm.add_constant(latest_q, has_constant="add")
            nowcast_val = float(self._model.predict(X_new).iloc[0])

        return {
            "nowcast_value": nowcast_val,
            "bridge_r2": r2,
        }


# Pre-configured bridge models for common use cases

def gdp_activity_bridge(**kwargs) -> BridgeEquation:
    """Bridge using real-sector activity indicators."""
    return BridgeEquation(
        indicators=[
            "PD37966AM",  # electricity production
            "PD37967GM",  # cement consumption yoy
            "PN31879GM",  # formal employment
            "PD37981AM",  # business expectations
        ],
        **kwargs,
    )


def gdp_financial_bridge(**kwargs) -> BridgeEquation:
    """Bridge using financial/monetary indicators."""
    return BridgeEquation(
        indicators=[
            "PN00518MM",  # private credit
            "PN01246PM",  # exchange rate
            "PD04722MM",  # policy rate
            "PN07807NM",  # TAMN lending rate
        ],
        **kwargs,
    )


def gdp_trade_bridge(**kwargs) -> BridgeEquation:
    """Bridge using external sector indicators."""
    return BridgeEquation(
        indicators=[
            "PN38714BM",  # exports
            "PN38718BM",  # imports
            "PN38923BM",  # terms of trade
            "PN01652XM",  # copper price
        ],
        **kwargs,
    )
