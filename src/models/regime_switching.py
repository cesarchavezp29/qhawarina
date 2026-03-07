"""Markov Regime-Switching models for detecting economic crises.

Implements a 2-regime Markov-Switching model to detect:
  - Regime 0: Normal growth (high mean, low volatility)
  - Regime 1: Crisis/recession (low mean, high volatility)

Uses statsmodels MarkovRegression with regime-dependent mean and variance.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

logger = logging.getLogger("nexus.regime_switching")


class RegimeSwitchingDetector:
    """Detect economic regimes (normal vs crisis) using Markov-Switching model.

    Parameters
    ----------
    k_regimes : int
        Number of regimes (default=2: normal, crisis).
    switching_variance : bool
        Allow variance to change across regimes (default=True).
    """

    def __init__(self, k_regimes: int = 2, switching_variance: bool = True):
        self.k_regimes = k_regimes
        self.switching_variance = switching_variance
        self.model_ = None
        self.result_ = None
        self.regime_labels_ = None

    def fit(
        self,
        y: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        min_obs: int = 24,
    ) -> "RegimeSwitchingDetector":
        """Fit Markov-Switching model to time series.

        Parameters
        ----------
        y : Series
            Target variable (e.g., GDP YoY growth, inflation).
        exog : DataFrame, optional
            Exogenous variables (e.g., financial stress index).
        min_obs : int
            Minimum observations required to fit.

        Returns
        -------
        self : RegimeSwitchingDetector
        """
        if len(y) < min_obs:
            raise ValueError(f"Need at least {min_obs} observations, got {len(y)}")

        # Drop NaNs
        if exog is not None:
            data = pd.concat([y, exog], axis=1).dropna()
            y_clean = data.iloc[:, 0]
            exog_clean = data.iloc[:, 1:]
        else:
            y_clean = y.dropna()
            exog_clean = None

        logger.info(
            "Fitting Markov-Switching model with %d obs, %d regimes",
            len(y_clean),
            self.k_regimes,
        )

        # Fit model
        try:
            self.model_ = MarkovRegression(
                endog=y_clean,
                k_regimes=self.k_regimes,
                exog=exog_clean,
                switching_variance=self.switching_variance,
            )
            # Use EM algorithm, which is more robust than MLE for regime-switching
            self.result_ = self.model_.fit(search_reps=20, maxiter=500, disp=False)

            # Get regime parameters
            regime_means = [
                self.result_.params[f"const[{i}]"] for i in range(self.k_regimes)
            ]

            # Identify regimes: crisis = regime with HIGHER VARIANCE (more volatile)
            if self.switching_variance:
                regime_vars = [
                    self.result_.params[f"sigma2[{i}]"] for i in range(self.k_regimes)
                ]
                crisis_regime = int(np.argmax(regime_vars))  # Higher variance = crisis
                logger.info("Using variance-based crisis identification")
            else:
                # Fall back to mean-based identification
                crisis_regime = int(np.argmin(regime_means))
                logger.info("Using mean-based crisis identification")

            normal_regime = 1 - crisis_regime if self.k_regimes == 2 else 0

            self.regime_labels_ = {
                "normal": normal_regime,
                "crisis": crisis_regime,
            }

            logger.info(
                "Model converged. Regime means: %s",
                {f"regime_{i}": f"{m:.3f}" for i, m in enumerate(regime_means)},
            )
            logger.info("Identified crisis regime: %d", crisis_regime)

        except Exception as e:
            logger.error("Failed to fit Markov-Switching model: %s", e)
            raise

        return self

    def predict_probabilities(
        self, y: pd.Series, exog: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Predict smoothed regime probabilities for each period.

        Parameters
        ----------
        y : Series
            Target variable (same as used in fit).
        exog : DataFrame, optional
            Exogenous variables.

        Returns
        -------
        DataFrame with columns: date, prob_regime_0, prob_regime_1, ..., prob_crisis
        """
        if self.result_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get smoothed probabilities (Kim filter)
        smoothed = self.result_.smoothed_marginal_probabilities

        # Match dates
        if exog is not None:
            data = pd.concat([y, exog], axis=1).dropna()
            dates = data.index
        else:
            dates = y.dropna().index

        # Build result DataFrame
        result = pd.DataFrame(index=dates)
        for i in range(self.k_regimes):
            result[f"prob_regime_{i}"] = smoothed.iloc[:, i].values

        # Add crisis probability
        crisis_regime = self.regime_labels_["crisis"]
        result["prob_crisis"] = result[f"prob_regime_{crisis_regime}"]

        result = result.reset_index()
        result.columns = ["date"] + list(result.columns[1:])

        return result

    def detect_crisis_periods(
        self, y: pd.Series, threshold: float = 0.5
    ) -> pd.DataFrame:
        """Detect periods with high crisis probability.

        Parameters
        ----------
        y : Series
            Target variable.
        threshold : float
            Crisis probability threshold (default=0.5).

        Returns
        -------
        DataFrame with columns: date, prob_crisis, is_crisis
        """
        probs = self.predict_probabilities(y)
        probs["is_crisis"] = probs["prob_crisis"] >= threshold

        logger.info(
            "Detected %d/%d periods as crisis (threshold=%.2f)",
            probs["is_crisis"].sum(),
            len(probs),
            threshold,
        )

        return probs[["date", "prob_crisis", "is_crisis"]]

    def get_transition_matrix(self) -> np.ndarray:
        """Get estimated transition probability matrix.

        Returns
        -------
        ndarray of shape (k_regimes, k_regimes)
            Element [i,j] = P(regime_t = j | regime_{t-1} = i)
        """
        if self.result_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.result_.regime_transition

    def get_regime_summary(self) -> pd.DataFrame:
        """Get summary statistics for each regime.

        Returns
        -------
        DataFrame with regime parameters (mean, variance, persistence).
        """
        if self.result_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        trans_mat = self.get_transition_matrix()

        summary = []
        for i in range(self.k_regimes):
            mean = self.result_.params[f"const[{i}]"]
            if self.switching_variance:
                var = self.result_.params[f"sigma2[{i}]"]
            else:
                var = self.result_.params["sigma2"]

            persistence = trans_mat[i, i]  # P(stay in regime i)

            summary.append({
                "regime": i,
                "label": "crisis" if i == self.regime_labels_["crisis"] else "normal",
                "mean": mean,
                "std": np.sqrt(var),
                "persistence": persistence,
            })

        return pd.DataFrame(summary)


def build_regime_detector_gdp(
    gdp_yoy: pd.Series,
    financial_stress: Optional[pd.Series] = None,
) -> Tuple[RegimeSwitchingDetector, pd.DataFrame]:
    """Build regime detector for GDP growth.

    Parameters
    ----------
    gdp_yoy : Series
        GDP year-over-year growth rate (%).
    financial_stress : Series, optional
        Financial stress index (z-score).

    Returns
    -------
    detector : RegimeSwitchingDetector
        Fitted model.
    probabilities : DataFrame
        Regime probabilities for each period.
    """
    detector = RegimeSwitchingDetector(k_regimes=2, switching_variance=True)

    if financial_stress is not None:
        # Align dates
        df = pd.concat([gdp_yoy, financial_stress], axis=1).dropna()
        y = df.iloc[:, 0]
        exog = df.iloc[:, 1:]
    else:
        y = gdp_yoy
        exog = None

    detector.fit(y, exog=exog)
    probabilities = detector.predict_probabilities(y, exog=exog)

    logger.info("GDP regime detector summary:")
    logger.info("\n%s", detector.get_regime_summary().to_string(index=False))

    return detector, probabilities


def build_regime_detector_inflation(
    inflation: pd.Series,
    financial_stress: Optional[pd.Series] = None,
) -> Tuple[RegimeSwitchingDetector, pd.DataFrame]:
    """Build regime detector for inflation.

    Parameters
    ----------
    inflation : Series
        Inflation rate (%).
    financial_stress : Series, optional
        Financial stress index (z-score).

    Returns
    -------
    detector : RegimeSwitchingDetector
        Fitted model.
    probabilities : DataFrame
        Regime probabilities for each period.
    """
    detector = RegimeSwitchingDetector(k_regimes=2, switching_variance=True)

    if financial_stress is not None:
        df = pd.concat([inflation, financial_stress], axis=1).dropna()
        y = df.iloc[:, 0]
        exog = df.iloc[:, 1:]
    else:
        y = inflation
        exog = None

    detector.fit(y, exog=exog)
    probabilities = detector.predict_probabilities(y, exog=exog)

    logger.info("Inflation regime detector summary:")
    logger.info("\n%s", detector.get_regime_summary().to_string(index=False))

    return detector, probabilities
