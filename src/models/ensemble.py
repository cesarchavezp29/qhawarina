"""Ensemble model combinator for nowcasting.

Combines multiple nowcasting models using configurable weighting strategies:
  - equal: simple average (Timmermann 2006 — hard to beat)
  - inverse_rmse: weight by 1/RMSE from recent performance
  - trimmed: drop worst model, average rest

All component models must implement the fit(panel_wide) / nowcast(panel_wide, target_df)
interface used by NowcastDFM, PhillipsCurveNowcaster, AR1Benchmark, etc.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("nexus.ensemble")


class EnsembleNowcaster:
    """Combine multiple nowcasting models.

    Parameters
    ----------
    models : list of (name, model) tuples
        Each model must have fit(panel_wide) and nowcast(panel_wide, target_df).
    method : str
        Weighting method: 'equal', 'inverse_rmse', or 'trimmed'.
    target : str
        'gdp' or 'inflation' — forwarded for compatibility.
    """

    def __init__(
        self,
        models: list[tuple[str, object]],
        method: str = "equal",
        target: str = "gdp",
    ):
        self.models = models
        self.method = method
        self.target = target
        self._weights = None
        self._recent_errors = {}

    def fit(self, panel_wide: pd.DataFrame):
        """Fit all component models."""
        for name, model in self.models:
            try:
                model.fit(panel_wide)
                logger.info("Fitted %s", name)
            except Exception as e:
                logger.warning("Failed to fit %s: %s", name, e)

    def set_weights(self, weights: dict[str, float]):
        """Manually set model weights (must sum to 1)."""
        self._weights = weights

    def update_errors(self, errors: dict[str, float]):
        """Update recent error history for inverse_rmse weighting.

        Parameters
        ----------
        errors : dict mapping model name to RMSE from recent evaluation.
        """
        self._recent_errors = errors

    def _compute_weights(self, available: list[str]) -> dict[str, float]:
        """Compute weights based on method and available models."""
        if self._weights:
            # Use manual weights, renormalize to available models
            w = {k: v for k, v in self._weights.items() if k in available}
            total = sum(w.values())
            if total > 0:
                return {k: v / total for k, v in w.items()}

        if self.method == "inverse_rmse" and self._recent_errors:
            # Weight by 1/RMSE (models with lower error get higher weight)
            inv = {}
            for name in available:
                rmse = self._recent_errors.get(name)
                if rmse and rmse > 0:
                    inv[name] = 1.0 / rmse
                else:
                    inv[name] = 1.0  # default if no error history
            total = sum(inv.values())
            return {k: v / total for k, v in inv.items()}

        if self.method == "trimmed" and len(available) >= 3:
            # Drop the model with the worst recent RMSE, equal-weight rest
            if self._recent_errors:
                worst = max(
                    (n for n in available if n in self._recent_errors),
                    key=lambda n: self._recent_errors[n],
                    default=None,
                )
                if worst:
                    trimmed = [n for n in available if n != worst]
                    logger.info("Trimmed: dropping %s", worst)
                    w = 1.0 / len(trimmed)
                    return {n: w for n in trimmed}

        # Default: equal weights
        w = 1.0 / len(available)
        return {n: w for n in available}

    def nowcast(
        self, panel_wide: pd.DataFrame, target_df: pd.DataFrame
    ) -> dict:
        """Generate ensemble nowcast from all component models.

        Returns
        -------
        dict with:
            nowcast_value: weighted combination
            component_nowcasts: dict of individual model predictions
            weights: dict of model weights used
        """
        predictions = {}

        for name, model in self.models:
            try:
                result = model.nowcast(panel_wide, target_df)
                val = result.get("nowcast_value", np.nan)
                if not np.isnan(val):
                    predictions[name] = val
            except Exception as e:
                logger.warning("Nowcast failed for %s: %s", name, e)

        if not predictions:
            return {"nowcast_value": np.nan, "component_nowcasts": {}, "weights": {}}

        available = list(predictions.keys())
        weights = self._compute_weights(available)

        # Weighted average
        combined = sum(predictions[n] * weights.get(n, 0) for n in available)

        logger.info(
            "Ensemble (%s): %.3f from %d models %s",
            self.method, combined, len(available),
            {n: f"{predictions[n]:.3f}*{weights.get(n,0):.2f}" for n in available},
        )

        return {
            "nowcast_value": combined,
            "component_nowcasts": predictions,
            "weights": weights,
        }
