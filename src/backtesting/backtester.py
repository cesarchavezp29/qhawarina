"""Rolling-window backtester for nowcasting models.

Runs an expanding-window evaluation loop that respects publication lags
via the VintageManager. At each evaluation date, only data available
at that point is used for model fitting and prediction.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.metrics import compute_all_metrics, relative_rmse, rmse
from src.backtesting.vintage import VintageManager
from src.utils.io import save_parquet

logger = logging.getLogger("nexus.backtester")


class RollingBacktester:
    """Expanding-window backtester with publication lag awareness.

    Parameters
    ----------
    vintage_manager : VintageManager
        Provides data vintages for each evaluation date.
    model : object
        Must implement fit(panel_wide) and nowcast(panel_wide, target_df).
    benchmarks : dict[str, object]
        Named benchmark models, each with fit() and nowcast() methods.
    target_df : pd.DataFrame
        Target variable DataFrame with 'date' column and target values.
    target_col : str
        Column name in target_df to nowcast.
    frequency : str
        'Q' for quarterly targets (GDP), 'M' for monthly (inflation).
    value_col : str
        Which panel column to use for wide-format data.
    """

    def __init__(
        self,
        vintage_manager: VintageManager,
        model,
        benchmarks: dict,
        target_df: pd.DataFrame,
        target_col: str,
        frequency: str = "Q",
        value_col: str = "value_dlog",
    ):
        self.vm = vintage_manager
        self.model = model
        self.benchmarks = benchmarks
        self.target_df = target_df.copy()
        self.target_df["date"] = pd.to_datetime(self.target_df["date"])
        self.target_col = target_col
        self.frequency = frequency
        self.value_col = value_col
        self.results: list[dict] = []

    def _generate_eval_dates(
        self, eval_start: str, eval_end: str
    ) -> list[pd.Timestamp]:
        """Generate evaluation dates at quarterly or monthly frequency.

        For quarterly: eval_date is 75 days after quarter end (Q+75d),
        i.e., mid-way through the following quarter when most data is available.
        For monthly: eval_date is 45 days after month end.
        """
        if self.frequency == "Q":
            # Generate quarter-end dates, then offset by 75 days
            quarters = pd.date_range(eval_start, eval_end, freq="QS")
            # Evaluate 75 days after quarter start (≈ when most data available)
            eval_dates = [q + pd.Timedelta(days=75) for q in quarters]
        else:
            # Monthly: evaluate 45 days after month start
            months = pd.date_range(eval_start, eval_end, freq="MS")
            eval_dates = [m + pd.Timedelta(days=45) for m in months]
        return eval_dates

    def _get_target_period(self, eval_date: pd.Timestamp) -> pd.Timestamp:
        """Map evaluation date to the target period being nowcast."""
        if self.frequency == "Q":
            # The quarter being nowcast is the one eval_date falls in minus 1Q
            # (we're nowcasting the just-ended quarter)
            quarter_start = eval_date - pd.offsets.QuarterBegin(startingMonth=1)
            return pd.Timestamp(quarter_start.year, quarter_start.month, 1)
        else:
            # Monthly: nowcasting the previous month
            return eval_date - pd.offsets.MonthBegin(1)

    def run(
        self,
        eval_start: str,
        eval_end: str,
        min_train_periods: int = 36,
    ) -> pd.DataFrame:
        """Run the expanding-window backtest.

        Parameters
        ----------
        eval_start : str
            First evaluation period (e.g., '2010-01-01' or '2010-Q1').
        eval_end : str
            Last evaluation period.
        min_train_periods : int
            Minimum number of training observations before first evaluation.

        Returns
        -------
        pd.DataFrame
            Results table with actual vs predicted values.
        """
        eval_dates = self._generate_eval_dates(eval_start, eval_end)
        self.results = []
        n_total = len(eval_dates)

        for i, eval_date in enumerate(eval_dates):
            target_period = self._get_target_period(eval_date)

            # Look up actual value
            target_row = self.target_df[self.target_df["date"] == target_period]
            if target_row.empty:
                continue
            actual = float(target_row[self.target_col].iloc[0])
            if np.isnan(actual):
                continue

            # Get vintage data
            panel_wide = self.vm.get_vintage_wide(eval_date, value_col=self.value_col)

            # Check minimum training length
            if len(panel_wide) < min_train_periods:
                logger.debug(
                    "Skipping %s: only %d periods (need %d)",
                    eval_date.strftime("%Y-%m-%d"), len(panel_wide), min_train_periods,
                )
                continue

            result = {
                "eval_date": eval_date,
                "target_period": target_period,
                "actual": actual,
            }

            # Filter target to exclude current period (prevent look-ahead bias)
            target_available = self.target_df[
                self.target_df["date"] < target_period
            ].copy()

            # Apply COVID filter if model supports it
            panel_for_fit = panel_wide
            if hasattr(self.model, "exclude_covid") and self.model.exclude_covid:
                panel_for_fit = self.model._filter_covid_periods(panel_wide, target_period)

            # Fit and nowcast with main model
            try:
                self.model.fit(panel_for_fit)
                nc = self.model.nowcast(panel_wide, target_available)
                result["dfm_nowcast"] = nc["nowcast_value"]
                result["dfm_error"] = nc["nowcast_value"] - actual
            except Exception as e:
                logger.warning("DFM failed at %s: %s", eval_date.strftime("%Y-%m-%d"), e)
                result["dfm_nowcast"] = np.nan
                result["dfm_error"] = np.nan

            # Fit and nowcast with benchmarks
            for bname, bmodel in self.benchmarks.items():
                try:
                    bmodel.fit(panel_wide)
                    bnc = bmodel.nowcast(panel_wide, target_available)
                    result[f"{bname}_nowcast"] = bnc["nowcast_value"]
                    result[f"{bname}_error"] = bnc["nowcast_value"] - actual
                except Exception as e:
                    logger.warning("%s failed at %s: %s", bname, eval_date.strftime("%Y-%m-%d"), e)
                    result[f"{bname}_nowcast"] = np.nan
                    result[f"{bname}_error"] = np.nan

            self.results.append(result)

            if (i + 1) % 10 == 0 or (i + 1) == n_total:
                logger.info("Backtest progress: %d/%d", i + 1, n_total)

        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results)

    def summary(self) -> dict:
        """Compute summary metrics for model and benchmarks.

        Returns
        -------
        dict
            Keyed by model name, each value is a metrics dict.
        """
        df = self.to_dataframe()
        if df.empty:
            return {}

        actual = df["actual"].values
        summaries = {}

        # DFM metrics
        if "dfm_nowcast" in df.columns:
            dfm_pred = df["dfm_nowcast"].values
            summaries["dfm"] = compute_all_metrics(actual, dfm_pred)

        # Benchmark metrics
        for bname in self.benchmarks:
            col = f"{bname}_nowcast"
            if col in df.columns:
                bpred = df[col].values
                summaries[bname] = compute_all_metrics(actual, bpred)

        # Add relative RMSE (vs each benchmark)
        if "dfm" in summaries:
            dfm_rmse = summaries["dfm"]["rmse"]
            for bname in self.benchmarks:
                if bname in summaries:
                    rel = relative_rmse(dfm_rmse, summaries[bname]["rmse"])
                    summaries["dfm"][f"rel_rmse_vs_{bname}"] = rel

        return summaries

    def save(self, output_path: Path):
        """Save results to parquet."""
        df = self.to_dataframe()
        if not df.empty:
            save_parquet(df, output_path)
            logger.info("Backtest results saved: %s (%d rows)", output_path, len(df))


class PovertyBacktester:
    """Expanding-window backtester for poverty nowcasting models.

    Evaluates models year by year, training on data up to year Y-1
    and nowcasting year Y. Compares model predictions against actual
    poverty outcomes across 24 departments.

    Parameters
    ----------
    dept_panel : pd.DataFrame
        Long-format departmental panel (monthly).
    model : object
        Primary model with fit(dept_features, poverty_df) and nowcast() methods.
    benchmarks : dict[str, object]
        Named benchmark models.
    poverty_df : pd.DataFrame
        Poverty targets with year, department_code, and target columns.
    target_col : str
        Primary target column for evaluation metrics.
    value_col : str
        Which panel value column to aggregate annually.
    """

    def __init__(
        self,
        dept_panel: pd.DataFrame,
        model,
        benchmarks: dict,
        poverty_df: pd.DataFrame,
        target_col: str = "poverty_rate",
        value_col: str = "value_yoy",
    ):
        self.dept_panel = dept_panel.copy()
        self.dept_panel["date"] = pd.to_datetime(self.dept_panel["date"])
        self.model = model
        self.benchmarks = benchmarks
        self.poverty_df = poverty_df.copy()
        self.target_col = target_col
        self.value_col = value_col
        self.results: list[dict] = []

    def run(
        self,
        eval_start_year: int = 2012,
        eval_end_year: int = 2024,
        min_train_years: int = 8,
    ) -> pd.DataFrame:
        """Run expanding-window poverty backtest.

        Parameters
        ----------
        eval_start_year : int
            First year to evaluate.
        eval_end_year : int
            Last year to evaluate.
        min_train_years : int
            Minimum years of poverty data required before first evaluation.

        Returns
        -------
        pd.DataFrame
            Results with year, department_code, actual, and model predictions.
        """
        from src.models.poverty import _aggregate_dept_panel_annual

        self.results = []

        for year in range(eval_start_year, eval_end_year + 1):
            # Skip COVID years if model has exclude_covid enabled
            if hasattr(self.model, "exclude_covid") and self.model.exclude_covid:
                if year in [2020, 2021]:
                    logger.info("Skipping COVID year %d (exclude_covid=True)", year)
                    continue

            # Training poverty: years before Y
            train_poverty = self.poverty_df[self.poverty_df["year"] < year].copy()

            if train_poverty["year"].nunique() < min_train_years:
                logger.debug(
                    "Skipping year %d: only %d training years (need %d)",
                    year, train_poverty["year"].nunique(), min_train_years,
                )
                continue

            # Filter panel to before July of year Y+1
            # (poverty is published ~6 months after year-end)
            cutoff = pd.Timestamp(f"{year + 1}-07-01")
            panel_available = self.dept_panel[self.dept_panel["date"] < cutoff]

            # Aggregate to annual features
            dept_features = _aggregate_dept_panel_annual(
                panel_available, value_col=self.value_col
            )

            # Actual poverty for year Y
            actual_poverty = self.poverty_df[
                self.poverty_df["year"] == year
            ].copy()
            if actual_poverty.empty:
                continue

            # Determine what to pass to model: DFM needs raw panel,
            # PanelPovertyNowcaster uses aggregated features
            model_input = (
                panel_available
                if hasattr(self.model, "pca_")
                else dept_features
            )

            # Fit and nowcast with primary model
            model_nowcasts = {}
            try:
                self.model.fit(model_input, train_poverty)
                nc = self.model.nowcast(model_input, train_poverty)
                model_nowcasts = nc.get("dept_nowcasts", {})
            except Exception as e:
                logger.warning("Model failed at year %d: %s", year, e)

            # Fit and nowcast with benchmarks
            bench_nowcasts = {}
            for bname, bmodel in self.benchmarks.items():
                try:
                    bmodel.fit(dept_features, train_poverty)
                    bnc = bmodel.nowcast(dept_features, train_poverty)
                    bench_nowcasts[bname] = bnc.get("dept_nowcasts", {})
                except Exception as e:
                    logger.warning(
                        "%s failed at year %d: %s", bname, year, e
                    )
                    bench_nowcasts[bname] = {}

            # Collect per-department results
            for _, row in actual_poverty.iterrows():
                dept = row["department_code"]
                actual = row.get(self.target_col, np.nan)
                if pd.isna(actual):
                    continue

                result = {
                    "year": year,
                    "department_code": dept,
                    "actual": float(actual),
                }

                # Model prediction
                if dept in model_nowcasts and self.target_col in model_nowcasts.get(dept, {}):
                    pred = model_nowcasts[dept][self.target_col]
                    result["panel_nowcast"] = pred
                    result["panel_error"] = pred - float(actual)
                else:
                    result["panel_nowcast"] = np.nan
                    result["panel_error"] = np.nan

                # Benchmark predictions
                for bname in self.benchmarks:
                    bn = bench_nowcasts.get(bname, {})
                    if dept in bn and self.target_col in bn.get(dept, {}):
                        pred = bn[dept][self.target_col]
                        result[f"{bname}_nowcast"] = pred
                        result[f"{bname}_error"] = pred - float(actual)
                    else:
                        result[f"{bname}_nowcast"] = np.nan
                        result[f"{bname}_error"] = np.nan

                self.results.append(result)

            logger.info("Poverty backtest: year %d done (%d depts)", year, len(actual_poverty))

        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results)

    def summary(self) -> dict:
        """Compute summary metrics by model.

        Returns
        -------
        dict
            Keyed by model name, each with RMSE, MAE, and Rel.RMSE.
        """
        df = self.to_dataframe()
        if df.empty:
            return {}

        actual = df["actual"].values
        summaries = {}

        # Panel model
        if "panel_nowcast" in df.columns:
            panel_pred = df["panel_nowcast"].values
            mask = ~np.isnan(panel_pred)
            if mask.sum() > 0:
                summaries["panel"] = compute_all_metrics(
                    actual[mask], panel_pred[mask]
                )

        # Benchmarks
        for bname in self.benchmarks:
            col = f"{bname}_nowcast"
            if col in df.columns:
                bpred = df[col].values
                mask = ~np.isnan(bpred)
                if mask.sum() > 0:
                    summaries[bname] = compute_all_metrics(
                        actual[mask], bpred[mask]
                    )

        # Relative RMSE
        if "panel" in summaries:
            panel_rmse = summaries["panel"]["rmse"]
            for bname in self.benchmarks:
                if bname in summaries:
                    rel = relative_rmse(panel_rmse, summaries[bname]["rmse"])
                    summaries["panel"][f"rel_rmse_vs_{bname}"] = rel

        return summaries

    def save(self, output_path: Path):
        """Save results to parquet."""
        df = self.to_dataframe()
        if not df.empty:
            save_parquet(df, output_path)
            logger.info(
                "Poverty backtest saved: %s (%d rows)", output_path, len(df)
            )


class MonthlyPovertyBacktester:
    """Monthly-resolution poverty backtester using rolling-window features.

    The model is trained once per year on annual data (same as PovertyBacktester),
    but predictions are produced at multiple months within each year using
    rolling-window features. This reveals how predictions evolve as new monthly
    data arrives and whether within-year noise is acceptable.

    Parameters
    ----------
    dept_panel : pd.DataFrame
        Long-format departmental panel (monthly).
    model : object
        Must implement fit(), nowcast_at_date(), and have a PanelPovertyNowcaster interface.
    benchmarks : dict[str, object]
        Named benchmark models (AR1, RW). Evaluated once per year.
    poverty_df : pd.DataFrame
        Poverty targets with year, department_code, and target columns.
    target_col : str
        Primary target column for evaluation.
    value_col : str
        Which panel value column to use.
    eval_months : list[int] or None
        Months to evaluate within each year. Default: [3, 6, 9, 12].
    window_months : int
        Rolling window size for feature aggregation. Default: 12.
    """

    def __init__(
        self,
        dept_panel: pd.DataFrame,
        model,
        benchmarks: dict,
        poverty_df: pd.DataFrame,
        target_col: str = "poverty_rate",
        value_col: str = "value_yoy",
        eval_months: list[int] | None = None,
        window_months: int = 12,
    ):
        self.dept_panel = dept_panel.copy()
        self.dept_panel["date"] = pd.to_datetime(self.dept_panel["date"])
        self.model = model
        self.benchmarks = benchmarks
        self.poverty_df = poverty_df.copy()
        self.target_col = target_col
        self.value_col = value_col
        self.eval_months = eval_months or [3, 6, 9, 12]
        self.window_months = window_months
        self.results: list[dict] = []

    def run(
        self,
        eval_start_year: int = 2012,
        eval_end_year: int = 2024,
        min_train_years: int = 8,
    ) -> pd.DataFrame:
        """Run monthly-resolution poverty backtest.

        For each year Y:
          1. Train model once on annual features (years < Y)
          2. Compute benchmarks once (AR1/RW don't update monthly)
          3. For each eval month M: produce rolling-window predictions

        Returns
        -------
        pd.DataFrame
            Results with year, month, department_code, actual, panel_nowcast, etc.
        """
        from src.models.poverty import _aggregate_dept_panel_annual

        self.results = []

        for year in range(eval_start_year, eval_end_year + 1):
            # Skip COVID years if model has exclude_covid enabled
            if hasattr(self.model, "exclude_covid") and self.model.exclude_covid:
                if year in [2020, 2021]:
                    logger.info("Skipping COVID year %d (exclude_covid=True)", year)
                    continue

            # Training poverty: years before Y
            train_poverty = self.poverty_df[self.poverty_df["year"] < year].copy()

            if train_poverty["year"].nunique() < min_train_years:
                logger.debug(
                    "Skipping year %d: only %d training years",
                    year, train_poverty["year"].nunique(),
                )
                continue

            # Actual poverty for year Y
            actual_poverty = self.poverty_df[
                self.poverty_df["year"] == year
            ].copy()
            if actual_poverty.empty:
                continue

            # Build actual lookup: {dept: actual_value}
            actual_lookup = {}
            for _, row in actual_poverty.iterrows():
                dept = row["department_code"]
                val = row.get(self.target_col, np.nan)
                if not pd.isna(val):
                    actual_lookup[dept] = float(val)

            # Filter panel for training (up to July Y+1 for annual features)
            cutoff_annual = pd.Timestamp(f"{year + 1}-07-01")
            panel_for_annual = self.dept_panel[self.dept_panel["date"] < cutoff_annual]

            # Aggregate annual features for training
            dept_features = _aggregate_dept_panel_annual(
                panel_for_annual, value_col=self.value_col
            )

            # Determine model input type
            model_input = (
                panel_for_annual
                if hasattr(self.model, "pca_")
                else dept_features
            )

            # Fit model ONCE for this year
            try:
                self.model.fit(model_input, train_poverty)
            except Exception as e:
                logger.warning("Model fit failed at year %d: %s", year, e)
                continue

            # Compute benchmarks ONCE (they don't change monthly)
            bench_nowcasts = {}
            for bname, bmodel in self.benchmarks.items():
                try:
                    bmodel.fit(dept_features, train_poverty)
                    bnc = bmodel.nowcast(dept_features, train_poverty)
                    bench_nowcasts[bname] = bnc.get("dept_nowcasts", {})
                except Exception as e:
                    logger.warning("%s failed at year %d: %s", bname, year, e)
                    bench_nowcasts[bname] = {}

            # For each eval month, produce rolling-window predictions
            for month in self.eval_months:
                end_date = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)

                # Filter panel to data available up to this date
                panel_available = self.dept_panel[self.dept_panel["date"] <= end_date]

                try:
                    nc = self.model.nowcast_at_date(
                        panel_available, train_poverty,
                        end_date=end_date,
                        window_months=self.window_months,
                        value_col=self.value_col,
                    )
                    model_nowcasts = nc.get("dept_nowcasts", {})
                    coverage = nc.get("coverage", {})
                except Exception as e:
                    logger.warning(
                        "Model nowcast_at_date failed: year=%d month=%d: %s",
                        year, month, e,
                    )
                    model_nowcasts = {}
                    coverage = {}

                # Collect per-department results
                for dept, actual_val in actual_lookup.items():
                    result = {
                        "year": year,
                        "month": month,
                        "department_code": dept,
                        "actual": actual_val,
                    }

                    # Model prediction
                    if dept in model_nowcasts and self.target_col in model_nowcasts.get(dept, {}):
                        pred = model_nowcasts[dept][self.target_col]
                        result["panel_nowcast"] = pred
                        result["panel_error"] = pred - actual_val
                    else:
                        result["panel_nowcast"] = np.nan
                        result["panel_error"] = np.nan

                    # Benchmark predictions (same for all months within year)
                    for bname in self.benchmarks:
                        bn = bench_nowcasts.get(bname, {})
                        if dept in bn and self.target_col in bn.get(dept, {}):
                            pred = bn[dept][self.target_col]
                            result[f"{bname}_nowcast"] = pred
                            result[f"{bname}_error"] = pred - actual_val
                        else:
                            result[f"{bname}_nowcast"] = np.nan
                            result[f"{bname}_error"] = np.nan

                    result["n_categories"] = coverage.get(dept, 0)
                    self.results.append(result)

            logger.info(
                "Monthly poverty backtest: year %d done (%d months × %d depts)",
                year, len(self.eval_months), len(actual_lookup),
            )

        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results)

    def convergence_analysis(self) -> pd.DataFrame:
        """Analyze how predictions improve as more monthly data arrives.

        Returns
        -------
        pd.DataFrame
            One row per eval_month with RMSE, MAE, within-year std,
            and revision magnitude.
        """
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()

        from src.backtesting.metrics import convergence_rmse_by_month, within_year_noise, revision_magnitude

        rmse_by_m = convergence_rmse_by_month(df)
        noise = within_year_noise(df)
        revisions = revision_magnitude(df)

        # Merge all analyses
        result = rmse_by_m.copy()
        if not noise.empty:
            result = result.merge(noise, on="month", how="left")
        if not revisions.empty:
            result = result.merge(revisions, on="month", how="left")

        return result

    def summary(self) -> dict:
        """Compute summary metrics aggregated across all months."""
        df = self.to_dataframe()
        if df.empty:
            return {}

        actual = df["actual"].values
        summaries = {}

        if "panel_nowcast" in df.columns:
            panel_pred = df["panel_nowcast"].values
            mask = ~np.isnan(panel_pred)
            if mask.sum() > 0:
                summaries["panel"] = compute_all_metrics(
                    actual[mask], panel_pred[mask]
                )

        for bname in self.benchmarks:
            col = f"{bname}_nowcast"
            if col in df.columns:
                bpred = df[col].values
                mask = ~np.isnan(bpred)
                if mask.sum() > 0:
                    summaries[bname] = compute_all_metrics(
                        actual[mask], bpred[mask]
                    )

        if "panel" in summaries:
            panel_rmse = summaries["panel"]["rmse"]
            for bname in self.benchmarks:
                if bname in summaries:
                    rel = relative_rmse(panel_rmse, summaries[bname]["rmse"])
                    summaries["panel"][f"rel_rmse_vs_{bname}"] = rel

        return summaries

    def save(self, output_path: Path):
        """Save results to parquet."""
        df = self.to_dataframe()
        if not df.empty:
            save_parquet(df, output_path)
            logger.info(
                "Monthly poverty backtest saved: %s (%d rows)",
                output_path, len(df),
            )
