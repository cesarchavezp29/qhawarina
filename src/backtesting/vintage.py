"""Data vintage manager — tracks publication lags for pseudo-real-time backtesting.

Provides the core anti-look-ahead-bias mechanism: given an 'as-of' date,
returns only the data that would have been publicly available at that point.

A series observation for reference month M is available at:
    M_end + publication_lag_days
where M_end is the last day of month M.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger("nexus.vintage")


class VintageManager:
    """Manages data vintages respecting publication lags.

    Parameters
    ----------
    panel_path : Path
        Path to the long-format national panel parquet.
    lags_config_path : Path or None
        Path to publication_lags.yaml. If None, uses lags from the panel.
    """

    def __init__(self, panel_path: Path, lags_config_path: Path | None = None):
        self.panel = pd.read_parquet(panel_path)
        self.panel["date"] = pd.to_datetime(self.panel["date"])

        # Load lag overrides from YAML if provided
        self._lag_overrides: dict[str, int] = {}
        self._lag_defaults: dict[str, int] = {}
        if lags_config_path and Path(lags_config_path).exists():
            with open(lags_config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self._lag_defaults = config.get("defaults", {})
            self._lag_overrides = config.get("overrides", {})

        # Apply overrides to panel lag column
        self._apply_lag_overrides()

        # Pre-compute publication dates for each observation
        self.panel["month_end"] = self.panel["date"] + pd.offsets.MonthEnd(0)
        self.panel["available_date"] = (
            self.panel["month_end"]
            + pd.to_timedelta(self.panel["publication_lag_days"], unit="D")
        )

    def _apply_lag_overrides(self):
        """Apply YAML overrides to publication_lag_days in the panel."""
        if not self._lag_overrides and not self._lag_defaults:
            return

        for series_id in self.panel["series_id"].unique():
            if series_id in self._lag_overrides:
                mask = self.panel["series_id"] == series_id
                self.panel.loc[mask, "publication_lag_days"] = self._lag_overrides[series_id]
            elif self._lag_defaults:
                # Apply category default if no override
                mask = self.panel["series_id"] == series_id
                category = self.panel.loc[mask, "category"].iloc[0]
                if category in self._lag_defaults:
                    self.panel.loc[mask, "publication_lag_days"] = self._lag_defaults[category]

    def get_vintage(self, as_of_date: str | pd.Timestamp) -> pd.DataFrame:
        """Return panel rows available as of the given date.

        Parameters
        ----------
        as_of_date : str or Timestamp
            The evaluation date. Only observations whose
            available_date <= as_of_date are returned.

        Returns
        -------
        pd.DataFrame
            Subset of the panel in long format.
        """
        as_of = pd.Timestamp(as_of_date)
        vintage = self.panel[self.panel["available_date"] <= as_of].copy()
        return vintage

    def get_vintage_wide(
        self,
        as_of_date: str | pd.Timestamp,
        value_col: str = "value_dlog",
    ) -> pd.DataFrame:
        """Return vintage pivoted to wide format (date × series matrix).

        Parameters
        ----------
        as_of_date : str or Timestamp
            The evaluation date.
        value_col : str
            Which value column to pivot (default: 'value_dlog').
            Use 'auto' to pick best column per series (value_yoy if
            available, otherwise value_raw).

        Returns
        -------
        pd.DataFrame
            Wide-format DataFrame with DatetimeIndex and series_id columns.
        """
        vintage = self.get_vintage(as_of_date)

        if value_col == "auto":
            return self._pivot_auto(vintage)

        wide = vintage.pivot_table(
            index="date",
            columns="series_id",
            values=value_col,
            aggfunc="first",
        )
        wide = wide.sort_index()
        return wide

    def _pivot_auto(self, vintage: pd.DataFrame) -> pd.DataFrame:
        """Build wide panel picking best value column per series.

        For each series:
          - If value_yoy has >= 50% non-null → use value_yoy
          - Else → use value_raw (covers rate_pct, var_pct series)
        """
        frames = []
        for sid in vintage["series_id"].unique():
            sub = vintage[vintage["series_id"] == sid][["date", "value_yoy", "value_raw"]]
            n = len(sub)
            yoy_ok = sub["value_yoy"].notna().sum()

            if yoy_ok >= 0.5 * n:
                col_data = sub.set_index("date")["value_yoy"].rename(sid)
            else:
                col_data = sub.set_index("date")["value_raw"].rename(sid)

            frames.append(col_data)

        if not frames:
            return pd.DataFrame()

        wide = pd.concat(frames, axis=1).sort_index()
        return wide

    def available_series(self, as_of_date: str | pd.Timestamp) -> list[str]:
        """List series IDs with at least one observation available.

        Parameters
        ----------
        as_of_date : str or Timestamp
            The evaluation date.

        Returns
        -------
        list[str]
            Series IDs available as of the date.
        """
        vintage = self.get_vintage(as_of_date)
        return sorted(vintage["series_id"].unique().tolist())

    def diagnose_vintage(self, as_of_date: str | pd.Timestamp) -> pd.DataFrame:
        """Show the ragged edge: last available date per series.

        Parameters
        ----------
        as_of_date : str or Timestamp
            The evaluation date.

        Returns
        -------
        pd.DataFrame
            DataFrame with series_id, category, last_date, lag_days,
            and n_obs columns, sorted by last_date ascending.
        """
        vintage = self.get_vintage(as_of_date)
        as_of = pd.Timestamp(as_of_date)

        diag = (
            vintage.groupby(["series_id", "category"])
            .agg(
                last_date=("date", "max"),
                n_obs=("date", "count"),
                lag_days=("publication_lag_days", "first"),
            )
            .reset_index()
        )
        diag["days_stale"] = (as_of - diag["last_date"]).dt.days
        diag = diag.sort_values("last_date", ascending=True).reset_index(drop=True)
        return diag
