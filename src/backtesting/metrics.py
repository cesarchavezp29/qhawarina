"""Backtesting metrics: RMSE, MAE, directional accuracy, relative RMSE.

Also includes convergence metrics for monthly poverty nowcasting:
  - convergence_rmse_by_month: RMSE at each evaluation month
  - within_year_noise: std of predictions across months for same (year, dept)
  - revision_magnitude: mean |pred_m - pred_{m-1}| by month
"""

import numpy as np
import pandas as pd


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(actual[mask] - predicted[mask])))


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Fraction of correct sign predictions (0-1)."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if mask.sum() == 0:
        return np.nan
    correct = np.sign(actual[mask]) == np.sign(predicted[mask])
    return float(correct.mean())


def relative_rmse(rmse_model: float, rmse_benchmark: float) -> float:
    """Ratio of model RMSE to benchmark RMSE. < 1 means model wins."""
    if rmse_benchmark == 0 or np.isnan(rmse_benchmark):
        return np.nan
    return rmse_model / rmse_benchmark


def compute_all_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute all metrics at once.

    Returns
    -------
    dict
        Keys: rmse, mae, directional_accuracy, n_obs
    """
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    return {
        "rmse": rmse(actual, predicted),
        "mae": mae(actual, predicted),
        "directional_accuracy": directional_accuracy(actual, predicted),
        "n_obs": int(mask.sum()),
    }


# ── Convergence metrics for monthly poverty nowcasting ────────────────────


def convergence_rmse_by_month(results: pd.DataFrame) -> pd.DataFrame:
    """RMSE at each evaluation month across all (year, dept) pairs.

    Parameters
    ----------
    results : pd.DataFrame
        Monthly backtest results with month, actual, panel_nowcast columns.

    Returns
    -------
    pd.DataFrame
        One row per month with columns: month, rmse, mae, n_obs.
    """
    rows = []
    for m, grp in results.groupby("month"):
        actual = grp["actual"].values
        pred = grp["panel_nowcast"].values
        mask = np.isfinite(actual) & np.isfinite(pred)
        if mask.sum() == 0:
            continue
        rows.append({
            "month": int(m),
            "rmse": float(np.sqrt(np.mean((actual[mask] - pred[mask]) ** 2))),
            "mae": float(np.mean(np.abs(actual[mask] - pred[mask]))),
            "n_obs": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def within_year_noise(results: pd.DataFrame) -> pd.DataFrame:
    """Std of predictions for same (year, dept) across months.

    High noise means predictions change too much month-to-month,
    which reduces credibility. Target: std < 2pp.

    Returns
    -------
    pd.DataFrame
        One row per month with mean within-year std at that point.
    """
    df = results.dropna(subset=["panel_nowcast"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["month", "within_year_std"])

    # For each month M, compute std of predictions across months [first, M]
    # for each (year, dept) pair
    rows = []
    months_sorted = sorted(df["month"].unique())
    for m in months_sorted:
        # Include all months up to and including m
        subset = df[df["month"] <= m]
        stds = (
            subset.groupby(["year", "department_code"])["panel_nowcast"]
            .std()
            .dropna()
        )
        rows.append({
            "month": int(m),
            "within_year_std": float(stds.mean()) if len(stds) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def revision_magnitude(results: pd.DataFrame) -> pd.DataFrame:
    """Mean |prediction_m - prediction_{m-1}| by month.

    Shows how much predictions revise from one evaluation month to the next.

    Returns
    -------
    pd.DataFrame
        One row per month (except first) with mean revision magnitude.
    """
    df = results.dropna(subset=["panel_nowcast"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["month", "mean_revision"])

    months_sorted = sorted(df["month"].unique())
    if len(months_sorted) < 2:
        return pd.DataFrame(columns=["month", "mean_revision"])

    rows = []
    for i in range(1, len(months_sorted)):
        m_prev = months_sorted[i - 1]
        m_curr = months_sorted[i]

        prev = df[df["month"] == m_prev].set_index(["year", "department_code"])["panel_nowcast"]
        curr = df[df["month"] == m_curr].set_index(["year", "department_code"])["panel_nowcast"]

        common = prev.index.intersection(curr.index)
        if len(common) == 0:
            continue

        revisions = np.abs(curr.loc[common].values - prev.loc[common].values)
        rows.append({
            "month": int(m_curr),
            "mean_revision": float(np.mean(revisions)),
        })
    return pd.DataFrame(rows)
