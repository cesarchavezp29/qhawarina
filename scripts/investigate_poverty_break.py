"""Investigate structural break in poverty nowcasting around 2018.

Analyzes:
1. Error patterns over time
2. Feature importance changes
3. Potential methodological changes
4. Series correlation shifts
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import RESULTS_DIR, TARGETS_DIR, PROCESSED_DEPARTMENTAL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nexus.poverty_investigation")

# Output directory for plots
PLOTS_DIR = Path("D:/Nexus/nexus/plots")
PLOTS_DIR.mkdir(exist_ok=True)


def load_data():
    """Load poverty backtest results and targets."""
    backtest = pd.read_parquet(RESULTS_DIR / "backtest_poverty.parquet")
    targets = pd.read_parquet(TARGETS_DIR / "poverty_departmental.parquet")

    logger.info(f"Loaded backtest: {len(backtest)} rows")
    logger.info(f"Year range: {backtest['year'].min()} - {backtest['year'].max()}")

    return backtest, targets


def analyze_error_patterns(backtest: pd.DataFrame):
    """Analyze how errors change over time."""
    print("\n" + "="*60)
    print("ERROR PATTERN ANALYSIS")
    print("="*60)

    # National-level aggregation
    national = backtest.groupby("year").agg({
        "actual": "mean",
        "panel_nowcast": "mean",
        "panel_error": "mean",
        "ar1_error": "mean",
    }).reset_index()

    national["abs_error"] = national["panel_error"].abs()
    national["ar1_abs_error"] = national["ar1_error"].abs()

    # Pre/post 2018 split
    pre_2018 = national[national["year"] < 2018]
    post_2018 = national[national["year"] >= 2018]

    print("\nPRE-2018 (2012-2017):")
    print(f"  Mean absolute error: {pre_2018['abs_error'].mean()*100:.2f}pp")
    print(f"  RMSE: {np.sqrt((pre_2018['panel_error']**2).mean())*100:.2f}pp")
    print(f"  Bias: {pre_2018['panel_error'].mean()*100:.2f}pp")

    print("\nPOST-2018 (2018-2024):")
    print(f"  Mean absolute error: {post_2018['abs_error'].mean()*100:.2f}pp")
    print(f"  RMSE: {np.sqrt((post_2018['panel_error']**2).mean())*100:.2f}pp")
    print(f"  Bias: {post_2018['panel_error'].mean()*100:.2f}pp")

    # Trend over time
    print("\nERROR TREND:")
    for year in sorted(national["year"].unique()):
        row = national[national["year"] == year].iloc[0]
        print(f"  {year}: RMSE={row['panel_error']**2:.4f}  Bias={row['panel_error']*100:+.2f}pp")

    # Statistical test for break
    pre_errors = backtest[backtest["year"] < 2018]["panel_error"]
    post_errors = backtest[backtest["year"] >= 2018]["panel_error"]

    t_stat, p_value = stats.ttest_ind(pre_errors.abs(), post_errors.abs())
    print(f"\nT-test for error magnitude change:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")

    # Plot error over time
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: RMSE over time
    ax1 = axes[0]
    ax1.plot(national["year"], national["abs_error"] * 100, marker="o", linewidth=2, label="Panel GBR")
    ax1.plot(national["year"], national["ar1_abs_error"] * 100, marker="s", linewidth=2, label="AR(1) Benchmark", alpha=0.7)
    ax1.axvline(2018, color="red", linestyle="--", alpha=0.5, label="2018 Break")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Mean Absolute Error (pp)")
    ax1.set_title("Poverty Nowcast Error Over Time")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Bias over time
    ax2 = axes[1]
    ax2.plot(national["year"], national["panel_error"] * 100, marker="o", linewidth=2)
    ax2.axhline(0, color="black", linestyle="-", alpha=0.3)
    ax2.axvline(2018, color="red", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Bias (pp)")
    ax2.set_title("Nowcast Bias Over Time")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "poverty_error_trend.png", dpi=150)
    logger.info(f"Saved: {PLOTS_DIR / 'poverty_error_trend.png'}")
    plt.close()

    return national


def analyze_departmental_patterns(backtest: pd.DataFrame):
    """Check if break affects all departments equally."""
    print("\n" + "="*60)
    print("DEPARTMENTAL BREAK ANALYSIS")
    print("="*60)

    dept_codes = backtest["department_code"].unique()

    pre_rmse = []
    post_rmse = []
    dept_names = []

    for dept in sorted(dept_codes):
        dept_data = backtest[backtest["department_code"] == dept]
        pre = dept_data[dept_data["year"] < 2018]
        post = dept_data[dept_data["year"] >= 2018]

        if len(pre) > 0 and len(post) > 0:
            pre_r = np.sqrt((pre["panel_error"]**2).mean()) * 100
            post_r = np.sqrt((post["panel_error"]**2).mean()) * 100

            pre_rmse.append(pre_r)
            post_rmse.append(post_r)
            dept_names.append(dept)

    # Departments with biggest degradation
    degradation = np.array(post_rmse) - np.array(pre_rmse)
    worst_idx = np.argsort(degradation)[-5:]

    print("\nTop 5 departments with worst degradation:")
    for idx in worst_idx:
        print(f"  {dept_names[idx]}: {pre_rmse[idx]:.1f}pp -> {post_rmse[idx]:.1f}pp (delta={degradation[idx]:+.1f}pp)")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pre_rmse, post_rmse, alpha=0.6, s=100)

    # 45-degree line
    max_val = max(max(pre_rmse), max(post_rmse))
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label="No change")

    # Label worst departments
    for idx in worst_idx:
        ax.annotate(dept_names[idx], (pre_rmse[idx], post_rmse[idx]),
                   fontsize=8, alpha=0.7)

    ax.set_xlabel("Pre-2018 RMSE (pp)")
    ax.set_ylabel("Post-2018 RMSE (pp)")
    ax.set_title("Departmental Error Changes: Pre vs Post 2018")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "poverty_dept_degradation.png", dpi=150)
    logger.info(f"Saved: {PLOTS_DIR / 'poverty_dept_degradation.png'}")
    plt.close()


def check_covid_impact(backtest: pd.DataFrame):
    """Separate 2018 break from COVID impact."""
    print("\n" + "="*60)
    print("COVID VS 2018 BREAK ANALYSIS")
    print("="*60)

    national = backtest.groupby("year")["panel_error"].apply(
        lambda x: np.sqrt((x**2).mean()) * 100
    ).reset_index()
    national.columns = ["year", "rmse"]

    periods = {
        "Pre-2018 (2012-2017)": (2012, 2017),
        "2018-2019": (2018, 2019),
        "COVID (2020-2021)": (2020, 2021),
        "Post-COVID (2022-2024)": (2022, 2024),
    }

    for period, (start, end) in periods.items():
        mask = (national["year"] >= start) & (national["year"] <= end)
        period_rmse = national[mask]["rmse"].mean()
        print(f"{period:30s}: {period_rmse:.2f}pp")

    # Check if 2018-2019 already shows degradation (before COVID)
    pre_2018 = national[national["year"] < 2018]["rmse"].mean()
    yr_2018_2019 = national[(national["year"] >= 2018) & (national["year"] < 2020)]["rmse"].mean()

    print(f"\nDegradation 2018-2019 vs Pre-2018: {yr_2018_2019 - pre_2018:+.2f}pp")
    print("-> Break started BEFORE COVID" if yr_2018_2019 > pre_2018 + 1.0 else "-> Break is primarily COVID-driven")


def recommend_solutions(national: pd.DataFrame):
    """Recommend modeling approaches based on findings."""
    print("\n" + "="*60)
    print("RECOMMENDED SOLUTIONS")
    print("="*60)

    post_2018 = national[national["year"] >= 2018]
    post_rmse = np.sqrt((post_2018["panel_error"]**2).mean()) * 100

    print("\n1. TEMPORAL SPLITTING STRATEGY")
    print("   Option A: Train only on 2015+ (discard old data)")
    print("   Option B: Add post-2018 dummy variable")
    print("   Option C: Separate models pre/post 2018")

    print("\n2. FEATURE ENGINEERING")
    print("   - Check if NTL relationship changed")
    print("   - Add COVID-related indicators for 2020-2021")
    print("   - Include dept-specific trends")

    print("\n3. MODEL COMPLEXITY")
    print("   Current: GBR with default params")
    print("   Try: Tuned GBR, Random Forest, or Neural Network")

    print("\n4. NEXT STEPS")
    print("   [ ] Re-run with 2015+ training data only")
    print("   [ ] Add post-2018 dummy to features")
    print("   [ ] Hyperparameter tuning for GBR")
    print("   [ ] Check INEI methodology changes")


def main():
    logger.info("="*60)
    logger.info("POVERTY NOWCAST STRUCTURAL BREAK INVESTIGATION")
    logger.info("="*60)

    # Load data
    backtest, targets = load_data()

    # Run analyses
    national = analyze_error_patterns(backtest)
    analyze_departmental_patterns(backtest)
    check_covid_impact(backtest)
    recommend_solutions(national)

    logger.info("\n✓ Analysis complete. Plots saved to: %s", PLOTS_DIR)


if __name__ == "__main__":
    main()
