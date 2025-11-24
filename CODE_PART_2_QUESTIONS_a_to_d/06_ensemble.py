#!/usr/bin/env python3
"""
Ensemble Model: Combining Traditional Policy-Based Model + Gemini LLM

Tests whether ensemble methods can outperform individual models,
especially for balance sheet forecasts.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import config
from typing import Dict, List, Tuple


def symmetric_mean_absolute_percentage_error(actual: float, forecast: float) -> float:
    """Calculate sMAPE between actual and forecast."""
    if actual == 0 and forecast == 0:
        return 0.0
    return 100 * abs(forecast - actual) / (abs(actual) + abs(forecast))


def load_forecasts_and_actuals():
    """
    Load Gemini forecasts, Traditional forecasts, and actual 2024 data from CSV files.

    Returns:
        DataFrame with all three datasets aligned
    """
    # Load Gemini evaluation results
    gemini_results = pd.read_csv(config.RESULTS_DIR / "gemini_evaluation.csv")
    gemini_evaluated = gemini_results[gemini_results['status'] == 'evaluated'].copy()

    # Load Traditional model results from PART_1
    traditional_results = pd.read_csv(config.PART_1_RESULTS)
    traditional_evaluated = traditional_results[traditional_results['status'] == 'evaluated'].copy()

    # Merge on ticker
    merged = gemini_evaluated.merge(
        traditional_evaluated[['ticker', 'forecast_revenue', 'forecast_net_income',
                              'forecast_equity', 'forecast_total_assets', 'forecast_cash']],
        on='ticker',
        suffixes=('_gemini', '_traditional')
    )

    print(f"Found {len(merged)} companies with both Gemini and Traditional forecasts")

    return merged


def create_ensemble_forecasts_from_row(row, method: str = 'simple_average',
                                      weights: Tuple[float, float] = (0.5, 0.5)) -> Dict[str, float]:
    """
    Create ensemble forecasts from a dataframe row containing both Gemini and Traditional forecasts.

    Args:
        row: DataFrame row with Gemini and Traditional forecast columns
        method: Ensemble method
        weights: (gemini_weight, traditional_weight) for weighted_average

    Returns:
        Dictionary with ensemble forecasts for key metrics
    """
    ensemble = {}

    # Balance sheet items (focus on these for the experiment)
    bs_items = {
        'equity': ('forecast_equity_gemini', 'forecast_equity'),
        'total_assets': ('forecast_total_assets', 'forecast_total_assets'),  # Same column name
        'cash': ('forecast_cash', 'forecast_cash'),  # Same column name
    }

    # Income statement items
    is_items = {
        'revenue': ('forecast_revenue_gemini', 'forecast_revenue'),
        'net_income': ('forecast_net_income_gemini', 'forecast_net_income'),
    }

    all_items = {**bs_items, **is_items}

    for item_name, (gemini_col, trad_col) in all_items.items():
        # Handle column naming differences
        gemini_val = row.get(gemini_col, row.get(f'forecast_{item_name}_gemini', 0))
        trad_val = row.get(trad_col, row.get(f'forecast_{item_name}_traditional', row.get(f'forecast_{item_name}', 0)))

        if pd.isna(gemini_val):
            gemini_val = 0
        if pd.isna(trad_val):
            trad_val = 0

        if method == 'simple_average':
            ensemble[f'forecast_{item_name}'] = (gemini_val + trad_val) / 2

        elif method == 'weighted_average':
            gemini_weight, trad_weight = weights
            ensemble[f'forecast_{item_name}'] = (
                gemini_weight * gemini_val + trad_weight * trad_val
            )

        elif method == 'median':
            ensemble[f'forecast_{item_name}'] = np.median([gemini_val, trad_val])

        elif method == 'favor_traditional_bs':
            # Use traditional for balance sheet items, average for others
            if item_name in bs_items:
                ensemble[f'forecast_{item_name}'] = trad_val
            else:
                ensemble[f'forecast_{item_name}'] = (gemini_val + trad_val) / 2

        elif method == 'favor_gemini_is':
            # Use Gemini for income statement, traditional for balance sheet
            if item_name in bs_items:
                ensemble[f'forecast_{item_name}'] = trad_val
            elif item_name in is_items:
                ensemble[f'forecast_{item_name}'] = gemini_val
            else:
                ensemble[f'forecast_{item_name}'] = (gemini_val + trad_val) / 2

    return ensemble


def evaluate_bs_smape_from_row(row, forecast_prefix='forecast') -> float:
    """
    Calculate balance sheet sMAPE from a dataframe row.

    Args:
        row: DataFrame row with actual and forecast columns
        forecast_prefix: Prefix for forecast columns

    Returns:
        Average sMAPE for balance sheet items
    """
    smapes = []

    # Balance sheet items to evaluate
    bs_items = ['equity', 'total_assets', 'cash']

    for item in bs_items:
        actual_col = f'actual_{item}'
        forecast_col = f'{forecast_prefix}_{item}'

        if actual_col in row and forecast_col in row:
            actual_val = row[actual_col]
            forecast_val = row[forecast_col]

            if pd.notna(actual_val) and pd.notna(forecast_val):
                smape = symmetric_mean_absolute_percentage_error(actual_val, forecast_val)
                smapes.append(smape)

    return np.mean(smapes) if smapes else None


def run_ensemble_experiment():
    """
    Run ensemble model experiment with multiple strategies.
    """
    print("=" * 70)
    print("ENSEMBLE MODEL EXPERIMENT")
    print("=" * 70)
    print("Testing multiple ensemble strategies for balance sheet forecasts")
    print("=" * 70)

    # Load all data
    print("\nLoading forecasts and actuals...")
    df = load_forecasts_and_actuals()
    print(f"Loaded data for {len(df)} companies")

    # Define ensemble methods to test
    ensemble_methods = [
        ('simple_average', 'Simple Average (50/50)', {}),
        ('weighted_average', 'Weighted (30% Gemini, 70% Traditional)', {'weights': (0.3, 0.7)}),
        ('weighted_average', 'Weighted (20% Gemini, 80% Traditional)', {'weights': (0.2, 0.8)}),
        ('median', 'Median', {}),
        ('favor_traditional_bs', 'Hybrid: Traditional for BS, Average for IS', {}),
        ('favor_gemini_is', 'Hybrid: Gemini for IS, Traditional for BS', {}),
    ]

    # Results storage
    all_results = []

    # Evaluate each ensemble method
    for method, method_name, kwargs in ensemble_methods:
        print(f"Evaluating: {method_name}...")

        for idx, row in df.iterrows():
            ticker = row['ticker']

            # Create ensemble forecast from row
            ensemble_forecasts = create_ensemble_forecasts_from_row(row, method=method, **kwargs)

            # Add ensemble forecasts to row
            ensemble_row = {**row, **ensemble_forecasts}

            # Evaluate balance sheet sMAPE
            bs_smape = evaluate_bs_smape_from_row(ensemble_row, forecast_prefix='forecast')

            if bs_smape is not None:
                all_results.append({
                    'ticker': ticker,
                    'method': method_name,
                    'bs_avg_smape': bs_smape
                })

    # Also get individual model results for comparison
    print("Calculating individual model performance...")

    for idx, row in df.iterrows():
        ticker = row['ticker']

        # Gemini
        gemini_smape = evaluate_bs_smape_from_row(row, forecast_prefix='forecast_gemini')
        if gemini_smape is not None:
            all_results.append({
                'ticker': ticker,
                'method': 'Gemini (Individual)',
                'bs_avg_smape': gemini_smape
            })

        # Traditional
        trad_smape = evaluate_bs_smape_from_row(row, forecast_prefix='forecast')
        if trad_smape is not None:
            all_results.append({
                'ticker': ticker,
                'method': 'Traditional (Individual)',
                'bs_avg_smape': trad_smape
            })

    # Create full results DataFrame
    full_results_df = pd.DataFrame(all_results)

    # Calculate summary statistics
    print("\n" + "=" * 70)
    print("BALANCE SHEET PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"\n{'Method':<50} {'Mean sMAPE':<12} {'Median sMAPE':<12} {'Std Dev'}")
    print("-" * 70)

    summary_stats = []

    for method_name in full_results_df['method'].unique():
        method_data = full_results_df[full_results_df['method'] == method_name]

        mean_smape = method_data['bs_avg_smape'].mean()
        median_smape = method_data['bs_avg_smape'].median()
        std_smape = method_data['bs_avg_smape'].std()

        print(f"{method_name:<50} {mean_smape:>11.2f}% {median_smape:>11.2f}% {std_smape:>11.2f}%")

        summary_stats.append({
            'method': method_name,
            'mean_smape': mean_smape,
            'median_smape': median_smape,
            'std_smape': std_smape
        })

    summary_df = pd.DataFrame(summary_stats).sort_values('mean_smape')

    # Find best ensemble method
    best_method = summary_df.iloc[0]

    print("\n" + "=" * 70)
    print("BEST PERFORMING METHOD")
    print("=" * 70)
    print(f"Method: {best_method['method']}")
    print(f"Mean sMAPE: {best_method['mean_smape']:.2f}%")
    print(f"Median sMAPE: {best_method['median_smape']:.2f}%")

    # Compare to individual models
    gemini_mean = full_results_df[full_results_df['method'] == 'Gemini (Individual)']['bs_avg_smape'].mean()
    trad_mean = full_results_df[full_results_df['method'] == 'Traditional (Individual)']['bs_avg_smape'].mean()

    print("\n" + "=" * 70)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 70)
    print(f"Gemini Individual Mean sMAPE:      {gemini_mean:.2f}%")
    print(f"Traditional Individual Mean sMAPE: {trad_mean:.2f}%")
    print(f"Best Ensemble Mean sMAPE:          {best_method['mean_smape']:.2f}%")
    print(f"\nImprovement vs Gemini:      {gemini_mean - best_method['mean_smape']:+.2f}%")
    print(f"Improvement vs Traditional: {trad_mean - best_method['mean_smape']:+.2f}%")

    if best_method['mean_smape'] < min(gemini_mean, trad_mean):
        print("\n✅ Ensemble outperforms BOTH individual models!")
    elif best_method['mean_smape'] < max(gemini_mean, trad_mean):
        print("\n✅ Ensemble outperforms the weaker model but not the stronger one")
    else:
        print("\n❌ Individual models outperform ensemble")

    # Save results
    output_file = config.RESULTS_DIR / "ensemble_results.csv"
    full_results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

    summary_file = config.RESULTS_DIR / "ensemble_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")

    # Generate report
    report = f"""
ENSEMBLE MODEL EXPERIMENT REPORT
{'=' * 70}

Research Question:
  Can combining Traditional Policy-Based Model + Gemini LLM create an
  ensemble that outperforms individual models for balance sheet forecasts?

Dataset: {len(df)} S&P 500 companies with complete data
Target: Balance Sheet Forecasts for 2024

RESULTS:
  Best Ensemble Method: {best_method['method']}
  Best Ensemble Mean sMAPE: {best_method['mean_smape']:.2f}%

  Gemini Individual:      {gemini_mean:.2f}%
  Traditional Individual: {trad_mean:.2f}%

  Improvement vs Gemini:      {gemini_mean - best_method['mean_smape']:+.2f}%
  Improvement vs Traditional: {trad_mean - best_method['mean_smape']:+.2f}%

CONCLUSION:
  {'Ensemble model successfully outperforms both individual models!' if best_method['mean_smape'] < min(gemini_mean, trad_mean) else
   'Ensemble provides marginal improvement over the weaker model.' if best_method['mean_smape'] < max(gemini_mean, trad_mean) else
   'Individual models are sufficient; ensemble does not improve performance.'}

RECOMMENDATION:
  {'Deploy the ensemble model for production balance sheet forecasting.' if best_method['mean_smape'] < min(gemini_mean, trad_mean) else
   'Use the traditional model for balance sheet forecasts.' if trad_mean < gemini_mean else
   'Use the Gemini model for balance sheet forecasts.'}

{'=' * 70}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    report_file = config.RESULTS_DIR / "ensemble_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"Report saved to: {report_file}")
    print("=" * 70 + "\n")

    return full_results_df, summary_df


if __name__ == "__main__":
    results_df, summary_df = run_ensemble_experiment()

    print("✅ Ensemble experiment complete!")
    print(f"   Tested {len(summary_df)} ensemble strategies")
    print(f"   Best method: {summary_df.iloc[0]['method']} ({summary_df.iloc[0]['mean_smape']:.2f}% sMAPE)")