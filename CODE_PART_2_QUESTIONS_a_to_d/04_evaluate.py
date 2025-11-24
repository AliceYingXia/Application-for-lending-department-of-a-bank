#!/usr/bin/env python3
"""
Evaluation Script for Gemini Forecasts

Calculates sMAPE (Symmetric Mean Absolute Percentage Error) for Gemini forecasts
by comparing them against actual 2024 financial data.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import config


def calculate_smape(actual, forecast):
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE).

    sMAPE = 100% * |actual - forecast| / ((|actual| + |forecast|) / 2)

    Args:
        actual: Actual value
        forecast: Forecasted value

    Returns:
        sMAPE as a percentage (0-200%)
    """
    if actual is None or forecast is None:
        return np.nan

    if np.isnan(actual) or np.isnan(forecast):
        return np.nan

    numerator = abs(actual - forecast)
    denominator = (abs(actual) + abs(forecast)) / 2.0

    if denominator == 0:
        return 0.0 if numerator == 0 else np.nan

    return 100.0 * numerator / denominator


def evaluate_forecast(ticker: str) -> dict:
    """
    Evaluate a single company's forecast against actual 2024 data.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with evaluation metrics
    """
    # Load forecast
    forecast_file = config.FORECASTS_DIR / "gemini" / f"{ticker}.json"
    if not forecast_file.exists():
        return {"ticker": ticker, "status": "no_forecast"}

    with open(forecast_file, 'r') as f:
        forecast_data = json.load(f)

    if forecast_data.get('status') != 'success':
        return {"ticker": ticker, "status": "forecast_failed"}

    forecast = forecast_data['forecast']['forecast_2024']

    # Load actual 2024 data
    actual_file = config.DATA_DIR / "actual" / f"{ticker}.json"
    if not actual_file.exists():
        return {"ticker": ticker, "status": "no_actual"}

    with open(actual_file, 'r') as f:
        actual_data = json.load(f)

    if '2024' not in actual_data:
        return {"ticker": ticker, "status": "no_2024_data"}

    actual = actual_data['2024']

    # Calculate sMAPE for Income Statement items
    is_metrics = ['revenue', 'net_income', 'operating_income']
    is_smapes = []

    for metric in is_metrics:
        actual_value = actual['income_statement'].get(metric)
        forecast_value = forecast['income_statement'].get(metric)

        smape = calculate_smape(actual_value, forecast_value)
        if not np.isnan(smape):
            is_smapes.append(smape)

    # Calculate sMAPE for Balance Sheet items
    bs_metrics = ['total_assets', 'equity', 'cash']
    bs_smapes = []

    for metric in bs_metrics:
        actual_value = actual['balance_sheet'].get(metric)
        forecast_value = forecast['balance_sheet'].get(metric)

        smape = calculate_smape(actual_value, forecast_value)
        if not np.isnan(smape):
            bs_smapes.append(smape)

    # Calculate average sMAPE
    all_smapes = is_smapes + bs_smapes
    avg_smape = np.mean(all_smapes) if all_smapes else np.nan
    is_avg_smape = np.mean(is_smapes) if is_smapes else np.nan
    bs_avg_smape = np.mean(bs_smapes) if bs_smapes else np.nan

    return {
        "ticker": ticker,
        "status": "evaluated",
        "avg_smape": avg_smape,
        "is_avg_smape": is_avg_smape,
        "bs_avg_smape": bs_avg_smape,
        "revenue_smape": calculate_smape(
            actual['income_statement'].get('revenue'),
            forecast['income_statement'].get('revenue')
        ),
        "net_income_smape": calculate_smape(
            actual['income_statement'].get('net_income'),
            forecast['income_statement'].get('net_income')
        ),
        "equity_smape": calculate_smape(
            actual['balance_sheet'].get('equity'),
            forecast['balance_sheet'].get('equity')
        ),
        "actual_revenue": actual['income_statement'].get('revenue'),
        "forecast_revenue": forecast['income_statement'].get('revenue'),
        "actual_net_income": actual['income_statement'].get('net_income'),
        "forecast_net_income": forecast['income_statement'].get('net_income'),
        "actual_equity": actual['balance_sheet'].get('equity'),
        "forecast_equity": forecast['balance_sheet'].get('equity')
    }


def evaluate_all_forecasts():
    """
    Evaluate all Gemini forecasts against actual 2024 data.
    """
    print("=" * 70)
    print("GEMINI FORECAST EVALUATION")
    print("=" * 70)

    # Load test sample
    sample_file = config.DATA_DIR / "test_sample.csv"
    sample = pd.read_csv(sample_file)
    print(f"\nCompanies to evaluate: {len(sample)}")

    # Evaluate each company
    print("\nEvaluating forecasts...")
    results = []
    evaluated_count = 0
    failed_count = 0

    for idx, row in tqdm(sample.iterrows(), total=len(sample), desc="Evaluating"):
        ticker = row['ticker']
        result = evaluate_forecast(ticker)
        results.append(result)

        if result['status'] == 'evaluated':
            evaluated_count += 1
        else:
            failed_count += 1
            print(f"  ⚠️  {ticker}: {result['status']}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_file = config.RESULTS_DIR / "gemini_evaluation.csv"
    results_df.to_csv(output_file, index=False)

    # Calculate summary statistics
    evaluated_df = results_df[results_df['status'] == 'evaluated']

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total companies: {len(sample)}")
    print(f"Successfully evaluated: {evaluated_count}")
    print(f"Failed evaluations: {failed_count}")

    if evaluated_count > 0:
        print(f"\nOverall Performance:")
        print(f"  Mean sMAPE: {evaluated_df['avg_smape'].mean():.2f}%")
        print(f"  Median sMAPE: {evaluated_df['avg_smape'].median():.2f}%")
        print(f"  Min sMAPE: {evaluated_df['avg_smape'].min():.2f}%")
        print(f"  Max sMAPE: {evaluated_df['avg_smape'].max():.2f}%")

        print(f"\nIncome Statement Performance:")
        print(f"  Mean sMAPE: {evaluated_df['is_avg_smape'].mean():.2f}%")
        print(f"  Median sMAPE: {evaluated_df['is_avg_smape'].median():.2f}%")

        print(f"\nBalance Sheet Performance:")
        print(f"  Mean sMAPE: {evaluated_df['bs_avg_smape'].mean():.2f}%")
        print(f"  Median sMAPE: {evaluated_df['bs_avg_smape'].median():.2f}%")

        # Top 5 best forecasts
        print(f"\nTop 5 Best Forecasts:")
        top5 = evaluated_df.nsmallest(5, 'avg_smape')[['ticker', 'avg_smape']]
        for idx, row in top5.iterrows():
            print(f"  {row['ticker']:6s} - sMAPE: {row['avg_smape']:6.2f}%")

        # Top 5 worst forecasts
        print(f"\nTop 5 Worst Forecasts:")
        bottom5 = evaluated_df.nlargest(5, 'avg_smape')[['ticker', 'avg_smape']]
        for idx, row in bottom5.iterrows():
            print(f"  {row['ticker']:6s} - sMAPE: {row['avg_smape']:6.2f}%")

    print(f"\nResults saved to: {output_file}")
    print("=" * 70 + "\n")

    return evaluated_count, failed_count, results_df


if __name__ == "__main__":
    evaluated, failed, results_df = evaluate_all_forecasts()

    if evaluated > 0:
        print(f"✅ Evaluation complete!")
        print(f"   Evaluated {evaluated} forecasts")
        if failed > 0:
            print(f"   ⚠️  {failed} forecasts could not be evaluated")
    else:
        print(f"❌ Evaluation failed!")
        print(f"   No forecasts could be evaluated")