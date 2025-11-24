#!/usr/bin/env python3
"""
Comparison Script: Gemini vs Traditional Model

Compares the performance of Gemini 2.5 Flash forecasts against the traditional
policy-based model from PART_1.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import config


def compare_models():
    """
    Compare Gemini vs Traditional model performance.
    """
    print("=" * 70)
    print("GEMINI VS TRADITIONAL MODEL COMPARISON")
    print("=" * 70)

    # Load Gemini evaluation results
    gemini_results = pd.read_csv(config.RESULTS_DIR / "gemini_evaluation.csv")
    gemini_evaluated = gemini_results[gemini_results['status'] == 'evaluated'].copy()

    # Load PART_1 results
    part1_results = pd.read_csv(config.PART_1_RESULTS)

    # Merge on ticker to get paired comparison
    comparison = gemini_evaluated.merge(
        part1_results[['ticker', 'avg_smape', 'is_avg_smape', 'bs_avg_smape']],
        on='ticker',
        suffixes=('_gemini', '_traditional')
    )

    print(f"\nCompanies with both forecasts: {len(comparison)}")

    # Calculate statistics
    gemini_mean = comparison['avg_smape_gemini'].mean()
    traditional_mean = comparison['avg_smape_traditional'].mean()

    gemini_median = comparison['avg_smape_gemini'].median()
    traditional_median = comparison['avg_smape_traditional'].median()

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(
        comparison['avg_smape_gemini'],
        comparison['avg_smape_traditional']
    )

    # Wilcoxon signed-rank test (non-parametric alternative)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
        comparison['avg_smape_gemini'],
        comparison['avg_smape_traditional']
    )

    # Count wins
    gemini_wins = (comparison['avg_smape_gemini'] < comparison['avg_smape_traditional']).sum()
    traditional_wins = (comparison['avg_smape_gemini'] > comparison['avg_smape_traditional']).sum()
    ties = (comparison['avg_smape_gemini'] == comparison['avg_smape_traditional']).sum()

    # Print results
    print("\n" + "=" * 70)
    print("OVERALL PERFORMANCE")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'Gemini 2.5':<15} {'Traditional':<15} {'Difference'}")
    print("-" * 70)
    print(f"{'Mean sMAPE':<30} {gemini_mean:>14.2f}% {traditional_mean:>14.2f}% {gemini_mean - traditional_mean:>+12.2f}%")
    print(f"{'Median sMAPE':<30} {gemini_median:>14.2f}% {traditional_median:>14.2f}% {gemini_median - traditional_median:>+12.2f}%")
    print(f"{'Min sMAPE':<30} {comparison['avg_smape_gemini'].min():>14.2f}% {comparison['avg_smape_traditional'].min():>14.2f}%")
    print(f"{'Max sMAPE':<30} {comparison['avg_smape_gemini'].max():>14.2f}% {comparison['avg_smape_traditional'].max():>14.2f}%")
    print(f"{'Std Dev':<30} {comparison['avg_smape_gemini'].std():>14.2f}% {comparison['avg_smape_traditional'].std():>14.2f}%")

    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 70)
    print(f"Gemini Wins:      {gemini_wins:3d} ({gemini_wins/len(comparison)*100:5.1f}%)")
    print(f"Traditional Wins: {traditional_wins:3d} ({traditional_wins/len(comparison)*100:5.1f}%)")
    print(f"Ties:             {ties:3d} ({ties/len(comparison)*100:5.1f}%)")

    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 70)
    print(f"Paired t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'} (α=0.05)")

    print(f"\nWilcoxon signed-rank test:")
    print(f"  statistic: {wilcoxon_stat:.4f}")
    print(f"  p-value: {wilcoxon_p:.4f}")
    print(f"  Significant: {'YES' if wilcoxon_p < 0.05 else 'NO'} (α=0.05)")

    # Income Statement comparison
    print("\n" + "=" * 70)
    print("INCOME STATEMENT PERFORMANCE")
    print("=" * 70)
    is_gemini_mean = comparison['is_avg_smape_gemini'].mean()
    is_traditional_mean = comparison['is_avg_smape_traditional'].mean()
    print(f"Gemini Mean sMAPE:      {is_gemini_mean:6.2f}%")
    print(f"Traditional Mean sMAPE: {is_traditional_mean:6.2f}%")
    print(f"Difference:             {is_gemini_mean - is_traditional_mean:+6.2f}%")

    # Balance Sheet comparison
    print("\n" + "=" * 70)
    print("BALANCE SHEET PERFORMANCE")
    print("=" * 70)
    bs_gemini_mean = comparison['bs_avg_smape_gemini'].mean()
    bs_traditional_mean = comparison['bs_avg_smape_traditional'].mean()
    print(f"Gemini Mean sMAPE:      {bs_gemini_mean:6.2f}%")
    print(f"Traditional Mean sMAPE: {bs_traditional_mean:6.2f}%")
    print(f"Difference:             {bs_gemini_mean - bs_traditional_mean:+6.2f}%")

    # Top improvers and decliners
    comparison['improvement'] = comparison['avg_smape_traditional'] - comparison['avg_smape_gemini']

    print("\n" + "=" * 70)
    print("TOP 5 GEMINI IMPROVEMENTS (vs Traditional)")
    print("=" * 70)
    top_improvements = comparison.nlargest(5, 'improvement')[['ticker', 'avg_smape_gemini', 'avg_smape_traditional', 'improvement']]
    for idx, row in top_improvements.iterrows():
        print(f"{row['ticker']:6s}: {row['avg_smape_traditional']:6.2f}% → {row['avg_smape_gemini']:6.2f}% (improved by {row['improvement']:+6.2f}%)")

    print("\n" + "=" * 70)
    print("TOP 5 GEMINI REGRESSIONS (vs Traditional)")
    print("=" * 70)
    top_regressions = comparison.nsmallest(5, 'improvement')[['ticker', 'avg_smape_gemini', 'avg_smape_traditional', 'improvement']]
    for idx, row in top_regressions.iterrows():
        print(f"{row['ticker']:6s}: {row['avg_smape_traditional']:6.2f}% → {row['avg_smape_gemini']:6.2f}% (regressed by {-row['improvement']:+6.2f}%)")

    # Save detailed comparison
    comparison_export = comparison[[
        'ticker', 'avg_smape_gemini', 'avg_smape_traditional', 'improvement',
        'is_avg_smape_gemini', 'is_avg_smape_traditional',
        'bs_avg_smape_gemini', 'bs_avg_smape_traditional'
    ]].copy()

    output_file = config.RESULTS_DIR / "model_comparison.csv"
    comparison_export.to_csv(output_file, index=False)

    print(f"\nDetailed comparison saved to: {output_file}")
    print("=" * 70 + "\n")

    # Generate summary report
    summary_report = f"""
GEMINI VS TRADITIONAL MODEL COMPARISON REPORT
{'=' * 70}

Dataset: {len(comparison)} S&P 500 companies with both forecasts
Forecast Year: 2024
Gemini Model: gemini-2.5-flash
Traditional Model: Policy-based (Vélez-Pareja)

OVERALL RESULTS:
  Gemini 2.5 Flash Mean sMAPE:    {gemini_mean:.2f}%
  Traditional Model Mean sMAPE:   {traditional_mean:.2f}%
  Difference:                     {gemini_mean - traditional_mean:+.2f}%

  Gemini 2.5 Flash Median sMAPE:  {gemini_median:.2f}%
  Traditional Model Median sMAPE: {traditional_median:.2f}%
  Difference:                     {gemini_median - traditional_median:+.2f}%

HEAD-TO-HEAD:
  Gemini Wins:      {gemini_wins} ({gemini_wins/len(comparison)*100:.1f}%)
  Traditional Wins: {traditional_wins} ({traditional_wins/len(comparison)*100:.1f}%)

STATISTICAL SIGNIFICANCE:
  Paired t-test p-value: {p_value:.4f}
  Result: {'Statistically significant difference' if p_value < 0.05 else 'No statistically significant difference'} (α=0.05)

INTERPRETATION:
  {'The traditional policy-based model outperforms Gemini 2.5 Flash on average.' if traditional_mean < gemini_mean else 'Gemini 2.5 Flash outperforms the traditional model on average.'}
  {'However, the difference is NOT statistically significant.' if p_value >= 0.05 else 'The difference is statistically significant.'}

INCOME STATEMENT:
  Gemini Mean sMAPE:      {is_gemini_mean:.2f}%
  Traditional Mean sMAPE: {is_traditional_mean:.2f}%

BALANCE SHEET:
  Gemini Mean sMAPE:      {bs_gemini_mean:.2f}%
  Traditional Mean sMAPE: {bs_traditional_mean:.2f}%

{'=' * 70}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    summary_file = config.RESULTS_DIR / "comparison_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(summary_report)

    print(f"Summary report saved to: {summary_file}\n")

    return comparison


if __name__ == "__main__":
    comparison_df = compare_models()

    print("✅ Comparison complete!")
    print(f"   Compared {len(comparison_df)} companies")