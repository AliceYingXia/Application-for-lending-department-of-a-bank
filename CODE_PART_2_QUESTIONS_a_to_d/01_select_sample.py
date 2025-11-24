#!/usr/bin/env python3
"""
Sample Selection for Gemini Forecasting Experiment

Selects 50 companies from PART_1 results stratified by performance:
- Top 10 most accurate (lowest sMAPE)
- Bottom 10 least accurate (highest sMAPE)
- Random 30 from middle performers

This ensures we test Gemini across the full performance spectrum.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import config

def select_sample(n_top=10, n_bottom=10, n_random=30):
    """
    Select stratified sample from PART_1 results.

    Args:
        n_top: Number of top performers to include
        n_bottom: Number of bottom performers to include
        n_random: Number of random middle performers to include

    Returns:
        DataFrame with selected companies
    """
    print("=" * 70)
    print("SAMPLE SELECTION FOR GEMINI FORECASTING EXPERIMENT")
    print("=" * 70)

    # Load PART_1 results
    print(f"\nLoading PART_1 results from: {config.PART_1_RESULTS}")
    df = pd.read_csv(config.PART_1_RESULTS)

    # Filter only successfully evaluated companies
    df_success = df[df['status'] == 'evaluated'].copy()
    print(f"Total successfully evaluated companies: {len(df_success)}")

    # Sort by average sMAPE
    df_sorted = df_success.sort_values('avg_smape').reset_index(drop=True)

    # Select top performers
    top_performers = df_sorted.head(n_top).copy()
    top_performers['performance_tier'] = 'top'
    print(f"\nTop {n_top} performers (lowest sMAPE):")
    for idx, row in top_performers.iterrows():
        print(f"  {row['ticker']:6s} - sMAPE: {row['avg_smape']:6.2f}%")

    # Select bottom performers
    bottom_performers = df_sorted.tail(n_bottom).copy()
    bottom_performers['performance_tier'] = 'bottom'
    print(f"\nBottom {n_top} performers (highest sMAPE):")
    for idx, row in bottom_performers.iterrows():
        print(f"  {row['ticker']:6s} - sMAPE: {row['avg_smape']:6.2f}%")

    # Select random middle performers
    # Exclude top and bottom to get middle tier
    middle_performers = df_sorted.iloc[n_top:-n_bottom]

    # Set random seed for reproducibility
    np.random.seed(42)
    random_indices = np.random.choice(middle_performers.index, size=n_random, replace=False)
    random_performers = middle_performers.loc[random_indices].copy()
    random_performers['performance_tier'] = 'random'

    print(f"\nRandom {n_random} middle performers:")
    for idx, row in random_performers.sort_values('avg_smape').iterrows():
        print(f"  {row['ticker']:6s} - sMAPE: {row['avg_smape']:6.2f}%")

    # Combine all selections
    selected = pd.concat([top_performers, random_performers, bottom_performers], ignore_index=True)

    # Sort by ticker for easier lookup
    selected = selected.sort_values('ticker').reset_index(drop=True)

    # Select relevant columns to save
    columns_to_save = [
        'ticker',
        'performance_tier',
        'avg_smape',
        'is_avg_smape',
        'bs_avg_smape',
        'forecast_revenue',
        'forecast_net_income',
        'forecast_equity',
        'actual_revenue',
        'actual_net_income',
        'actual_equity'
    ]

    selected_export = selected[columns_to_save].copy()

    # Save to CSV
    output_file = config.DATA_DIR / "test_sample.csv"
    selected_export.to_csv(output_file, index=False)

    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total companies selected: {len(selected)}")
    print(f"  - Top performers: {n_top}")
    print(f"  - Random performers: {n_random}")
    print(f"  - Bottom performers: {n_bottom}")
    print(f"\nSample statistics:")
    print(f"  Mean sMAPE: {selected['avg_smape'].mean():.2f}%")
    print(f"  Median sMAPE: {selected['avg_smape'].median():.2f}%")
    print(f"  Min sMAPE: {selected['avg_smape'].min():.2f}%")
    print(f"  Max sMAPE: {selected['avg_smape'].max():.2f}%")
    print(f"\nOutput saved to: {output_file}")
    print("=" * 70 + "\n")

    return selected_export

if __name__ == "__main__":
    # Select sample with default parameters
    sample = select_sample(
        n_top=config.TEST_SAMPLE_SIZE // 5,      # 10 companies
        n_bottom=config.TEST_SAMPLE_SIZE // 5,   # 10 companies
        n_random=config.TEST_SAMPLE_SIZE * 3 // 5  # 30 companies
    )

    print(f"✅ Sample selection complete!")
    print(f"   Selected {len(sample)} companies for Gemini forecasting experiment")