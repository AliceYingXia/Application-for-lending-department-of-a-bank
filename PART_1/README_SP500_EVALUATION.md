# S&P 500 Forecast Evaluation Guide

## Overview

The `evaluate_sp500.py` script evaluates the financial forecast model on S&P 500 stocks for a specified year (default: 2025).

## Features

- ✓ Automatically fetches S&P 500 ticker list from Wikipedia
- ✓ Runs forecasts for all S&P 500 stocks (or a sample)
- ✓ Evaluates forecast accuracy using sMAPE metrics (if actual data is available)
- ✓ Saves detailed results to CSV files
- ✓ Generates comprehensive summary reports
- ✓ Includes progress checkpoints and error handling
- ✓ Provides detailed statistics and rankings

## Quick Start

### 1. Test with Sample (Recommended First)

Test with a small sample of 10 stocks:

```bash
python3 evaluate_sp500.py --sample 10 --output test
```

### 2. Run on All S&P 500 Stocks

```bash
python3 evaluate_sp500.py --year 2025 --output sp500_2025
```

### 3. Custom Year

Forecast for a different year:

```bash
python3 evaluate_sp500.py --year 2024 --sample 50 --output sp500_2024
```

## Command-Line Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--year` | Forecast year | 2025 | `--year 2024` |
| `--sample` | Sample size (None = all) | None | `--sample 50` |
| `--output` | Output file prefix | sp500 | `--output test` |

## Output Files

The script generates three types of files:

### 1. Results CSV (`{prefix}_results_{timestamp}.csv`)

Contains detailed results for each stock:

| Column | Description |
|--------|-------------|
| ticker | Stock ticker symbol |
| status | Processing status |
| forecast_completed | Whether forecast succeeded |
| evaluation_completed | Whether evaluation succeeded |
| forecast_revenue | Forecasted revenue |
| forecast_net_income | Forecasted net income |
| avg_smape | Average sMAPE across all metrics |
| is_avg_smape | Income statement average sMAPE |
| bs_avg_smape | Balance sheet average sMAPE |
| actual_revenue | Actual revenue (if available) |
| error | Error message (if failed) |

### 2. Summary Report (`{prefix}_summary_{timestamp}.txt`)

Contains:
- Processing statistics
- Forecast accuracy metrics
- sMAPE distribution
- Top 20 most accurate forecasts
- Bottom 20 least accurate forecasts
- List of failed stocks

### 3. Checkpoint File (`{prefix}_results_temp.csv`)

Automatically saved every 10 stocks for recovery if interrupted.

## Example Output

```
================================================================================
S&P 500 FORECAST EVALUATION - YEAR 2025
Started: 2025-11-20 19:00:00
================================================================================
Fetching S&P 500 ticker list...
Found 503 S&P 500 companies

[1/503] Processing AAPL...
Using last available year (2024) for policy parameters
✓ Forecast completed (2025 actual data not yet available)

[2/503] Processing MSFT...
Using last available year (2024) for policy parameters
✓ Forecast completed (2025 actual data not yet available)

...

================================================================================
SUMMARY STATISTICS
================================================================================

Processing Summary:
  Total stocks attempted: 503
  Forecasts completed: 487 (96.8%)
  Evaluations completed: 0 (0.0%)
  Failed: 16 (3.2%)

Note: 2025 actual data not yet available for evaluation
Forecasts were successfully generated for 487 stocks
```

## Use Cases

### 1. **Model Validation**
Run on a historical year (e.g., 2023) where actual data is available:
```bash
python3 evaluate_sp500.py --year 2023 --output validation_2023
```

### 2. **Production Forecasts**
Generate forecasts for the upcoming year:
```bash
python3 evaluate_sp500.py --year 2025 --output production_2025
```

### 3. **Quick Testing**
Test changes with a small sample:
```bash
python3 evaluate_sp500.py --sample 20 --output quick_test
```

## Error Handling

The script handles common issues:

- **Missing data**: Stocks with insufficient data are skipped
- **Network errors**: Individual failures don't stop the entire run
- **Yahoo Finance limits**: Built-in delays to avoid rate limiting
- **Checkpoints**: Results saved every 10 stocks for recovery

## Performance

- **Single stock**: ~2-5 seconds
- **50 stocks**: ~3-5 minutes
- **All 500+ stocks**: ~40-60 minutes

The script includes 0.5-second delays between stocks to avoid overwhelming Yahoo Finance API.

## Tips

1. **Start small**: Always test with `--sample 10` first
2. **Check logs**: Monitor console output for errors
3. **Use checkpoints**: If interrupted, you can resume by checking the temp file
4. **Validate first**: Run on a historical year to verify accuracy before production use
5. **Monitor progress**: The script prints progress every 10 stocks

## Troubleshooting

### Issue: "Missing financial statements from Yahoo Finance"

**Solution**: This stock doesn't have data on Yahoo Finance. The script will skip it and continue.

### Issue: Script is slow

**Solution**:
- Use `--sample` to test on fewer stocks
- The 0.5s delay is intentional to avoid rate limiting
- Run overnight for full S&P 500 evaluation

### Issue: Checkpoint file exists from previous run

**Solution**: Delete `*_temp.csv` files before starting a new run, or they will be overwritten.

## Advanced Usage

### Programmatic Use

```python
from evaluate_sp500 import evaluate_sp500_forecasts

# Run on sample
results = evaluate_sp500_forecasts(
    forecast_year=2025,
    sample_size=50,
    output_prefix='my_test'
)

# Access results
print(f"Success rate: {results['forecast_completed'].mean():.1%}")
```

### Custom Ticker List

Modify the `get_sp500_tickers()` function to use your own list:

```python
def get_sp500_tickers():
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # Your custom list
```