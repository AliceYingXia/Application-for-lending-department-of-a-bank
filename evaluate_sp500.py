"""
Evaluate forecast model on S&P 500 stocks for 2025

This script:
1. Fetches S&P 500 ticker list
2. Runs forecasts for each stock
3. Evaluates forecast accuracy (if 2025 data is available)
4. Saves results to CSV files
5. Generates summary statistics
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time
from forecast_model import (
    forecast_company_year,
    fetch_yahoo_statements,
    build_historical_dataset,
    evaluate_forecast
)

def get_sp500_tickers():
# S&P 500 tickers as a Python list of strings
    sp500_tickers = [
        "AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "BRK-B", "WMT", "LLY",
        "JPM", "V", "MA", "ORCL", "XOM", "UNH", "COST", "PG", "HD", "NFLX", "JNJ", "BAC", "CRM",
        "ABBV", "KO", "TMUS", "CVX", "MRK", "WFC", "CSCO", "ACN", "NOW", "AXP", "MCD", "PEP", "BX",
        "IBM", "DIS", "LIN", "TMO", "MS", "ABT", "ADBE", "AMD", "PM", "ISRG", "PLTR", "GE", "INTU",
        "GS", "CAT", "TXN", "QCOM", "VZ", "BKNG", "DHR", "T", "BLK", "RTX", "SPGI", "PFE", "HON",
        "NEE", "CMCSA", "ANET", "AMGN", "PGR", "LOW", "SYK", "UNP", "TJX", "KKR", "SCHW", "ETN",
        "AMAT", "BA", "BSX", "C", "UBER", "COP", "PANW", "ADP", "DE", "FI", "BMY", "LMT", "GILD",
        "NKE", "CB", "UPS", "ADI", "MMC", "MDT", "VRTX", "MU", "SBUX", "PLD", "GEV", "LRCX", "MO",
        "SO", "EQIX", "CRWD", "PYPL", "SHW", "ICE", "CME", "AMT", "APH", "ELV", "TT", "MCO", "CMG",
        "INTC", "KLAC", "ABNB", "DUK", "PH", "CDNS", "WM", "DELL", "MDLZ", "MAR", "MSI", "WELL", "AON",
        "REGN", "CI", "HCA", "PNC", "ITW", "SNPS", "CTAS", "CL", "USB", "FTNT", "ZTS", "MCK", "GD",
        "TDG", "CEG", "AJG", "EMR", "MMM", "ORLY", "NOC", "COF", "ECL", "EOG", "FDX", "BDX", "APD",
        "WMB", "SPG", "ADSK", "RCL", "RSG", "CARR", "CSX", "HLT", "DLR", "TGT", "KMI", "OKE", "TFC",
        "AFL", "GM", "BK", "ROP", "MET", "CPRT", "FCX", "CVS", "PCAR", "SRE", "AZO", "TRV", "NXPI",
        "JCI", "GWW", "NSC", "PSA", "SLB", "AMP", "ALL", "FICO", "MNST", "PAYX", "CHTR", "AEP", "ROST",
        "PWR", "CMI", "AXON", "VST", "URI", "MSCI", "LULU", "O", "PSX", "AIG", "FANG", "D", "HWM",
        "DHI", "KR", "NDAQ", "OXY", "EW", "COR", "KDP", "FIS", "KMB", "NEM", "DFS", "PCG", "TEL",
        "MPC", "FAST", "AME", "PEG", "PRU", "KVUE", "STZ", "GLW", "LHX", "GRMN", "BKR", "CBRE", "CTVA",
        "HES", "CCI", "DAL", "CTSH", "F", "VRSK", "EA", "ODFL", "XEL", "TRGP", "A", "IT", "LVS", "SYY",
        "VLO", "OTIS", "LEN", "EXC", "IR", "YUM", "KHC", "GEHC", "IQV", "GIS", "CCL", "RMD", "VMC",
        "HSY", "ACGL", "IDXX", "WAB", "ROK", "MLM", "EXR", "DD", "ETR", "DECK", "EFX", "UAL", "WTW",
        "TTWO", "HIG", "RJF", "AVB", "MTB", "DXCM", "ED", "EBAY", "HPQ", "IRM", "EIX", "LYV", "VICI",
        "CNC", "WEC", "MCHP", "HUM", "ANSS", "BRO", "CSGP", "MPWR", "GDDY", "TSCO", "STT", "CAH",
        "GPN", "FITB", "XYL", "HPE", "KEYS", "DOW", "EQR", "ON", "PPG", "K", "SW", "NUE", "EL", "BR",
        "WBD", "TPL", "CHD", "MTD", "DOV", "TYL", "FTV", "TROW", "VLTO", "EQT", "SYF", "NVR", "DTE",
        "VTR", "AWK", "ADM", "NTAP", "WST", "CPAY", "PPL", "LYB", "AEE", "EXPE", "HBAN", "CDW", "FE",
        "HUBB", "HAL", "ROL", "PHM", "CINF", "PTC", "WRB", "DRI", "FOXA", "FOX", "IFF", "SBAC", "WAT",
        "ERIE", "TDY", "ATO", "RF", "BIIB", "ZBH", "CNP", "MKC", "ES", "WDC", "TSN", "TER", "STE",
        "PKG", "CLX", "NTRS", "ZBRA", "DVN", "CBOE", "WY", "LUV", "ULTA", "CMS", "INVH", "FSLR",
        "BF-B", "LDOS", "CFG", "LH", "VRSN", "IP", "ESS", "PODD", "COO", "SMCI", "STX", "MAA", "FDS",
        "NRG", "BBY", "SNA", "L", "PFG", "STLD", "TRMB", "OMC", "CTRA", "HRL", "ARE", "BLDR", "JBHT",
        "GEN", "DGX", "KEY", "NI", "MOH", "PNR", "J", "DG", "BALL", "NWS", "NWSA", "UDR", "HOLX",
        "JBL", "GPC", "IEX", "MAS", "KIM", "ALGN", "DLTR", "EXPD", "EG", "MRNA", "LNT", "AVY", "BAX",
        "TPR", "VTRS", "CF", "FFIV", "DPZ", "AKAM", "RL", "TXT", "SWKS", "EVRG", "EPAM", "DOC", "APTV",
        "RVTY", "AMCR", "REG", "POOL", "INCY", "BXP", "KMX", "CAG", "HST", "JKHY", "SWK", "DVA", "CPB",
        "CHRW", "JNPR", "CPT", "TAP", "NDSN", "PAYC", "UHS", "NCLH", "DAY", "SJM", "TECH", "SOLV",
        "ALLE", "BG", "AIZ", "IPG", "BEN", "EMN", "ALB", "MGM", "AOS", "WYNN", "PNW", "ENPH", "LKQ",
        "FRT", "CRL", "GNRC", "AES", "GL", "LW", "HSIC", "MKTX", "MTCH", "TFX", "WBA", "HAS", "IVZ",
        "APA", "MOS", "PARA", "MHK", "CE", "HII", "CZR", "BWA", "QRVO", "FMC", "AMTM"
    ]
    return sp500_tickers


def evaluate_single_stock(ticker, forecast_year=2025):
    """
    Evaluate forecast for a single stock

    Args:
        ticker: Stock ticker symbol
        forecast_year: Year to forecast

    Returns:
        Dict with forecast results and evaluation metrics
    """
    result = {
        'ticker': ticker,
        'status': 'failed',
        'error': None,
        'forecast_completed': False,
        'evaluation_completed': False,
    }

    try:
        # Make forecast
        print(f"\n{'='*80}")
        print(f"Processing {ticker}...")
        print(f"{'='*80}")

        forecast_result = forecast_company_year(ticker, forecast_year)
        result['forecast_completed'] = True
        result['status'] = 'forecast_completed'

        # Extract forecast values
        is_forecast = forecast_result['income_statement'].loc[forecast_year]
        bs_forecast = forecast_result['balance_sheet'].loc[forecast_year]

        result['forecast_revenue'] = is_forecast['Revenue']
        result['forecast_cogs'] = is_forecast['COGS']
        result['forecast_sga'] = is_forecast['SGA']
        result['forecast_net_income'] = is_forecast['NetIncome']
        result['forecast_ebit'] = is_forecast['EBIT']
        result['forecast_cash'] = bs_forecast['Cash']
        result['forecast_equity'] = bs_forecast['Equity']
        result['forecast_total_assets'] = bs_forecast['Assets']

        # Try to evaluate if 2025 data is available
        try:
            fin_all, bs_all, cf_all = fetch_yahoo_statements(ticker, end_year=None)
            hist_all = build_historical_dataset(fin_all, bs_all, cf_all, debug=False)

            if forecast_year in hist_all.index:
                eval_df = evaluate_forecast(forecast_result, hist_all, forecast_year)
                result['evaluation_completed'] = True
                result['status'] = 'evaluated'

                # Extract evaluation metrics
                result['avg_smape'] = eval_df['sMAPE_%'].mean()
                result['median_smape'] = eval_df['sMAPE_%'].median()

                # Income statement sMAPE
                is_eval = eval_df[eval_df['Statement'] == 'Income Statement']
                result['is_avg_smape'] = is_eval['sMAPE_%'].mean()
                result['is_median_smape'] = is_eval['sMAPE_%'].median()

                # Balance sheet sMAPE
                bs_eval = eval_df[eval_df['Statement'] == 'Balance Sheet']
                result['bs_avg_smape'] = bs_eval['sMAPE_%'].mean()
                result['bs_median_smape'] = bs_eval['sMAPE_%'].median()

                # Get actual values
                actual = hist_all.loc[forecast_year]
                result['actual_revenue'] = actual['S']
                result['actual_net_income'] = actual['NI']
                result['actual_equity'] = actual['Equity']

                print(f"✓ Evaluation completed: Avg sMAPE = {result['avg_smape']:.2f}%")
            else:
                print(f"✓ Forecast completed (2025 actual data not yet available)")

        except Exception as eval_error:
            print(f"✓ Forecast completed (evaluation skipped: {str(eval_error)})")

    except Exception as e:
        result['error'] = str(e)
        print(f"✗ Failed: {str(e)}")

    return result


def evaluate_sp500_forecasts(forecast_year=2025, sample_size=None, output_prefix='sp500'):
    """
    Evaluate forecasts for S&P 500 stocks

    Args:
        forecast_year: Year to forecast (default: 2025)
        sample_size: Number of stocks to test (None = all stocks)
        output_prefix: Prefix for output files

    Returns:
        DataFrame with results
    """
    start_time = datetime.now()
    print("="*80)
    print(f"S&P 500 FORECAST EVALUATION - YEAR {forecast_year}")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Get ticker list
    tickers = get_sp500_tickers()

    # Sample if requested
    if sample_size is not None and sample_size < len(tickers):
        print(f"\nUsing random sample of {sample_size} stocks")
        tickers = np.random.choice(tickers, size=sample_size, replace=False).tolist()

    # Initialize results
    results = []
    total = len(tickers)

    # Process each stock
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{total}] Processing {ticker}...")

        result = evaluate_single_stock(ticker, forecast_year)
        results.append(result)

        # Save intermediate results every 10 stocks
        if i % 10 == 0:
            df_temp = pd.DataFrame(results)
            df_temp.to_csv(f'{output_prefix}_results_temp.csv', index=False)
            print(f"\n>>> Checkpoint: Saved {i} results to {output_prefix}_results_temp.csv")

        # Small delay to avoid overwhelming Yahoo Finance API
        time.sleep(0.5)

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Save full results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'{output_prefix}_results_{timestamp}.csv'
    df_results.to_csv(results_file, index=False)
    print(f"\n✓ Full results saved to: {results_file}")

    # Generate summary statistics
    print_summary_statistics(df_results, forecast_year)

    # Save summary
    summary_file = f'{output_prefix}_summary_{timestamp}.txt'
    save_summary_report(df_results, forecast_year, summary_file, start_time)
    print(f"✓ Summary report saved to: {summary_file}")

    return df_results


def print_summary_statistics(df_results, forecast_year):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    total = len(df_results)
    forecasts_completed = df_results['forecast_completed'].sum()
    evaluations_completed = df_results['evaluation_completed'].sum()
    failed = total - forecasts_completed

    print(f"\nProcessing Summary:")
    print(f"  Total stocks attempted: {total}")
    print(f"  Forecasts completed: {forecasts_completed} ({forecasts_completed/total*100:.1f}%)")
    print(f"  Evaluations completed: {evaluations_completed} ({evaluations_completed/total*100:.1f}%)")
    print(f"  Failed: {failed} ({failed/total*100:.1f}%)")

    if evaluations_completed > 0:
        evaluated = df_results[df_results['evaluation_completed'] == True]

        print(f"\nForecast Accuracy Metrics (n={evaluations_completed}):")
        print(f"  Overall Average sMAPE: {evaluated['avg_smape'].mean():.2f}%")
        print(f"  Overall Median sMAPE: {evaluated['avg_smape'].median():.2f}%")
        print(f"  Income Statement Avg sMAPE: {evaluated['is_avg_smape'].mean():.2f}%")
        print(f"  Balance Sheet Avg sMAPE: {evaluated['bs_avg_smape'].mean():.2f}%")

        print(f"\nsMAPE Distribution:")
        print(f"  Best (lowest): {evaluated['avg_smape'].min():.2f}% ({evaluated.loc[evaluated['avg_smape'].idxmin(), 'ticker']})")
        print(f"  Worst (highest): {evaluated['avg_smape'].max():.2f}% ({evaluated.loc[evaluated['avg_smape'].idxmax(), 'ticker']})")
        print(f"  25th percentile: {evaluated['avg_smape'].quantile(0.25):.2f}%")
        print(f"  75th percentile: {evaluated['avg_smape'].quantile(0.75):.2f}%")

        # Top 10 most accurate forecasts
        print(f"\nTop 10 Most Accurate Forecasts:")
        top10 = evaluated.nsmallest(10, 'avg_smape')[['ticker', 'avg_smape']]
        for idx, (_, row) in enumerate(top10.iterrows(), 1):
            print(f"  {idx}. {row['ticker']}: {row['avg_smape']:.2f}% sMAPE")

    if forecasts_completed > 0 and evaluations_completed == 0:
        print(f"\nNote: {forecast_year} actual data not yet available for evaluation")
        print(f"Forecasts were successfully generated for {forecasts_completed} stocks")

    if failed > 0:
        print(f"\nFailed Stocks ({failed}):")
        failed_stocks = df_results[df_results['forecast_completed'] == False]
        for _, row in failed_stocks.head(10).iterrows():
            print(f"  {row['ticker']}: {row['error']}")
        if failed > 10:
            print(f"  ... and {failed - 10} more")


def save_summary_report(df_results, forecast_year, filename, start_time):
    """Save detailed summary report to file"""
    end_time = datetime.now()
    duration = end_time - start_time

    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"S&P 500 FORECAST EVALUATION - YEAR {forecast_year}\n")
        f.write("="*80 + "\n\n")

        f.write(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {duration}\n\n")

        total = len(df_results)
        forecasts_completed = df_results['forecast_completed'].sum()
        evaluations_completed = df_results['evaluation_completed'].sum()
        failed = total - forecasts_completed

        f.write("PROCESSING SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Total stocks attempted: {total}\n")
        f.write(f"Forecasts completed: {forecasts_completed} ({forecasts_completed/total*100:.1f}%)\n")
        f.write(f"Evaluations completed: {evaluations_completed} ({evaluations_completed/total*100:.1f}%)\n")
        f.write(f"Failed: {failed} ({failed/total*100:.1f}%)\n\n")

        if evaluations_completed > 0:
            evaluated = df_results[df_results['evaluation_completed'] == True]

            f.write("FORECAST ACCURACY METRICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Number of evaluated stocks: {evaluations_completed}\n\n")
            f.write(f"Overall Average sMAPE: {evaluated['avg_smape'].mean():.2f}%\n")
            f.write(f"Overall Median sMAPE: {evaluated['avg_smape'].median():.2f}%\n")
            f.write(f"Income Statement Avg sMAPE: {evaluated['is_avg_smape'].mean():.2f}%\n")
            f.write(f"Balance Sheet Avg sMAPE: {evaluated['bs_avg_smape'].mean():.2f}%\n\n")

            f.write("sMAPE DISTRIBUTION\n")
            f.write("-"*80 + "\n")
            f.write(f"Minimum: {evaluated['avg_smape'].min():.2f}% ({evaluated.loc[evaluated['avg_smape'].idxmin(), 'ticker']})\n")
            f.write(f"25th percentile: {evaluated['avg_smape'].quantile(0.25):.2f}%\n")
            f.write(f"Median: {evaluated['avg_smape'].median():.2f}%\n")
            f.write(f"75th percentile: {evaluated['avg_smape'].quantile(0.75):.2f}%\n")
            f.write(f"Maximum: {evaluated['avg_smape'].max():.2f}% ({evaluated.loc[evaluated['avg_smape'].idxmax(), 'ticker']})\n\n")

            f.write("TOP 20 MOST ACCURATE FORECASTS\n")
            f.write("-"*80 + "\n")
            top20 = evaluated.nsmallest(20, 'avg_smape')[['ticker', 'avg_smape', 'is_avg_smape', 'bs_avg_smape']]
            for idx, (_, row) in enumerate(top20.iterrows(), 1):
                f.write(f"{idx:2d}. {row['ticker']:6s} - Overall: {row['avg_smape']:6.2f}%, "
                       f"IS: {row['is_avg_smape']:6.2f}%, BS: {row['bs_avg_smape']:6.2f}%\n")

            f.write("\nBOTTOM 20 LEAST ACCURATE FORECASTS\n")
            f.write("-"*80 + "\n")
            bottom20 = evaluated.nlargest(20, 'avg_smape')[['ticker', 'avg_smape', 'is_avg_smape', 'bs_avg_smape']]
            for idx, (_, row) in enumerate(bottom20.iterrows(), 1):
                f.write(f"{idx:2d}. {row['ticker']:6s} - Overall: {row['avg_smape']:6.2f}%, "
                       f"IS: {row['is_avg_smape']:6.2f}%, BS: {row['bs_avg_smape']:6.2f}%\n")

        if failed > 0:
            f.write("\nFAILED STOCKS\n")
            f.write("-"*80 + "\n")
            failed_stocks = df_results[df_results['forecast_completed'] == False]
            for _, row in failed_stocks.iterrows():
                f.write(f"{row['ticker']}: {row['error']}\n")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate forecast model on S&P 500 stocks')
    parser.add_argument('--year', type=int, default=2025, help='Forecast year (default: 2025)')
    parser.add_argument('--sample', type=int, default=None, help='Sample size (default: all stocks)')
    parser.add_argument('--output', type=str, default='sp500', help='Output file prefix (default: sp500)')

    args = parser.parse_args()

    # Run evaluation
    results = evaluate_sp500_forecasts(
        forecast_year=args.year,
        sample_size=args.sample,
        output_prefix=args.output
    )

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved with prefix: {args.output}")
    print(f"Total stocks processed: {len(results)}")
    print(f"Successful forecasts: {results['forecast_completed'].sum()}")

    return results


if __name__ == "__main__":
    # Example: Run on sample of 50 stocks for testing
    # results = evaluate_sp500_forecasts(forecast_year=2025, sample_size=50, output_prefix='sp500_sample')

    # Run on all S&P 500 stocks
    results = main()