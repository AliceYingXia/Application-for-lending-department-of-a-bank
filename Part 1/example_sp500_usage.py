"""
Example usage of the S&P 500 evaluation script

This demonstrates different ways to use the evaluation functionality
"""

from evaluate_sp500 import (
    evaluate_sp500_forecasts,
    evaluate_single_stock,
    get_sp500_tickers
)
import pandas as pd


def example_1_single_stock():
    """Example 1: Evaluate a single stock"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Stock Evaluation")
    print("="*80)

    result = evaluate_single_stock('AAPL', forecast_year=2025)

    print("\nResult:")
    for key, value in result.items():
        print(f"  {key}: {value}")


def example_2_small_sample():
    """Example 2: Evaluate a small sample"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Small Sample (10 stocks)")
    print("="*80)

    results = evaluate_sp500_forecasts(
        forecast_year=2025,
        sample_size=10,
        output_prefix='example_sample'
    )

    print("\nResults summary:")
    print(results[['ticker', 'status', 'forecast_completed']].to_string(index=False))


def example_3_custom_ticker_list():
    """Example 3: Use a custom ticker list"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Ticker List (FAANG stocks)")
    print("="*80)

    # Custom list of FAANG stocks
    custom_tickers = ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL']

    results = []
    for ticker in custom_tickers:
        print(f"\nProcessing {ticker}...")
        result = evaluate_single_stock(ticker, forecast_year=2025)
        results.append(result)

    df = pd.DataFrame(results)

    print("\nForecast Summary:")
    print(df[['ticker', 'forecast_revenue', 'forecast_net_income']].to_string(index=False))


def example_4_analyze_results():
    """Example 4: Load and analyze previous results"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Analyze Previous Results")
    print("="*80)

    # This assumes you've already run the evaluation and have a results file
    # Replace with your actual filename
    try:
        # Try to load a recent results file
        import glob
        files = glob.glob('sp500_results_*.csv')
        if files:
            latest_file = max(files)
            df = pd.read_csv(latest_file)

            print(f"\nLoaded results from: {latest_file}")
            print(f"Total stocks: {len(df)}")
            print(f"Successful forecasts: {df['forecast_completed'].sum()}")

            if df['evaluation_completed'].sum() > 0:
                evaluated = df[df['evaluation_completed'] == True]
                print(f"Evaluated stocks: {len(evaluated)}")
                print(f"Average sMAPE: {evaluated['avg_smape'].mean():.2f}%")

                # Show top 5 best forecasts
                print("\nTop 5 Most Accurate Forecasts:")
                top5 = evaluated.nsmallest(5, 'avg_smape')[['ticker', 'avg_smape']]
                print(top5.to_string(index=False))

        else:
            print("\nNo previous results found. Run the evaluation first:")
            print("  python3 evaluate_sp500.py --sample 10")

    except Exception as e:
        print(f"\nCould not load previous results: {e}")
        print("Run the evaluation first:")
        print("  python3 evaluate_sp500.py --sample 10")


def example_5_sector_analysis():
    """Example 5: Analyze by sector (requires sector data)"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Sector-Based Analysis")
    print("="*80)

    # Get S&P 500 with sector information
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_info = tables[0]

        # Sample a few stocks from different sectors
        sectors_to_test = ['Information Technology', 'Health Care', 'Financials']
        sample_size_per_sector = 2

        results = []
        for sector in sectors_to_test:
            print(f"\nProcessing {sector} sector...")
            sector_stocks = sp500_info[sp500_info['GICS Sector'] == sector]['Symbol'].head(sample_size_per_sector)

            for ticker in sector_stocks:
                ticker = ticker.replace('.', '-')
                result = evaluate_single_stock(ticker, forecast_year=2025)
                result['sector'] = sector
                results.append(result)

        df = pd.DataFrame(results)

        print("\nResults by Sector:")
        print(df[['ticker', 'sector', 'status', 'forecast_completed']].to_string(index=False))

    except Exception as e:
        print(f"Error in sector analysis: {e}")


def main():
    """Run all examples"""
    print("="*80)
    print("S&P 500 EVALUATION - USAGE EXAMPLES")
    print("="*80)
    print("\nThese examples demonstrate different ways to use the evaluation script.")
    print("Choose which examples to run:\n")

    examples = {
        '1': ('Single stock evaluation', example_1_single_stock),
        '2': ('Small sample (10 stocks)', example_2_small_sample),
        '3': ('Custom ticker list (FAANG)', example_3_custom_ticker_list),
        '4': ('Analyze previous results', example_4_analyze_results),
        '5': ('Sector-based analysis', example_5_sector_analysis),
    }

    for key, (description, _) in examples.items():
        print(f"  {key}. {description}")
    print(f"  0. Run all examples")
    print(f"  q. Quit")

    choice = input("\nEnter your choice (or press Enter to run example 1): ").strip() or '1'

    if choice == 'q':
        print("Exiting...")
        return

    if choice == '0':
        # Run all examples
        for _, func in examples.values():
            try:
                func()
            except Exception as e:
                print(f"Error: {e}")
            input("\nPress Enter to continue to next example...")
    elif choice in examples:
        # Run selected example
        _, func = examples[choice]
        func()
    else:
        print(f"Invalid choice: {choice}")

    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)


if __name__ == "__main__":
    # Quick non-interactive demo - run example 1
    print("Running Example 1: Single Stock Evaluation")
    print("(Edit this file to run other examples or run interactively)")
    example_1_single_stock()

    # To run interactive menu, uncomment:
    # main()