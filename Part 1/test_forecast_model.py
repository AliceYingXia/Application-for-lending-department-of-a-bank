"""
Test suite for forecast_model.py

This test file validates all components of the financial forecasting model
and outputs detailed results to test_results.txt
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime
from forecast_model import (
    calculate_smape,
    fetch_yahoo_statements,
    build_historical_dataset,
    estimate_policies,
    forecast_one_year,
    forecast_company_year,
    evaluate_forecast,
    _safe_div
)


class TestLogger:
    """Logger to write test results to both console and file"""
    def __init__(self, filename="test_results.txt"):
        self.filename = filename
        self.file = open(filename, 'w', encoding='utf-8')
        self.test_count = 0
        self.passed_count = 0
        self.failed_count = 0

    def log(self, message):
        """Write message to both console and file"""
        print(message)
        self.file.write(message + '\n')
        self.file.flush()

    def test_header(self, test_name):
        """Print a formatted test header"""
        self.test_count += 1
        header = f"\n{'='*80}\nTEST {self.test_count}: {test_name}\n{'='*80}"
        self.log(header)

    def test_result(self, passed, message=""):
        """Log test result"""
        if passed:
            self.passed_count += 1
            self.log(f"✓ PASSED: {message}")
        else:
            self.failed_count += 1
            self.log(f"✗ FAILED: {message}")

    def section(self, title):
        """Print a section header"""
        self.log(f"\n{'-'*80}\n{title}\n{'-'*80}")

    def close(self):
        """Close the log file and print summary"""
        summary = f"""
{'='*80}
TEST SUMMARY
{'='*80}
Total Tests: {self.test_count}
Passed: {self.passed_count}
Failed: {self.failed_count}
Success Rate: {(self.passed_count/self.test_count*100):.1f}%
{'='*80}
"""
        self.log(summary)
        self.file.close()


def test_utility_functions(logger):
    """Test utility functions"""
    logger.test_header("Utility Functions - calculate_smape")

    # Test 1: Perfect forecast
    smape = calculate_smape(100, 100)
    passed = abs(smape - 0.0) < 0.01
    logger.test_result(passed, f"Perfect forecast: sMAPE = {smape:.2f}% (expected 0%)")

    # Test 2: 50% overestimate
    smape = calculate_smape(100, 150)
    expected = 40.0  # |100-150| / ((100+150)/2) * 100 = 40%
    passed = abs(smape - expected) < 0.01
    logger.test_result(passed, f"50% overestimate: sMAPE = {smape:.2f}% (expected {expected}%)")

    # Test 3: 50% underestimate
    smape = calculate_smape(100, 50)
    expected = 66.67  # |100-50| / ((100+50)/2) * 100 = 66.67%
    passed = abs(smape - expected) < 0.01
    logger.test_result(passed, f"50% underestimate: sMAPE = {smape:.2f}% (expected {expected:.2f}%)")

    # Test 4: NaN handling
    smape = calculate_smape(np.nan, 100)
    passed = np.isnan(smape)
    logger.test_result(passed, f"NaN input handling: sMAPE = {smape} (expected NaN)")

    logger.test_header("Utility Functions - _safe_div")

    # Test 5: Normal division
    result = _safe_div(10, 2)
    passed = abs(result - 5.0) < 0.01
    logger.test_result(passed, f"Normal division: 10/2 = {result} (expected 5.0)")

    # Test 6: Division by zero
    result = _safe_div(10, 0)
    passed = np.isnan(result)
    logger.test_result(passed, f"Division by zero: 10/0 = {result} (expected NaN)")


def test_data_fetching(logger, ticker="AAPL"):
    """Test data fetching from Yahoo Finance"""
    logger.test_header(f"Data Fetching - Yahoo Finance for {ticker}")

    try:
        # Fetch data
        fin, bs, cf = fetch_yahoo_statements(ticker, end_year=None)

        # Test that we got data
        passed = not fin.empty and not bs.empty and not cf.empty
        logger.test_result(passed, f"Data fetched successfully")

        if passed:
            logger.section("Data Summary")
            logger.log(f"Years available: {sorted(fin.index.tolist())}")
            logger.log(f"Income statement columns: {len(fin.columns)}")
            logger.log(f"Balance sheet columns: {len(bs.columns)}")
            logger.log(f"Cash flow columns: {len(cf.columns)}")

        return fin, bs, cf

    except Exception as e:
        logger.test_result(False, f"Data fetch failed: {str(e)}")
        return None, None, None


def test_historical_dataset(logger, fin, bs, cf):
    """Test building historical dataset"""
    logger.test_header("Historical Dataset Building")

    if fin is None or bs is None or cf is None:
        logger.test_result(False, "Cannot build dataset - input data is None")
        return None

    try:
        hist = build_historical_dataset(fin, bs, cf, debug=False)

        # Test that dataset was built
        passed = hist is not None and len(hist) > 0
        logger.test_result(passed, f"Dataset built with {len(hist)} years")

        if passed:
            logger.section("Dataset Summary")
            logger.log(f"Years: {sorted(hist.index.tolist())}")
            logger.log(f"Columns: {list(hist.columns)}")

            # Check for missing data
            missing_pct = (hist.isna().sum() / len(hist) * 100).round(1)
            logger.log("\nMissing Data Percentage:")
            for col, pct in missing_pct.items():
                status = "⚠️" if pct > 50 else "✓"
                logger.log(f"  {status} {col}: {pct}%")

            # Test key columns exist
            required_cols = ['S', 'COGS', 'SGA', 'NI', 'Cash', 'Equity']
            all_exist = all(col in hist.columns for col in required_cols)
            logger.test_result(all_exist, f"All required columns present: {required_cols}")

        return hist

    except Exception as e:
        logger.test_result(False, f"Dataset building failed: {str(e)}")
        return None


def test_policy_estimation(logger, hist):
    """Test policy parameter estimation"""
    logger.test_header("Policy Parameter Estimation")

    if hist is None or len(hist) < 2:
        logger.test_result(False, "Cannot estimate policies - insufficient data")
        return None

    try:
        # Get the second-to-last year for testing
        test_year = sorted(hist.index)[-1]

        policies = estimate_policies(hist, up_to_year=test_year, lookback_years=3)

        # Test that policies were estimated
        passed = policies is not None
        logger.test_result(passed, "Policies estimated successfully")

        if passed:
            logger.section("Policy Parameters")
            logger.log(f"Revenue Growth (g_S): {policies.g_S:.4f} ({policies.g_S*100:.2f}%)")
            logger.log(f"COGS Margin (m_cogs): {policies.m_cogs:.4f} ({policies.m_cogs*100:.2f}%)")
            logger.log(f"SGA Margin (m_sga): {policies.m_sga:.4f} ({policies.m_sga*100:.2f}%)")
            logger.log(f"Depreciation Rate: {policies.dep_rate:.4f}")
            logger.log(f"Interest Rate on Debt: {policies.r_d:.4f}")
            logger.log(f"Tax Rate: {policies.tax_rate:.4f} ({policies.tax_rate*100:.2f}%)")
            logger.log(f"Payout Ratio: {policies.payout_ratio:.4f}")
            logger.log(f"DSO (days): {policies.dso:.2f}")
            logger.log(f"DIH (days): {policies.dih:.2f}")
            logger.log(f"DPO (days): {policies.dpo:.2f}")
            logger.log(f"Capex Ratio: {policies.capex_ratio:.4f}")
            logger.log(f"Min Cash Ratio: {policies.min_cash_ratio:.4f}")
            logger.log(f"OA Growth: {policies.g_OA:.4f}")
            logger.log(f"OL Growth: {policies.g_OL:.4f}")

            # Test that values are reasonable (not NaN or extreme)
            reasonable = (
                not np.isnan(policies.g_S) and abs(policies.g_S) < 1.0 and
                not np.isnan(policies.m_cogs) and 0 < policies.m_cogs < 1.0 and
                not np.isnan(policies.m_sga) and 0 < policies.m_sga < 1.0
            )
            logger.test_result(reasonable, "Policy parameters are reasonable")

        return policies

    except Exception as e:
        logger.test_result(False, f"Policy estimation failed: {str(e)}")
        return None


def test_one_year_forecast(logger, hist, policies):
    """Test one-year forecast"""
    logger.test_header("One-Year Forecast")

    if hist is None or policies is None:
        logger.test_result(False, "Cannot forecast - missing hist or policies")
        return None

    try:
        # Forecast the last year in our dataset
        forecast_year = sorted(hist.index)[-1]

        result = forecast_one_year(hist, policies, target_year=forecast_year)

        # Test that forecast was made
        passed = result is not None and 'income_statement' in result and 'balance_sheet' in result
        logger.test_result(passed, f"Forecast created for year {forecast_year}")

        if passed:
            logger.section(f"Income Statement Forecast - {forecast_year}")
            is_df = result['income_statement']
            logger.log(is_df.to_string())

            logger.section(f"Balance Sheet Forecast - {forecast_year}")
            bs_df = result['balance_sheet']
            logger.log(bs_df.to_string())

            # Test accounting identity
            assets_diff = bs_df.loc[forecast_year, 'Assets_minus_LiabEq']
            balance_ok = abs(assets_diff) < 1e-6
            logger.test_result(balance_ok,
                f"Balance sheet balances: Assets - Liab&Eq = {assets_diff:.2e}")

        return result

    except Exception as e:
        logger.test_result(False, f"One-year forecast failed: {str(e)}")
        return None


def test_full_forecast(logger, ticker="AAPL", forecast_year=2024):
    """Test full forecast workflow"""
    logger.test_header(f"Full Forecast Workflow - {ticker} {forecast_year}")

    try:
        result = forecast_company_year(ticker, forecast_year)

        passed = result is not None
        logger.test_result(passed, f"Full forecast completed for {ticker} {forecast_year}")

        if passed:
            logger.section(f"Forecast Results for {forecast_year}")
            logger.log("\nIncome Statement:")
            logger.log(result['income_statement'].round(2).to_string())
            logger.log("\nBalance Sheet:")
            logger.log(result['balance_sheet'].round(2).to_string())

        return result

    except Exception as e:
        logger.test_result(False, f"Full forecast failed: {str(e)}")
        return None


def test_forecast_evaluation(logger, ticker="AAPL", forecast_year=2024):
    """Test forecast evaluation with sMAPE"""
    logger.test_header(f"Forecast Evaluation - {ticker} {forecast_year}")

    try:
        # Make forecast
        forecast_result = forecast_company_year(ticker, forecast_year)

        # Get actual data
        fin_all, bs_all, cf_all = fetch_yahoo_statements(ticker, end_year=None)
        hist_all = build_historical_dataset(fin_all, bs_all, cf_all, debug=False)

        if forecast_year not in hist_all.index:
            logger.test_result(False, f"Actual data for {forecast_year} not available yet")
            return None

        # Evaluate
        eval_df = evaluate_forecast(forecast_result, hist_all, forecast_year)

        passed = eval_df is not None and len(eval_df) > 0
        logger.test_result(passed, "Evaluation completed successfully")

        if passed:
            logger.section("Detailed Evaluation Results")
            logger.log(eval_df.to_string(index=False))

            logger.section("Summary Statistics")
            logger.log(f"Average sMAPE: {eval_df['sMAPE_%'].mean():.2f}%")
            logger.log(f"Median sMAPE: {eval_df['sMAPE_%'].median():.2f}%")
            logger.log(f"Best prediction (lowest sMAPE): {eval_df.loc[eval_df['sMAPE_%'].idxmin(), 'Metric']} "
                      f"({eval_df['sMAPE_%'].min():.2f}%)")
            logger.log(f"Worst prediction (highest sMAPE): {eval_df.loc[eval_df['sMAPE_%'].idxmax(), 'Metric']} "
                      f"({eval_df['sMAPE_%'].max():.2f}%)")

        return eval_df

    except Exception as e:
        logger.test_result(False, f"Evaluation failed: {str(e)}")
        return None


def test_multiple_tickers(logger, tickers=["AAPL", "MSFT"], forecast_year=2024):
    """Test forecasting for multiple companies"""
    logger.test_header(f"Multiple Ticker Forecasts - {', '.join(tickers)}")

    results = {}
    for ticker in tickers:
        logger.section(f"Testing {ticker}")
        try:
            result = forecast_company_year(ticker, forecast_year)
            results[ticker] = result

            revenue = result['income_statement'].loc[forecast_year, 'Revenue']
            net_income = result['income_statement'].loc[forecast_year, 'NetIncome']

            logger.test_result(True, f"{ticker} forecast: Revenue=${revenue:,.0f}, NI=${net_income:,.0f}")

        except Exception as e:
            logger.test_result(False, f"{ticker} forecast failed: {str(e)}")
            results[ticker] = None

    return results


def create_mock_historical_data():
    """Create mock historical data for testing without internet"""
    data = {
        2020: {'S': 274515000000, 'COGS': 169559000000, 'SGA': 38668000000, 'Dep': 11056000000,
               'IntExp': 2873000000, 'Tax': 9680000000, 'NI': 57411000000,
               'Cash': 38016000000, 'AR': 37445000000, 'Inv': 4061000000, 'PPE': 37378000000,
               'OA': 42522000000, 'AP': 42296000000, 'STD': 13769000000, 'LTD': 98667000000,
               'OL': 43770000000, 'Equity': 65339000000, 'TotalAssets': 323888000000,
               'TotalLiab': 258549000000, 'Capex': -7309000000, 'DivPaid': -14081000000},
        2021: {'S': 365817000000, 'COGS': 212981000000, 'SGA': 43887000000, 'Dep': 11284000000,
               'IntExp': 2645000000, 'Tax': 14527000000, 'NI': 94680000000,
               'Cash': 34940000000, 'AR': 51506000000, 'Inv': 6580000000, 'PPE': 39440000000,
               'OA': 48849000000, 'AP': 54763000000, 'STD': 15613000000, 'LTD': 109106000000,
               'OL': 47493000000, 'Equity': 63090000000, 'TotalAssets': 351002000000,
               'TotalLiab': 287912000000, 'Capex': -11085000000, 'DivPaid': -14467000000},
        2022: {'S': 394328000000, 'COGS': 223546000000, 'SGA': 25094000000, 'Dep': 11104000000,
               'IntExp': 2931000000, 'Tax': 19300000000, 'NI': 99803000000,
               'Cash': 23646000000, 'AR': 60932000000, 'Inv': 4946000000, 'PPE': 42117000000,
               'OA': 54428000000, 'AP': 64115000000, 'STD': 21110000000, 'LTD': 98959000000,
               'OL': 60845000000, 'Equity': 50672000000, 'TotalAssets': 352755000000,
               'TotalLiab': 302083000000, 'Capex': -10708000000, 'DivPaid': -14841000000},
        2023: {'S': 383285000000, 'COGS': 214137000000, 'SGA': 24932000000, 'Dep': 11519000000,
               'IntExp': 3933000000, 'Tax': 16741000000, 'NI': 96995000000,
               'Cash': 29965000000, 'AR': 62467000000, 'Inv': 6511000000, 'PPE': 43715000000,
               'OA': 48304000000, 'AP': 62611000000, 'STD': 15000000000, 'LTD': 95000000000,
               'OL': 56000000000, 'Equity': 62146000000, 'TotalAssets': 190962000000,
               'TotalLiab': 128816000000, 'Capex': -10959000000, 'DivPaid': -15025000000},
    }

    df = pd.DataFrame(data).T
    df.index.name = 'year'
    return df


def test_with_mock_data(logger):
    """Test using mock data"""
    logger.test_header("Mock Data Tests - Full Workflow")

    # Create mock data
    hist = create_mock_historical_data()
    logger.log(f"Created mock dataset with years: {sorted(hist.index.tolist())}")

    # Test policy estimation
    try:
        policies = estimate_policies(hist, up_to_year=2023, lookback_years=3)
        logger.test_result(True, "Policy estimation with mock data successful")

        # Test that values are reasonable (not NaN or extreme)
        reasonable = (
            not np.isnan(policies.g_S) and abs(policies.g_S) < 1.0 and
            not np.isnan(policies.m_cogs) and 0 < policies.m_cogs < 1.0 and
            not np.isnan(policies.m_sga) and 0 < policies.m_sga < 1.0
        )
        logger.test_result(reasonable, "Policy parameters are reasonable")

        # Test forecast
        result = forecast_one_year(hist, policies, target_year=2023)
        logger.test_result(True, "Forecast with mock data successful")

        if result:
            logger.section("Mock Data Forecast Results")
            logger.log("\nIncome Statement:")
            logger.log(result['income_statement'].round(2).to_string())
            logger.log("\nBalance Sheet:")
            logger.log(result['balance_sheet'].round(2).to_string())

            # Test evaluation
            eval_df = evaluate_forecast(result, hist, 2023)
            logger.test_result(True, "Evaluation with mock data successful")

            logger.section("Mock Data Evaluation Summary")
            logger.log(f"Average sMAPE: {eval_df['sMAPE_%'].mean():.2f}%")
            logger.log(f"Median sMAPE: {eval_df['sMAPE_%'].median():.2f}%")

    except Exception as e:
        logger.test_result(False, f"Mock data test failed: {str(e)}")


def main():
    """Run all tests"""
    logger = TestLogger("test_results.txt")

    # Print header
    logger.log("="*80)
    logger.log("FINANCIAL FORECAST MODEL - TEST SUITE")
    logger.log(f"Test run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("="*80)

    # Test 1: Utility functions
    test_utility_functions(logger)

    # Test 2: Mock data tests (these always work)
    test_with_mock_data(logger)

    # Test 3: Data fetching (may fail due to network issues)
    logger.section("Live Data Tests (may fail due to network/SSL issues)")
    ticker = "AAPL"
    fin, bs, cf = test_data_fetching(logger, ticker)

    if fin is not None:
        # Test 4: Historical dataset
        hist = test_historical_dataset(logger, fin, bs, cf)

        if hist is not None:
            # Test 5: Policy estimation
            policies = test_policy_estimation(logger, hist)

            # Test 6: One-year forecast
            forecast_result = test_one_year_forecast(logger, hist, policies)

        # Test 7: Full forecast workflow
        full_result = test_full_forecast(logger, ticker="AAPL", forecast_year=2024)

        # Test 8: Forecast evaluation (if actual data is available)
        test_forecast_evaluation(logger, ticker="AAPL", forecast_year=2024)

        # Test 9: Multiple tickers
        test_multiple_tickers(logger, tickers=["AAPL", "MSFT"], forecast_year=2024)

    # Close logger and print summary
    logger.close()

    print(f"\n✓ Test results saved to: {logger.filename}")
    print(f"✓ Total tests: {logger.test_count}")
    print(f"✓ Passed: {logger.passed_count}")
    print(f"✓ Failed: {logger.failed_count}")


if __name__ == "__main__":
    main()