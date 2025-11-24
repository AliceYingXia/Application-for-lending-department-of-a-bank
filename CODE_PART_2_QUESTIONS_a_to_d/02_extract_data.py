#!/usr/bin/env python3
"""
Data Extraction for Gemini Forecasting Experiment

Fetches historical (2020-2023) and actual (2024) financial data from Yahoo Finance
for the selected sample companies. Formats data as JSON for LLM prompts.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import json
from pathlib import Path
from tqdm import tqdm
import config


def fetch_financial_data(ticker: str, start_year: int, end_year: int):
    """
    Fetch financial data from Yahoo Finance for a given ticker and year range.

    Args:
        ticker: Stock ticker symbol
        start_year: First year to include
        end_year: Last year to include (inclusive)

    Returns:
        dict: Financial data organized by year
    """
    try:
        tkr = yf.Ticker(ticker)

        # Fetch statements
        income_stmt = tkr.financials.T  # Transpose to get years as rows
        balance_sheet = tkr.balance_sheet.T
        cash_flow = tkr.cashflow.T

        # Convert index to year integers
        income_stmt.index = income_stmt.index.year
        balance_sheet.index = balance_sheet.index.year
        cash_flow.index = cash_flow.index.year

        # Filter years
        years = sorted(set(income_stmt.index) & set(balance_sheet.index) & set(cash_flow.index))
        years = [y for y in years if start_year <= y <= end_year]

        if not years:
            return None

        # Extract key financial metrics
        financial_data = {}

        for year in years:
            is_row = income_stmt.loc[year]
            bs_row = balance_sheet.loc[year]
            cf_row = cash_flow.loc[year]

            # Helper function to get first available field
            def get_field(row, field_names):
                for name in field_names:
                    if name in row.index and pd.notna(row[name]):
                        return float(row[name])
                return None

            # Extract Income Statement metrics
            revenue = get_field(is_row, [
                "Total Revenue", "TotalRevenue", "Revenue", "Revenues"
            ])
            cogs = get_field(is_row, [
                "Cost Of Revenue", "CostOfRevenue", "Total Cost Of Revenue"
            ])
            gross_profit = get_field(is_row, [
                "Gross Profit", "GrossProfit"
            ])
            operating_expense = get_field(is_row, [
                "Operating Expense", "OperatingExpense", "Total Operating Expenses"
            ])
            operating_income = get_field(is_row, [
                "Operating Income", "OperatingIncome", "EBIT"
            ])
            net_income = get_field(is_row, [
                "Net Income", "NetIncome", "Net Income Common Stockholders"
            ])

            # Extract Balance Sheet metrics
            total_assets = get_field(bs_row, [
                "Total Assets", "TotalAssets"
            ])
            current_assets = get_field(bs_row, [
                "Current Assets", "CurrentAssets"
            ])
            cash = get_field(bs_row, [
                "Cash And Cash Equivalents", "CashAndCashEquivalents",
                "Cash", "Cash Cash Equivalents And Short Term Investments"
            ])
            total_liabilities = get_field(bs_row, [
                "Total Liabilities Net Minority Interest", "Total Liabilities",
                "TotalLiabilities", "TotalLiabilitiesNetMinorityInterest"
            ])
            current_liabilities = get_field(bs_row, [
                "Current Liabilities", "CurrentLiabilities"
            ])
            equity = get_field(bs_row, [
                "Stockholders Equity", "StockholdersEquity", "Total Equity Gross Minority Interest",
                "Common Stock Equity", "CommonStockEquity"
            ])

            # Extract Cash Flow metrics
            operating_cash_flow = get_field(cf_row, [
                "Operating Cash Flow", "OperatingCashFlow", "Cash Flow From Operating Activities"
            ])
            capex = get_field(cf_row, [
                "Capital Expenditure", "CapitalExpenditure", "Capital Expenditures"
            ])
            free_cash_flow = get_field(cf_row, [
                "Free Cash Flow", "FreeCashFlow"
            ])

            financial_data[year] = {
                "income_statement": {
                    "revenue": revenue,
                    "cogs": cogs,
                    "gross_profit": gross_profit,
                    "operating_expense": operating_expense,
                    "operating_income": operating_income,
                    "net_income": net_income
                },
                "balance_sheet": {
                    "total_assets": total_assets,
                    "current_assets": current_assets,
                    "cash": cash,
                    "total_liabilities": total_liabilities,
                    "current_liabilities": current_liabilities,
                    "equity": equity
                },
                "cash_flow": {
                    "operating_cash_flow": operating_cash_flow,
                    "capex": capex,
                    "free_cash_flow": free_cash_flow
                }
            }

        return financial_data

    except Exception as e:
        print(f"  ❌ Error fetching data for {ticker}: {e}")
        return None


def extract_all_data():
    """
    Extract historical and actual data for all companies in the test sample.
    """
    print("=" * 70)
    print("DATA EXTRACTION FOR GEMINI FORECASTING EXPERIMENT")
    print("=" * 70)

    # Load test sample
    sample_file = config.DATA_DIR / "test_sample.csv"
    print(f"\nLoading test sample from: {sample_file}")
    sample = pd.read_csv(sample_file)
    print(f"Companies to process: {len(sample)}")

    # Statistics
    success_count = 0
    failed_tickers = []

    # Process each company
    print("\nExtracting financial data from Yahoo Finance...")
    for idx, row in tqdm(sample.iterrows(), total=len(sample), desc="Processing"):
        ticker = row['ticker']

        # Fetch historical data (2020-2023) for training
        historical_data = fetch_financial_data(ticker, start_year=2020, end_year=2023)

        # Fetch actual 2024 data for evaluation
        actual_data = fetch_financial_data(ticker, start_year=2024, end_year=2024)

        if historical_data and actual_data:
            # Save historical data
            historical_file = config.DATA_DIR / "historical" / f"{ticker}.json"
            with open(historical_file, 'w') as f:
                json.dump(historical_data, f, indent=2)

            # Save actual data
            actual_file = config.DATA_DIR / "actual" / f"{ticker}.json"
            with open(actual_file, 'w') as f:
                json.dump(actual_data, f, indent=2)

            success_count += 1
        else:
            failed_tickers.append(ticker)
            print(f"  ⚠️  Failed to fetch complete data for {ticker}")

    # Summary
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Total companies processed: {len(sample)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_tickers)}")

    if failed_tickers:
        print(f"\nFailed tickers: {', '.join(failed_tickers)}")

    print(f"\nData saved to:")
    print(f"  Historical (2020-2023): {config.DATA_DIR / 'historical'}")
    print(f"  Actual (2024): {config.DATA_DIR / 'actual'}")
    print("=" * 70 + "\n")

    return success_count, failed_tickers


if __name__ == "__main__":
    success, failed = extract_all_data()

    if success > 0:
        print(f"✅ Data extraction complete!")
        print(f"   Extracted data for {success} companies")
        if failed:
            print(f"   ⚠️  {len(failed)} companies had incomplete data")
    else:
        print(f"❌ Data extraction failed!")
        print(f"   No companies had complete data")