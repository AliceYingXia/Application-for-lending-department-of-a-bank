"""
Simple non-circular forecast of earnings and balance sheet
using Yahoo Finance data and Vélez-Pareja-style policies.

Requirements:
    pip install yfinance pandas numpy
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import Dict, Any, Optional


# ------------------------ Utilities ------------------------ #

def _pick_first_available(row: pd.Series, candidates) -> float:
    """
    Given a Series representing one column of a financials DataFrame
    (all rows, one date/year), pick the first non-NaN from a list of row names.
    """
    for name in candidates:
        if name in row.index and pd.notna(row[name]):
            return float(row[name])
    return np.nan


def _safe_div(a, b, default=np.nan):
    if b is None or b == 0 or np.isnan(b):
        return default
    return a / b


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
    if np.isnan(actual) or np.isnan(forecast):
        return np.nan

    numerator = abs(actual - forecast)
    denominator = (abs(actual) + abs(forecast)) / 2.0

    if denominator == 0:
        return 0.0 if numerator == 0 else np.nan

    return 100.0 * numerator / denominator


# ------------------------ Data structures ------------------------ #

@dataclass
class PolicyParams:
    g_S: float
    m_cogs: float
    m_sga: float
    dep_rate: float
    r_d: float
    tax_rate: float
    payout_ratio: float
    dso: float
    dih: float
    dpo: float
    capex_ratio: float
    min_cash_ratio: float
    g_OA: float
    g_OL: float


# ------------------------ Yahoo pulling & preprocessing ------------------------ #

def fetch_yahoo_statements(ticker: str, end_year: Optional[int] = None):
    """
    Fetch annual financial statements from Yahoo via yfinance.
    Note: Yahoo Finance typically provides only the last 4 fiscal years.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        end_year: Optional. The year to forecast (e.g., 2015).
                 If provided, only keeps data for years before this year.

    Returns:
        financials (IS), balance_sheet (BS), cashflow (CF) as DataFrames indexed by year (int).
    """
    tkr = yf.Ticker(ticker)

    # yfinance gives DataFrames with columns as periods, index as line items
    fin = tkr.financials  # Income statement
    bs = tkr.balance_sheet
    cf = tkr.cashflow

    if fin.empty or bs.empty or cf.empty:
        raise ValueError("Missing financial statements from Yahoo Finance.")

    # transpose: rows = dates, cols = line items
    fin = fin.T
    bs = bs.T
    cf = cf.T

    # convert index (Timestamp) to year int
    fin.index = fin.index.year
    bs.index = bs.index.year
    cf.index = cf.index.year

    # Keep only intersection of years
    years = sorted(set(fin.index) & set(bs.index) & set(cf.index))

    # Filter to only include years before end_year if specified
    if end_year is not None:
        years = [y for y in years if y < end_year]
        print(f"Yahoo Finance provided {len(years)} years of data before {end_year}: {years}")

    if not years:
        raise ValueError(f"No data available for the specified year range.")

    fin = fin.loc[years]
    bs = bs.loc[years]
    cf = cf.loc[years]

    return fin, bs, cf


def build_historical_dataset(fin: pd.DataFrame,
                             bs: pd.DataFrame,
                             cf: pd.DataFrame,
                             debug: bool = True) -> pd.DataFrame:
    """
    Build a clean historical panel with all key variables.
    One row per year.
    """
    if debug and len(fin.index) > 0:
        print("\n=== Available field names (first year) ===")
        print("Income Statement fields:", list(fin.columns[:20]))
        print("Balance Sheet fields:", list(bs.columns[:20]))
        print("Cash Flow fields:", list(cf.columns[:20]))
        print("=" * 50 + "\n")

    rows = []
    for year in fin.index:
        fin_row = fin.loc[year]
        bs_row = bs.loc[year]
        cf_row = cf.loc[year]

        # Income statement - comprehensive field name matching
        S = _pick_first_available(fin_row, [
            "Total Revenue", "TotalRevenue", "Total Revenues",
            "Revenue", "Revenues", "Sales"
        ])

        COGS = _pick_first_available(fin_row, [
            "Cost Of Revenue", "CostOfRevenue", "Cost of Revenue",
            "Total Cost Of Revenue", "TotalCostOfRevenue"
        ])

        # Try to find SG&A in multiple ways
        SGA = _pick_first_available(fin_row, [
            "Selling General And Administrative", "SellingGeneralAndAdministrative",
            "Selling General Administrative", "SellingGeneralAdministrative",
            "SG&A", "SGA",
            "Selling And Marketing Expense", "General And Administrative Expense"
        ])

        # If SGA is not found, try to calculate from operating expenses
        if np.isnan(SGA):
            operating_exp = _pick_first_available(fin_row, [
                "Operating Expense", "OperatingExpense", "Total Operating Expenses",
                "Operating Expenses", "OperatingExpenses"
            ])
            if not np.isnan(operating_exp) and not np.isnan(COGS):
                # Operating expenses might include COGS, or might be COGS + SGA
                # Try to derive SGA safely
                if operating_exp > COGS:
                    SGA = operating_exp - COGS
                else:
                    # If operating_exp <= COGS, assume operating_exp IS the SGA
                    SGA = operating_exp

        # Ensure SGA is positive (expenses should be positive in our model)
        if not np.isnan(SGA) and SGA < 0:
            SGA = abs(SGA)

        Dep = _pick_first_available(cf_row, [
            "Depreciation And Amortization", "DepreciationAndAmortization",
            "Depreciation", "Depreciation Amortization Depletion",
            "Reconciled Depreciation"
        ])

        IntExp = _pick_first_available(fin_row, [
            "Interest Expense", "InterestExpense", "Interest Expense Non Operating",
            "Net Interest Income", "Interest Income Expense"
        ])
        # Interest expense should be positive, but sometimes it's negative in data
        if not np.isnan(IntExp) and IntExp < 0:
            IntExp = -IntExp

        Tax = _pick_first_available(fin_row, [
            "Tax Provision", "TaxProvision", "Income Tax Expense",
            "IncomeTaxExpense", "Tax Effect Of Unusual Items"
        ])

        NI = _pick_first_available(fin_row, [
            "Net Income", "NetIncome", "Net Income Common Stockholders",
            "NetIncomeCommonStockholders", "Normalized Income"
        ])

        # Balance sheet - comprehensive matching
        Cash = _pick_first_available(bs_row, [
            "Cash And Cash Equivalents", "CashAndCashEquivalents",
            "Cash Cash Equivalents And Short Term Investments",
            "Cash", "Cash And Short Term Investments"
        ])

        AR = _pick_first_available(bs_row, [
            "Accounts Receivable", "AccountsReceivable", "Receivables",
            "Net Receivables", "NetReceivables", "Accounts Receivable Net"
        ])

        Inv = _pick_first_available(bs_row, [
            "Inventory", "Inventories", "Net Inventory"
        ])

        PPE = _pick_first_available(bs_row, [
            "Net PPE", "NetPPE", "Property Plant Equipment Net",
            "PropertyPlantEquipmentNet", "Net Property Plant And Equipment",
            "Gross PPE", "GrossPPE"
        ])

        TotalAssets = _pick_first_available(bs_row, [
            "Total Assets", "TotalAssets", "Assets"
        ])

        AP = _pick_first_available(bs_row, [
            "Accounts Payable", "AccountsPayable", "Payables",
            "Accounts Payable And Accrued Expenses"
        ])

        # Try multiple debt fields
        ShortDebt = _pick_first_available(bs_row, [
            "Current Debt", "CurrentDebt", "Short Term Debt",
            "Short Long Term Debt", "ShortLongTermDebt",
            "Current Debt And Capital Lease Obligation"
        ])

        LongDebt = _pick_first_available(bs_row, [
            "Long Term Debt", "LongTermDebt", "Long Term Debt And Capital Lease Obligation",
            "Total Debt", "TotalDebt"
        ])

        # Handle case where only total debt is available
        if np.isnan(LongDebt) and np.isnan(ShortDebt):
            total_debt = _pick_first_available(bs_row, [
                "Total Debt", "TotalDebt", "Net Debt"
            ])
            if not np.isnan(total_debt):
                # Assume 70% long-term, 30% short-term as approximation
                LongDebt = total_debt * 0.7
                ShortDebt = total_debt * 0.3

        TotalLiab = _pick_first_available(bs_row, [
            "Total Liabilities Net Minority Interest", "TotalLiabilitiesNetMinorityInterest",
            "Total Liabilities", "TotalLiabilities"
        ])

        Equity = _pick_first_available(bs_row, [
            "Stockholders Equity", "StockholdersEquity",
            "Total Equity Gross Minority Interest", "TotalEquityGrossMinorityInterest",
            "Total Stockholder Equity", "TotalStockholderEquity",
            "Common Stock Equity", "CommonStockEquity"
        ])

        # Derived "other" buckets with safe calculations
        if not np.isnan(TotalAssets):
            known_assets = sum(x for x in [Cash, AR, Inv, PPE] if not np.isnan(x))
            OA = TotalAssets - known_assets
        else:
            OA = np.nan

        if not np.isnan(ShortDebt) or not np.isnan(LongDebt):
            interest_bearing_debt = (ShortDebt if not np.isnan(ShortDebt) else 0) + \
                                   (LongDebt if not np.isnan(LongDebt) else 0)
        else:
            interest_bearing_debt = 0

        if not np.isnan(TotalLiab):
            known_liab = (AP if not np.isnan(AP) else 0) + interest_bearing_debt
            OL = TotalLiab - known_liab
        else:
            OL = np.nan

        # Cashflow items
        Capex = _pick_first_available(cf_row, [
            "Capital Expenditure", "CapitalExpenditure",
            "Capital Expenditures", "CapitalExpenditures",
            "Purchase Of PPE", "PurchaseOfPPE"
        ])

        DivPaid = _pick_first_available(cf_row, [
            "Cash Dividends Paid", "CashDividendsPaid",
            "Dividends Paid", "DividendsPaid",
            "Common Stock Dividend Paid", "Payment Of Dividends"
        ])

        rows.append({
            "year": year,
            "S": S,
            "COGS": COGS,
            "SGA": SGA,
            "Dep": Dep,
            "IntExp": IntExp,
            "Tax": Tax,
            "NI": NI,
            "Cash": Cash,
            "AR": AR,
            "Inv": Inv,
            "PPE": PPE,
            "OA": OA,
            "AP": AP,
            "STD": ShortDebt,
            "LTD": LongDebt,
            "OL": OL,
            "Equity": Equity,
            "TotalAssets": TotalAssets,
            "TotalLiab": TotalLiab,
            "Capex": Capex,
            "DivPaid": DivPaid,
        })

    df = pd.DataFrame(rows).set_index("year").sort_index()

    # Basic cleaning: replace inf and drop impossible rows
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if debug:
        print("\n=== Data extraction summary ===")
        print("Number of years:", len(df))
        print("\nMissing data percentage by field:")
        missing_pct = (df.isna().sum() / len(df) * 100).round(1)
        for field, pct in missing_pct.items():
            status = "⚠️" if pct > 50 else "✓" if pct == 0 else "○"
            print(f"  {status} {field}: {pct}% missing")
        print("=" * 50 + "\n")

    return df


# ------------------------ Policy estimation ------------------------ #

def estimate_policies(hist: pd.DataFrame, up_to_year: int, lookback_years: int = 3) -> PolicyParams:
    """
    Estimate policy parameters using the last available values before up_to_year.
    Uses the most recent year's data for forecasting.
    """
    # Filter to years < up_to_year
    df = hist.loc[hist.index < up_to_year].copy()

    if df.shape[0] < 2:
        raise ValueError(f"Not enough historical years to estimate policies. "
                        f"Need at least 2 years of data before {up_to_year}.")

    # Calculate all metrics using the full dataset
    # Revenue growth - use the last available percentage change
    df["g_S"] = df["S"].pct_change()

    # Margins (these don't need previous year, so can use all years)
    df["m_cogs"] = df["COGS"] / df["S"]
    df["m_sga"] = df["SGA"] / df["S"]

    # Depreciation rate (Dep / PPE_{t-1})
    df["PPE_prev"] = df["PPE"].shift(1)
    df["dep_rate"] = df["Dep"] / df["PPE_prev"]

    # Interest rate on debt
    df["Debt_prev"] = (df["STD"].shift(1) + df["LTD"].shift(1))
    df["r_d"] = df["IntExp"] / df["Debt_prev"]

    # Tax rate: Tax / EBT
    df["EBIT"] = df["S"] - df["COGS"] - df["SGA"] - df["Dep"]
    df["EBT"] = df["EBIT"] - df["IntExp"]
    df["tax_rate"] = df.apply(
        lambda row: _safe_div(row["Tax"], max(row["EBT"], 1e-6)), axis=1
    )

    # Dividend payout ratio: Div_t / NI_{t-1}
    df["NI_prev"] = df["NI"].shift(1)
    df["Div"] = -df["DivPaid"]  # dividends paid are usually negative
    df["payout"] = df.apply(
        lambda row: _safe_div(row["Div"], max(row["NI_prev"], 1e-6)), axis=1
    )

    # Working capital days
    df["DSO"] = df.apply(
        lambda row: _safe_div(row["AR"], row["S"] / 365.0, default=np.nan), axis=1
    )
    df["DIH"] = df.apply(
        lambda row: _safe_div(row["Inv"], row["COGS"] / 365.0, default=np.nan), axis=1
    )
    df["DPO"] = df.apply(
        lambda row: _safe_div(row["AP"], row["COGS"] / 365.0, default=np.nan), axis=1
    )

    # Capex and minimum cash ratios
    df["capex_ratio"] = df.apply(
        lambda row: _safe_div(-row["Capex"], row["S"]), axis=1
    )
    df["min_cash_ratio"] = df.apply(
        lambda row: _safe_div(row["Cash"], row["S"]), axis=1
    )

    # Growth of OA and OL (trend)
    df["g_OA"] = df["OA"].pct_change()
    df["g_OL"] = df["OL"].pct_change()

    # Use the LAST AVAILABLE value for each metric (most recent year)
    # Get the last valid (non-NaN) value for each metric
    last_idx = df.index[-1]  # Most recent year

    def get_last_valid(series):
        """Get the last non-NaN value from a series"""
        valid_values = series.dropna()
        if len(valid_values) == 0:
            return np.nan
        return float(valid_values.iloc[-1])

    g_S = get_last_valid(df["g_S"])
    m_cogs = get_last_valid(df["m_cogs"])
    m_sga = get_last_valid(df["m_sga"])
    dep_rate = get_last_valid(df["dep_rate"])
    r_d = get_last_valid(df["r_d"])
    tax_rate = get_last_valid(df["tax_rate"])
    payout_ratio = get_last_valid(df["payout"])
    dso = get_last_valid(df["DSO"])
    dih = get_last_valid(df["DIH"])
    dpo = get_last_valid(df["DPO"])
    capex_ratio = get_last_valid(df["capex_ratio"])
    min_cash_ratio = get_last_valid(df["min_cash_ratio"])
    g_OA = get_last_valid(df["g_OA"])
    g_OL = get_last_valid(df["g_OL"])

    print(f"Using last available year ({last_idx}) for policy parameters")

    # Apply reasonable bounds to ensure parameters are valid
    def apply_bounds(value, min_val, max_val, default, param_name):
        """Apply bounds to parameter values and handle NaN"""
        if np.isnan(value) or np.isinf(value):
            print(f"  Warning: {param_name} is NaN/Inf, using default {default}")
            return default
        if value < min_val or value > max_val:
            capped = max(min_val, min(max_val, value))
            print(f"  Warning: {param_name} = {value:.4f} out of bounds [{min_val}, {max_val}], capped to {capped:.4f}")
            return capped
        return value

    # Apply bounds to each parameter
    g_S = apply_bounds(g_S, -0.5, 0.5, 0.05, "g_S (revenue growth)")
    m_cogs = apply_bounds(m_cogs, 0.01, 0.99, 0.60, "m_cogs (COGS margin)")
    m_sga = apply_bounds(m_sga, 0.01, 0.99, 0.20, "m_sga (SGA margin)")
    dep_rate = apply_bounds(dep_rate, 0.01, 0.50, 0.10, "dep_rate")
    r_d = apply_bounds(r_d, 0.0, 0.20, 0.03, "r_d (interest rate)")
    tax_rate = apply_bounds(tax_rate, 0.0, 0.50, 0.21, "tax_rate")
    payout_ratio = apply_bounds(payout_ratio, 0.0, 1.0, 0.30, "payout_ratio")
    dso = apply_bounds(dso, 1, 365, 45, "DSO (days)")
    dih = apply_bounds(dih, 1, 365, 30, "DIH (days)")
    dpo = apply_bounds(dpo, 1, 365, 60, "DPO (days)")
    capex_ratio = apply_bounds(capex_ratio, 0.0, 0.50, 0.05, "capex_ratio")
    min_cash_ratio = apply_bounds(min_cash_ratio, 0.01, 0.50, 0.10, "min_cash_ratio")
    g_OA = apply_bounds(g_OA, -0.5, 0.5, 0.05, "g_OA (OA growth)")
    g_OL = apply_bounds(g_OL, -0.5, 0.5, 0.05, "g_OL (OL growth)")

    return PolicyParams(
        g_S=float(g_S),
        m_cogs=float(m_cogs),
        m_sga=float(m_sga),
        dep_rate=float(dep_rate),
        r_d=float(r_d),
        tax_rate=float(tax_rate),
        payout_ratio=float(payout_ratio),
        dso=float(dso),
        dih=float(dih),
        dpo=float(dpo),
        capex_ratio=float(capex_ratio),
        min_cash_ratio=float(min_cash_ratio),
        g_OA=float(g_OA),
        g_OL=float(g_OL),
    )


# ------------------------ Forecast for one year ------------------------ #

def forecast_one_year(hist: pd.DataFrame,
                      policies: PolicyParams,
                      target_year: int) -> Dict[str, pd.DataFrame]:
    """
    Forecast income statement and balance sheet for target_year.
    Uses all years < target_year for policy estimation and
    uses the last year < target_year as base year (t-1).
    """
    years_before = hist.index[hist.index < target_year]
    if len(years_before) == 0:
        raise ValueError("No historical data before target_year.")

    base_year = years_before.max()
    base = hist.loc[base_year]

    p = policies  # alias

    # ---- 1. Income statement for year t ---- #
    S0 = base["S"]
    S_t = S0 * (1.0 + p.g_S)

    COGS_t = p.m_cogs * S_t
    SGA_t = p.m_sga * S_t

    Dep_t = p.dep_rate * base["PPE"]

    EBIT_t = S_t - COGS_t - SGA_t - Dep_t

    debt_prev = base["STD"] + base["LTD"]
    IntExp_t = p.r_d * debt_prev

    EBT_t = EBIT_t - IntExp_t
    Tax_t = p.tax_rate * max(EBT_t, 0.0)
    NI_t = EBT_t - Tax_t

    # Dividends based on previous NI (year base_year)
    Div_t = p.payout_ratio * max(base["NI"], 0.0)

    # ---- 2. Working capital and PPE ---- #
    AR_t = p.dso * (S_t / 365.0)
    Inv_t = p.dih * (COGS_t / 365.0)
    AP_t = p.dpo * (COGS_t / 365.0)

    NWC_t = AR_t + Inv_t - AP_t
    NWC_prev = base["AR"] + base["Inv"] - base["AP"]

    Capex_t = p.capex_ratio * S_t
    PPE_t = base["PPE"] + Capex_t - Dep_t

    # Other assets / liabilities growth
    OA_t = base["OA"] * (1.0 + (p.g_OA if not np.isnan(p.g_OA) else 0.0))
    OL_t = base["OL"] * (1.0 + (p.g_OL if not np.isnan(p.g_OL) else 0.0))

    # ---- 3. Cash flow and financing ---- #
    EBIT_after_tax = EBIT_t * (1.0 - p.tax_rate)
    OCF_t = EBIT_after_tax + Dep_t - (NWC_t - NWC_prev)
    FCFF_t = OCF_t - Capex_t

    Cash_prev = base["Cash"]
    Cash_pre = Cash_prev + FCFF_t - Div_t - IntExp_t  # cash before financing

    Cash_min = p.min_cash_ratio * S_t

    deficit = max(0.0, Cash_min - Cash_pre)
    surplus = max(0.0, Cash_pre - Cash_min)

    # Simple policy: adjust only short-term debt
    dSTD_plus = deficit
    dSTD_minus = min(surplus, base["STD"])

    STD_t = base["STD"] + dSTD_plus - dSTD_minus
    LTD_t = base["LTD"]  # keep constant (you can add amortization if wanted)

    Cash_t = Cash_pre + dSTD_plus - dSTD_minus

    # ---- 4. Equity update ---- #
    Equity_prev = base["Equity"]
    delta_RE = NI_t - Div_t
    Equity_t = Equity_prev + delta_RE

    # ---- 5. Construct output DataFrames ---- #
    is_forecast = pd.DataFrame(
        {
            "Revenue": [S_t],
            "COGS": [COGS_t],
            "SGA": [SGA_t],
            "Depreciation": [Dep_t],
            "EBIT": [EBIT_t],
            "InterestExpense": [IntExp_t],
            "EBT": [EBT_t],
            "Tax": [Tax_t],
            "NetIncome": [NI_t],
            "Dividends": [Div_t],
        },
        index=[target_year],
    )

    bs_forecast = pd.DataFrame(
        {
            "Cash": [Cash_t],
            "AR": [AR_t],
            "Inv": [Inv_t],
            "PPE": [PPE_t],
            "OA": [OA_t],
            "AP": [AP_t],
            "STD": [STD_t],
            "LTD": [LTD_t],
            "OL": [OL_t],
            "Equity": [Equity_t],
        },
        index=[target_year],
    )

    # Check accounting identity
    bs_forecast["Assets"] = (
        bs_forecast["Cash"] + bs_forecast["AR"] + bs_forecast["Inv"]
        + bs_forecast["PPE"] + bs_forecast["OA"]
    )
    bs_forecast["LiabEq"] = (
        bs_forecast["AP"] + bs_forecast["STD"] + bs_forecast["LTD"]
        + bs_forecast["OL"] + bs_forecast["Equity"]
    )
    bs_forecast["Assets_minus_LiabEq"] = bs_forecast["Assets"] - bs_forecast["LiabEq"]

    return {
        "income_statement": is_forecast,
        "balance_sheet": bs_forecast,
    }


# ------------------------ Forecast Evaluation ------------------------ #

def evaluate_forecast(forecast_result: Dict[str, pd.DataFrame],
                     actual_hist: pd.DataFrame,
                     forecast_year: int) -> pd.DataFrame:
    """
    Evaluate forecast accuracy using sMAPE metrics.

    Args:
        forecast_result: Dict with 'income_statement' and 'balance_sheet' forecasts
        actual_hist: Historical dataset containing the actual values for forecast_year
        forecast_year: The year that was forecasted

    Returns:
        DataFrame with sMAPE values for each metric
    """
    if forecast_year not in actual_hist.index:
        raise ValueError(f"Actual data for {forecast_year} not found in historical dataset")

    actual = actual_hist.loc[forecast_year]
    is_forecast = forecast_result["income_statement"].loc[forecast_year]
    bs_forecast = forecast_result["balance_sheet"].loc[forecast_year]

    # Map forecast column names to historical data column names
    is_mapping = {
        "Revenue": "S",
        "COGS": "COGS",
        "SGA": "SGA",
        "Depreciation": "Dep",
        "EBIT": None,  # Calculated field
        "InterestExpense": "IntExp",
        "EBT": None,  # Calculated field
        "Tax": "Tax",
        "NetIncome": "NI",
        "Dividends": None,  # Not directly in historical data
    }

    bs_mapping = {
        "Cash": "Cash",
        "AR": "AR",
        "Inv": "Inv",
        "PPE": "PPE",
        "OA": "OA",
        "AP": "AP",
        "STD": "STD",
        "LTD": "LTD",
        "OL": "OL",
        "Equity": "Equity",
    }

    results = []

    # Evaluate Income Statement
    for forecast_col, hist_col in is_mapping.items():
        if hist_col is not None and hist_col in actual.index:
            forecast_val = is_forecast[forecast_col]
            actual_val = actual[hist_col]
            smape_val = calculate_smape(actual_val, forecast_val)

            results.append({
                "Statement": "Income Statement",
                "Metric": forecast_col,
                "Actual": actual_val,
                "Forecast": forecast_val,
                "Error": forecast_val - actual_val,
                "Error_%": ((forecast_val - actual_val) / actual_val * 100) if actual_val != 0 else np.nan,
                "sMAPE_%": smape_val
            })

    # Evaluate Balance Sheet
    for forecast_col, hist_col in bs_mapping.items():
        if hist_col in actual.index:
            forecast_val = bs_forecast[forecast_col]
            actual_val = actual[hist_col]
            smape_val = calculate_smape(actual_val, forecast_val)

            results.append({
                "Statement": "Balance Sheet",
                "Metric": forecast_col,
                "Actual": actual_val,
                "Forecast": forecast_val,
                "Error": forecast_val - actual_val,
                "Error_%": ((forecast_val - actual_val) / actual_val * 100) if actual_val != 0 else np.nan,
                "sMAPE_%": smape_val
            })

    eval_df = pd.DataFrame(results)

    # Calculate summary statistics
    print("\n" + "=" * 80)
    print(f"FORECAST EVALUATION FOR YEAR {forecast_year}")
    print("=" * 80)
    print(f"\nIncome Statement Metrics:")
    is_eval = eval_df[eval_df["Statement"] == "Income Statement"]
    print(f"  Average sMAPE: {is_eval['sMAPE_%'].mean():.2f}%")
    print(f"  Median sMAPE:  {is_eval['sMAPE_%'].median():.2f}%")

    print(f"\nBalance Sheet Metrics:")
    bs_eval = eval_df[eval_df["Statement"] == "Balance Sheet"]
    print(f"  Average sMAPE: {bs_eval['sMAPE_%'].mean():.2f}%")
    print(f"  Median sMAPE:  {bs_eval['sMAPE_%'].median():.2f}%")

    print(f"\nOverall:")
    print(f"  Average sMAPE: {eval_df['sMAPE_%'].mean():.2f}%")
    print(f"  Median sMAPE:  {eval_df['sMAPE_%'].median():.2f}%")
    print("=" * 80 + "\n")

    return eval_df


# ------------------------ High-level API ------------------------ #

def forecast_company_year(ticker: str, forecast_year: int, lookback_years: int = 3) -> Dict[str, pd.DataFrame]:
    """
    High-level helper: fetch Yahoo data, estimate policies, forecast one year.
    Note: Yahoo Finance provides approximately 4 years of historical data.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        forecast_year: Year to forecast (e.g., 2015)
        lookback_years: Number of historical years to use for calculating medians (default: 3)
                       Uses the last 3 years of available data before forecast_year

    Example:
        forecast_company_year("AAPL", 2015)  # Uses last 3 years available to forecast 2015
    """
    print(f"\nFetching financial data for {ticker}...")
    fin, bs, cf = fetch_yahoo_statements(ticker, end_year=forecast_year)
    hist = build_historical_dataset(fin, bs, cf)

    # Determine which years are available
    available_years = sorted(hist.index[hist.index < forecast_year])

    if len(available_years) == 0:
        raise ValueError(f"No historical data available before {forecast_year}")

    # Use the last 'lookback_years' years for median calculation
    years_to_use = available_years[-lookback_years:] if len(available_years) >= lookback_years else available_years

    print(f"\nUsing the last {len(years_to_use)} years of data: {years_to_use}")
    print(f"Forecasting year: {forecast_year}\n")

    policies = estimate_policies(hist, up_to_year=forecast_year, lookback_years=lookback_years)
    result = forecast_one_year(hist, policies, target_year=forecast_year)
    return result


if __name__ == "__main__":
    # Example: forecast Apple 2024 using data up to 2023
    ticker = "AAPL"
    forecast_year = 2025

    # Make forecast
    res = forecast_company_year(ticker, forecast_year)
    print("\n=== Forecast Income Statement ===")
    print(res["income_statement"].round(2))
    print("\n=== Forecast Balance Sheet ===")
    print(res["balance_sheet"].round(2))

    # Evaluate forecast if actual data is available
    # To evaluate, fetch all available data (including forecast_year) and compare
    try:
        print("\n" + "="*80)
        print("EVALUATING FORECAST ACCURACY")
        print("="*80)

        # Fetch all available data including the forecast year
        fin_all, bs_all, cf_all = fetch_yahoo_statements(ticker, end_year=None)
        hist_all = build_historical_dataset(fin_all, bs_all, cf_all, debug=False)

        if forecast_year in hist_all.index:
            # Evaluate the forecast
            eval_results = evaluate_forecast(res, hist_all, forecast_year)

            print("\n=== Detailed Evaluation Results ===")
            print(eval_results.to_string(index=False))
        else:
            print(f"\nActual data for {forecast_year} not yet available for evaluation.")

    except Exception as e:
        print(f"\nCould not evaluate forecast: {e}")
