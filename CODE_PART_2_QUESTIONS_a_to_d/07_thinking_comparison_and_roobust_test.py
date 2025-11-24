#!/usr/bin/env python3
"""
Thinking Mode Comparison Test

Compares 4 configurations across 2 thinking modes × 2 temperatures:

  NO THINKING (thinking_budget=0, max_tokens=8192):
    - Config A: No Thinking + Temperature 0.0
    - Config B: No Thinking + Temperature 0.7

  WITH THINKING (thinking_budget=8192, max_tokens=16384):
    - Config C: With Thinking + Temperature 0.0
    - Config D: With Thinking + Temperature 0.7

Tests on 3 stocks: REGN (best), AMZN (medium), MET (poor)
10 runs per configuration = 120 total forecasts
"""

import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
import config
from datetime import datetime
from google import genai
from google.genai import types

# ─── Test Configuration ────────────────────────────────────────────────────────

TEST_TICKERS        = ['REGN', 'AMZN', 'MET']
TEMPERATURES        = [0.0, 0.7]
RUNS_PER_CONFIG     = 10

# Token limits
TOKENS_NO_THINKING  = 8192
TOKENS_WITH_THINKING = 16384   # doubled

# Thinking budgets
BUDGET_NO_THINKING  = 0        # Disabled
BUDGET_WITH_THINKING = 8192    # Enabled (8k thinking tokens)

ORIGINAL_SMAPE = {
    'REGN': 4.05,
    'AMZN': 14.01,
    'MET':  166.68,
}

# ─── Rate Limiter ──────────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, calls_per_minute=15):
        self.min_interval = 60.0 / calls_per_minute
        self.last_call_time = 0

    def wait_if_needed(self):
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call_time = time.time()


# ─── Data Loading ──────────────────────────────────────────────────────────────

def load_company_data(ticker: str) -> dict:
    historical_file = config.DATA_DIR / "historical" / f"{ticker}.json"
    actual_file     = config.DATA_DIR / "actual"     / f"{ticker}.json"
    with open(historical_file) as f:
        historical = json.load(f)
    with open(actual_file) as f:
        actual = json.load(f)
    return {'ticker': ticker, 'historical': historical, 'actual': actual}


# ─── Prompt ────────────────────────────────────────────────────────────────────

def create_forecast_prompt(ticker: str, historical_data: dict) -> str:
    years = sorted(historical_data.keys())
    prompt = f"Forecast 2024 financials for {ticker} based on {years[0]}-{years[-1]} data.\n\nHistorical Data:\n"
    for year in years:
        data    = historical_data[year]
        is_data = data['income_statement']
        bs_data = data['balance_sheet']
        prompt += f"\n{year}: "
        if is_data.get('revenue'):
            prompt += f"Rev ${is_data['revenue']:,.0f}, "
        if is_data.get('net_income'):
            prompt += f"NI ${is_data['net_income']:,.0f}, "
        if bs_data.get('total_assets'):
            prompt += f"Assets ${bs_data['total_assets']:,.0f}, "
        if bs_data.get('equity'):
            prompt += f"Equity ${bs_data['equity']:,.0f}"
        prompt += "\n"

    prompt += """
Task: Analyze trends and forecast 2024.

Output ONLY this JSON (NUMBERS ONLY, NO TEXT):

{
  "forecast_2024": {
    "income_statement": {
      "revenue": <number>,
      "cogs": <number>,
      "gross_profit": <number>,
      "operating_expense": <number>,
      "operating_income": <number>,
      "net_income": <number>
    },
    "balance_sheet": {
      "total_assets": <number>,
      "current_assets": <number>,
      "cash": <number>,
      "total_liabilities": <number>,
      "current_liabilities": <number>,
      "equity": <number>
    },
    "cash_flow": {
      "operating_cash_flow": <number>,
      "capex": <number>,
      "free_cash_flow": <number>
    }
  }
}

Return ONLY valid JSON with numbers. NO text fields."""
    return prompt


# ─── Single Forecast ───────────────────────────────────────────────────────────

def generate_forecast(
    client,
    company_data: dict,
    temperature: float,
    thinking_mode: str,     # "no_thinking" | "with_thinking"
    rate_limiter: RateLimiter,
    run_num: int = 0,
    debug: bool = False,
) -> dict:
    """Generate one forecast under the given thinking mode and temperature."""
    rate_limiter.wait_if_needed()

    ticker   = company_data['ticker']
    prompt   = create_forecast_prompt(ticker, company_data['historical'])

    # Select token / thinking settings
    if thinking_mode == "no_thinking":
        max_tokens     = TOKENS_NO_THINKING
        thinking_cfg   = types.ThinkingConfig(thinking_budget=BUDGET_NO_THINKING)
    else:  # "with_thinking"
        max_tokens     = TOKENS_WITH_THINKING
        thinking_cfg   = types.ThinkingConfig(thinking_budget=BUDGET_WITH_THINKING)

    try:
        t0 = time.time()
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                thinking_config=thinking_cfg,
            )
        )
        latency = time.time() - t0

        # Collect token usage if available
        thinking_tokens = 0
        output_tokens   = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            um = response.usage_metadata
            output_tokens   = getattr(um, 'candidates_token_count', 0) or 0
            thinking_tokens = getattr(um, 'thoughts_token_count', 0) or 0

        response_text = response.text.strip()

        if debug and run_num == 0:
            print(f"\n    [DEBUG] Raw response (first 400 chars):\n{response_text[:400]}\n")
            print(f"    [DEBUG] Thinking tokens: {thinking_tokens}, Output tokens: {output_tokens}")

        # Clean JSON
        start_idx = response_text.find('{')
        end_idx   = response_text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            response_text = response_text[start_idx:end_idx+1]

        forecast_data = json.loads(response_text)

        forecast_2024 = forecast_data.get('forecast_2024', forecast_data)
        income  = forecast_2024.get('income_statement', {})
        balance = forecast_2024.get('balance_sheet', {})

        return {
            'success':         True,
            'revenue':         income.get('revenue', 0),
            'net_income':      income.get('net_income', 0),
            'total_assets':    balance.get('total_assets', 0),
            'equity':          balance.get('equity', 0),
            'cash':            balance.get('cash', 0),
            'latency_s':       round(latency, 2),
            'thinking_tokens': thinking_tokens,
            'output_tokens':   output_tokens,
        }

    except Exception as e:
        print(f"    ❌ Error: {str(e)[:120]}")
        return {
            'success':         False,
            'revenue': 0, 'net_income': 0, 'total_assets': 0,
            'equity': 0, 'cash': 0,
            'latency_s': 0, 'thinking_tokens': 0, 'output_tokens': 0,
            'error': str(e),
        }


# ─── CV Helper ─────────────────────────────────────────────────────────────────

def cv(values: list) -> float:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return np.nan
    mean = np.mean(vals)
    if mean == 0:
        return np.nan
    return np.std(vals) / abs(mean) * 100


def smape(actual: float, forecast: float) -> float:
    if actual == 0 and forecast == 0:
        return 0.0
    return 100 * abs(forecast - actual) / (abs(actual) + abs(forecast))


# ─── Main Test ─────────────────────────────────────────────────────────────────

def run_thinking_comparison():
    THINKING_MODES = ["no_thinking", "with_thinking"]
    CONFIGS = [
        ("no_thinking",   0.0, TOKENS_NO_THINKING,   BUDGET_NO_THINKING),
        ("no_thinking",   0.7, TOKENS_NO_THINKING,   BUDGET_NO_THINKING),
        ("with_thinking", 0.0, TOKENS_WITH_THINKING, BUDGET_WITH_THINKING),
        ("with_thinking", 0.7, TOKENS_WITH_THINKING, BUDGET_WITH_THINKING),
    ]
    total = len(TEST_TICKERS) * len(CONFIGS) * RUNS_PER_CONFIG

    print("=" * 80)
    print("THINKING MODE COMPARISON TEST")
    print("=" * 80)
    print(f"\nModel: {config.GEMINI_MODEL}")
    print(f"Stocks: {TEST_TICKERS}")
    print(f"\nConfigurations (4 total):")
    print(f"  A. No Thinking  + Temp 0.0  (max_tokens={TOKENS_NO_THINKING})")
    print(f"  B. No Thinking  + Temp 0.7  (max_tokens={TOKENS_NO_THINKING})")
    print(f"  C. With Thinking + Temp 0.0 (max_tokens={TOKENS_WITH_THINKING}, budget={BUDGET_WITH_THINKING})")
    print(f"  D. With Thinking + Temp 0.7 (max_tokens={TOKENS_WITH_THINKING}, budget={BUDGET_WITH_THINKING})")
    print(f"\nRuns per config: {RUNS_PER_CONFIG}")
    print(f"Total forecasts: {total}")
    print("=" * 80)

    client       = genai.Client(api_key=config.GEMINI_API_KEY)
    rate_limiter = RateLimiter(calls_per_minute=config.MAX_REQUESTS_PER_MINUTE)

    all_rows = []

    for ticker in TEST_TICKERS:
        company_data = load_company_data(ticker)
        actual       = company_data['actual']

        print(f"\n{'=' * 80}")
        print(f"STOCK: {ticker}  (Original sMAPE: {ORIGINAL_SMAPE[ticker]:.2f}%)")
        print("=" * 80)

        for thinking_mode, temperature, max_tokens, t_budget in CONFIGS:
            label = (f"{'NO' if thinking_mode == 'no_thinking' else 'WITH'} THINKING"
                     f"  |  Temp {temperature}"
                     f"  |  max_tokens={max_tokens}")
            print(f"\n  [{label}]")

            forecasts = []

            for run in range(RUNS_PER_CONFIG):
                print(f"    Run {run+1}/{RUNS_PER_CONFIG}...", end=' ', flush=True)

                fc = generate_forecast(
                    client, company_data, temperature, thinking_mode,
                    rate_limiter, run_num=run, debug=True
                )

                # sMAPE calculation
                avg_smape = np.nan
                if fc['success']:
                    smapes = []
                    for metric in ['revenue', 'net_income', 'equity']:
                        a = actual.get(metric) or 0
                        f = fc.get(metric) or 0
                        if a != 0 or f != 0:
                            smapes.append(smape(a, f))
                    avg_smape = float(np.mean(smapes)) if smapes else np.nan
                    print(f"✅  (latency={fc['latency_s']:.1f}s"
                          f"  thinking_tokens={fc['thinking_tokens']}"
                          f"  output_tokens={fc['output_tokens']})")
                    forecasts.append(fc)
                else:
                    print("❌")

                all_rows.append({
                    'ticker':          ticker,
                    'thinking_mode':   thinking_mode,
                    'temperature':     temperature,
                    'max_tokens':      max_tokens,
                    'thinking_budget': t_budget,
                    'run':             run + 1,
                    'success':         fc['success'],
                    'revenue':         fc.get('revenue'),
                    'net_income':      fc.get('net_income'),
                    'equity':          fc.get('equity'),
                    'total_assets':    fc.get('total_assets'),
                    'cash':            fc.get('cash'),
                    'smape':           avg_smape,
                    'latency_s':       fc.get('latency_s', 0),
                    'thinking_tokens': fc.get('thinking_tokens', 0),
                    'output_tokens':   fc.get('output_tokens', 0),
                })

            # Per-config summary
            n = len(forecasts)
            print(f"\n    ── Summary ──────────────────────────────────────")
            print(f"    Success rate: {n}/{RUNS_PER_CONFIG}")
            if n > 0:
                rev_cv = cv([f['revenue']    for f in forecasts])
                ni_cv  = cv([f['net_income'] for f in forecasts])
                eq_cv  = cv([f['equity']     for f in forecasts])
                avg_l  = np.mean([f['latency_s']       for f in forecasts])
                avg_tt = np.mean([f['thinking_tokens']  for f in forecasts])
                avg_ot = np.mean([f['output_tokens']    for f in forecasts])
                print(f"    Revenue CV:    {rev_cv:.2f}%")
                print(f"    Net Income CV: {ni_cv:.2f}%")
                print(f"    Equity CV:     {eq_cv:.2f}%")
                print(f"    Avg Latency:   {avg_l:.1f}s")
                print(f"    Avg Thinking Tokens: {avg_tt:.0f}")
                print(f"    Avg Output Tokens:   {avg_ot:.0f}")

    # ── Save detailed results ──────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    detail_path = config.RESULTS_DIR / "thinking_comparison_detailed.csv"
    df.to_csv(detail_path, index=False)
    print(f"\n\nDetailed results saved to: {detail_path}")

    # ── Build summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    header = (f"{'Ticker':<6} {'Mode':<15} {'Temp':<5} "
              f"{'OK%':<6} {'RevCV%':<8} {'NI_CV%':<8} {'EqCV%':<8} "
              f"{'AvgCV%':<8} {'Latency':>8} {'ThinkTok':>10} {'OutTok':>8}")
    print(header)
    print("-" * 80)

    summary_rows = []

    for ticker in TEST_TICKERS:
        td = df[df['ticker'] == ticker]
        for thinking_mode, temperature, max_tokens, t_budget in CONFIGS:
            sub  = td[(td['thinking_mode'] == thinking_mode) & (td['temperature'] == temperature)]
            ok   = sub[sub['success'] == True]
            n    = len(ok)
            pct  = n / RUNS_PER_CONFIG * 100

            rev_cv = cv(ok['revenue'].tolist())   if n > 0 else np.nan
            ni_cv  = cv(ok['net_income'].tolist()) if n > 0 else np.nan
            eq_cv  = cv(ok['equity'].tolist())     if n > 0 else np.nan
            avg_cv_val = float(np.nanmean([rev_cv, ni_cv, eq_cv]))

            avg_lat  = ok['latency_s'].mean()       if n > 0 else np.nan
            avg_tt   = ok['thinking_tokens'].mean() if n > 0 else 0
            avg_ot   = ok['output_tokens'].mean()   if n > 0 else np.nan

            mode_short = "NoThink" if thinking_mode == "no_thinking" else "Thinking"
            row_str = (f"{ticker:<6} {mode_short:<15} {temperature:<5.1f} "
                       f"{pct:<6.0f} {rev_cv:<8.2f} {ni_cv:<8.2f} {eq_cv:<8.2f} "
                       f"{avg_cv_val:<8.2f} {avg_lat:>8.1f} {avg_tt:>10.0f} {avg_ot:>8.0f}")
            print(row_str)

            summary_rows.append({
                'ticker':          ticker,
                'thinking_mode':   thinking_mode,
                'temperature':     temperature,
                'max_tokens':      max_tokens,
                'thinking_budget': t_budget,
                'success_rate':    pct,
                'revenue_cv':      rev_cv,
                'net_income_cv':   ni_cv,
                'equity_cv':       eq_cv,
                'avg_cv':          avg_cv_val,
                'avg_latency_s':   avg_lat,
                'avg_thinking_tokens': avg_tt,
                'avg_output_tokens':   avg_ot,
                'original_smape':  ORIGINAL_SMAPE[ticker],
            })

        print()  # blank line between tickers

    summary_df = pd.DataFrame(summary_rows)
    summary_path = config.RESULTS_DIR / "thinking_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    # ── Cross-mode aggregate comparison ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("AGGREGATE: No Thinking vs With Thinking")
    print("=" * 80)

    for mode in ["no_thinking", "with_thinking"]:
        mode_df  = summary_df[summary_df['thinking_mode'] == mode]
        label    = "No Thinking  " if mode == "no_thinking" else "With Thinking"
        avg_ok   = mode_df['success_rate'].mean()
        avg_cv_m = mode_df['avg_cv'].mean()
        avg_lat  = mode_df['avg_latency_s'].mean()
        avg_tt   = mode_df['avg_thinking_tokens'].mean()
        avg_ot   = mode_df['avg_output_tokens'].mean()
        print(f"  {label}: "
              f"Success={avg_ok:.0f}%  "
              f"AvgCV={avg_cv_m:.2f}%  "
              f"Latency={avg_lat:.1f}s  "
              f"ThinkTok={avg_tt:.0f}  "
              f"OutTok={avg_ot:.0f}")

    print("\n" + "=" * 80)
    print("AGGREGATE: Temperature 0.0 vs 0.7")
    print("=" * 80)
    for temp in [0.0, 0.7]:
        temp_df  = summary_df[summary_df['temperature'] == temp]
        avg_ok   = temp_df['success_rate'].mean()
        avg_cv_m = temp_df['avg_cv'].mean()
        avg_lat  = temp_df['avg_latency_s'].mean()
        print(f"  Temp {temp}: "
              f"Success={avg_ok:.0f}%  "
              f"AvgCV={avg_cv_m:.2f}%  "
              f"Latency={avg_lat:.1f}s")

    generate_comparison_report(summary_df)
    return df, summary_df


# ─── Report ────────────────────────────────────────────────────────────────────

def generate_comparison_report(summary_df: pd.DataFrame):
    nt   = summary_df[summary_df['thinking_mode'] == 'no_thinking']
    wt   = summary_df[summary_df['thinking_mode'] == 'with_thinking']

    nt_ok    = nt['success_rate'].mean()
    wt_ok    = wt['success_rate'].mean()
    nt_cv    = nt['avg_cv'].mean()
    wt_cv    = wt['avg_cv'].mean()
    nt_lat   = nt['avg_latency_s'].mean()
    wt_lat   = wt['avg_latency_s'].mean()
    nt_ttok  = nt['avg_thinking_tokens'].mean()
    wt_ttok  = wt['avg_thinking_tokens'].mean()

    report = f"""
THINKING MODE COMPARISON REPORT
{'=' * 80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model:     {config.GEMINI_MODEL}

CONFIGURATIONS
  No Thinking:   thinking_budget=0,    max_output_tokens={TOKENS_NO_THINKING}
  With Thinking: thinking_budget={BUDGET_WITH_THINKING}, max_output_tokens={TOKENS_WITH_THINKING}

STOCKS TESTED:
  REGN (Best  - sMAPE: {ORIGINAL_SMAPE['REGN']:.2f}%)
  AMZN (Mid   - sMAPE: {ORIGINAL_SMAPE['AMZN']:.2f}%)
  MET  (Poor  - sMAPE: {ORIGINAL_SMAPE['MET']:.2f}%)

─────────────────────────────────────────────────────
OVERALL RESULTS (averaged across stocks and temps)
─────────────────────────────────────────────────────
Metric                 No Thinking    With Thinking   Delta
Success Rate           {nt_ok:>8.1f}%    {wt_ok:>8.1f}%    {wt_ok-nt_ok:>+8.1f}%
Avg CV (consistency)   {nt_cv:>8.2f}%    {wt_cv:>8.2f}%    {wt_cv-nt_cv:>+8.2f}%
Avg Latency            {nt_lat:>8.1f}s    {wt_lat:>8.1f}s    {wt_lat-nt_lat:>+8.1f}s
Avg Thinking Tokens    {nt_ttok:>8.0f}     {wt_ttok:>8.0f}

─────────────────────────────────────────────────────
PER-STOCK / PER-CONFIG BREAKDOWN
─────────────────────────────────────────────────────
"""

    header = (f"{'Ticker':<6} {'Mode':<15} {'Temp':<5} "
              f"{'Success%':<10} {'AvgCV%':<9} {'Latency':>8}")
    report += header + "\n" + "-" * 60 + "\n"

    for _, row in summary_df.iterrows():
        mode_s = "NoThink" if row['thinking_mode'] == 'no_thinking' else "Thinking"
        report += (f"{row['ticker']:<6} {mode_s:<15} {row['temperature']:<5.1f} "
                   f"{row['success_rate']:<10.0f} {row['avg_cv']:<9.2f} "
                   f"{row['avg_latency_s']:>8.1f}\n")

    # Interpretation
    better_mode = "No Thinking" if nt_cv < wt_cv else "With Thinking"
    faster_mode = "No Thinking" if nt_lat < wt_lat else "With Thinking"

    report += f"""
─────────────────────────────────────────────────────
KEY FINDINGS
─────────────────────────────────────────────────────

1. Consistency (lower CV = better):
   {"✅ No Thinking is more consistent" if nt_cv < wt_cv else "✅ With Thinking is more consistent"}
   No Thinking CV={nt_cv:.2f}%  vs  With Thinking CV={wt_cv:.2f}%
   Delta: {abs(wt_cv - nt_cv):.2f} percentage points

2. Success Rate (token usage):
   {"✅ No Thinking has fewer failures" if nt_ok >= wt_ok else "✅ With Thinking has fewer failures"}
   No Thinking={nt_ok:.0f}%  vs  With Thinking={wt_ok:.0f}%
   (With Thinking uses 2× token limit, reducing truncation)

3. Latency:
   {"✅ No Thinking is faster" if nt_lat <= wt_lat else "⚠️ No Thinking is slower (unexpected)"}
   No Thinking={nt_lat:.1f}s  vs  With Thinking={wt_lat:.1f}s
   Overhead: {abs(wt_lat - nt_lat):.1f}s per forecast

4. Thinking Token Usage:
   No Thinking thinking tokens:   {nt_ttok:.0f} (should be 0)
   With Thinking thinking tokens: {wt_ttok:.0f}

─────────────────────────────────────────────────────
PRODUCTION RECOMMENDATION
─────────────────────────────────────────────────────

For banking lending systems:

  Accuracy-focused:  {"Use With Thinking (higher token limit reduces failures)" if wt_ok > nt_ok else "Use No Thinking (equal or better success)"}
  Speed-focused:     Use No Thinking (faster, lower latency)
  Consistency:       Use {better_mode} (lower CV)
  Cost-sensitive:    Use No Thinking (fewer thinking tokens consumed)

Suggested production config:
  temperature=0.0 + no_thinking + max_tokens=8192  → highest consistency
  temperature=0.0 + with_thinking + max_tokens=16384 → higher success on large companies

{'=' * 80}
"""

    report_path = config.RESULTS_DIR / "thinking_comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start = time.time()
    df, summary = run_thinking_comparison()
    elapsed = time.time() - start
    ok = df['success'].sum()
    print(f"\n✅ Thinking comparison test complete!")
    print(f"   Total time:       {elapsed/60:.1f} minutes")
    print(f"   Total forecasts:  {len(df)}")
    print(f"   Overall success:  {ok}/{len(df)} ({ok/len(df)*100:.1f}%)")